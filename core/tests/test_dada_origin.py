"""Tests for the DADA-2000 **original** release preprocessor and its xlsx reader.

Covers `core.data.xlsx` (sheet detection, the three cell encodings) and
`core.data.dada_origin` (annotation parsing, the absolute-index label
convention, the type-stratified source split, the T2 window build and the
symlink farm).

No downloads and no ``openpyxl``: the fixture workbook is written with
``zipfile`` and hand-rolled OOXML, mirroring what the reader has to survive --
a decoy sheet whose *name* suggests it holds the table, a non-breaking space
inside a header, and both shared and inline strings.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from core import constants
from core.data import dada_origin
from core.data.dataset_files import TEST_IDS_FILENAME, TRAIN_IDS_FILENAME
from core.data.video_io import list_frame_folders
from core.data.xlsx import find_sheet, normalize_header, read_sheets

STRIDE = 2
FRAME_SIZE = 8
SEED = 11
#: 64 raw frames -> 32 sampled at STRIDE 2, so a 16-frame window at hop 8 gives
#: three windows per clip -- enough for the T2 shape (normal prefix, then the
#: anomaly) inside a fixture that stays fast.
RAW_FRAMES = 64
ANOMALY_START = 40
ANOMALY_END = 56

HEADERS = (
    "video",
    "weather(sunny,rainy,snowy,foggy)1-4",
    "light(day,night)1-2",
    "scenes(highway,tunnel,mountain,urban,rural)1-5",
    # NBSP inside the header, exactly as the real workbook ships it.
    "linear(arterials,curve,intersection,t-junction,ramp)\xa01-5",
    "type",
    "whether an accident occurred (1/0)",
    "abnormal start frame",
    "accident frame",
    "abnormal end frame",
    "total frames",
    "texts",
    "causes",
    "measures",
)

#: (type, video). Video numbers repeat ACROSS types on purpose -- the bare
#: on-disk folder name is the video number alone, so these five folders collapse
#: to three ids unless the (type, video) key is used (lesson C26).
CLIPS = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]


# --------------------------------------------------------------------------
# fixture workbook
# --------------------------------------------------------------------------
def _sheet_xml(rows: list[list[str]]) -> str:
    """Rows of inline strings -- no shared-string table, the other cell encoding."""
    out = ['<?xml version="1.0"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
           "<sheetData>"]
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{_ref(c, r)}" t="inlineStr"><is><t>{_escape(value)}</t></is></c>'
            for c, value in enumerate(row)
            if value != ""
        )
        out.append(f'<row r="{r}">{cells}</row>')
    out.append("</sheetData></worksheet>")
    return "".join(out)


def _ref(column: int, row: int) -> str:
    letters = ""
    column += 1
    while column:
        column, rem = divmod(column - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"{letters}{row}"


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[str]]]]) -> Path:
    """A minimal but real ``.xlsx``: workbook part, rels, one part per sheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    doc_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="xml" '
            'ContentType="application/xml"/></Types>',
        )
        entries = "".join(
            f'<sheet name="{_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(sheets, start=1)
        )
        archive.writestr(
            "xl/workbook.xml",
            f'<?xml version="1.0"?><workbook xmlns="{main}" '
            f'xmlns:r="{doc_rel}"><sheets>{entries}</sheets></workbook>',
        )
        rels = "".join(
            f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" '
            f'Type="{doc_rel}/worksheet"/>'
            for i in range(1, len(sheets) + 1)
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<?xml version="1.0"?><Relationships xmlns="{pkg_rel}">{rels}</Relationships>',
        )
        for i, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))
    return path


def _replace_in_zip(path: Path, parts: dict[str, str]) -> None:
    """Rewrite an archive with ``parts`` replaced -- appending would duplicate names."""
    with zipfile.ZipFile(path) as archive:
        kept = {n: archive.read(n) for n in archive.namelist() if n not in parts}
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in kept.items():
            archive.writestr(name, payload)
        for name, text in parts.items():
            archive.writestr(name, text)


def _row(type_id: int, video: int, *, flag: str = "1", start: int = ANOMALY_START,
         end: int = ANOMALY_END, total: int = RAW_FRAMES) -> list[str]:
    return [
        str(video), "1", "1", "4", "2", str(type_id), flag,
        str(start), str((start + end) // 2), str(end), str(total),
        f"[CLS]incident of type {type_id}[SEP]", f"cause {type_id}", f"measure {type_id}",
    ]


def write_annotation(path: Path, rows: list[list[str]] | None = None) -> Path:
    """The fixture workbook: a DECOY sheet first, the real table second.

    The real file's sheets are named the opposite of their contents, so a reader
    that trusts sheet order or sheet names parses the taxonomy and reports zero
    clips. The decoy is listed first for exactly that reason.
    """
    body = rows if rows is not None else [_row(t, v) for t, v in CLIPS]
    decoy = [["id", "sentence"], ["1", "a vehicle turns left"], ["2", "a pedestrian crosses"]]
    return write_xlsx(path, [("Sheet1", decoy), ("text", [list(HEADERS), *body])])


def write_frames(frames_dir: Path, clips=CLIPS, frames: dict[tuple[int, int], int] | None = None):
    from torchvision.io import write_png

    generator = torch.Generator().manual_seed(SEED)
    for type_id, video in clips:
        folder = (
            dada_origin.clip_folder(frames_dir, type_id, video)
            / constants.DADA_ORIGIN_IMAGES_SUBDIR
        )
        folder.mkdir(parents=True, exist_ok=True)
        count = (frames or {}).get((type_id, video), RAW_FRAMES)
        for index in range(count):
            image = (torch.rand(3, FRAME_SIZE, FRAME_SIZE, generator=generator) * 255).to(
                torch.uint8
            )
            write_png(image, str(folder / f"{index:06d}.png"))
    return frames_dir


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    annotation = write_annotation(tmp_path / "anno.xlsx")
    frames = write_frames(tmp_path / "frames")
    return {"annotation": annotation, "frames": frames, "out": tmp_path / "out"}


# --------------------------------------------------------------------------
class TestXlsxReader:
    def test_sheet_is_detected_by_columns_not_by_name(self, tmp_path: Path) -> None:
        path = write_annotation(tmp_path / "a.xlsx")
        assert [s.name for s in read_sheets(path)] == ["Sheet1", "text"]
        # the decoy is named "Sheet1" and comes first; the table is named "text"
        assert find_sheet(path, dada_origin.REQUIRED_COLUMNS).name == "text"

    def test_non_breaking_space_in_a_header_still_matches(self, tmp_path: Path) -> None:
        sheet = find_sheet(write_annotation(tmp_path / "a.xlsx"), dada_origin.REQUIRED_COLUMNS)
        nbsp_header = "linear(arterials,curve,intersection,t-junction,ramp)\xa01-5"
        assert normalize_header(nbsp_header) in sheet.header
        assert sheet.column(nbsp_header) == HEADERS.index(nbsp_header)

    def test_missing_column_raises_and_lists_every_sheet(self, tmp_path: Path) -> None:
        path = write_xlsx(tmp_path / "a.xlsx", [("only", [["a", "b"], ["1", "2"]])])
        with pytest.raises(ValueError, match="No sheet in") as excinfo:
            find_sheet(path, dada_origin.REQUIRED_COLUMNS)
        assert "[only]" in str(excinfo.value)

    def test_shared_strings_are_resolved(self, tmp_path: Path) -> None:
        """The other cell encoding: a <c t="s"> pointing into sharedStrings.xml."""
        path = tmp_path / "shared.xlsx"
        write_xlsx(path, [("s", [["h"]])])
        main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        sheet_xml = (
            f'<?xml version="1.0"?><worksheet xmlns="{main}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>1</v></c></row>'
            "</sheetData></worksheet>"
        )
        strings = (
            f'<?xml version="1.0"?><sst xmlns="{main}"><si><t>type</t></si>'
            "<si><t>42</t></si></sst>"
        )
        _replace_in_zip(path, {"xl/worksheets/sheet1.xml": sheet_xml,
                               "xl/sharedStrings.xml": strings})
        sheet = read_sheets(path)[0]
        assert sheet.header == ["type"]
        assert sheet.rows == [{0: "42"}]


class TestParseAnnotation:
    def test_parses_every_accident_row_with_its_meta_columns(self, corpus) -> None:
        rows = dada_origin.parse_annotation(corpus["annotation"])
        assert [(r.type_id, r.video) for r in rows] == CLIPS
        assert rows[0].attributes["causes"] == "cause 1"
        assert rows[0].attributes["linear"] == "2"

    def test_rows_without_the_accident_flag_are_skipped(self, tmp_path: Path) -> None:
        path = write_annotation(
            tmp_path / "a.xlsx",
            [_row(1, 1), _row(1, 2, flag="0"), _row(2, 1)],
        )
        assert len(dada_origin.parse_annotation(path)) == 2

    def test_a_window_outside_the_clip_is_skipped(self, tmp_path: Path) -> None:
        path = write_annotation(
            tmp_path / "a.xlsx", [_row(1, 1), _row(1, 2, start=90, end=99)]
        )
        assert [r.video for r in dada_origin.parse_annotation(path)] == [1]

    def test_a_duplicate_type_video_key_raises(self, tmp_path: Path) -> None:
        path = write_annotation(tmp_path / "a.xlsx", [_row(1, 1), _row(1, 1)])
        with pytest.raises(ValueError, match="duplicate annotation row"):
            dada_origin.parse_annotation(path)

    def test_ids_are_unique_where_bare_folder_names_collide(self, corpus) -> None:
        """Lesson C26: the on-disk folder name is the video number alone."""
        rows = dada_origin.parse_annotation(corpus["annotation"])
        assert len({r.video for r in rows}) < len(rows)          # bare names collide
        assert len({r.video_id for r in rows}) == len(rows)      # ours do not


class TestAbsoluteSpan:
    """Plan §3.1 -- the label convention, and the bug it exists to avoid."""

    def test_span_is_normalized_against_the_on_disk_count(self) -> None:
        row = dada_origin.OriginRow(1, 1, 30, 50, 60, total_frames=100, attributes={})
        record = dada_origin.make_record(row, frames_on_disk=100)
        assert record is not None
        assert record.span == (0.30, 0.60)

    def test_a_trimmed_clip_keeps_its_anomaly_at_the_end(self) -> None:
        """``t05_v040``: on-disk equals the anomaly's end, exactly.

        Under the fraction convention (``span`` over the *annotation's* total) the
        window rescales by ``D/A`` and lands in the middle of the clip. Absolute
        indices keep it where it belongs -- at the end.
        """
        row = dada_origin.OriginRow(5, 40, 285, 300, 382, total_frames=482, attributes={})
        record = dada_origin.make_record(row, frames_on_disk=382)
        assert record is not None
        span = record.span
        assert span == (285 / 382, 1.0)
        fraction_convention = (285 / 482, 382 / 482)
        assert span is not None and span[1] - fraction_convention[1] > 0.20

    def test_a_clip_trimmed_before_its_anomaly_is_dropped(self) -> None:
        row = dada_origin.OriginRow(1, 1, 300, 310, 320, total_frames=400, attributes={})
        assert dada_origin.make_record(row, frames_on_disk=250) is None

    def test_labels_land_on_the_frames_the_annotation_named(self, corpus) -> None:
        from core.data.dada import sampled_frame_labels

        rows = dada_origin.parse_annotation(corpus["annotation"])
        record = dada_origin.make_record(rows[0], RAW_FRAMES)
        assert record is not None
        labels = sampled_frame_labels(record, STRIDE)
        positive = [i for i, v in enumerate(labels) if v]
        assert positive[0] == ANOMALY_START // STRIDE
        assert positive[-1] == ANOMALY_END // STRIDE - 1


class TestResolveRecords:
    def test_missing_frames_raise_unless_allowed(self, corpus) -> None:
        rows = dada_origin.parse_annotation(corpus["annotation"])
        counts = dada_origin.frame_census(rows, corpus["frames"])
        counts.pop(rows[0].video_id)
        with pytest.raises(ValueError, match="no readable frames"):
            dada_origin.resolve_records(rows, counts)
        assert len(dada_origin.resolve_records(rows, counts, allow_missing=True)) == len(rows) - 1

    def test_an_empty_folder_is_not_evidence_of_data(self, corpus) -> None:
        """Lesson C10 -- a created-but-empty folder must not enter the census."""
        empty = (
            dada_origin.clip_folder(corpus["frames"], 9, 9)
            / constants.DADA_ORIGIN_IMAGES_SUBDIR
        )
        empty.mkdir(parents=True)
        rows = dada_origin.parse_annotation(corpus["annotation"])
        assert dada_origin.video_id(9, 9) not in dada_origin.frame_census(rows, corpus["frames"])


class TestSplit:
    def test_stratifies_by_type_and_is_deterministic(self, corpus) -> None:
        rows = dada_origin.parse_annotation(corpus["annotation"])
        counts = dada_origin.frame_census(rows, corpus["frames"])
        records = dada_origin.resolve_records(rows, counts)
        rows_by_id = {r.video_id: r for r in rows}

        train, test = dada_origin.split_by_type(records, rows_by_id, seed=SEED, test_ratio=0.5)
        again, _ = dada_origin.split_by_type(records, rows_by_id, seed=SEED, test_ratio=0.5)
        assert [r.video_id for r in train] == [r.video_id for r in again]
        assert not {r.video_id for r in train} & {r.video_id for r in test}
        # every type with two clips contributes one to each side
        for type_id in (1, 2):
            assert any(rows_by_id[r.video_id].type_id == type_id for r in test)
            assert any(rows_by_id[r.video_id].type_id == type_id for r in train)

    def test_an_empty_side_raises(self, corpus) -> None:
        rows = dada_origin.parse_annotation(corpus["annotation"])
        counts = dada_origin.frame_census(rows, corpus["frames"])
        records = dada_origin.resolve_records(rows, counts)
        with pytest.raises(ValueError, match="one side is empty"):
            dada_origin.split_by_type(
                records, {r.video_id: r for r in rows}, seed=SEED, test_ratio=0.0
            )


class TestPreprocess:
    @staticmethod
    def _build(corpus, **kwargs):
        return dada_origin.preprocess(
            annotation=corpus["annotation"],
            frames_dir=corpus["frames"],
            out_dir=corpus["out"],
            stride=STRIDE,
            seed=SEED,
            test_ratio=0.4,
            window_length=16,
            window_stride=8,
            **kwargs,
        )

    def test_writes_the_five_windowed_files(self, corpus) -> None:
        self._build(corpus)
        for name in (
            constants.LABELS_TRAIN_FILENAME, constants.FRAME_LABELS_TEST_FILENAME,
            constants.DEFS_FILENAME, constants.META_FILENAME, constants.WINDOWS_FILENAME,
            TRAIN_IDS_FILENAME, TEST_IDS_FILENAME,
        ):
            assert (corpus["out"] / name).is_file(), name

    def test_negatives_come_from_inside_the_accident_videos(self, corpus) -> None:
        """The whole point of T2 -- there is no normal source clip anywhere."""
        self._build(corpus)
        labels = json.loads((corpus["out"] / constants.LABELS_TRAIN_FILENAME).read_text())
        assert set(labels.values()) == {0, 1}
        meta = json.loads((corpus["out"] / constants.META_FILENAME).read_text())
        assert all(entry["source_is_abnormal"] for entry in meta.values())

    def test_no_source_clip_appears_in_both_splits(self, corpus) -> None:
        self._build(corpus)
        windows = json.loads((corpus["out"] / constants.WINDOWS_FILENAME).read_text())
        meta = json.loads((corpus["out"] / constants.META_FILENAME).read_text())
        by_split: dict[str, set[str]] = {"train": set(), "test": set()}
        for wid, entry in meta.items():
            by_split[entry["split"]].add(windows[wid]["source"])
        assert not by_split["train"] & by_split["test"]

    def test_meta_carries_the_annotation_columns_and_labels_do_not(self, corpus) -> None:
        self._build(corpus)
        meta = json.loads((corpus["out"] / constants.META_FILENAME).read_text())
        entry = next(iter(meta.values()))
        for key in dada_origin.META_KEYS:
            assert key in entry, key
        assert entry["texts"].startswith("[CLS]")
        labels = json.loads((corpus["out"] / constants.LABELS_TRAIN_FILENAME).read_text())
        assert all(isinstance(v, int) for v in labels.values())

    def test_extra_meta_cannot_overwrite_a_corpus_defining_field(self, corpus) -> None:
        self._build(corpus)
        meta = json.loads((corpus["out"] / constants.META_FILENAME).read_text())
        entry = next(iter(meta.values()))
        # "type" is an annotation column; total_frames is ours and must survive
        assert entry["total_frames"] == RAW_FRAMES
        assert entry["sampled_frames"] == 16

    def test_the_census_route_reproduces_the_disk_route(self, corpus, tmp_path: Path) -> None:
        self._build(corpus)
        from_disk = json.loads((corpus["out"] / constants.META_FILENAME).read_text())

        rows = dada_origin.parse_annotation(corpus["annotation"])
        counts = dada_origin.frame_census(rows, corpus["frames"])
        out2 = tmp_path / "out2"
        dada_origin.preprocess(
            annotation=corpus["annotation"], out_dir=out2, counts=counts,
            stride=STRIDE, seed=SEED, test_ratio=0.4, window_length=16, window_stride=8,
        )
        assert json.loads((out2 / constants.META_FILENAME).read_text()) == from_disk

    def test_passing_both_or_neither_frame_source_raises(self, corpus) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            dada_origin.preprocess(
                annotation=corpus["annotation"], out_dir=corpus["out"],
                frames_dir=corpus["frames"], counts={"x": 1},
            )
        with pytest.raises(ValueError, match="exactly one"):
            dada_origin.preprocess(annotation=corpus["annotation"], out_dir=corpus["out"])

    def test_dry_run_writes_nothing(self, corpus) -> None:
        self._build(corpus, dry_run=True)
        assert not corpus["out"].exists()


class TestFlatFramesDir:
    def test_links_at_the_images_level_so_the_extractor_can_walk_them(self, corpus) -> None:
        """Lesson C26 + the pathlib symlink guard measured in Phase 1."""
        flat = corpus["out"].parent / "flat"
        dada_origin.preprocess(
            annotation=corpus["annotation"], frames_dir=corpus["frames"],
            out_dir=corpus["out"], stride=STRIDE, seed=SEED, test_ratio=0.4,
            window_length=16, window_stride=8, flat_frames_dir=flat,
        )
        folders = list_frame_folders(flat)
        assert len(folders) == len(CLIPS)
        assert {f.name for f in folders} == {dada_origin.video_id(t, v) for t, v in CLIPS}
        link = flat / dada_origin.video_id(1, 1)
        assert link.is_symlink()
        assert link.resolve().name == constants.DADA_ORIGIN_IMAGES_SUBDIR

    def test_a_flat_dir_without_frames_raises(self, corpus, tmp_path: Path) -> None:
        rows = dada_origin.parse_annotation(corpus["annotation"])
        counts = dada_origin.frame_census(rows, corpus["frames"])
        with pytest.raises(ValueError, match="needs --frames-dir"):
            dada_origin.preprocess(
                annotation=corpus["annotation"], out_dir=tmp_path / "o", counts=counts,
                stride=STRIDE, test_ratio=0.4, window_length=16, window_stride=8,
                flat_frames_dir=tmp_path / "flat",
            )


class TestCli:
    def test_census_only_writes_a_census_and_no_dataset(self, corpus, tmp_path: Path) -> None:
        census = tmp_path / "counts" / "00.json"
        dada_origin.main([
            "--annotation", str(corpus["annotation"]),
            "--frames-dir", str(corpus["frames"]),
            "--out-dir", str(corpus["out"]),
            "--counts-out", str(census), "--census-only",
        ])
        assert json.loads(census.read_text()) == {
            dada_origin.video_id(t, v): RAW_FRAMES for t, v in CLIPS
        }
        assert not corpus["out"].exists()

    def test_counts_file_route_builds_without_frames(self, corpus, tmp_path: Path) -> None:
        census = tmp_path / "counts" / "00.json"
        dada_origin.main([
            "--annotation", str(corpus["annotation"]), "--frames-dir", str(corpus["frames"]),
            "--counts-out", str(census), "--census-only", "--out-dir", str(corpus["out"]),
        ])
        dada_origin.main([
            "--annotation", str(corpus["annotation"]),
            "--counts-file", str(tmp_path / "counts" / "*.json"),
            "--out-dir", str(corpus["out"]), "--stride", str(STRIDE),
            "--window-length", "16", "--window-stride", "8", "--test-ratio", "0.4",
        ])
        assert (corpus["out"] / constants.WINDOWS_FILENAME).is_file()

    def test_both_frame_sources_is_rejected(self, corpus) -> None:
        with pytest.raises(SystemExit, match="exactly one"):
            dada_origin.main([
                "--annotation", str(corpus["annotation"]),
                "--frames-dir", str(corpus["frames"]),
                "--counts-file", "x.json",
            ])


class TestCensusPass:
    """The sharded route: census + farm, with no dataset and no split."""

    def test_writes_a_census_and_a_farm_for_what_is_on_disk(self, corpus, tmp_path: Path) -> None:
        shard = write_frames(tmp_path / "shard_frames", clips=CLIPS[:2])
        census = tmp_path / "counts" / "00.json"
        farm = tmp_path / "farm_00"
        counts = dada_origin.census_pass(corpus["annotation"], shard, census, farm)
        assert set(counts) == {dada_origin.video_id(t, v) for t, v in CLIPS[:2]}
        assert json.loads(census.read_text()) == counts
        assert {f.name for f in list_frame_folders(farm)} == set(counts)

    def test_census_only_cli_builds_the_farm_too(self, corpus, tmp_path: Path) -> None:
        farm = tmp_path / "farm"
        dada_origin.main([
            "--annotation", str(corpus["annotation"]),
            "--frames-dir", str(corpus["frames"]),
            "--out-dir", str(corpus["out"]),
            "--counts-out", str(tmp_path / "c.json"),
            "--flat-frames-dir", str(farm), "--census-only",
        ])
        assert len(list_frame_folders(farm)) == len(CLIPS)
        assert not corpus["out"].exists()
