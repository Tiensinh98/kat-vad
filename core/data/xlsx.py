"""Read-only minimal ``.xlsx`` reader (stdlib only).

Exists because exactly one artifact in this project is a spreadsheet -- the
DADA-2000 **original** annotation ``dada标注.xlsx`` -- and pulling ``openpyxl``
or ``pandas`` into a tree that pins ``torch==2.4.*`` buys one file's worth of
parsing at the cost of a supply-chain surface (lesson **C4**). An ``.xlsx`` is a
zip of XML; reading cell text out of it needs three element types.

**Read-only, text-only, on purpose.** Numbers come back as the strings the file
stores, dates are not decoded and formatting is ignored -- every consumer here
casts with ``int()`` and validates. Anything richer should use a real library.

Sheets are addressed by **content**, never by name: this workbook is a
hand-maintained export whose sheet names are the opposite of their contents
(``"text"`` holds the type taxonomy, ``"Sheet1"`` holds the 1,962-row per-clip
table). :func:`find_sheet` picks the first sheet carrying every required column
and, on failure, reports every sheet's header so the mismatch is visible in one
run instead of reading as a corrupt file.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ElementTree  # nosec B405 -- local, trusted artifact
import zipfile
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_WORKBOOK_PATH = "xl/workbook.xml"
_WORKBOOK_RELS_PATH = "xl/_rels/workbook.xml.rels"
_SHARED_STRINGS_PATH = "xl/sharedStrings.xml"

_COLUMN_RE = re.compile(r"^([A-Z]+)")

#: Row = {zero-based column index: cell text}. Sparse: xlsx omits empty cells.
Row = dict[int, str]


@dataclass(frozen=True)
class Sheet:
    """One worksheet's normalized header and its data rows."""

    name: str
    header: list[str]
    rows: list[Row]

    def column(self, name: str) -> int:
        """Zero-based index of ``name`` in the normalized header."""
        return self.header.index(normalize_header(name))


def normalize_header(value: object) -> str:
    """Case- and whitespace-insensitive header key.

    The DADA workbook uses a non-breaking space inside several headers, so a
    plain ``.strip().lower()`` misses them; collapsing every run of whitespace
    (NBSP included) is what makes the lookup survive a re-export.
    """
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().lower()


def _column_index(cell_ref: str) -> int:
    """``"AB12"`` -> 27. Letters are base-26 with no zero digit."""
    match = _COLUMN_RE.match(cell_ref)
    if match is None:
        raise ValueError(f"Cell reference {cell_ref!r} has no column letters")
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """The workbook's string table; empty when the file inlines every string."""
    try:
        payload = archive.read(_SHARED_STRINGS_PATH)
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)  # nosec B314 -- local, trusted artifact
    return [
        "".join(node.text or "" for node in item.iter(f"{_MAIN_NS}t"))
        for item in root.findall(f"{_MAIN_NS}si")
    ]


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str | None:
    """Text of one ``<c>``: shared-table lookup, inline run, or literal value."""
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_MAIN_NS}t"))
    value = cell.find(f"{_MAIN_NS}v")
    if value is None or value.text is None:
        return None
    if cell.get("t") == "s":
        index = int(value.text)
        if not 0 <= index < len(shared):
            raise ValueError(f"Shared-string index {index} outside the string table")
        return shared[index]
    return value.text


def _sheet_rows(archive: zipfile.ZipFile, path: str, shared: list[str]) -> list[Row]:
    root = ElementTree.fromstring(archive.read(path))  # nosec B314 -- trusted artifact
    rows: list[Row] = []
    for row in root.iter(f"{_MAIN_NS}row"):
        cells: Row = {}
        for cell in row.findall(f"{_MAIN_NS}c"):
            ref = cell.get("r")
            if ref is None:
                continue
            text = _cell_text(cell, shared)
            if text is not None:
                cells[_column_index(ref)] = text
        rows.append(cells)
    return rows


def _sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """``[(sheet name, part path)]`` in workbook order."""
    workbook = ElementTree.fromstring(archive.read(_WORKBOOK_PATH))  # nosec B314
    rels_root = ElementTree.fromstring(archive.read(_WORKBOOK_RELS_PATH))  # nosec B314
    targets = {
        rel.get("Id"): rel.get("Target", "")
        for rel in rels_root.iter(f"{_PKG_REL_NS}Relationship")
    }
    out: list[tuple[str, str]] = []
    for sheet in workbook.iter(f"{_MAIN_NS}sheet"):
        name = sheet.get("name") or ""
        target = targets.get(sheet.get(f"{_DOC_REL_NS}id") or "")
        if not target:
            raise ValueError(f"Sheet {name!r} has no relationship target in {_WORKBOOK_RELS_PATH}")
        target = target.lstrip("/")
        out.append((name, target if target.startswith("xl/") else f"xl/{target}"))
    return out


def read_sheets(path: Path) -> list[Sheet]:
    """Every worksheet, first row taken as the (normalized) header."""
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheets: list[Sheet] = []
        for name, part in _sheet_paths(archive):
            rows = _sheet_rows(archive, part, shared)
            if not rows:
                sheets.append(Sheet(name, [], []))
                continue
            head = rows[0]
            width = max(head) + 1 if head else 0
            header = [normalize_header(head.get(i)) for i in range(width)]
            sheets.append(Sheet(name, header, rows[1:]))
    return sheets


def find_sheet(path: Path, required_columns: tuple[str, ...]) -> Sheet:
    """The first sheet carrying **every** required column, detected not named.

    Raises with every sheet's header listed: a hardcoded sheet name that stops
    matching parses zero rows and reads exactly like a corrupt file, which is
    the failure this function exists to make impossible.
    """
    sheets = read_sheets(path)
    for sheet in sheets:
        if all(normalize_header(column) in sheet.header for column in required_columns):
            LOGGER.info(
                "%s: using sheet %r (%d columns, %d data rows)",
                path.name, sheet.name, len(sheet.header), len(sheet.rows),
            )
            return sheet
    listing = "\n".join(f"  [{s.name}] {s.header}" for s in sheets)
    raise ValueError(
        f"No sheet in {path} carries all of {required_columns}.\n"
        f"Sheets found:\n{listing}"
    )


__all__ = ["Row", "Sheet", "find_sheet", "normalize_header", "read_sheets"]
