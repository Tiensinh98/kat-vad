"""Convert PDF papers in a folder to Markdown using pymupdf4llm."""

import argparse
import sys
from pathlib import Path

import pymupdf4llm


def convert_all(input_dir: Path, output_dir: Path, overwrite: bool = False) -> int:
    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        return 0

    failures = 0
    for pdf in pdfs:
        out_path = output_dir / pdf.relative_to(input_dir).with_suffix(".md")
        if out_path.exists() and not overwrite:
            print(f"skip  {pdf.name} (already converted)")
            continue
        try:
            text = pymupdf4llm.to_markdown(str(pdf))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"ok    {pdf.name} -> {out_path}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {pdf.name}: {e}", file=sys.stderr)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=Path("papers"), help="folder with PDFs (default: papers)")
    parser.add_argument("-o", "--output", type=Path, default=Path("markdown"), help="output folder (default: markdown)")
    parser.add_argument("-f", "--overwrite", action="store_true", help="re-convert files that already exist")
    args = parser.parse_args()

    failures = convert_all(args.input, args.output, args.overwrite)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
