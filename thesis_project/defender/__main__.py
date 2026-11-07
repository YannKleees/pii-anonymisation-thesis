"""Command-line entry point for the defender pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..shared import Config
from .pipeline import iter_document_inputs, process_many, write_audit_jsonl


def main():
    """Parse args, process documents, and write audit log."""
    parser = argparse.ArgumentParser(description="Run the defender on DOCX and PDF files.")
    parser.add_argument("input_path", type=Path, help="DOCX/PDF file or folder")
    parser.add_argument("--output-root", type=Path, default=None, help="Output folder")
    parser.add_argument("--redact-quasi", action="store_true", help="Redact quasi-identifiers")
    parser.add_argument("--ner-model", default=None, help="NER model path or name")
    parser.add_argument("--ner-local-files-only", action="store_true", help="Use only local NER files")
    parser.add_argument("--no-pdf", action="store_true", help="Skip DOCX-to-PDF export")
    parser.add_argument("--pdf-render-dpi", type=int, default=200, help="DPI for rendering image-only PDF pages for OCR")
    parser.add_argument("--pdf-ocr-all-pages", action="store_true", help="Render every PDF page for OCR, even pages with a text layer")
    args = parser.parse_args()

    config = Config(
        output=args.output_root,
        quasi=args.redact_quasi,
        model=args.ner_model,
        local=args.ner_local_files_only,
        pdf=not args.no_pdf,
        pdf_render_dpi=args.pdf_render_dpi,
        pdf_ocr_all_pages=args.pdf_ocr_all_pages,
    )

    paths = iter_document_inputs(args.input_path)
    results = process_many(paths, config)
    output_root = args.output_root or (
        (args.input_path if args.input_path.is_dir() else args.input_path.parent) / "anonymized_documents"
    )
    audit_path = write_audit_jsonl(results, output_root)
    print(f"Processed {len(results)} document(s). Audit log written to {audit_path}")


if __name__ == "__main__":
    main()

