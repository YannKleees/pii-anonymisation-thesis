"""Document ingestion for DOCX and PDF sources."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docx import Document
from document_generator.marker import iter_body_paragraphs, iter_headers_footers, normalize_text

from ..shared import Block, DocumentRecord, ImageArtifact, normalize_region

EMBED_KEY = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
SUPPORTED_INPUT_SUFFIXES = {".docx", ".pdf"}
DEFAULT_PDF_RENDER_DPI = 200


class PdfBackendUnavailable(RuntimeError):
    """Raised when no optional PDF backend is available for a requested action."""


def iter_all_paragraphs(doc):
    """Yield all paragraphs from headers, footers, and body."""
    yield from iter_headers_footers(doc)
    yield from iter_body_paragraphs(doc)


def extract(para, region, path, seen):
    """Extract embedded image artifacts from paragraph."""
    images = []
    for idx, blip in enumerate(para._element.xpath('.//*[local-name()="blip"]')):
        rid = blip.get(EMBED_KEY)
        if not rid or rid not in para.part.related_parts:
            continue
        part = para.part.related_parts[rid]
        blob = getattr(part, "blob", None)
        media = str(getattr(part, "partname", "")).lstrip("/")
        if not blob or not media.startswith("word/media/"):
            continue
        key = (path, media)
        if key in seen:
            continue
        seen.add(key)
        images.append(
            ImageArtifact(
                image_id=f"{path}:image:{idx}",
                region=region,
                path=path,
                media_path=media,
                blob=blob,
                metadata={"document_type": "docx"},
            )
        )
    return images


def _parse_docx(document_path: Path):
    """Parse DOCX and extract blocks, images, full text."""
    doc = Document(str(document_path))
    blocks = []
    images = []
    full = []
    cursor = 0
    seen = set()

    for raw, path, para in iter_all_paragraphs(doc):
        source = para.text or ""
        text = normalize_text(source)
        region = normalize_region(raw)
        if full:
            full.append("\n")
            cursor += 1
        start = cursor
        full.append(text)
        cursor += len(text)
        blocks.append(
            Block(
                region=region,
                path=path,
                text=text,
                raw_text=source,
                start=start,
                end=cursor,
                metadata={"document_type": "docx"},
            )
        )
        images.extend(extract(para, region, path, seen))

    return DocumentRecord(
        doc_id=document_path.stem,
        document_path=document_path,
        blocks=blocks,
        images=images,
        metadata={"full_text": "".join(full), "document_type": "docx"},
    )


def _load_fitz():
    """Load PyMuPDF module or raise if unavailable."""
    try:
        import fitz
    except Exception as exc:
        raise PdfBackendUnavailable(
            "PDF ingestion needs PyMuPDF (`pip install PyMuPDF`) for cross-platform text extraction and rendering."
        ) from exc
    return fitz


def text_blocks(page, page_index: int) -> Iterable[Block]:
    """Extract text blocks from PDF page."""
    for block_index, item in enumerate(page.get_text("blocks") or []):
        if len(item) < 5:
            continue
        x0, y0, x1, y1, raw = item[:5]
        source = str(raw or "")
        text = normalize_text(source).strip("\n")
        if not text.strip():
            continue
        yield Block(
            region="body",
            path=f"pdf:page:{page_index + 1}:block:{block_index}",
            text=text,
            raw_text=source,
            metadata={
                "document_type": "pdf",
                "page_index": page_index,
                "page_number": page_index + 1,
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "has_text_offsets": True,
            },
        )


def _render_page_image(page, page_index: int, document_path: Path, dpi: int) -> ImageArtifact:
    """Render PDF page to image at specified DPI."""
    fitz = _load_fitz()
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    rect = page.rect
    return ImageArtifact(
        image_id=f"{document_path.stem}:pdf_page:{page_index + 1}",
        region="body",
        path=f"pdf:page:{page_index + 1}",
        media_path=f"pdf/pages/page_{page_index + 1:04d}.png",
        blob=pixmap.tobytes("png"),
        metadata={
            "document_type": "pdf",
            "page_index": page_index,
            "page_number": page_index + 1,
            "image_size": (int(pixmap.width), int(pixmap.height)),
            "pdf_page_rect": (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
            "render_dpi": int(dpi),
        },
    )


def _parse_pdf(document_path: Path, *, render_dpi: int = DEFAULT_PDF_RENDER_DPI, ocr_all_pages: bool = False):
    """Parse PDF and extract text blocks and rendered images."""
    fitz = _load_fitz()
    blocks = []
    images = []
    full = []
    cursor = 0
    page_text_counts = []

    with fitz.open(str(document_path)) as pdf:
        for page_index, page in enumerate(pdf):
            page_blocks = list(text_blocks(page, page_index))
            page_text_len = sum(len(block.text.strip()) for block in page_blocks)
            page_text_counts.append(page_text_len)

            for block in page_blocks:
                if full:
                    full.append("\n")
                    cursor += 1
                start = cursor
                full.append(block.text)
                cursor += len(block.text)
                block.start = start
                block.end = cursor
                blocks.append(block)

            if ocr_all_pages or page_text_len == 0:
                images.append(_render_page_image(page, page_index, document_path, render_dpi))

        page_count = int(pdf.page_count)

    return DocumentRecord(
        doc_id=document_path.stem,
        document_path=document_path,
        blocks=blocks,
        images=images,
        metadata={
            "full_text": "".join(full),
            "document_type": "pdf",
            "page_count": page_count,
            "pdf_text_pages": sum(1 for count in page_text_counts if count > 0),
            "pdf_ocr_pages": len(images),
            "pdf_render_dpi": int(render_dpi),
            "pdf_backend": "pymupdf",
        },
    )


def parse_document(document_path: Path, *, pdf_render_dpi: int | None = None, pdf_ocr_all_pages: bool = False):
    """Parse DOCX or PDF document; dispatch to appropriate parser."""
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()
    if suffix == ".docx":
        return _parse_docx(document_path)
    if suffix == ".pdf":
        return _parse_pdf(
            document_path,
            render_dpi=int(pdf_render_dpi or DEFAULT_PDF_RENDER_DPI),
            ocr_all_pages=bool(pdf_ocr_all_pages),
        )
    raise ValueError(
        f"Unsupported document type '{document_path.suffix}'. Supported inputs: {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}."
    )


def normalise_blocks(document):
    """Normalize text in blocks, recalculate offsets, update full_text metadata."""
    blocks = []
    full = []
    cursor = 0

    for block in document.blocks:
        text = normalize_text(block.text)
        if full:
            full.append("\n")
            cursor += 1
        start = cursor
        full.append(text)
        cursor += len(text)
        blocks.append(
            Block(
                region=block.region,
                path=block.path,
                text=text,
                raw_text=block.raw_text or block.text,
                start=start,
                end=cursor,
                metadata=dict(block.metadata or {}),
            )
        )

    document.blocks = blocks
    document.metadata["full_text"] = "".join(full)
    return document
