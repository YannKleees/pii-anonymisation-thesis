"""Writers for DOCX and PDF anonymized outputs."""
from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from docx import Document
from PIL import Image, ImageDraw
from document_generator.marker import replace_paragraph_text

from ..shared import Block, Config, DocumentRecord, RedactionAction
from .parser import PdfBackendUnavailable, iter_all_paragraphs


def by_path(plan):
    """Group text redaction actions by path and sort in reverse start order."""
    grouped = defaultdict(list)
    for action in plan:
        if action.action == "mask_text":
            grouped[action.path].append(action)
    for key in grouped:
        grouped[key].sort(key=lambda action: int(action.start or 0), reverse=True)
    return grouped


def by_media(plan):
    """Group image box redactions by DOCX media path."""
    grouped = defaultdict(list)
    for action in plan:
        if action.action == "mask_image_box":
            media = str(action.metadata.get("media_path", ""))
            if media:
                grouped[media].append(action)
    return grouped


def by_page(plan):
    """Group PDF page box redactions by page index."""
    grouped = defaultdict(list)
    for action in plan:
        if action.action != "mask_image_box":
            continue
        if action.metadata.get("document_type") != "pdf":
            continue
        page_index = action.metadata.get("page_index")
        if page_index is None and action.metadata.get("page_number") is not None:
            page_index = int(action.metadata.get("page_number")) - 1
        if page_index is not None:
            grouped[int(page_index)].append(action)
    return grouped


def text(blocks, plan):
    """Apply in-memory text replacements to parsed blocks."""
    grouped = by_path(plan)
    updated = []
    cursor = 0
    parts = []

    for block in blocks:
        value = block.text
        for action in grouped.get(block.path, []):
            start = int(action.start or 0)
            end = int(action.end or 0)
            if 0 <= start < end <= len(value):
                value = value[:start] + action.replacement + value[end:]
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(value)
        cursor += len(value)
        updated.append(
            Block(
                region=block.region,
                path=block.path,
                text=value,
                raw_text=value,
                start=start,
                end=cursor,
                metadata=dict(block.metadata or {}),
            )
        )

    return updated


def out(source, root):
    """Return output path for anonymized document."""
    root = root or (source.parent / "anonymized_documents")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{source.stem}_anonymized{source.suffix.lower()}"


def blob(data, plan):
    """Mask boxes inside a binary image payload."""
    image = Image.open(io.BytesIO(data))
    image_format = image.format
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if image_format == "PNG" else "RGB")
    draw = ImageDraw.Draw(image)
    fill = (0, 0, 0, 255) if image.mode == "RGBA" else (0, 0, 0)
    for action in plan:
        box = action.metadata.get("bbox")
        if not box:
            continue
        left, top, right, bottom = (int(value) for value in box)
        pad = 2
        draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=fill)
    fmt = (
        image_format
        or Path(str(plan[0].metadata.get("media_path", "image.png"))).suffix.lstrip(".").upper()
        or "PNG"
    )
    if fmt == "JPG":
        fmt = "JPEG"
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def apply(docx_path, plan):
    """Patch embedded DOCX media parts with box masks."""
    grouped = by_media(plan)
    if not grouped:
        return

    tmp = docx_path.with_suffix(docx_path.suffix + ".tmp")
    with zipfile.ZipFile(docx_path, "r") as src, zipfile.ZipFile(tmp, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            actions = grouped.get(info.filename)
            if actions:
                try:
                    data = blob(data, actions)
                except Exception:
                    pass
            dst.writestr(info, data)
    tmp.replace(docx_path)


def render(docx_path, config):
    """Render anonymized DOCX to PDF when LibreOffice is available."""
    if not config.pdf:
        return None

    office = str(config.office or "").strip()
    if office and Path(office).exists():
        office_cmd = office
    else:
        candidates = [
            office,
            "soffice",
            "libreoffice",
            "C:/Program Files/LibreOffice/program/soffice.exe",
            "C:/Program Files (x86)/LibreOffice/program/soffice.exe",
        ]
        office_cmd = next((item for item in candidates if item and shutil.which(item)), None)

    if not office_cmd:
        return None

    try:
        subprocess.run(
            [
                office_cmd,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(docx_path.parent),
                str(docx_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    pdf = docx_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def _write_docx(document, plan, config):
    """Write anonymized DOCX with text and image redactions."""
    actions = list(plan)
    blocks = text(document.blocks, actions)
    docx = Document(str(document.document_path))
    paras = {path: paragraph for _region, path, paragraph in iter_all_paragraphs(docx)}
    original = {block.path: block.text for block in document.blocks}
    for block in blocks:
        if block.text == original.get(block.path):
            continue
        paragraph = paras.get(block.path)
        if paragraph is not None:
            replace_paragraph_text(paragraph, block.text, preserve_non_text=True)
    output = out(document.document_path, config.output)
    docx.save(str(output))
    apply(output, actions)
    pdf = render(output, config)
    meta = {
        **dict(document.metadata or {}),
        "source_document_path": str(document.document_path),
        "document_type": "docx",
    }
    if pdf is not None:
        meta["pdf_path"] = str(pdf)
    return DocumentRecord(
        doc_id=document.doc_id,
        document_path=output,
        blocks=blocks,
        images=document.images,
        metadata=meta,
    )


def _load_fitz():
    """Load PyMuPDF or raise if unavailable."""
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise PdfBackendUnavailable(
            "PDF redaction needs PyMuPDF (`pip install PyMuPDF`)."
        ) from exc
    return fitz


def rect_from_box(fitz, action):
    """Project image bbox coordinates to PDF page coordinates."""
    box = action.metadata.get("bbox")
    image_size = action.metadata.get("image_size")
    page_rect = action.metadata.get("pdf_page_rect")
    if not box or not image_size or not page_rect:
        return None
    left, top, right, bottom = [float(value) for value in box]
    width, height = [float(value) for value in image_size]
    px0, py0, px1, py1 = [float(value) for value in page_rect]
    if width <= 0 or height <= 0:
        return None
    page_width = px1 - px0
    page_height = py1 - py0
    return fitz.Rect(
        px0 + (left / width) * page_width,
        py0 + (top / height) * page_height,
        px0 + (right / width) * page_width,
        py0 + (bottom / height) * page_height,
    )


def redact_text(pdf, document, plan):
    """Add text redaction annotations to PDF pages."""
    fitz = _load_fitz()
    block_by_path = {block.path: block for block in document.blocks}
    for action in plan:
        if action.action != "mask_text":
            continue
        block = block_by_path.get(action.path)
        if block is None:
            continue
        page_index = block.metadata.get("page_index", action.metadata.get("page_index"))
        if page_index is None or not (0 <= int(page_index) < pdf.page_count):
            continue
        start = int(action.start or 0)
        end = int(action.end or 0)
        if not (0 <= start < end <= len(block.text)):
            continue
        surface = block.text[start:end]
        if not surface.strip():
            continue
        page = pdf[int(page_index)]
        clip = None
        if block.metadata.get("bbox"):
            try:
                clip = fitz.Rect(*[float(value) for value in block.metadata["bbox"]])
            except Exception:
                clip = None
        rects = []
        try:
            rects = page.search_for(surface, clip=clip) if clip is not None else page.search_for(surface)
        except Exception:
            rects = []
        if not rects and clip is not None:
            rects = [clip]
        for rect in rects:
            try:
                page.add_redact_annot(
                    rect,
                    text=action.replacement or "",
                    fill=(0, 0, 0),
                    text_color=(1, 1, 1),
                    fontsize=8,
                    fontname="helv",
                )
            except TypeError:
                page.add_redact_annot(rect, text=action.replacement or "", fill=(0, 0, 0))


def redact_images(pdf, plan):
    """Add image box redaction annotations to PDF pages."""
    fitz = _load_fitz()
    for page_index, actions in by_page(plan).items():
        if not (0 <= int(page_index) < pdf.page_count):
            continue
        page = pdf[int(page_index)]
        for action in actions:
            rect = rect_from_box(fitz, action)
            if rect is None:
                continue
            page.add_redact_annot(rect, fill=(0, 0, 0))


def _apply_pdf_redactions(pdf):
    """Apply redaction annotations to PDF pages, removing marked content."""
    fitz = _load_fitz()
    image_mode = getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", None)
    for page in pdf:
        try:
            if image_mode is None:
                page.apply_redactions()
            else:
                page.apply_redactions(images=image_mode)
        except TypeError:
            page.apply_redactions()


def _write_pdf(document, plan, config):
    """Write anonymized PDF with text and image redactions."""
    actions = list(plan)
    blocks = text(document.blocks, actions)
    output = out(document.document_path, config.output)
    fitz = _load_fitz()
    with fitz.open(str(document.document_path)) as pdf:
        redact_text(pdf, document, actions)
        redact_images(pdf, actions)
        _apply_pdf_redactions(pdf)
        save_kwargs = {"garbage": 4, "deflate": True}
        try:
            pdf.save(str(output), **save_kwargs)
        except TypeError:
            pdf.save(str(output))

    meta = {
        **dict(document.metadata or {}),
        "source_document_path": str(document.document_path),
        "document_type": "pdf",
        "pdf_path": str(output),
    }
    return DocumentRecord(
        doc_id=document.doc_id,
        document_path=output,
        blocks=blocks,
        images=document.images,
        metadata=meta,
    )


def write(document, plan, config):
    """Write anonymized output using the source document format."""
    suffix = document.document_path.suffix.lower()
    if suffix == ".docx":
        return _write_docx(document, plan, config)
    if suffix == ".pdf":
        return _write_pdf(document, plan, config)
    raise ValueError(f"Unsupported output document type: {document.document_path.suffix}")
