"""
RQ3 Document Labelling & Anonymisation Review UI - label1.py
============================================================

Updated version focused on reliability and RQ3 export quality:
  - Real PDF layout preview with clickable masks.
  - Reliable select-text workflow: select text -> popup -> choose label -> Add.
  - PDF drag workflow for scanned PDFs: drag over text/area -> popup -> choose label -> Add.
  - Full-page EasyOCR fallback for PDFs when OCR is enabled and easyocr is installed.
  - DOCX preview rendered as document-style pages where possible.
  - Per-document editable attacker database tab.
  - Per-document database generation uses annotated names, observed student numbers, and course codes first.
  - First-name/last-name annotations can regenerate full-name rows for faster review.
  - First-only full-name rows are completed with a reused matching full name or a generated surname.
  - Generated Maastricht email addresses use first-initial.last-name with student/staff domains.
  - Institutional student-number format supports i followed by seven digits, e.g. i1234567.
  - True redacted PDF export using PyMuPDF redaction annotations.
  - Export ZIP includes redacted documents/text, per-document databases, combined database,
    review states, audit logs, label summaries, redaction verification, and README.

Run:
    pip install flask python-docx pymupdf
    pip install easyocr   # optional OCR
    python label1.py

Open:
    http://127.0.0.1:7998
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_file, url_for

try:
    import pymupdf as fitz
except Exception:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        fitz = None  # type: ignore
        PYMUPDF_ERROR = exc
    else:
        PYMUPDF_ERROR = None
else:
    PYMUPDF_ERROR = None

APP_DIR = Path(__file__).resolve().parent
WORKSPACE = APP_DIR / "rq3_labeling_workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = WORKSPACE / "document_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DIRECT_LABELS = ("FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email")
QUASI_LABELS = ("Course code", "Group label", "Programme", "Date", "Role")
LABELS = (*DIRECT_LABELS, *QUASI_LABELS)
LABEL_OPTIONS = [
    "FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email",
    "Course code", "Group label", "Programme", "Date", "Role", "Name", "Other",
]
DOC_EXTENSIONS = {".docx", ".pdf", ".txt", ".md"}

MASK_MAP = {
    "FULL_NAME": "[NAME]",
    "FIRST_NAME": "[NAME]",
    "LAST_NAME": "[NAME]",
    "Name": "[NAME]",
    "Student number": "[STUDENT_ID]",
    "Institutional email": "[EMAIL]",
    "Course code": "[COURSE]",
    "Group label": "[GROUP]",
    "Programme": "[PROGRAMME]",
    "Date": "[DATE]",
    "Role": "[ROLE]",
}

DB_COLUMNS = [
    ("record", "Record"),
    ("entity_type", "Type"),
    ("full_name", "Full name"),
    ("student_number", "Student #"),
    ("institutional_email", "Email"),
    ("programme", "Programme"),
    ("course_codes", "Courses"),
    ("group_labels", "Groups"),
    ("roles", "Roles"),
    ("dates", "Dates"),
    ("generated_fields", "Generated"),
    ("source_documents", "Sources"),
]
DB_FIELDNAMES = [k for k, _ in DB_COLUMNS]

PDF_RENDER_ZOOM = 2.0
OCR_RENDER_ZOOM = 2.5
OCR_MIN_CONFIDENCE = 0.30

FALLBACK_COURSE_CODES = ["KEN3034", "KEN3125", "KEN3450", "KEN4110"]
FALLBACK_PROGRAMMES = ["Data Science and Artificial Intelligence (Bachelor)", "Computer Science (Bachelor)"]
FALLBACK_GROUPS = ["Group 01", "Group 02", "Group 03", "Group 04"]
FALLBACK_ROLES = ["student"]
STUDENT_EMAIL_DOMAIN = "student.maastrichtuniversity.nl"
STAFF_EMAIL_DOMAIN = "maastrichtuniversity.nl"


# -- repo / import wiring -----------------------------------------------------


def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "document_generator").exists() and (p / "thesis_project").exists():
            return p
    for p in [APP_DIR, *APP_DIR.parents]:
        if (p / "document_generator").exists() and (p / "thesis_project").exists():
            return p
    return cwd


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from thesis_project.shared import Config, canon
    from thesis_project.defender.parser import parse_document, normalise_blocks
    from thesis_project.defender.pipeline import detect_entities, process_document
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on user's project
    Config = parse_document = normalise_blocks = detect_entities = process_document = None
    IMPORT_ERROR = exc

    def canon(x):
        return str(x)


# -- utilities ---------------------------------------------------------------


def safe_obj(value: Any) -> Any:
    if is_dataclass(value):
        return safe_obj(asdict(value))
    if isinstance(value, dict):
        return {str(k): safe_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_obj(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return safe_obj(vars(value))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def obj_get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "item"


def normalise(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def entity_kind(label: str) -> str:
    c = canon(label)
    return "direct" if c in DIRECT_LABELS or c == "Name" else "quasi"


def mask_for_label(label: str) -> str:
    return MASK_MAP.get(canon(label), "[REDACTED]")


def valid_label(label: str) -> str:
    label = canon(str(label or "Other"))
    return label if label in LABEL_OPTIONS else "Other"


def h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def ordered_unique(values):
    seen = set()
    out = []
    for value in values or []:
        s = str(value or "").strip()
        k = normalise(s)
        if not s or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# -- persistence -------------------------------------------------------------


def session_dir(sid):
    return WORKSPACE / sid


def session_json_path(sid):
    return session_dir(sid) / "session.json"


def doc_state_path(sid, doc_id):
    return session_dir(sid) / "states" / f"{doc_id}.json"


def load_session(sid):
    p = session_json_path(sid)
    if not p.exists():
        raise FileNotFoundError(f"Unknown session: {sid}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_session(sid, data):
    p = session_json_path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(safe_obj(data), indent=2, ensure_ascii=False), encoding="utf-8")


def load_doc_state(sid, doc_id):
    return json.loads(doc_state_path(sid, doc_id).read_text(encoding="utf-8"))


def save_doc_state(sid, state):
    p = doc_state_path(sid, state["doc_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(safe_obj(state), indent=2, ensure_ascii=False), encoding="utf-8")


# -- session / document helpers ---------------------------------------------


def make_initial_state(doc_meta):
    return {
        **doc_meta,
        "processing_status": "queued",
        "progress_message": "Queued",
        "blocks": [],
        "entities": [],
        "pipeline_anonymized_path": "",
        "pipeline_audit": [],
        "errors": [],
        "review_status": "not reviewed",
        "annotation_complete_at": None,
        "attacker_database_csv": "",
        "database_complete": False,
        "database_completed_at": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def block_to_dict(block, idx):
    return {
        "path": str(obj_get(block, "path", obj_get(block, "block_path", f"block_{idx}"))),
        "text": str(obj_get(block, "text", obj_get(block, "surface", "")) or ""),
        "region": str(obj_get(block, "region", "body") or "body"),
        "index": idx,
    }


def entity_to_dict(entity, idx, blocks):
    label = valid_label(str(obj_get(entity, "label", "Other") or "Other"))
    surface = str(obj_get(entity, "surface", "") or "")
    path = str(obj_get(entity, "path", obj_get(entity, "block_path", "")) or "")
    start = obj_get(entity, "start", None)
    end = obj_get(entity, "end", None)
    region = str(obj_get(entity, "region", "") or "")
    score = obj_get(entity, "score", obj_get(entity, "confidence", ""))
    source = obj_get(entity, "source", obj_get(entity, "method", obj_get(entity, "provenance", "model")))

    if not path and surface:
        for b in blocks:
            pos = b["text"].find(surface)
            if pos >= 0:
                path = b["path"]
                start = pos
                end = pos + len(surface)
                region = region or b["region"]
                break
    try:
        start, end = int(start), int(end)
    except Exception:
        start = end = -1

    return {
        "id": f"e_{idx:05d}",
        "label": label,
        "surface": surface,
        "path": path,
        "start": start,
        "end": end,
        "region": region,
        "score": score,
        "source": source,
        "status": "accepted",
        "review_note": "",
        "manual": False,
    }


def accepted_entities(state):
    return [e for e in state.get("entities", []) if e.get("status") in {"accepted", "manual"}]


# -- parsing, pattern detection, OCR -----------------------------------------


def parse_pdf_text_fallback(path: Path):
    blocks = []
    if fitz is None:
        return blocks
    try:
        with fitz.open(str(path)) as doc:
            for page_no, page in enumerate(doc):
                try:
                    page_blocks = page.get_text("blocks") or []
                except Exception:
                    page_blocks = []
                if page_blocks:
                    for i, b in enumerate(page_blocks):
                        text = str(b[4] if len(b) > 4 else "").strip()
                        if not text:
                            continue
                        blocks.append({
                            "path": f"pdf:{page_no}:block:{i}",
                            "text": text,
                            "region": f"pdf_page_{page_no + 1}",
                            "index": len(blocks),
                            "page": page_no,
                        })
                else:
                    text = str(page.get_text("text") or "").strip()
                    if text:
                        blocks.append({
                            "path": f"pdf:{page_no}",
                            "text": text,
                            "region": f"pdf_page_{page_no + 1}",
                            "index": len(blocks),
                            "page": page_no,
                        })
    except Exception:
        return []
    return blocks


def parse_text_fallback(path: Path):
    if path.suffix.lower() == ".pdf":
        blocks = parse_pdf_text_fallback(path)
        if blocks:
            return blocks
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = f"Preview not available for {path.name}."
    return [{"path": "body:0", "text": text, "region": "body", "index": 0}]


_PATTERN_DEFS = [
    ("Institutional email", re.compile(r"(?<![\w@.-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w@.-])", re.I), 0),
    # Thesis/local format: i followed by seven digits, e.g. i1234567. The broader s-format is kept for compatibility.
    ("Student number", re.compile(r"(?<![\w@.-])(?:i\d{7}|s\d{6,9})(?![\w@.-])", re.I), 0),
    ("Student number", re.compile(r"\bstudent\s*(?:id|no\.?|number|#)?\s*[:#]?\s*([is]?\d{6,9})\b", re.I), 1),
    ("Date", re.compile(r"(?<!\w)(?:(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:\d{2}|\d{4})|(?:\d{4})[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})(?!\w)", re.I), 0),
    ("Course code", re.compile(r"(?<!\w)[A-Z]{2,6}\s?\d{3,5}[A-Z]?(?!\w)"), 0),
    ("FULL_NAME", re.compile(r"\b(?:name|student|author|lecturer|teacher|supervisor|coordinator|ta)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"), 1),
]


def _rect_for_text_slice(block_rect, text, start, end):
    rects = clean_pdf_rects(block_rect or [])
    if not rects or not text:
        return []
    r = rects[0]
    n = max(len(text), 1)
    frac0 = max(0.0, min(1.0, start / n))
    frac1 = max(frac0 + 0.01, min(1.0, end / n))
    width = r["x1"] - r["x0"]
    return [{
        "page": r["page"],
        "x0": r["x0"] + width * frac0,
        "y0": r["y0"],
        "x1": r["x0"] + width * frac1,
        "y1": r["y1"],
    }]


def simple_pattern_detections(blocks):
    out = []
    seen = set()
    serial = 0
    for block in blocks:
        text = str(block.get("text", "") or "")
        for label, rx, group in _PATTERN_DEFS:
            for m in rx.finditer(text):
                if group:
                    try:
                        surface = m.group(group).strip()
                        start, end = m.start(group), m.end(group)
                    except Exception:
                        continue
                else:
                    start, end = m.start(), m.end()
                    surface = m.group(0).strip()
                if not surface:
                    continue
                key = (block.get("path", ""), start, end, label, surface.lower())
                if key in seen:
                    continue
                seen.add(key)
                ent = {
                    "id": f"p_{serial:05d}",
                    "label": label,
                    "surface": surface,
                    "path": str(block.get("path", "")),
                    "start": int(start),
                    "end": int(end),
                    "region": str(block.get("region", "")),
                    "score": "pattern_fallback",
                    "source": "pattern_fallback",
                    "status": "accepted",
                    "review_note": "",
                    "manual": False,
                }
                pdf_rects = _rect_for_text_slice(block.get("pdf_rects", []), text, start, end)
                if pdf_rects:
                    ent["pdf_rects"] = pdf_rects
                out.append(ent)
                serial += 1
    return out


_EASYOCR_READER = None
_EASYOCR_LOCK = threading.Lock()


def easyocr_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("easyocr") is not None
    except Exception:
        return False


def get_easyocr_reader():
    global _EASYOCR_READER
    with _EASYOCR_LOCK:
        if _EASYOCR_READER is None:
            import easyocr  # type: ignore
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _EASYOCR_READER


def ocr_pdf_blocks(path: Path, sid: str, doc_id: str, progress=None):
    """Run OCR over every PDF page and return line-level blocks with PDF coordinates."""
    if fitz is None:
        raise RuntimeError(f"PyMuPDF is not available: {PYMUPDF_ERROR}")
    if not easyocr_available():
        raise RuntimeError("easyocr is not installed")

    reader = get_easyocr_reader()
    out_dir = session_dir(sid) / "ocr_pages" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    with fitz.open(str(path)) as doc:
        for page_no, page in enumerate(doc):
            if progress:
                progress(page_no + 1, len(doc))
            pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_ZOOM, OCR_RENDER_ZOOM), alpha=False)
            img_path = out_dir / f"page_{page_no + 1:04d}.png"
            pix.save(str(img_path))
            try:
                results = reader.readtext(str(img_path), detail=1, paragraph=False)
            except Exception as exc:
                raise RuntimeError(f"easyocr failed on page {page_no + 1}: {exc}")
            for i, item in enumerate(results or []):
                try:
                    box, text, conf = item[0], str(item[1] or "").strip(), float(item[2])
                except Exception:
                    continue
                if not text or conf < OCR_MIN_CONFIDENCE:
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                page_x0 = float(page.rect.x0)
                page_y0 = float(page.rect.y0)
                rect = {
                    "page": page_no,
                    "x0": page_x0 + min(xs) / OCR_RENDER_ZOOM,
                    "y0": page_y0 + min(ys) / OCR_RENDER_ZOOM,
                    "x1": page_x0 + max(xs) / OCR_RENDER_ZOOM,
                    "y1": page_y0 + max(ys) / OCR_RENDER_ZOOM,
                }
                blocks.append({
                    "path": f"ocr_pdf:{page_no}:line:{i}",
                    "text": text,
                    "region": f"ocr_page_{page_no + 1}",
                    "index": len(blocks),
                    "page": page_no,
                    "ocr": True,
                    "ocr_confidence": conf,
                    "pdf_rects": [rect],
                })
    return blocks


# -- PDF preview / redaction helpers -----------------------------------------


def pdf_preview_dir(sid, doc_id):
    return session_dir(sid) / "pdf_preview" / doc_id


def pdf_page_png_path(sid, doc_id, page_no, zoom=PDF_RENDER_ZOOM):
    zoom_tag = str(int(round(float(zoom) * 100)))
    return pdf_preview_dir(sid, doc_id) / f"page_{page_no + 1:04d}_z{zoom_tag}.png"


def clean_pdf_rects(rects):
    cleaned = []
    for r in rects or []:
        try:
            page = int(r.get("page", 0))
            x0 = float(r.get("x0"))
            y0 = float(r.get("y0"))
            x1 = float(r.get("x1"))
            y1 = float(r.get("y1"))
            if page >= 0 and x1 > x0 and y1 > y0:
                cleaned.append({"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        except Exception:
            continue
    return cleaned[:1000]


def ensure_pdf_page_image(sid, state, page_no, zoom=PDF_RENDER_ZOOM):
    if fitz is None:
        raise RuntimeError(f"PyMuPDF is not available: {PYMUPDF_ERROR}")
    src = Path(state.get("absolute_path", ""))
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = pdf_preview_dir(sid, state["doc_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = pdf_page_png_path(sid, state["doc_id"], page_no, zoom)
    if img_path.exists() and img_path.stat().st_mtime >= src.stat().st_mtime:
        return img_path
    with fitz.open(str(src)) as doc:
        if page_no < 0 or page_no >= len(doc):
            raise IndexError(page_no)
        pix = doc[page_no].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(img_path))
    return img_path


def masked_pdf_file_path(sid, doc_id):
    return session_dir(sid) / "masked_pdf_preview" / f"{doc_id}.pdf"


def masked_pdf_page_png_path(sid, doc_id, page_no, zoom=PDF_RENDER_ZOOM):
    zoom_tag = str(int(round(float(zoom) * 100)))
    return session_dir(sid) / "masked_pdf_preview" / "pages" / doc_id / f"page_{page_no + 1:04d}_z{zoom_tag}.png"


def ensure_masked_pdf_file(sid, state):
    if fitz is None:
        raise RuntimeError(f"PyMuPDF is not available: {PYMUPDF_ERROR}")
    original = Path(state.get("absolute_path", ""))
    if not original.exists():
        raise FileNotFoundError(original)
    if original.suffix.lower() != ".pdf":
        raise ValueError("Masked PDF can only be generated for PDF files")
    out = masked_pdf_file_path(sid, state["doc_id"])
    state_path = doc_state_path(sid, state["doc_id"])
    newest_source = max(original.stat().st_mtime, state_path.stat().st_mtime if state_path.exists() else 0)
    if out.exists() and out.stat().st_mtime >= newest_source:
        return out
    ok = write_redacted_pdf(original, state, out)
    if not ok:
        raise RuntimeError("Could not generate masked PDF")
    return out


def ensure_masked_pdf_page_image(sid, state, page_no, zoom=PDF_RENDER_ZOOM):
    if fitz is None:
        raise RuntimeError(f"PyMuPDF is not available: {PYMUPDF_ERROR}")
    masked_pdf = ensure_masked_pdf_file(sid, state)
    out = masked_pdf_page_png_path(sid, state["doc_id"], page_no, zoom)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_mtime >= masked_pdf.stat().st_mtime:
        return out
    with fitz.open(str(masked_pdf)) as doc:
        if page_no < 0 or page_no >= len(doc):
            raise IndexError(page_no)
        pix = doc[page_no].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(out))
    return out


def _entity_surface_variants(surface):
    surface = str(surface or "").strip()
    if not surface:
        return []
    variants = {surface, re.sub(r"\s+", " ", surface), surface.replace("\n", " "), surface.replace("\u00a0", " ")}
    return [v.strip() for v in variants if len(v.strip()) >= 2]


def _pdf_overlap_area(a, b):
    x0 = max(float(a["x0"]), float(b["x0"]))
    y0 = max(float(a["y0"]), float(b["y0"]))
    x1 = min(float(a["x1"]), float(b["x1"]))
    y1 = min(float(a["y1"]), float(b["y1"]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _pdf_area(r):
    return max(0.0, float(r["x1"]) - float(r["x0"])) * max(0.0, float(r["y1"]) - float(r["y0"]))


def _normalise_pdf_overlays(overlays_by_page):
    result = {}
    for page_no, overlays in overlays_by_page.items():
        ordered = sorted(
            overlays,
            key=lambda ov: (
                linked_label_priority(ov.get("label", "")),
                -_pdf_area(ov),
                str(ov.get("surface", "")).lower(),
            ),
        )
        kept = []
        for ov in ordered:
            area = max(_pdf_area(ov), 1.0)
            if any(_pdf_overlap_area(ov, ex) / area > 0.72 for ex in kept):
                continue
            kept.append(ov)
        result[page_no] = kept
    return result


def build_pdf_overlay_rects(pdf_doc, state):
    overlays_by_page = {}
    seen = set()
    for e in accepted_entities(state):
        eid = str(e.get("id", ""))
        label = valid_label(e.get("label", "Other"))
        surface = str(e.get("surface", "") or "").strip()
        for r in clean_pdf_rects(e.get("pdf_rects", [])):
            key = (r["page"], round(r["x0"], 2), round(r["y0"], 2), round(r["x1"], 2), round(r["y1"], 2), label)
            if key in seen:
                continue
            seen.add(key)
            overlays_by_page.setdefault(r["page"], []).append({
                "id": eid,
                "label": label,
                "surface": surface,
                "manual": bool(e.get("manual")),
                "linked": bool(e.get("linked")),
                "x0": r["x0"],
                "y0": r["y0"],
                "x1": r["x1"],
                "y1": r["y1"],
            })
        if not surface or not re.search(r"\w", surface):
            continue
        for needle in _entity_surface_variants(surface):
            for page_no, page in enumerate(pdf_doc):
                try:
                    hits = page.search_for(needle)
                except Exception:
                    hits = []
                for rect in hits:
                    key = (page_no, round(float(rect.x0), 2), round(float(rect.y0), 2), round(float(rect.x1), 2), round(float(rect.y1), 2), label)
                    if key in seen:
                        continue
                    seen.add(key)
                    overlays_by_page.setdefault(page_no, []).append({
                        "id": eid,
                        "label": label,
                        "surface": surface,
                        "manual": bool(e.get("manual")),
                        "linked": bool(e.get("linked")),
                        "x0": float(rect.x0),
                        "y0": float(rect.y0),
                        "x1": float(rect.x1),
                        "y1": float(rect.y1),
                    })
    return _normalise_pdf_overlays(overlays_by_page)


# -- linking -----------------------------------------------------------------


LINK_LABELS = {"FULL_NAME", "FIRST_NAME", "LAST_NAME", "Name", "Student number", "Institutional email"}


def linked_label_priority(label):
    return {
        "Institutional email": 0,
        "Student number": 1,
        "FULL_NAME": 2,
        "Name": 3,
        "FIRST_NAME": 4,
        "LAST_NAME": 5,
    }.get(canon(label), 99)


def surface_link_allowed(e):
    label = valid_label(e.get("label", ""))
    surface = str(e.get("surface", "") or "").strip()
    if e.get("status") == "rejected" or label not in LINK_LABELS or len(surface) < 2:
        return False
    if label in {"FIRST_NAME", "LAST_NAME"} and len(surface) < 3:
        return False
    return True


def find_surface_spans(text, surface):
    surface = str(surface or "").strip()
    if not surface:
        return []
    pattern = re.compile(rf"(?<![\w@.-]){re.escape(surface)}(?![\w@.-])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text or "")]


def spans_overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def entity_span_key(e):
    return (str(e.get("path", "")), int(e.get("start", -1)), int(e.get("end", -1)), valid_label(e.get("label", "")))


def apply_cross_region_linking(state, *, source="cross_region_link"):
    entities = list(state.get("entities", []))
    blocks = list(state.get("blocks", []))
    seeds = sorted(
        [e for e in entities if surface_link_allowed(e)],
        key=lambda e: (linked_label_priority(e.get("label", "")), -len(str(e.get("surface", ""))), str(e.get("surface", "")).lower()),
    )
    existing = {entity_span_key(e) for e in entities}
    additions = []
    serial = 0
    for seed in seeds:
        label = valid_label(seed.get("label", ""))
        surface = str(seed.get("surface", "") or "").strip()
        if not surface:
            continue
        for block in blocks:
            text = str(block.get("text", "") or "")
            path = str(block.get("path", ""))
            for start, end in find_surface_spans(text, surface):
                if (path, start, end, label) in existing:
                    continue
                blocked = any(
                    str(ex.get("path", "")) == path
                    and ex.get("status") != "rejected"
                    and spans_overlap(start, end, int(ex.get("start", -1)), int(ex.get("end", -1)))
                    and linked_label_priority(ex.get("label", "")) <= linked_label_priority(label)
                    for ex in entities + additions
                )
                if blocked:
                    continue
                serial += 1
                linked = {
                    "id": f"l_{uuid.uuid4().hex[:8]}_{serial}",
                    "label": label,
                    "surface": text[start:end],
                    "path": path,
                    "start": start,
                    "end": end,
                    "region": str(block.get("region", "")),
                    "score": seed.get("score", "linked"),
                    "source": source,
                    "status": "accepted",
                    "review_note": f"linked from {seed.get('id', '')}",
                    "manual": False,
                    "linked": True,
                    "linked_from": seed.get("id", ""),
                    "reviewed_at": time.time(),
                }
                if block.get("pdf_rects"):
                    linked["pdf_rects"] = _rect_for_text_slice(block.get("pdf_rects"), text, start, end)
                additions.append(linked)
                existing.add((path, start, end, label))
    if additions:
        state["entities"] = entities + additions
        state["updated_at"] = time.time()
    return state


def remove_linked_for_seed(state, seed_id):
    state["entities"] = [
        e for e in state.get("entities", [])
        if not (e.get("linked") and e.get("linked_from") == seed_id)
    ]
    state["updated_at"] = time.time()
    return state


# -- workers / pipeline -------------------------------------------------------


WORKERS: dict[str, threading.Thread] = {}
WORKERS_LOCK = threading.Lock()
CLAIM_LOCK = threading.Lock()
STALE_AFTER = 300


def build_config(sid):
    if Config is None:
        raise RuntimeError(f"Could not import thesis_project: {IMPORT_ERROR}")
    candidate = ROOT / "notebooks" / "workspace" / "tuned_models" / "name_candidate_model.joblib"
    settings = load_session(sid).get("settings", {})
    return Config(
        output=session_dir(sid) / "pipeline_output",
        patterns=True,
        ner=bool(settings.get("ner", True)),
        ocr=False,  # OCR is handled here explicitly so every PDF page is covered.
        link=True,
        strict=True,
        direct=True,
        quasi=True,
        labels=QUASI_LABELS,
        regions=(),
        pdf=False,
        candidate=str(candidate) if candidate.exists() else None,
        ner_backend="hybrid",
        gliner_model="urchade/gliner_multi-v2.1",
        gliner_threshold=0.35,
        fallback=True,
    )


def safe_extract_zip(zip_path, out_dir):
    extracted = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            raw = member.filename.replace("\\", "/")
            if raw.startswith("/") or ".." in Path(raw).parts:
                continue
            target = out_dir / raw
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def zip_cache_key(zip_path):
    digest = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_cached_documents(zip_path, extract_dir):
    cache_dir = CACHE_ROOT / zip_cache_key(zip_path) / "source_documents"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        shutil.copytree(cache_dir, extract_dir, dirs_exist_ok=True)
        return [p for p in extract_dir.rglob("*") if p.is_file()]

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    files = safe_extract_zip(zip_path, extract_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    shutil.copytree(extract_dir, cache_dir)
    return files


def collect_documents(files, root_dir):
    supported = sorted(
        [p for p in files if p.suffix.lower() in DOC_EXTENSIONS],
        key=lambda p: (p.stat().st_size if p.exists() else 0, p.relative_to(root_dir).as_posix()),
    )
    return [
        {
            "doc_id": f"doc_{i:04d}_{slug(p.stem)}",
            "file_name": p.name,
            "relative_path": p.relative_to(root_dir).as_posix(),
            "absolute_path": str(p),
            "extension": p.suffix.lower(),
            "file_size_bytes": p.stat().st_size if p.exists() else 0,
        }
        for i, p in enumerate(supported)
    ]


def _merge_blocks(existing, new_blocks):
    seen = {normalise(b.get("text", "")) for b in existing if normalise(b.get("text", ""))}
    merged = list(existing)
    for b in new_blocks:
        key = normalise(b.get("text", ""))
        if not key or key in seen:
            continue
        b = dict(b)
        b["index"] = len(merged)
        merged.append(b)
        seen.add(key)
    return merged


def _entity_dedupe_key(e):
    return (
        str(e.get("path", "")),
        int(e.get("start", -1)),
        int(e.get("end", -1)),
        normalise(e.get("surface", "")),
        valid_label(e.get("label", "")),
    )


def run_defender_for_doc(sid, doc_meta, preserve_manual=True):
    state = load_doc_state(sid, doc_meta["doc_id"]) if doc_state_path(sid, doc_meta["doc_id"]).exists() else make_initial_state(doc_meta)
    manual = [e for e in state.get("entities", []) if e.get("manual")] if preserve_manual else []
    previous_db_csv = state.get("attacker_database_csv", "")
    previous_db_complete = bool(state.get("database_complete"))
    previous_db_completed_at = state.get("database_completed_at")
    state.update({"processing_status": "processing", "progress_message": "Parsing document", "updated_at": time.time()})
    save_doc_state(sid, state)

    settings = load_session(sid).get("settings", {})
    path = Path(doc_meta["absolute_path"])
    blocks = []
    entities = []
    errors = []

    try:
        cfg = build_config(sid)
    except Exception as exc:
        cfg = None
        errors.append(f"Defender config/import failed: {IMPORT_ERROR or exc}")

    try:
        if parse_document is None or normalise_blocks is None or detect_entities is None or cfg is None:
            raise RuntimeError("Defender pipeline unavailable")
        parsed = normalise_blocks(parse_document(path))
        raw_blks = list(obj_get(parsed, "blocks", []) or [])
        blocks = [block_to_dict(b, i) for i, b in enumerate(raw_blks)]
        if not blocks and path.suffix.lower() == ".pdf":
            blocks = parse_pdf_text_fallback(path)
        state.update({"blocks": blocks, "entities": manual, "progress_message": f"Detecting PII in {len(blocks)} blocks", "updated_at": time.time()})
        save_doc_state(sid, state)
        detections = list(detect_entities(parsed, cfg) or [])
        entities = [entity_to_dict(e, i, blocks) for i, e in enumerate(detections)]
    except Exception as exc:
        errors.append(f"detect_entities failed: {exc}")
        blocks = blocks or parse_text_fallback(path)

    if path.suffix.lower() == ".pdf" and settings.get("ocr"):
        if easyocr_available():
            try:
                def progress(page_i, total):
                    state.update({"progress_message": f"OCR page {page_i}/{total}", "updated_at": time.time()})
                    save_doc_state(sid, state)
                ocr_blocks = ocr_pdf_blocks(path, sid, doc_meta["doc_id"], progress=progress)
                blocks = _merge_blocks(blocks, ocr_blocks)
                state.setdefault("ocr", {})["enabled"] = True
                state["ocr"]["blocks"] = len(ocr_blocks)
                state["ocr"]["min_confidence"] = OCR_MIN_CONFIDENCE
            except Exception as exc:
                errors.append(f"OCR failed: {exc}")
        else:
            errors.append("OCR was enabled, but easyocr is not installed. Run: pip install easyocr")

    pattern_entities = simple_pattern_detections(blocks)
    # Avoid duplicates without collapsing repeated mentions in different blocks.
    existing_keys = {_entity_dedupe_key(e) for e in entities}
    for e in pattern_entities:
        key = _entity_dedupe_key(e)
        if key not in existing_keys:
            entities.append(e)
            existing_keys.add(key)

    state.update({
        "blocks": blocks,
        "entities": entities + manual,
        "errors": errors,
        "processing_status": "ready",
        "progress_message": "Ready for review" if not errors else "Ready with warnings",
        "review_status": state.get("review_status", "not reviewed"),
        "attacker_database_csv": previous_db_csv,
        "database_complete": previous_db_complete,
        "database_completed_at": previous_db_completed_at,
        "updated_at": time.time(),
    })
    state = apply_cross_region_linking(state)
    save_doc_state(sid, state)
    return state


def worker_running(sid):
    with WORKERS_LOCK:
        return any(k.startswith(sid) and t.is_alive() for k, t in WORKERS.items())


def claim_next_doc(sid, doc_ids=None):
    now = time.time()
    with CLAIM_LOCK:
        session = load_session(sid)
        for doc in sorted(session.get("documents", []), key=lambda d: (int(d.get("file_size_bytes", 0)), d.get("relative_path", ""))):
            if doc_ids is not None and doc["doc_id"] not in doc_ids:
                continue
            state = load_doc_state(sid, doc["doc_id"])
            status = state.get("processing_status", "queued")
            stale = status == "processing" and now - float(state.get("updated_at", 0)) > STALE_AFTER
            if status == "queued" or stale or (doc_ids is not None and status != "ready"):
                state.update({"processing_status": "processing", "progress_message": "Claimed", "updated_at": now})
                save_doc_state(sid, state)
                return doc
    return None


def worker_loop(sid, doc_ids=None):
    while True:
        doc = claim_next_doc(sid, doc_ids)
        if doc is None:
            break
        run_defender_for_doc(sid, doc, preserve_manual=True)


def start_workers(sid, doc_ids=None, count=None):
    session = load_session(sid)
    n = max(1, min(int(count or session.get("settings", {}).get("workers", 2) or 2), 4))
    with WORKERS_LOCK:
        if doc_ids is None and any(k.startswith(sid) and t.is_alive() for k, t in WORKERS.items()):
            return
        for i in range(n if doc_ids is None else 1):
            key = f"{sid}:w{i}:{uuid.uuid4().hex[:6]}"
            t = threading.Thread(target=worker_loop, args=(sid, doc_ids), daemon=True)
            WORKERS[key] = t
            t.start()


def status_counts(sid):
    session = load_session(sid)
    docs = []
    counts = {"queued": 0, "processing": 0, "ready": 0, "error": 0}
    first_ready = ""
    for d in session.get("documents", []):
        st = load_doc_state(sid, d["doc_id"])
        status = st.get("processing_status", "queued")
        counts[status] = counts.get(status, 0) + 1
        accepted = sum(1 for e in st.get("entities", []) if e.get("status") in {"accepted", "manual"})
        if status == "ready" and not first_ready:
            first_ready = d["doc_id"]
        docs.append({
            **d,
            "processing_status": status,
            "progress_message": st.get("progress_message", ""),
            "accepted": accepted,
            "review_status": st.get("review_status", "not reviewed"),
            "database_complete": bool(st.get("database_complete")),
        })
    return {
        "sid": sid,
        "docs": docs,
        "counts": counts,
        "total": max(len(docs), 1),
        "ready_share": counts.get("ready", 0) / max(len(docs), 1),
        "first_ready": first_ready,
        "worker_running": worker_running(sid),
    }


# -- database ----------------------------------------------------------------


def _split_values(value: str):
    return [v.strip() for v in str(value or "").split("|") if v.strip()]


def parse_csv_rows(csv_text: str):
    if not str(csv_text or "").strip():
        return []
    try:
        return list(csv.DictReader(io.StringIO(csv_text)))
    except Exception:
        return []


def rows_to_csv(rows, fieldnames=None):
    fieldnames = fieldnames or DB_FIELDNAMES
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows or []:
        w.writerow({k: row.get(k, "") for k in fieldnames})
    return out.getvalue()


def _name_tokens(value: str):
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’-]+", str(value or ""))
    cleaned = []
    for token in tokens:
        token = token.strip("'’-")
        if token:
            cleaned.append(token)
    return cleaned


def _person_name_parts(full_name: str, index: int = 1):
    """Return first and last-ish tokens for generated institutional email addresses."""
    cleaned = _name_tokens(full_name)
    if not cleaned:
        return "synthetic", f"person{index:03d}"
    first = cleaned[0]
    # If only a first name was annotated, keep the row useful but generate a stable placeholder surname.
    last = cleaned[-1] if len(cleaned) > 1 else f"person{index:03d}"
    return first, last


def _complete_partial_full_name(full_name: str, index: int, candidate_full_names):
    """Complete a first-name-only full_name before generating an email.

    If the row contains only a first name and another annotated full name starts with
    the same first name, reuse that full name. Otherwise append a deterministic
    synthetic surname so the database and generated email are no longer incomplete.
    """
    value = str(full_name or "").strip()
    tokens = _name_tokens(value)
    if len(tokens) != 1:
        return value, False

    first = tokens[0]
    first_key = normalise(first)
    for candidate in ordered_unique(candidate_full_names or []):
        cand_tokens = _name_tokens(candidate)
        if len(cand_tokens) >= 2 and normalise(cand_tokens[0]) == first_key:
            return str(candidate).strip(), False

    return f"{first} Syntheticlast{index:03d}", True


def _email_token(value: str) -> str:
    token = normalise(value)
    token = token.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token or "x"


def _is_staff_or_tutor(role_text: str = "", entity_type: str = "") -> bool:
    haystack = normalise(f"{role_text} {entity_type}")
    staff_terms = (
        "tutor", "coordinator", "teacher", "lecturer", "supervisor", "staff",
        "teaching assistant", "assistant", "ta", "instructor", "mentor",
    )
    return any(term in haystack for term in staff_terms)


def _generated_institutional_email(full_name: str, role_text: str, entity_type: str, index: int, used_emails: set[str]) -> str:
    first, last = _person_name_parts(full_name, index=index)
    local = f"{_email_token(first)[0]}.{_email_token(last)}"
    domain = STAFF_EMAIL_DOMAIN if _is_staff_or_tutor(role_text, entity_type) else STUDENT_EMAIL_DOMAIN
    candidate = f"{local}@{domain}"
    suffix = 1
    while normalise(candidate) in used_emails:
        suffix += 1
        candidate = f"{local}{suffix}@{domain}"
    return candidate


def _paired_name_values(first_names, last_names):
    """Create name rows from first/last annotations, allowing incomplete names.

    Examples:
    - 7 first names and 7 last names -> 7 full names.
    - 7 first names and no last names -> 7 incomplete full_name values using the first names.
    - no first names and 7 last names -> 7 incomplete full_name values using the last names.
    """
    first_names = ordered_unique(first_names)
    last_names = ordered_unique(last_names)
    count = max(len(first_names), len(last_names))
    names = []
    for i in range(count):
        first = first_names[i] if i < len(first_names) else ""
        last = last_names[i] if i < len(last_names) else ""
        # Reuse a single counterpart only when there is exactly one. This helps with documents
        # that list many first names and one shared surname, while avoiding broad repetition.
        if not first and len(first_names) == 1:
            first = first_names[0]
        if not last and len(last_names) == 1:
            last = last_names[0]
        name = " ".join(x for x in [first, last] if x).strip()
        if name:
            names.append(name)
    return ordered_unique(names)


def _merge_full_and_partial_names(full_names, first_names, last_names):
    full_names = ordered_unique(full_names)
    partial_names = _paired_name_values(first_names, last_names)
    if not full_names:
        return partial_names
    merged = list(full_names)
    for name in partial_names:
        n = normalise(name)
        # Do not add a partial name when it is already contained in a complete full name.
        if any(n == normalise(existing) or n in normalise(existing) for existing in merged):
            continue
        merged.append(name)
    return ordered_unique(merged)



def _shared_programme_value(values):
    values = ordered_unique(values)
    for value in values:
        n = normalise(value)
        if "data science" in n or "artificial intelligence" in n:
            return "Data Science and Artificial Intelligence (Bachelor)"
        if "computer science" in n:
            return "Computer Science (Bachelor)"
    return FALLBACK_PROGRAMMES[0]


def _shared_course_code(values):
    values = ordered_unique(values)
    for value in values:
        compact = re.sub(r"\s+", "", str(value or "")).upper()
        if re.fullmatch(r"KEN\d{3,5}[A-Z]?", compact):
            return compact
    for value in values:
        compact = re.sub(r"\s+", "", str(value or "")).upper()
        if compact:
            return compact
    return FALLBACK_COURSE_CODES[0]


def _shared_group_label(values):
    values = ordered_unique(values)
    return values[0] if values else FALLBACK_GROUPS[0]


def _shared_date_value(values):
    values = ordered_unique(values)
    return values[0] if values else "2026-01-01"


def _shared_role_value(values=None):
    # The generated attacker database should use student rows only unless the user explicitly edits it.
    return "student"

def build_attacker_rows(states):
    profiles: dict[str, dict[str, Any]] = {}

    def key_for(full_name="", student_number="", email=""):
        if email:
            return "email:" + normalise(email)
        if student_number:
            return "student:" + normalise(student_number)
        if full_name:
            return "name:" + normalise(full_name)
        return "unknown:" + uuid.uuid4().hex[:8]

    for state in states:
        ents = accepted_entities(state)
        by_label: dict[str, list[str]] = {}
        for e in ents:
            label = valid_label(e.get("label", ""))
            surface = str(e.get("surface", "")).strip()
            if surface:
                by_label.setdefault(label, []).append(surface)

        name_values = _merge_full_and_partial_names(
            by_label.get("FULL_NAME", []) + by_label.get("Name", []),
            by_label.get("FIRST_NAME", []),
            by_label.get("LAST_NAME", []),
        )
        student_nos = ordered_unique(by_label.get("Student number", []))
        emails = ordered_unique(by_label.get("Institutional email", []))
        count = max(len(name_values), len(student_nos), len(emails), 1)

        # Keep one shared context value per document. This avoids creating huge databases
        # with hundreds of repeated dates/course codes/groups copied into every student row.
        doc_programme = _shared_programme_value(by_label.get("Programme", []))
        doc_group = _shared_group_label(by_label.get("Group label", []))
        doc_course = _shared_course_code(by_label.get("Course code", []))
        doc_role = _shared_role_value(by_label.get("Role", []))
        doc_date = _shared_date_value(by_label.get("Date", []))

        for i in range(count):
            # Do not repeat a single direct identifier across many generated rows; incomplete
            # rows are safer than incorrectly linking one name to every student number.
            full = name_values[i] if i < len(name_values) else ""
            student = student_nos[i] if i < len(student_nos) else ""
            email = emails[i] if i < len(emails) else ""
            k = key_for(full, student, email)
            row = profiles.setdefault(k, {
                "record": f"profile_{len(profiles) + 1}",
                "entity_type": "student",
                "full_name": full,
                "student_number": student,
                "institutional_email": email,
                "programme": set(),
                "group_labels": set(),
                "course_codes": set(),
                "roles": set(),
                "dates": set(),
                "source_documents": set(),
                "generated_fields": set(),
            })
            if full and not row.get("full_name"):
                row["full_name"] = full
            if student and not row.get("student_number"):
                row["student_number"] = student
            if email and not row.get("institutional_email"):
                row["institutional_email"] = email
            if doc_programme:
                row["programme"].add(doc_programme)
            if doc_group:
                row["group_labels"].add(doc_group)
            if doc_course:
                row["course_codes"].add(doc_course)
            if doc_role:
                row["roles"].add(doc_role)
            if doc_date:
                row["dates"].add(doc_date)
            row["source_documents"].add(state.get("relative_path", ""))

    rows = []
    for i, row in enumerate(profiles.values(), 1):
        rows.append({
            "record": row.get("record") or f"profile_{i}",
            "entity_type": row.get("entity_type", "student"),
            "full_name": row.get("full_name", ""),
            "student_number": row.get("student_number", ""),
            "institutional_email": row.get("institutional_email", ""),
            "programme": "|".join(ordered_unique(row.get("programme", set()))),
            "group_labels": "|".join(ordered_unique(row.get("group_labels", set()))),
            "course_codes": "|".join(ordered_unique(row.get("course_codes", set()))),
            "roles": "|".join(ordered_unique(row.get("roles", set()))),
            "dates": "|".join(ordered_unique(row.get("dates", set()))),
            "generated_fields": "|".join(ordered_unique(row.get("generated_fields", set()))),
            "source_documents": "|".join(ordered_unique(row.get("source_documents", set()))),
        })
    return rows


def _fake_slug(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", ".", normalise(value)).strip(".")
    return base or "synthetic.person"


def collect_identifier_pools(states, extra_rows=None):
    pools = {
        "student_numbers": [],
        "course_codes": [],
        "programmes": [],
        "group_labels": [],
        "roles": [],
        "dates": [],
    }
    for state in states or []:
        for e in accepted_entities(state):
            label = valid_label(e.get("label", ""))
            surface = str(e.get("surface", "") or "").strip()
            if not surface:
                continue
            if label == "Student number":
                pools["student_numbers"].append(surface)
            elif label == "Course code":
                pools["course_codes"].append(surface)
            elif label == "Programme":
                pools["programmes"].append(surface)
            elif label == "Group label":
                pools["group_labels"].append(surface)
            elif label == "Role":
                pools["roles"].append(surface)
            elif label == "Date":
                pools["dates"].append(surface)
    for row in extra_rows or []:
        if row.get("student_number"):
            pools["student_numbers"].extend(_split_values(row.get("student_number", "")))
        if row.get("course_codes"):
            pools["course_codes"].extend(_split_values(row.get("course_codes", "")))
        if row.get("programme"):
            pools["programmes"].extend(_split_values(row.get("programme", "")))
        if row.get("group_labels"):
            pools["group_labels"].extend(_split_values(row.get("group_labels", "")))
        if row.get("roles"):
            pools["roles"].extend(_split_values(row.get("roles", "")))
        if row.get("dates"):
            pools["dates"].extend(_split_values(row.get("dates", "")))
    return {k: ordered_unique(v) for k, v in pools.items()}


def identifier_pools_for_session(sid, focus_state=None):
    session = load_session(sid)
    states = []
    if focus_state:
        states.append(focus_state)
    for d in session.get("documents", []):
        p = doc_state_path(sid, d["doc_id"])
        if not p.exists():
            continue
        st = load_doc_state(sid, d["doc_id"])
        if not focus_state or st.get("doc_id") != focus_state.get("doc_id"):
            states.append(st)
    rows = []
    for st in states:
        rows.extend(parse_csv_rows(st.get("attacker_database_csv", "")))
    return collect_identifier_pools(states, rows)


def identifier_pools_for_document(state, extra_rows=None):
    # Per-document databases should be generated from values in this document only.
    # If a course/group/date is absent from the document, the generator falls back to KEN...,
    # Group 01, and 2026-01-01 rather than borrowing from another document.
    rows = list(extra_rows or [])
    return collect_identifier_pools([state], rows)


def _next_institutional_student_number(index, used_students):
    # i + seven digits. Start near the Maastricht-style example i1234567 but keep deterministic synthetic values.
    n = 6340000 + index
    candidate = f"i{n:07d}"
    while normalise(candidate) in used_students:
        n += 1
        candidate = f"i{n:07d}"
    return candidate


def generate_missing_identifiers(rows, pools=None):
    """Fill missing database identifiers deterministically, preserving real annotations.

    The generator now favours the values you annotated in the current document/session:
    names, student numbers, course codes, programmes, groups and dates are reused first.
    Only missing values fall back to Maastricht-style synthetic values.
    """
    pools = pools or {}
    rows = list(rows or [])
    generated = []

    student_pool = ordered_unique(pools.get("student_numbers", []))
    course_pool = ordered_unique(pools.get("course_codes", [])) or FALLBACK_COURSE_CODES
    programme_pool = ordered_unique(pools.get("programmes", [])) or FALLBACK_PROGRAMMES
    group_pool = ordered_unique(pools.get("group_labels", [])) or FALLBACK_GROUPS
    date_pool = ordered_unique(pools.get("dates", [])) or ["2026-01-01"]
    shared_programme = _shared_programme_value(programme_pool)
    shared_group = _shared_group_label(group_pool)
    shared_course = _shared_course_code(course_pool)
    shared_date = _shared_date_value(date_pool)
    shared_role = _shared_role_value()

    used_students = {normalise(r.get("student_number", "")) for r in rows if r.get("student_number")}
    used_emails = {normalise(r.get("institutional_email", "")) for r in rows if r.get("institutional_email")}
    candidate_full_names = [r.get("full_name", "") for r in rows if len(_name_tokens(r.get("full_name", ""))) >= 2]

    for i, row in enumerate(rows, 1):
        r = {k: row.get(k, "") for k in DB_FIELDNAMES}
        r["record"] = r.get("record") or f"profile_{i}"
        r["entity_type"] = r.get("entity_type") or "student"
        generated_fields = set(_split_values(r.get("generated_fields", "")))

        if not str(r.get("full_name", "")).strip():
            r["full_name"] = f"Synthetic Person {i:03d}"
            generated_fields.add("full_name")
        else:
            completed_name, generated_last = _complete_partial_full_name(r.get("full_name", ""), i, candidate_full_names)
            if completed_name and completed_name != r.get("full_name", ""):
                r["full_name"] = completed_name
                if generated_last:
                    generated_fields.add("last_name")

        if not str(r.get("student_number", "")).strip():
            candidate = ""
            for possible in student_pool:
                if normalise(possible) not in used_students:
                    candidate = possible
                    break
            if not candidate:
                candidate = _next_institutional_student_number(i, used_students)
            r["student_number"] = candidate
            used_students.add(normalise(candidate))
            generated_fields.add("student_number")

        # Use one shared programme/course/group/date/role for the whole document database.
        # This keeps the attacker database compact and avoids copying every detected date into every row.
        if r.get("programme") != shared_programme:
            if not str(r.get("programme", "")).strip():
                generated_fields.add("programme")
            else:
                generated_fields.add("programme_normalised")
            r["programme"] = shared_programme
        if r.get("group_labels") != shared_group:
            if not str(r.get("group_labels", "")).strip():
                generated_fields.add("group_labels")
            else:
                generated_fields.add("group_labels_normalised")
            r["group_labels"] = shared_group
        if r.get("course_codes") != shared_course:
            if not str(r.get("course_codes", "")).strip():
                generated_fields.add("course_codes")
            else:
                generated_fields.add("course_codes_normalised")
            r["course_codes"] = shared_course
        if r.get("roles") != shared_role:
            if not str(r.get("roles", "")).strip():
                generated_fields.add("roles")
            else:
                generated_fields.add("roles_normalised")
            r["roles"] = shared_role
        if r.get("dates") != shared_date:
            if not str(r.get("dates", "")).strip():
                generated_fields.add("dates")
            else:
                generated_fields.add("dates_normalised")
            r["dates"] = shared_date

        if not str(r.get("institutional_email", "")).strip():
            candidate = _generated_institutional_email(
                r.get("full_name", f"Synthetic Person {i:03d}"),
                r.get("roles", ""),
                r.get("entity_type", ""),
                i,
                used_emails,
            )
            r["institutional_email"] = candidate
            used_emails.add(normalise(candidate))
            generated_fields.add("institutional_email")

        r["generated_fields"] = "|".join(sorted(generated_fields))
        generated.append(r)
    return generated


def db_rows_from_state_or_annotations(state):
    state_csv = state.get("attacker_database_csv", "")
    rows = parse_csv_rows(state_csv)
    if rows:
        return rows
    return build_attacker_rows([state])


def db_rows_from_session_annotations(sid):
    session = load_session(sid)
    states = [load_doc_state(sid, d["doc_id"]) for d in session.get("documents", []) if doc_state_path(sid, d["doc_id"]).exists()]
    rows = []
    for st in states:
        doc_rows = parse_csv_rows(st.get("attacker_database_csv", "")) or build_attacker_rows([st])
        rows.extend(doc_rows)
    return rows

# -- HTML renderers -----------------------------------------------------------


def entity_span_html(e, original_text, label_override=None):
    label = valid_label(label_override or e.get("label", "Other"))
    kind = entity_kind(label)
    eid = h(e.get("id", ""))
    surface = h(original_text)
    mask = h(mask_for_label(label))
    manual_cls = " manual" if e.get("manual") else ""
    linked_cls = " linked" if e.get("linked") else ""
    return (
        f'<span class="entity {kind}{manual_cls}{linked_cls}" data-id="{eid}" data-label="{h(label)}" '
        f'data-masked="1" onclick="openExistingEntity(event, this)" title="Click to edit / reveal">'
        f'<span class="mask-val">{mask}</span><span class="orig-val" style="display:none">{surface}</span></span>'
    )


def render_text_with_position_entities(text, ents):
    text = str(text or "")
    ordered = sorted(ents, key=lambda x: (int(x.get("start", 0)), -(int(x.get("end", 0)) - int(x.get("start", 0)))))
    segs = []
    cursor = 0
    used = 0
    for e in ordered:
        s = max(0, int(e.get("start", 0)))
        en = min(len(text), int(e.get("end", 0)))
        if s < used or en <= s:
            continue
        segs.append(h(text[cursor:s]))
        segs.append(entity_span_html(e, text[s:en]))
        cursor = en
        used = en
    segs.append(h(text[cursor:]))
    return "".join(segs)


def render_text_with_surface_entities(text, state):
    text = str(text or "")
    candidates = []
    for e in accepted_entities(state):
        surface = str(e.get("surface", "") or "").strip()
        if len(surface) < 2:
            continue
        start = 0
        low_text = text.lower()
        low_surface = surface.lower()
        while True:
            idx = low_text.find(low_surface, start)
            if idx < 0:
                break
            candidates.append((idx, idx + len(surface), e))
            start = idx + max(len(surface), 1)
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    segs = []
    cursor = 0
    used = 0
    for s, en, e in candidates:
        if s < used or en <= s:
            continue
        segs.append(h(text[cursor:s]))
        segs.append(entity_span_html(e, text[s:en]))
        cursor = en
        used = en
    segs.append(h(text[cursor:]))
    return "".join(segs)


def html_preview_blocks(state):
    entities_by_path: dict[str, list] = {}
    for e in state.get("entities", []):
        if e.get("status") == "rejected":
            continue
        if int(e.get("start", -1)) < 0 or int(e.get("end", -1)) <= int(e.get("start", -1)):
            continue
        entities_by_path.setdefault(str(e.get("path", "")), []).append(e)

    rendered = []
    for block in state.get("blocks", []):
        text = str(block.get("text", "") or "")
        path = str(block.get("path", ""))
        rendered.append(
            f'<section class="block-card">'
            f'<div class="blk-meta">{h(block.get("region", ""))} · {h(path)}</div>'
            f'<pre class="block selectable-text" data-block="{h(path)}">{render_text_with_position_entities(text, entities_by_path.get(path, []))}</pre>'
            f'</section>'
        )
    return "\n".join(rendered) or '<div class="empty-state">No text blocks were extracted.</div>'


def html_preview_docx_visual(state):
    path = Path(state.get("absolute_path", ""))
    if path.suffix.lower() != ".docx" or not path.exists():
        return html_preview_blocks(state)
    try:
        from docx import Document
        doc = Document(str(path))
        parts = ['<div class="docx-page">']
        for p in doc.paragraphs:
            txt = "".join(run.text for run in p.runs).strip()
            if not txt:
                parts.append('<p class="docx-p blank">&nbsp;</p>')
            else:
                parts.append(f'<p class="docx-p selectable-text" data-block="visual">{render_text_with_surface_entities(txt, state)}</p>')
        for table in doc.tables:
            parts.append('<table class="docx-table">')
            for row in table.rows:
                parts.append('<tr>')
                for cell in row.cells:
                    txt = "\n".join(p.text for p in cell.paragraphs).strip()
                    parts.append(f'<td class="selectable-text" data-block="visual">{render_text_with_surface_entities(txt, state)}</td>')
                parts.append('</tr>')
            parts.append('</table>')
        parts.append('</div>')
        return "".join(parts)
    except Exception:
        return html_preview_blocks(state)


def html_preview_pdf_real(sid, state, zoom=PDF_RENDER_ZOOM):
    if fitz is None:
        return '<div class="warn-box">PyMuPDF is not installed. Install with <code>pip install pymupdf</code>.</div>' + html_preview_blocks(state)
    pdf_path = Path(state.get("absolute_path", ""))
    if not pdf_path.exists():
        return '<div class="warn-box">PDF file not found.</div>'

    ocr_blocks_by_page: dict[int, list] = {}
    for b in state.get("blocks", []):
        rects = clean_pdf_rects(b.get("pdf_rects", []))
        if b.get("ocr") and rects:
            page = int(rects[0]["page"])
            ocr_blocks_by_page.setdefault(page, []).append(b)

    pages_html = []
    total_words = 0
    total_overlays = 0
    with fitz.open(str(pdf_path)) as doc:
        overlays_by_page = build_pdf_overlay_rects(doc, state)
        total_overlays = sum(len(v) for v in overlays_by_page.values())
        for page_no, page in enumerate(doc):
            ensure_pdf_page_image(sid, state, page_no, zoom=zoom)
            page_x0 = float(page.rect.x0)
            page_y0 = float(page.rect.y0)
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            width = int(round(page_w * zoom))
            height = int(round(page_h * zoom))
            img_url = f"{url_for('pdf_page_image', sid=sid, doc_id=state['doc_id'], page_no=page_no)}?v={int(pdf_path.stat().st_mtime)}"
            masked_url = f"{url_for('masked_pdf_page_image', sid=sid, doc_id=state['doc_id'], page_no=page_no)}?v={int(doc_state_path(sid, state['doc_id']).stat().st_mtime)}"

            word_spans = []
            try:
                words = page.get_text("words") or []
            except Exception:
                words = []
            total_words += len(words)
            for wi, w in enumerate(words):
                try:
                    x0, y0, x1, y1, txt = w[:5]
                    txt = str(txt or "")
                    if not txt.strip():
                        continue
                    left, top = (float(x0) - page_x0) * zoom, (float(y0) - page_y0) * zoom
                    ww, hh = max(1.0, (float(x1) - float(x0)) * zoom), max(1.0, (float(y1) - float(y0)) * zoom)
                    fs = max(5.0, hh * 0.78)
                    word_spans.append(
                        f'<span class="pdf-word selectable-token" data-page="{page_no}" data-word-index="{wi}" data-text="{h(txt)}" '
                        f'data-x0="{float(x0):.4f}" data-y0="{float(y0):.4f}" data-x1="{float(x1):.4f}" data-y1="{float(y1):.4f}" '
                        f'style="left:{left:.2f}px;top:{top:.2f}px;width:{ww:.2f}px;height:{hh:.2f}px;font-size:{fs:.2f}px">{h(txt)}</span>'
                    )
                except Exception:
                    continue
            # OCR selectable line layer for scanned PDFs.
            for oi, b in enumerate(ocr_blocks_by_page.get(page_no, [])):
                rects = clean_pdf_rects(b.get("pdf_rects", []))
                if not rects:
                    continue
                r = rects[0]
                text = str(b.get("text", "") or "")
                left, top = (r["x0"] - page_x0) * zoom, (r["y0"] - page_y0) * zoom
                ww, hh = max(2, (r["x1"] - r["x0"]) * zoom), max(2, (r["y1"] - r["y0"]) * zoom)
                fs = max(6, hh * 0.72)
                word_spans.append(
                    f'<span class="pdf-ocr-line selectable-token" data-page="{page_no}" data-word-index="{100000 + oi}" data-text="{h(text)}" '
                    f'data-x0="{r["x0"]:.4f}" data-y0="{r["y0"]:.4f}" data-x1="{r["x1"]:.4f}" data-y1="{r["y1"]:.4f}" '
                    f'style="left:{left:.2f}px;top:{top:.2f}px;width:{ww:.2f}px;height:{hh:.2f}px;font-size:{fs:.2f}px">{h(text)}</span>'
                )

            mask_spans = []
            for ov in overlays_by_page.get(page_no, []):
                label = valid_label(ov.get("label", "Other"))
                kind = entity_kind(label)
                pad = 1.6
                left = max(0.0, ((float(ov["x0"]) - page_x0) * zoom) - pad)
                top = max(0.0, ((float(ov["y0"]) - page_y0) * zoom) - pad)
                ww = max(2.0, ((float(ov["x1"]) - float(ov["x0"])) * zoom) + pad * 2)
                hh = max(2.0, ((float(ov["y1"]) - float(ov["y0"])) * zoom) + pad * 2)
                manual_cls = " manual" if ov.get("manual") else ""
                linked_cls = " linked" if ov.get("linked") else ""
                mask_spans.append(
                    f'<span class="pdf-entity {kind}{manual_cls}{linked_cls}" data-id="{h(ov.get("id", ""))}" data-label="{h(label)}" data-masked="1" '
                    f'onclick="openExistingPdfEntity(event, this)" title="Click to edit / reveal" '
                    f'style="left:{left:.2f}px;top:{top:.2f}px;width:{ww:.2f}px;height:{hh:.2f}px">{h(mask_for_label(label))}</span>'
                )

            pages_html.append(
                f'<div class="pdf-page-shell">'
                f'<div class="pdf-page-label">Page {page_no + 1}</div>'
                f'<div class="pdf-page selectable-root" data-page="{page_no}" data-zoom="{zoom}" '
                f'data-pdf-x0="{page_x0:.6f}" data-pdf-y0="{page_y0:.6f}" '
                f'data-pdf-width="{page_w:.6f}" data-pdf-height="{page_h:.6f}" '
                f'style="width:{width}px;height:{height}px">'
                f'<img class="pdf-img pdf-img-original" src="{img_url}" width="{width}" height="{height}" alt="Original PDF page {page_no + 1}">'
                f'<img class="pdf-img pdf-img-masked" loading="lazy" src="{masked_url}" width="{width}" height="{height}" alt="Masked PDF page {page_no + 1}">'
                f'<div class="pdf-text-layer">{"".join(word_spans)}</div>'
                f'<div class="pdf-mask-layer">{"".join(mask_spans)}</div>'
                f'</div></div>'
            )

    notes = []
    if state.get("ocr", {}).get("enabled"):
        notes.append(f"OCR blocks available: {int(state.get('ocr', {}).get('blocks', 0))}. Drag over OCR text or any area to label it.")
    if accepted_entities(state) and total_overlays == 0:
        notes.append("Detections exist, but no matching PDF coordinates were found. Drag an area on the PDF page to place masks manually.")
    if total_words == 0 and not ocr_blocks_by_page:
        notes.append("This PDF has no selectable text layer. Enable OCR on upload, rerun the document, or drag an area manually.")
    note_html = "".join(f'<div class="pdf-note">{h(n)}</div>' for n in notes)
    return f'<div class="pdf-doc-real">{note_html}{"".join(pages_html)}</div>'


# -- CSS ----------------------------------------------------------------------


CSS = """
<style>
:root{
  --bg:#f4f6fb;--surface:#fff;--surface2:#f8fafc;--border:#d8dee9;--ink:#101828;--muted:#667085;
  --accent:#2563eb;--accent2:#7c3aed;--ok:#15803d;--bad:#b91c1c;--warn:#b45309;
  --direct:#1d4ed8;--direct-bg:#dbeafe;--quasi:#b45309;--quasi-bg:#fef3c7;
  --sans:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono',monospace;
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px} a{text-decoration:none;color:inherit}
.shell{height:100vh;display:grid;grid-template-columns:230px minmax(0,1fr) 320px;grid-template-rows:56px 1fr;overflow:hidden}
.topbar{grid-column:1/-1;background:#0f172a;color:white;display:flex;align-items:center;gap:10px;padding:0 14px;box-shadow:0 2px 12px rgba(15,23,42,.18);z-index:5}
.brand{font-weight:800;letter-spacing:.02em;margin-right:8px}.spacer{flex:1}.top-stat{font-size:12px;color:rgba(255,255,255,.68);white-space:nowrap}.tab-btn,.top-btn{border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.08);color:white;border-radius:999px;padding:7px 12px;font-weight:700;font-size:12px;cursor:pointer}.tab-btn:hover,.top-btn:hover{background:rgba(255,255,255,.16)}.tab-btn.active{background:white;color:#0f172a}.tab-btn.done{background:#dcfce7;color:#15803d;border-color:#86efac}.top-btn.green{background:#15803d;border-color:#15803d}.top-btn.warn{background:#f59e0b;border-color:#f59e0b;color:#111827}.top-btn.active{box-shadow:0 0 0 3px rgba(245,158,11,.25)}
.sidebar{background:var(--surface);border-right:1px solid var(--border);overflow:auto;padding:12px 0}.sidebar h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;padding:0 14px 8px}.doc-row{display:grid;grid-template-columns:30px minmax(0,1fr);align-items:stretch;border-bottom:1px solid #f1f5f9}.doc-row.active{background:#eff6ff;border-left:3px solid var(--accent)}.doc-row.review-complete{background:#fef3c7}.doc-row.review-complete.active{background:#fde68a}.doc-check{display:flex;align-items:center;justify-content:center}.doc-check input{width:16px;height:16px;accent-color:#f59e0b}.doc-item{display:block;padding:10px 14px}.doc-item:hover{background:#f8fafc}.fname{display:block;font-weight:700;font-size:12px;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.fmeta{font-size:11px;color:var(--muted);margin-top:2px}.badge{display:inline-block;border-radius:999px;padding:2px 7px;font-size:10px;font-weight:800}.badge.ready{background:#dcfce7;color:#15803d}.badge.processing{background:#dbeafe;color:#1d4ed8}.badge.queued{background:#f1f5f9;color:#64748b}.badge.error{background:#fee2e2;color:#b91c1c}
.workspace{overflow:auto;padding:16px}.doc-head{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:14px 16px;margin-bottom:12px;display:flex;align-items:center;gap:12px}.doc-head h2{font-size:15px;margin:2px 0}.doc-head .meta{font-size:12px;color:var(--muted)}.head-actions{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}.soft-btn{background:white;border:1px solid var(--border);border-radius:10px;padding:8px 11px;font-size:12px;font-weight:700;cursor:pointer}.soft-btn:hover{border-color:#94a3b8;box-shadow:0 2px 10px rgba(15,23,42,.08)}.soft-btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}.soft-btn.danger{background:#b91c1c;border-color:#b91c1c;color:white}.view-panel.hidden{display:none}.doc-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 8px 22px rgba(15,23,42,.05)}.pdf-card{overflow:auto;padding:16px}.empty-state{color:var(--muted);padding:24px;text-align:center}.warn-box,.info-box,.pdf-note{border-radius:12px;padding:10px 12px;font-size:12px;margin-bottom:10px}.warn-box{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}.info-box,.pdf-note{background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a}
.rpanel{background:var(--surface);border-left:1px solid var(--border);overflow:auto}.rsection{padding:14px;border-bottom:1px solid var(--border)}.rsection h4{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 10px}.hint{font-size:12px;color:var(--muted);line-height:1.45}.edit-form{display:none;gap:8px;flex-direction:column}.edit-form.open{display:flex}.edit-form label{font-size:11px;font-weight:700;color:var(--muted)}.edit-form input,.edit-form select,.label-pop select{width:100%;border:1px solid var(--border);border-radius:10px;padding:8px 10px;font-family:var(--sans);font-size:13px}.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.entity{display:inline;border-radius:5px;padding:1px 4px;cursor:pointer;border-bottom:2px solid transparent;user-select:none}.entity.direct{background:var(--direct-bg);border-bottom-color:var(--direct)}.entity.quasi{background:var(--quasi-bg);border-bottom-color:var(--quasi)}.entity.manual{outline:2px dashed var(--ok);outline-offset:1px}.entity.linked{box-shadow:inset 0 0 0 1px rgba(0,0,0,.08)}.orig-val{background:rgba(255,255,255,.45)}
.block-card{border:1px solid var(--border);border-radius:12px;background:#fff;margin-bottom:10px;padding:12px}.blk-meta{font-size:11px;color:var(--muted);margin-bottom:6px}.block{white-space:pre-wrap;font-family:var(--mono);font-size:13px;line-height:1.75;margin:0}.docx-page{max-width:900px;margin:0 auto;background:white;border:1px solid var(--border);box-shadow:0 12px 30px rgba(15,23,42,.08);padding:54px 64px;min-height:900px}.docx-p{line-height:1.75;margin:0 0 10px}.docx-p.blank{height:12px}.docx-table{border-collapse:collapse;width:100%;margin:14px 0}.docx-table td{border:1px solid #cbd5e1;padding:8px;vertical-align:top;line-height:1.5}
.pdf-doc-real{display:flex;flex-direction:column;align-items:center;gap:18px;min-width:max-content;padding-bottom:28px}.pdf-page-shell{width:max-content}.pdf-page-label{font-size:11px;color:var(--muted);margin:0 0 6px 2px}.pdf-page{box-sizing:content-box;position:relative;background:white;overflow:hidden;box-shadow:0 10px 30px rgba(15,23,42,.2);border:1px solid var(--border)}.pdf-img{position:absolute;inset:0;z-index:1;display:block;user-select:none}.pdf-img-masked{display:none}.pdf-text-layer{position:absolute;inset:0;z-index:2;overflow:hidden}.selectable-token{position:absolute;color:rgba(0,0,0,.008);user-select:text;white-space:pre;line-height:1;font-family:serif}.selectable-token::selection{background:rgba(37,99,235,.35);color:transparent}.pdf-mask-layer{position:absolute;inset:0;z-index:3;pointer-events:none}.pdf-entity{position:absolute;pointer-events:auto;border-radius:4px;display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap;font-size:9px;font-weight:800;font-family:var(--sans);cursor:pointer;border:1px solid rgba(0,0,0,.1);user-select:none}.pdf-entity.direct{background:rgba(29,78,216,.92);color:white}.pdf-entity.quasi{background:rgba(180,83,9,.92);color:white}.pdf-entity.manual{outline:2px dashed var(--ok);outline-offset:1px}.pdf-entity[data-masked="0"]{background:rgba(255,255,255,0);color:transparent;border:1px dashed rgba(37,99,235,.5)}body.pdf-view-masked .pdf-img-original{display:none}body.pdf-view-masked .pdf-img-masked{display:block}body.pdf-view-masked .pdf-mask-layer,body.pdf-view-masked .pdf-text-layer{display:none}.pdf-draw-rect{position:absolute;z-index:10;border:2px solid #f59e0b;background:rgba(245,158,11,.2);pointer-events:none}
.label-pop{position:fixed;z-index:1000;width:310px;background:white;border:1px solid var(--border);border-radius:16px;box-shadow:0 18px 50px rgba(15,23,42,.24);padding:12px;display:none}.label-pop.open{display:block}.pop-title{font-weight:800;font-size:13px;margin-bottom:6px}.pop-preview{font-size:12px;color:var(--muted);background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:8px;max-height:80px;overflow:auto;margin-bottom:8px}.pop-surface{width:100%;min-height:78px;resize:vertical;border:1px solid var(--border);border-radius:10px;padding:8px;font-family:var(--sans);font-size:12px;margin-bottom:8px}.quick-labels{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}.quick-labels button{border:0;border-radius:999px;background:#eef2ff;color:#3730a3;padding:5px 8px;font-size:11px;font-weight:800;cursor:pointer}.pop-actions{display:flex;gap:8px;margin-top:8px}.pop-actions button{flex:1}
.db-card{background:white;border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:0 8px 22px rgba(15,23,42,.05)}.db-toolbar{display:flex;align-items:center;gap:8px;padding:12px;border-bottom:1px solid var(--border);background:#f8fafc;flex-wrap:wrap}.db-toolbar h3{margin:0;font-size:14px}.db-complete-btn.done{background:#15803d;border-color:#15803d;color:white}.db-scroll{overflow:auto;max-height:calc(100vh - 190px)}.db-table{width:100%;border-collapse:collapse;font-size:12px}.db-table th{position:sticky;top:0;background:#f8fafc;z-index:2;text-align:left;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:9px;border-bottom:1px solid var(--border)}.db-table td{padding:6px;border-bottom:1px solid #f1f5f9}.db-cell{width:100%;min-width:140px;border:1px solid transparent;background:transparent;border-radius:8px;padding:7px;font-family:var(--mono);font-size:12px}.db-cell:focus{outline:none;border-color:#bfdbfe;background:#eff6ff}.del-row-btn{border:0;background:transparent;color:#94a3b8;cursor:pointer;font-weight:900}.del-row-btn:hover{color:var(--bad)}
.upload-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}.upload-card{width:520px;background:white;border:1px solid var(--border);border-radius:20px;box-shadow:0 14px 40px rgba(15,23,42,.1);padding:32px}.upload-card h1{margin:0 0 8px}.upload-card p{color:var(--muted);line-height:1.5}.upload-card input[type=file]{width:100%;border:1px dashed #94a3b8;border-radius:14px;padding:16px;margin:14px 0}.field-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}.field-row label{font-size:12px;font-weight:700}.submit{width:100%;padding:13px;border:0;border-radius:12px;background:var(--accent);color:#fff;font-weight:800}.dep-pill{font-size:12px;padding:8px;border-radius:10px;border:1px solid var(--border);background:#f8fafc;margin-top:6px}.dep-pill.ok{background:#dcfce7;color:#15803d}.dep-pill.bad{background:#fee2e2;color:#b91c1c}.queue-wrap{padding:24px}.queue-header{display:flex;gap:12px;align-items:center;margin-bottom:18px}.queue-card{background:white;border:1px solid var(--border);border-radius:16px;overflow:hidden}.q-row{display:flex;align-items:center;gap:10px;padding:12px 14px;border-bottom:1px solid #f1f5f9}.qfile{font-weight:800;flex:1}.qmeta{font-size:11px;color:var(--muted)}.progress-bar{height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;width:220px}.progress-bar>div{height:100%;background:var(--accent)}
body.db-tab .shell{grid-template-columns:230px minmax(0,1fr)}body.db-tab .rpanel{display:none}body.db-tab .workspace{grid-column:2}body.db-tab .doc-head .head-actions{display:none}
@media(max-width:1000px){.shell{grid-template-columns:1fr;grid-template-rows:56px auto 1fr auto}.sidebar{max-height:180px}.rpanel{max-height:320px}.topbar{overflow-x:auto}.docx-page{padding:28px}}
</style>
"""


# -- Page templates -----------------------------------------------------------


def dependency_pills():
    pills = []
    if IMPORT_ERROR is None:
        pills.append('<div class="dep-pill ok">Defender pipeline: available</div>')
    else:
        pills.append(f'<div class="dep-pill bad">Defender pipeline failed: {h(IMPORT_ERROR)}</div>')
    if fitz is not None:
        pills.append('<div class="dep-pill ok">PyMuPDF PDF preview/export: available</div>')
    else:
        pills.append(f'<div class="dep-pill bad">PyMuPDF missing: <code>pip install pymupdf</code></div>')
    if easyocr_available():
        pills.append('<div class="dep-pill ok">EasyOCR: available. Tick OCR to scan every PDF page.</div>')
    else:
        pills.append('<div class="dep-pill">EasyOCR optional: <code>pip install easyocr</code></div>')
    return "".join(pills)


UPLOAD_HTML = CSS + f"""
<div class="upload-wrap">
  <div class="upload-card">
    <h1>RQ3 Label &amp; Anonymise</h1>
    <p>Upload a ZIP. PDFs open as real pages. Each document now has its own editable identity database, saved into the export ZIP.</p>
    {dependency_pills()}
    <form method="post" action="/upload" enctype="multipart/form-data">
      <input type="file" name="zipfile" accept=".zip" required>
      <div class="field-row">
        <label><input type="checkbox" name="ner" checked> NER detection</label>
        <label><input type="checkbox" name="ocr" {"checked" if easyocr_available() else ""}> OCR every PDF page</label>
        <label>Workers <input type="number" name="workers" value="2" min="1" max="4" style="width:58px"></label>
      </div>
      <button class="submit" type="submit">Upload &amp; process</button>
    </form>
  </div>
</div>
"""


def render_queue(sid, stats):
    total = stats["total"]
    ready = stats["counts"].get("ready", 0)
    percent = int(stats["ready_share"] * 100)
    rows = []
    for d in stats["docs"]:
        review_button = ""
        if d["processing_status"] == "ready":
            review_button = f'<a href="{url_for("review_page", sid=sid, doc_id=d["doc_id"])}"><button class="soft-btn primary">Review</button></a>'
        db_text = "DB checked" if d.get("database_complete") else "DB open"
        rows.append(f"""
        <div class="q-row">
          <span class="badge {h(d['processing_status'])}">{h(d['processing_status'])}</span>
          <span class="qfile">{h(d['file_name'])}</span>
          <span class="qmeta">{h(d['relative_path'])} · {d['accepted']} found · {db_text}</span>
          {review_button}
        </div>""")
    return CSS + f"""
    <div class="queue-wrap">
      <div class="queue-header">
        <strong style="font-size:18px">Processing queue</strong>
        <div class="progress-bar"><div style="width:{percent}%"></div></div>
        <span style="font-size:13px;color:#667085">{ready}/{total} ready</span>
        <a href="{url_for('export_session', sid=sid)}"><button class="soft-btn primary">Export ZIP</button></a>
      </div>
      <div class="queue-card">{''.join(rows)}</div>
    </div>
    <script>
    const SID={js_json(sid)};
    async function poll(){{
      const r=await fetch('/api/status/'+SID); const d=await r.json();
      if(d.first_ready) window.location.href='/review/'+SID+'/'+d.first_ready;
      else setTimeout(poll,2500);
    }}
    if({str(ready == 0).lower()}) setTimeout(poll,1500);
    </script>
    """


def render_db_table(db_rows, db_cols):
    thead = "<tr><th>#</th>" + "".join(f"<th>{h(title)}</th>" for key, title in db_cols) + "<th></th></tr>"
    body = []
    for i, row in enumerate(db_rows):
        cells = "".join(
            f'<td><input class="db-cell" data-col="{h(key)}" value="{h(row.get(key, ""))}"></td>'
            for key, _ in db_cols
        )
        body.append(f'<tr id="dbrow-{i}"><td style="color:var(--muted);font-size:11px">{i+1}</td>{cells}<td><button class="del-row-btn" onclick="this.parentElement.parentElement.remove()">×</button></td></tr>')
    return f'<table class="db-table"><thead>{thead}</thead><tbody id="dbBody">{"".join(body)}</tbody></table>'


def render_review(sid, doc_id):
    session = load_session(sid)
    state = load_doc_state(sid, doc_id)
    docs = session.get("documents", [])
    is_pdf = state.get("extension") == ".pdf"

    accepted_n = sum(1 for e in state.get("entities", []) if e.get("status") in {"accepted", "manual"})
    rejected_n = sum(1 for e in state.get("entities", []) if e.get("status") == "rejected")

    doc_links = []
    for d in docs:
        st = load_doc_state(sid, d["doc_id"])
        status = st.get("processing_status", "queued")
        active = "active" if d["doc_id"] == doc_id else ""
        ann_done = st.get("review_status") == "annotation_complete"
        db_done = bool(st.get("database_complete"))
        done_cls = "review-complete" if ann_done and db_done else ("review-complete" if ann_done else "")
        checked = "checked" if ann_done else ""
        done_text = "IDs checked" if ann_done else "IDs open"
        db_text = "DB checked" if db_done else "DB open"
        acc = sum(1 for e in st.get("entities", []) if e.get("status") in {"accepted", "manual"})
        href = url_for("review_page", sid=sid, doc_id=d["doc_id"]) if status == "ready" else "#"
        doc_links.append(
            f'<div class="doc-row {active} {done_cls}" data-doc-id="{h(d["doc_id"])}">'
            f'<label class="doc-check" title="Mark document identifiers checked"><input type="checkbox" {checked} onclick="toggleDocComplete(event, {js_json(d["doc_id"])}, this.checked)"></label>'
            f'<a class="doc-item" href="{href}">'
            f'<span class="badge {h(status)}">{h(status)}</span><span class="fname">{h(d["file_name"])}</span>'
            f'<div class="fmeta">{acc} PII found · {done_text} · {db_text}</div></a></div>'
        )

    errors = state.get("errors", [])
    err_html = "".join(f'<div class="warn-box">{h(e)}</div>' for e in errors)

    review_html = html_preview_pdf_real(sid, state) if is_pdf else html_preview_docx_visual(state)
    blocks_html = html_preview_blocks(state)
    doc_card_class = "doc-card pdf-card" if is_pdf else "doc-card"

    db_rows = db_rows_from_state_or_annotations(state)
    db_complete = bool(state.get("database_complete"))
    db_tab_cls = "done" if db_complete else ""
    db_tab_text = "Database ✓" if db_complete else "Database"
    db_button_cls = "soft-btn db-complete-btn done" if db_complete else "soft-btn db-complete-btn"
    db_button_text = "Database complete ✓" if db_complete else "Mark this DB complete"
    db_table = render_db_table(db_rows, DB_COLUMNS)
    label_opts = "".join(f'<option value="{h(l)}">{h(l)}</option>' for l in LABEL_OPTIONS)
    quick_label_buttons = "".join(f'<button type="button" onclick="setPopupLabel({js_json(l)})">{h(l)}</button>' for l in ["FULL_NAME", "Student number", "Institutional email", "Course code", "Date", "Other"])

    pdf_download = ""
    ocr_btn = ""
    if is_pdf:
        pdf_download = f'<a href="{url_for("serve_document_file", sid=sid, doc_id=doc_id, kind="masked_pdf")}?download=1"><button class="top-btn">Masked PDF</button></a>'
        ocr_btn = f'<a href="{url_for("rerun_doc_ocr", sid=sid, doc_id=doc_id)}"><button class="top-btn">Run OCR</button></a>'

    script_vars = f"""
<script>
const SID = {js_json(sid)};
const DCID = {js_json(doc_id)};
const LABEL_OPTIONS = {js_json(LABEL_OPTIONS)};
let ENTITIES = {js_json(state.get('entities', []))};
let DB_COLS = {js_json(DB_FIELDNAMES)};
</script>
"""

    return CSS + f"""
<div class="shell">
  <div class="topbar">
    <span class="brand">RQ3 Review <span style="font-size:10px;opacity:.65">per-doc-db</span></span>
    <button class="tab-btn active" data-tab="review" onclick="setTab('review')">Document</button>
    <button class="tab-btn" data-tab="blocks" onclick="setTab('blocks')">Blocks</button>
    <button class="tab-btn {db_tab_cls}" data-tab="database" onclick="setTab('database')">{db_tab_text}</button>
    {ocr_btn}
    <span class="spacer"></span>
    <span class="top-stat">{accepted_n} accepted · {rejected_n} rejected</span>
    {pdf_download}
    <a href="{url_for('serve_document_file', sid=sid, doc_id=doc_id, kind='original')}?download=1"><button class="top-btn">Original</button></a>
    <a href="{url_for('export_session', sid=sid)}"><button class="top-btn green">Export ZIP</button></a>
  </div>

  <div class="sidebar"><h3>Documents</h3>{''.join(doc_links)}</div>

  <main class="workspace" id="workspace">
    <div class="doc-head">
      <div><div class="meta">{h(state['relative_path'])}</div><h2>{h(state['file_name'])}</h2></div>
      <div class="head-actions">
        <button class="soft-btn" onclick="revealAll()">Reveal masks</button>
        <button class="soft-btn" onclick="maskAll()">Hide masks</button>
        <button class="soft-btn danger" onclick="removeCurrentDoc()">Remove duplicate document</button>
      </div>
    </div>
    {err_html}

    <section id="panel-review" class="view-panel">
      <div class="{doc_card_class}" id="docCard">{review_html}</div>
    </section>

    <section id="panel-blocks" class="view-panel hidden">
      <div class="doc-card" id="blocksCard">{blocks_html}</div>
    </section>

    <section id="panel-database" class="view-panel hidden">
      <div class="db-card">
        <div class="db-toolbar">
          <h3>Identity database for this document</h3>
          <span class="hint">Saved separately for <strong>{h(state['file_name'])}</strong>. Generation reuses observed student numbers and course codes before synthetic fallbacks.</span>
          <span class="spacer"></span>
          <button class="soft-btn" onclick="addDbRow()">Add row</button>
          <button class="soft-btn" onclick="generateDb()">Generate from current labels</button>
          <button class="soft-btn primary" onclick="saveDb()">Save this DB</button>
          <button class="soft-btn" onclick="regenDb()">Regenerate from annotations</button>
          <button class="{db_button_cls}" onclick="toggleDbComplete({str(not db_complete).lower()})">{db_button_text}</button>
        </div>
        <div class="db-scroll">{db_table}</div>
      </div>
    </section>
  </main>

  <aside class="rpanel">
    <div class="rsection">
      <h4>Add PII</h4>
      <div class="hint">Select text in DOCX/text previews, or drag over text/areas on PDF pages. A label popup opens immediately.</div>
    </div>
    <div class="rsection">
      <h4>Edit selected detection</h4>
      <div class="hint" id="editEmpty">Click a highlighted mask to edit or remove it.</div>
      <div class="edit-form" id="editForm">
        <input type="hidden" id="eid">
        <label>Label</label><select id="eLabel">{label_opts}</select>
        <label>Surface text</label><input id="eSurface" type="text">
        <div class="row2"><div><label>Status</label><select id="eStatus"><option>accepted</option><option>manual</option><option>rejected</option></select></div><div><label>Source</label><input id="eSource" disabled></div></div>
        <div class="row2"><button class="soft-btn primary" onclick="saveEntity()">Save</button><button class="soft-btn danger" onclick="deleteEntity()">Delete this</button></div>
        <button class="soft-btn danger" onclick="deleteSimilarEntities()">Delete all same text</button>
      </div>
    </div>
  </aside>
</div>

<div class="label-pop" id="labelPopover">
  <div class="pop-title" id="popTitle">Label selected text</div>
  <div class="pop-preview" id="popPreview"></div>
  <textarea class="pop-surface" id="popSurface" placeholder="Edit the exact selected text before adding"></textarea>
  <select id="popLabel">{label_opts}</select>
  <div class="quick-labels">{quick_label_buttons}</div>
  <div class="pop-actions"><button class="soft-btn" onclick="closeLabelPopover()">Cancel</button><button class="soft-btn primary" id="popAddBtn" onclick="confirmPendingLabel()">Add</button></div>
</div>
{script_vars}
{REVIEW_JS}
"""


REVIEW_JS = r"""
<script>
let PENDING_ANNOTATION = null;
let DRAW_STATE = null;

function escSel(value){
  if (window.CSS && CSS.escape) return CSS.escape(String(value));
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function setTab(tab){
  ['review','blocks','database'].forEach(t => {
    const panel = document.getElementById('panel-' + t);
    if (panel) panel.classList.toggle('hidden', t !== tab);
    document.querySelectorAll(`.tab-btn[data-tab="${t}"]`).forEach(b => b.classList.toggle('active', t === tab));
  });
  document.body.classList.toggle('db-tab', tab === 'database');
  localStorage.setItem('rq3_active_tab', tab);
  closeLabelPopover();
}

function revealAll(){
  document.querySelectorAll('.entity').forEach(el => {
    const m = el.querySelector('.mask-val'), o = el.querySelector('.orig-val');
    if (m) m.style.display = 'none'; if (o) o.style.display = '';
    el.dataset.masked = '0';
  });
  document.querySelectorAll('.pdf-entity').forEach(el => { el.dataset.masked = '0'; });
}

function maskAll(){
  document.querySelectorAll('.entity').forEach(el => {
    const m = el.querySelector('.mask-val'), o = el.querySelector('.orig-val');
    if (m) m.style.display = ''; if (o) o.style.display = 'none';
    el.dataset.masked = '1';
  });
  document.querySelectorAll('.pdf-entity').forEach(el => { el.dataset.masked = '1'; });
}

function openExistingEntity(event, el){
  if (event) event.stopPropagation();
  const masked = el.dataset.masked !== '0';
  const m = el.querySelector('.mask-val'), o = el.querySelector('.orig-val');
  if (m) m.style.display = masked ? 'none' : '';
  if (o) o.style.display = masked ? '' : 'none';
  el.dataset.masked = masked ? '0' : '1';
  openEntityEditor(el.dataset.id);
}

function openExistingPdfEntity(event, el){
  if (event) event.stopPropagation();
  const id = el.dataset.id;
  const masked = el.dataset.masked !== '0';
  document.querySelectorAll(`.pdf-entity[data-id="${escSel(id)}"]`).forEach(box => { box.dataset.masked = masked ? '0' : '1'; });
  openEntityEditor(id);
}

function openEntityEditor(id){
  const e = ENTITIES.find(x => x.id === id);
  if (!e) return;
  document.getElementById('editEmpty').style.display = 'none';
  document.getElementById('editForm').classList.add('open');
  document.getElementById('eid').value = e.id;
  document.getElementById('eLabel').value = LABEL_OPTIONS.includes(e.label) ? e.label : 'Other';
  document.getElementById('eSurface').value = e.surface || '';
  document.getElementById('eStatus').value = e.status || 'accepted';
  document.getElementById('eSource').value = e.source || '';
  document.querySelectorAll('.entity,.pdf-entity').forEach(x => x.style.outline = '');
  document.querySelectorAll(`.entity[data-id="${escSel(id)}"],.pdf-entity[data-id="${escSel(id)}"]`).forEach(x => x.style.outline = '2px solid #2563eb');
}

async function postJson(url, payload){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function toggleDocComplete(event, docId, complete){
  if (event) event.stopPropagation();
  await postJson(`/api/doc/${SID}/${docId}/complete`, {complete});
  const row = document.querySelector(`.doc-row[data-doc-id="${escSel(docId)}"]`);
  if (row) row.classList.toggle('review-complete', complete);
}

async function generateDb(){
  await postJson(`/api/db/${SID}/${DCID}/generate`, {});
  location.reload();
}

async function toggleDbComplete(complete){
  await postJson(`/api/db/${SID}/${DCID}/complete`, {complete, rows: dbRowsFromTable()});
  location.reload();
}

async function saveEntity(){
  const id = document.getElementById('eid').value;
  if (!id) return alert('Click a detection first.');
  await postJson(`/api/entity/${SID}/${DCID}`, {
    id,
    label: document.getElementById('eLabel').value,
    surface: document.getElementById('eSurface').value,
    status: document.getElementById('eStatus').value
  });
  location.reload();
}

async function deleteEntity(){
  const id = document.getElementById('eid').value;
  if (!id) return alert('Click a detection first.');
  if (!confirm('Delete this detection?')) return;
  await postJson(`/api/entity/${SID}/${DCID}/delete`, {id});
  location.reload();
}

async function deleteSimilarEntities(){
  const id = document.getElementById('eid').value;
  if (!id) return alert('Click a detection first.');
  const e = ENTITIES.find(x => x.id === id);
  const label = e ? (e.label || 'Other') : 'this label';
  const surface = e ? (e.surface || '') : '';
  if (!confirm(`Delete all detections with text "${surface}" in this document?`)) return;
  await postJson(`/api/entity/${SID}/${DCID}/delete_similar`, {id});
  location.reload();
}

async function removeCurrentDoc(){
  if (!confirm('Remove this document from the session/export? Use this for duplicates. The saved state file will remain in the workspace, but it will no longer appear in this review session.')) return;
  const d = await postJson(`/api/doc/${SID}/${DCID}/remove`, {});
  if (d.next_doc) location.href = `/review/${SID}/${d.next_doc}`;
  else location.href = `/session/${SID}`;
}

function rangeRect(range){
  let rect = range.getBoundingClientRect();
  if ((!rect || (!rect.width && !rect.height)) && range.getClientRects().length) rect = range.getClientRects()[0];
  return rect;
}

function mergePdfRects(rects){
  if (!rects.length) return [];
  const sorted = rects.slice().sort((a,b) => a.page-b.page || a.y0-b.y0 || a.x0-b.x0);
  const groups = [];
  sorted.forEach(r => {
    const mid = (r.y0 + r.y1) / 2;
    const h = Math.max(1, r.y1-r.y0);
    const last = groups.length ? groups[groups.length-1] : null;
    if (last && last.page === r.page && Math.abs(last.mid - mid) <= Math.max(3, h*0.60)) {
      last.x0 = Math.min(last.x0, r.x0); last.y0 = Math.min(last.y0, r.y0);
      last.x1 = Math.max(last.x1, r.x1); last.y1 = Math.max(last.y1, r.y1); last.mid = (last.y0 + last.y1)/2;
    } else groups.push({...r, mid});
  });
  return groups.map(g => ({page:g.page, x0:g.x0, y0:g.y0, x1:g.x1, y1:g.y1}));
}

function captureSelection(){
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return null;
  const selectedText = sel.toString().trim();
  if (!selectedText) return null;
  const range = sel.getRangeAt(0);
  let node = range.commonAncestorContainer;
  if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
  if (!node || !document.getElementById('workspace').contains(node)) return null;
  if (node.closest && node.closest('#labelPopover')) return null;

  let pdfPage = node.closest ? node.closest('.pdf-page') : null;
  if (!pdfPage) {
    const maybeTokens = [...document.querySelectorAll('.selectable-token')].filter(w => {
      try { return range.intersectsNode(w); } catch { return false; }
    });
    if (maybeTokens.length) pdfPage = maybeTokens[0].closest('.pdf-page');
  }
  if (pdfPage) {
    const pageNo = Number(pdfPage.dataset.page);
    let tokens = [...pdfPage.querySelectorAll('.selectable-token')].filter(w => {
      try { return range.intersectsNode(w); } catch { return false; }
    });
    tokens = tokens.filter(w => w.dataset.text && w.dataset.text.trim()).sort((a,b) => Number(a.dataset.wordIndex)-Number(b.dataset.wordIndex));
    const tokenItems = tokens.map(w => ({
      text: w.dataset.text || '',
      rect: {page: pageNo, x0: Number(w.dataset.x0), y0: Number(w.dataset.y0), x1: Number(w.dataset.x1), y1: Number(w.dataset.y1)}
    })).filter(t => Number.isFinite(t.rect.x0) && Number.isFinite(t.rect.x1));
    const rects = mergePdfRects(tokenItems.map(t => t.rect));
    return {block_path:`pdf:${pageNo}`, start:-1, end:-1, surface:selectedText, pdf_rects:rects, token_items:tokenItems, kind:'selection'};
  }

  const block = node.closest ? node.closest('.block') : null;
  if (block) {
    const pre = document.createRange();
    pre.selectNodeContents(block);
    try { pre.setEnd(range.startContainer, range.startOffset); } catch {}
    const start = pre.toString().length;
    return {block_path:block.dataset.block || 'visual', start, end:start+selectedText.length, surface:selectedText, kind:'selection'};
  }

  const visual = node.closest ? node.closest('.selectable-text') : null;
  if (visual) return {block_path:visual.dataset.block || 'visual', start:-1, end:-1, surface:selectedText, kind:'selection'};
  return null;
}

function showLabelPopover(annotation, rect){
  PENDING_ANNOTATION = annotation;
  const pop = document.getElementById('labelPopover');
  document.getElementById('popTitle').textContent = annotation.kind === 'box' ? 'Label drawn box' : 'Label selected text';
  const preview = annotation.surface ? annotation.surface : `Box on page ${(annotation.pdf_rects?.[0]?.page ?? 0) + 1}`;
  document.getElementById('popPreview').textContent = preview;
  document.getElementById('popSurface').value = annotation.surface || '';
  const last = localStorage.getItem('rq3_last_label') || 'FULL_NAME';
  document.getElementById('popLabel').value = LABEL_OPTIONS.includes(last) ? last : 'FULL_NAME';
  pop.classList.add('open');
  const w = 310, h = 230;
  let left = rect ? rect.left : window.innerWidth/2 - w/2;
  let top = rect ? rect.bottom + 8 : window.innerHeight/2 - h/2;
  left = Math.max(8, Math.min(window.innerWidth - w - 8, left));
  top = Math.max(8, Math.min(window.innerHeight - h - 8, top));
  pop.style.left = left + 'px'; pop.style.top = top + 'px';
}

function closeLabelPopover(){
  document.getElementById('labelPopover').classList.remove('open');
  PENDING_ANNOTATION = null;
  const surfaceBox = document.getElementById('popSurface');
  if (surfaceBox) surfaceBox.value = '';
}

function normTok(s){ return String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim(); }

function rectsFromEditedSurface(annotation, editedSurface){
  if (!annotation.token_items || !annotation.token_items.length || !editedSurface.trim()) return annotation.pdf_rects || [];
  const wanted = normTok(editedSurface).split(/\s+/).filter(Boolean);
  const toks = annotation.token_items.map(t => ({...t, norm:normTok(t.text)})).filter(t => t.norm);
  if (!wanted.length || !toks.length) return annotation.pdf_rects || [];
  for (let i=0; i<=toks.length-wanted.length; i++){
    let ok = true;
    for (let j=0; j<wanted.length; j++){ if (toks[i+j].norm !== wanted[j]) { ok=false; break; } }
    if (ok) return mergePdfRects(toks.slice(i, i+wanted.length).map(t => t.rect));
  }
  const counts = {}; wanted.forEach(w => counts[w] = (counts[w] || 0) + 1);
  const picked = [];
  toks.forEach(t => { if (counts[t.norm] > 0) { picked.push(t.rect); counts[t.norm]--; } });
  return picked.length ? mergePdfRects(picked) : (annotation.pdf_rects || []);
}

function setPopupLabel(label){
  document.getElementById('popLabel').value = label;
}

async function confirmPendingLabel(){
  if (!PENDING_ANNOTATION) return;
  const label = document.getElementById('popLabel').value || 'Other';
  const editedSurface = document.getElementById('popSurface').value.trim();
  localStorage.setItem('rq3_last_label', label);
  const btn = document.getElementById('popAddBtn');
  btn.disabled = true;
  try {
    const payload = {...PENDING_ANNOTATION, surface: editedSurface, label, status:'manual'};
    if (PENDING_ANNOTATION.token_items) payload.pdf_rects = rectsFromEditedSurface(PENDING_ANNOTATION, editedSurface);
    delete payload.token_items;
    await postJson(`/api/entity/${SID}/${DCID}/add`, payload);
    location.reload();
  } catch (err) {
    btn.disabled = false;
    alert('Could not add label: ' + err.message);
  }
}

let selectionTimer = null;
document.addEventListener('mouseup', e => {
  if ((e.target.closest && e.target.closest('#labelPopover')) || DRAW_STATE) return;
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(() => {
    const info = captureSelection();
    if (!info) return;
    const sel = window.getSelection();
    const rect = sel && sel.rangeCount ? rangeRect(sel.getRangeAt(0)) : null;
    showLabelPopover(info, rect);
  }, 60);
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLabelPopover();
  if (PENDING_ANNOTATION && e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    confirmPendingLabel();
  }
});

function pdfPoint(page, event){
  const r = page.getBoundingClientRect();
  const style = window.getComputedStyle(page);
  const borderLeft = parseFloat(style.borderLeftWidth || '0') || 0;
  const borderTop = parseFloat(style.borderTopWidth || '0') || 0;
  const contentW = Math.max(1, page.clientWidth);
  const contentH = Math.max(1, page.clientHeight);
  const xCss = Math.max(0, Math.min(contentW, event.clientX - r.left - borderLeft));
  const yCss = Math.max(0, Math.min(contentH, event.clientY - r.top - borderTop));
  const pdfX0 = Number(page.dataset.pdfX0 || '0');
  const pdfY0 = Number(page.dataset.pdfY0 || '0');
  const pdfW = Number(page.dataset.pdfWidth || (contentW / Number(page.dataset.zoom || '1')));
  const pdfH = Number(page.dataset.pdfHeight || (contentH / Number(page.dataset.zoom || '1')));
  return {
    xCss, yCss,
    xPdf: pdfX0 + xCss * pdfW / contentW,
    yPdf: pdfY0 + yCss * pdfH / contentH
  };
}

document.addEventListener('mousedown', e => {
  const page = e.target.closest && e.target.closest('.pdf-page');
  if (!page) return;
  if (e.target.closest && (e.target.closest('.pdf-entity') || e.target.closest('#labelPopover'))) return;
  e.preventDefault();
  closeLabelPopover();
  const p = pdfPoint(page, e);
  const box = document.createElement('div');
  box.className = 'pdf-draw-rect';
  box.style.left = p.xCss + 'px'; box.style.top = p.yCss + 'px';
  box.style.width = '1px'; box.style.height = '1px';
  page.appendChild(box);
  DRAW_STATE = {page, box, start:p};
});

document.addEventListener('mousemove', e => {
  if (!DRAW_STATE) return;
  const p = pdfPoint(DRAW_STATE.page, e);
  const x0 = Math.min(DRAW_STATE.start.xCss, p.xCss), y0 = Math.min(DRAW_STATE.start.yCss, p.yCss);
  const x1 = Math.max(DRAW_STATE.start.xCss, p.xCss), y1 = Math.max(DRAW_STATE.start.yCss, p.yCss);
  DRAW_STATE.box.style.left = x0 + 'px'; DRAW_STATE.box.style.top = y0 + 'px';
  DRAW_STATE.box.style.width = (x1-x0) + 'px'; DRAW_STATE.box.style.height = (y1-y0) + 'px';
});

function overlapRatio(a,b){
  const x0 = Math.max(a.x0,b.x0), y0 = Math.max(a.y0,b.y0);
  const x1 = Math.min(a.x1,b.x1), y1 = Math.min(a.y1,b.y1);
  if (x1 <= x0 || y1 <= y0) return 0;
  const inter = (x1-x0)*(y1-y0);
  const bArea = Math.max(1e-6, (b.x1-b.x0)*(b.y1-b.y0));
  return inter / bArea;
}

document.addEventListener('mouseup', e => {
  if (!DRAW_STATE) return;
  const st = DRAW_STATE; DRAW_STATE = null;
  const p = pdfPoint(st.page, e);
  const x0 = Math.min(st.start.xCss, p.xCss), y0 = Math.min(st.start.yCss, p.yCss);
  const x1 = Math.max(st.start.xCss, p.xCss), y1 = Math.max(st.start.yCss, p.yCss);
  const px0 = Math.min(st.start.xPdf, p.xPdf), py0 = Math.min(st.start.yPdf, p.yPdf);
  const px1 = Math.max(st.start.xPdf, p.xPdf), py1 = Math.max(st.start.yPdf, p.yPdf);
  st.box.remove();
  if ((x1-x0) < 6 || (y1-y0) < 6) return;
  const pageNo = Number(st.page.dataset.page);
  const selected = {page:pageNo, x0:px0, y0:py0, x1:px1, y1:py1};
  let tokens = [...st.page.querySelectorAll('.selectable-token')].filter(tok => {
    const r = {page:pageNo, x0:Number(tok.dataset.x0), y0:Number(tok.dataset.y0), x1:Number(tok.dataset.x1), y1:Number(tok.dataset.y1)};
    return Number.isFinite(r.x0) && overlapRatio(selected, r) >= 0.35;
  });
  tokens = tokens.sort((a,b) => Number(a.dataset.y0)-Number(b.dataset.y0) || Number(a.dataset.x0)-Number(b.dataset.x0));
  const tokenItems = tokens.map(t => ({
    text: t.dataset.text || '',
    rect: {page:pageNo, x0:Number(t.dataset.x0), y0:Number(t.dataset.y0), x1:Number(t.dataset.x1), y1:Number(t.dataset.y1)}
  })).filter(t => Number.isFinite(t.rect.x0) && Number.isFinite(t.rect.x1));
  const surface = tokenItems.map(t => t.text).join(' ').trim();
  const rects = tokenItems.length ? mergePdfRects(tokenItems.map(t => t.rect)) : [selected];
  showLabelPopover({
    block_path:`pdf:${pageNo}`, start:-1, end:-1, surface, kind: surface ? 'selection' : 'box',
    pdf_rects: rects, token_items:tokenItems
  }, {left:e.clientX, bottom:e.clientY});
});

function addDbRow(){
  const tbody = document.getElementById('dbBody');
  const tr = document.createElement('tr');
  tr.innerHTML = '<td style="color:var(--muted);font-size:11px">' + (tbody.rows.length + 1) + '</td>' +
    DB_COLS.map(c => `<td><input class="db-cell" data-col="${c}" value=""></td>`).join('') +
    `<td><button class="del-row-btn" onclick="this.parentElement.parentElement.remove()">×</button></td>`;
  tbody.appendChild(tr);
}

function dbRowsFromTable(){
  const rows = [];
  document.querySelectorAll('#dbBody tr').forEach(tr => {
    const row = {};
    tr.querySelectorAll('.db-cell').forEach(inp => row[inp.dataset.col] = inp.value);
    if (Object.values(row).some(v => String(v || '').trim())) rows.push(row);
  });
  return rows;
}

async function saveDb(){
  await postJson(`/api/db/${SID}/${DCID}`, {rows: dbRowsFromTable()});
  alert('Document database saved.');
}

async function regenDb(){
  if (!confirm('Regenerate this document database from current annotations? This overwrites database edits for this document only.')) return;
  await fetch(`/api/db/${SID}/${DCID}/regen`, {method:'POST'});
  location.reload();
}

document.getElementById('labelPopover').addEventListener('mousedown', e => e.stopPropagation());
window.addEventListener('load', () => {
  const tab = localStorage.getItem('rq3_active_tab') || 'review';
  setTab(tab);
});
</script>
"""


# -- Flask app ---------------------------------------------------------------


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


@app.get("/")
def index():
    return UPLOAD_HTML


@app.post("/upload")
def upload():
    uploaded = request.files.get("zipfile")
    if not uploaded:
        return redirect(url_for("index"))
    sid = uuid.uuid4().hex[:12]
    sdir = session_dir(sid)
    upload_dir = sdir / "uploaded"
    extract_dir = sdir / "source_documents"
    upload_dir.mkdir(parents=True, exist_ok=True)
    zip_path = upload_dir / "input.zip"
    uploaded.save(zip_path)
    try:
        files = materialize_cached_documents(zip_path, extract_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(sdir, ignore_errors=True)
        return CSS + '<div class="upload-wrap"><div class="upload-card"><h1>Invalid ZIP</h1><p>The file could not be opened as a ZIP archive.</p><a href="/"><button class="submit">Back</button></a></div></div>', 400
    docs = collect_documents(files, extract_dir)
    if not docs:
        shutil.rmtree(sdir, ignore_errors=True)
        return CSS + '<div class="upload-wrap"><div class="upload-card"><h1>No supported documents</h1><p>The ZIP must contain .pdf, .docx, .txt or .md files.</p><a href="/"><button class="submit">Back</button></a></div></div>', 400
    try:
        workers = max(1, min(int(request.form.get("workers", 2)), 4))
    except Exception:
        workers = 2
    session = {
        "sid": sid,
        "created_at": time.time(),
        "root": str(ROOT),
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "documents": docs,
        "settings": {
            "ner": request.form.get("ner") == "on",
            "ocr": request.form.get("ocr") == "on",
            "workers": workers,
            "ocr_min_confidence": OCR_MIN_CONFIDENCE,
        },
    }
    save_session(sid, session)
    for doc in docs:
        save_doc_state(sid, make_initial_state(doc))
    start_workers(sid, count=workers)
    return redirect(url_for("session_page", sid=sid))


@app.get("/session/<sid>")
def session_page(sid):
    stats = status_counts(sid)
    if stats["first_ready"] and stats["counts"].get("processing", 0) == 0 and stats["counts"].get("queued", 0) == 0:
        return redirect(url_for("review_page", sid=sid, doc_id=stats["first_ready"]))
    return render_queue(sid, stats)


@app.get("/review/<sid>/<doc_id>")
def review_page(sid, doc_id):
    state = load_doc_state(sid, doc_id)
    if state.get("processing_status") in {"queued", "processing"}:
        return f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2>Processing {h(state['file_name'])}</h2><p>{h(state.get('progress_message',''))}</p>
        <script>setTimeout(()=>location.reload(),1500)</script></body></html>"""
    return render_review(sid, doc_id)


@app.get("/resume/<sid>")
def resume_session(sid):
    start_workers(sid)
    return redirect(url_for("session_page", sid=sid))


@app.get("/rerun/<sid>/<doc_id>")
def rerun_doc(sid, doc_id):
    state = load_doc_state(sid, doc_id)
    state.update({"processing_status": "queued", "progress_message": "Queued for re-run"})
    save_doc_state(sid, state)
    start_workers(sid, {doc_id}, count=1)
    return redirect(url_for("review_page", sid=sid, doc_id=doc_id))


@app.get("/rerun_ocr/<sid>/<doc_id>")
def rerun_doc_ocr(sid, doc_id):
    session = load_session(sid)
    session.setdefault("settings", {})["ocr"] = True
    save_session(sid, session)
    state = load_doc_state(sid, doc_id)
    state.update({"processing_status": "queued", "progress_message": "Queued for OCR re-run"})
    save_doc_state(sid, state)
    start_workers(sid, {doc_id}, count=1)
    return redirect(url_for("review_page", sid=sid, doc_id=doc_id))


@app.post("/api/doc/<sid>/<doc_id>/complete")
def api_doc_complete(sid, doc_id):
    data = request.get_json(force=True)
    state = load_doc_state(sid, doc_id)
    complete = bool(data.get("complete"))
    state["review_status"] = "annotation_complete" if complete else "in progress"
    state["annotation_complete_at"] = time.time() if complete else None
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "complete": complete})


@app.post("/api/doc/<sid>/<doc_id>/remove")
def api_remove_document(sid, doc_id):
    session = load_session(sid)
    docs = list(session.get("documents", []))
    removed = next((d for d in docs if d.get("doc_id") == doc_id), None)
    session["documents"] = [d for d in docs if d.get("doc_id") != doc_id]
    save_session(sid, session)
    if doc_state_path(sid, doc_id).exists():
        state = load_doc_state(sid, doc_id)
        state["removed"] = True
        state["removed_at"] = time.time()
        state["review_status"] = "removed"
        state["updated_at"] = time.time()
        save_doc_state(sid, state)
    next_doc = ""
    for d in session.get("documents", []):
        try:
            st = load_doc_state(sid, d["doc_id"])
        except Exception:
            continue
        if st.get("processing_status") == "ready":
            next_doc = d["doc_id"]
            break
    return jsonify({"ok": True, "removed": bool(removed), "next_doc": next_doc})


@app.post("/api/db/<sid>/<doc_id>")
def api_save_doc_db(sid, doc_id):
    data = request.get_json(force=True)
    rows = data.get("rows")
    if rows is None:
        rows = parse_csv_rows(data.get("csv", ""))
    state = load_doc_state(sid, doc_id)
    state["attacker_database_csv"] = rows_to_csv(rows, DB_FIELDNAMES)
    # Saving should not remove the database checkmark. Mark/unmark with the complete button.
    state["database_complete"] = bool(state.get("database_complete"))
    state["database_completed_at"] = state.get("database_completed_at")
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "rows": len(rows or [])})


@app.post("/api/db/<sid>/<doc_id>/generate")
def api_generate_doc_db(sid, doc_id):
    state = load_doc_state(sid, doc_id)
    # Refresh from the current accepted/manual annotations so newly labelled first names,
    # last names, student numbers and course codes immediately shape the database.
    rows = build_attacker_rows([state]) or db_rows_from_state_or_annotations(state)
    pools = identifier_pools_for_document(state, rows)
    rows = generate_missing_identifiers(rows, pools)
    state["attacker_database_csv"] = rows_to_csv(rows, DB_FIELDNAMES)
    state["database_complete"] = False
    state["database_completed_at"] = None
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "rows": len(rows)})


@app.post("/api/db/<sid>/<doc_id>/complete")
def api_complete_doc_db(sid, doc_id):
    data = request.get_json(force=True)
    complete = bool(data.get("complete"))
    state = load_doc_state(sid, doc_id)
    incoming_rows = data.get("rows")
    if incoming_rows is None:
        incoming_rows = parse_csv_rows(data.get("csv", "")) if data.get("csv") else None
    base_rows = incoming_rows if incoming_rows is not None else db_rows_from_state_or_annotations(state)
    if complete:
        # First save the current table as seen in the browser, then fill/normalise missing fields.
        pools = identifier_pools_for_document(state, base_rows)
        rows = generate_missing_identifiers(base_rows, pools)
        state["attacker_database_csv"] = rows_to_csv(rows, DB_FIELDNAMES)
        state["database_completed_at"] = time.time()
    else:
        state["attacker_database_csv"] = rows_to_csv(base_rows, DB_FIELDNAMES)
        state["database_completed_at"] = None
    state["database_complete"] = complete
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "complete": complete})


@app.post("/api/db/<sid>/<doc_id>/regen")
def api_regen_doc_db(sid, doc_id):
    state = load_doc_state(sid, doc_id)
    rows = build_attacker_rows([state])
    pools = identifier_pools_for_document(state, rows)
    rows = generate_missing_identifiers(rows, pools)
    state["attacker_database_csv"] = rows_to_csv(rows, DB_FIELDNAMES)
    state["database_complete"] = False
    state["database_completed_at"] = None
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "rows": len(rows)})


@app.get("/api/status/<sid>")
def api_status(sid):
    return jsonify(status_counts(sid))


@app.get("/api/pdf-page/<sid>/<doc_id>/<int:page_no>.png")
def pdf_page_image(sid, doc_id, page_no):
    state = load_doc_state(sid, doc_id)
    if state.get("extension") != ".pdf":
        return "Not a PDF", 404
    try:
        path = ensure_pdf_page_image(sid, state, page_no)
    except Exception as exc:
        return f"PDF render failed: {exc}", 500
    return send_file(path, mimetype="image/png")


@app.get("/api/masked-pdf-page/<sid>/<doc_id>/<int:page_no>.png")
def masked_pdf_page_image(sid, doc_id, page_no):
    state = load_doc_state(sid, doc_id)
    if state.get("extension") != ".pdf":
        return "Not a PDF", 404
    try:
        path = ensure_masked_pdf_page_image(sid, state, page_no)
    except Exception as exc:
        return f"Masked PDF render failed: {exc}", 500
    return send_file(path, mimetype="image/png")


@app.get("/file/<sid>/<doc_id>/<kind>")
def serve_document_file(sid, doc_id, kind):
    state = load_doc_state(sid, doc_id)
    if kind == "original":
        path = Path(state.get("absolute_path", ""))
        name = state.get("file_name", path.name)
    elif kind == "pipeline":
        path = Path(state.get("pipeline_anonymized_path", ""))
        name = f"pipeline_{state.get('file_name', '')}"
    elif kind == "masked_pdf":
        original = Path(state.get("absolute_path", ""))
        if original.suffix.lower() != ".pdf":
            return "Masked PDF is only available for PDF files", 404
        try:
            path = ensure_masked_pdf_file(sid, state)
        except Exception as exc:
            return f"Could not generate masked PDF: {exc}", 500
        name = f"masked_{state.get('file_name', original.name)}"
    else:
        return "Unknown", 404
    if not path.exists():
        return "File not available", 404
    return send_file(path, as_attachment=bool(request.args.get("download")), download_name=name)


@app.post("/api/entity/<sid>/<doc_id>")
def api_update_entity(sid, doc_id):
    data = request.get_json(force=True)
    state = load_doc_state(sid, doc_id)
    uid = str(data.get("id", ""))
    new_status = data.get("status", "accepted")
    changed_seed = False
    for e in state.get("entities", []):
        if str(e.get("id", "")) == uid:
            old_label = e.get("label", "")
            old_surface = e.get("surface", "")
            e["label"] = valid_label(data.get("label", e.get("label", "Other")))
            e["surface"] = data.get("surface", e.get("surface", ""))
            e["status"] = new_status if new_status in {"accepted", "manual", "rejected"} else "accepted"
            if e["status"] == "manual":
                e["manual"] = True
            e["reviewed_at"] = time.time()
            changed_seed = old_label != e["label"] or old_surface != e["surface"]
            break
    if changed_seed or new_status == "rejected":
        state = remove_linked_for_seed(state, uid)
    if new_status in {"accepted", "manual"}:
        state = apply_cross_region_linking(state, source="review")
    if state.get("review_status") != "annotation_complete":
        state["review_status"] = "in progress"
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True})


@app.post("/api/entity/<sid>/<doc_id>/delete")
def api_delete_entity(sid, doc_id):
    data = request.get_json(force=True)
    eid = str(data.get("id", ""))
    state = load_doc_state(sid, doc_id)
    before = len(state.get("entities", []))
    state["entities"] = [
        e for e in state.get("entities", [])
        if str(e.get("id", "")) != eid and not (e.get("linked") and e.get("linked_from") == eid)
    ]
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "deleted": before - len(state.get("entities", []))})


@app.post("/api/entity/<sid>/<doc_id>/delete_similar")
def api_delete_similar_entities(sid, doc_id):
    data = request.get_json(force=True)
    eid = str(data.get("id", ""))
    state = load_doc_state(sid, doc_id)
    target = next((e for e in state.get("entities", []) if str(e.get("id", "")) == eid), None)
    if not target:
        return jsonify({"ok": False, "error": "Entity not found"}), 404
    target_surface = normalise(target.get("surface", ""))
    if not target_surface:
        return jsonify({"ok": False, "error": "Selected entity has no surface text"}), 400
    delete_ids = {
        str(e.get("id", ""))
        for e in state.get("entities", [])
        if normalise(e.get("surface", "")) == target_surface
    }
    before = len(state.get("entities", []))
    state["entities"] = [
        e for e in state.get("entities", [])
        if str(e.get("id", "")) not in delete_ids and not (e.get("linked") and e.get("linked_from") in delete_ids)
    ]
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "deleted": before - len(state.get("entities", []))})


@app.post("/api/entity/<sid>/<doc_id>/add")
def api_add_entity(sid, doc_id):
    data = request.get_json(force=True)
    state = load_doc_state(sid, doc_id)
    bpath = str(data.get("block_path", "visual"))
    block = next((b for b in state.get("blocks", []) if b.get("path") == bpath), None)
    try:
        start = int(data.get("start", -1))
        end = int(data.get("end", -1))
    except Exception:
        start = end = -1
    if block and start >= 0 and end > start:
        surface = block.get("text", "")[start:end]
    else:
        surface = str(data.get("surface", ""))
    pdf_rects = clean_pdf_rects(data.get("pdf_rects", []))
    if not surface and not pdf_rects:
        return jsonify({"ok": False, "error": "No selected text or box coordinates received"}), 400
    new_id = f"m_{uuid.uuid4().hex[:8]}"
    new_entity = {
        "id": new_id,
        "label": valid_label(data.get("label", "Other")),
        "surface": surface,
        "path": bpath,
        "start": start,
        "end": end,
        "region": block.get("region", "body") if block else ("pdf" if bpath.startswith("pdf:") else "visual"),
        "score": "manual",
        "source": "manual_review",
        "status": "manual",
        "review_note": "manual addition",
        "manual": True,
        "reviewed_at": time.time(),
    }
    if pdf_rects:
        new_entity["pdf_rects"] = pdf_rects
    state.setdefault("entities", []).append(new_entity)
    state = apply_cross_region_linking(state, source="manual")
    if state.get("review_status") != "annotation_complete":
        state["review_status"] = "in progress"
    state["updated_at"] = time.time()
    save_doc_state(sid, state)
    return jsonify({"ok": True, "id": new_id})


# -- export ------------------------------------------------------------------


def write_redacted_text(state, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ents = accepted_entities(state)
    parts = []
    for block in state.get("blocks", []):
        text = block.get("text", "")
        for e in sorted(ents, key=lambda x: len(str(x.get("surface", ""))), reverse=True):
            s = str(e.get("surface", ""))
            if s:
                text = text.replace(s, mask_for_label(e.get("label", "")))
        parts.append(f"\n--- {block.get('region', '')} | {block.get('path', '')} ---\n{text}")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_redacted_docx(original, state, out_path):
    if original.suffix.lower() != ".docx":
        return False
    try:
        from docx import Document
        doc = Document(str(original))
        entities = accepted_entities(state)

        def redact(text):
            for e in sorted(entities, key=lambda x: len(str(x.get("surface", ""))), reverse=True):
                s = str(e.get("surface", ""))
                if s:
                    text = text.replace(s, mask_for_label(e.get("label", "")))
            return text

        def visit(container):
            for p in getattr(container, "paragraphs", []):
                full = "".join(r.text for r in p.runs)
                redacted = redact(full)
                if redacted != full:
                    if p.runs:
                        for r in p.runs:
                            r.text = ""
                        p.runs[0].text = redacted
                    else:
                        p.add_run(redacted)
            for table in getattr(container, "tables", []):
                for row in table.rows:
                    for cell in row.cells:
                        visit(cell)
        visit(doc)
        for section in doc.sections:
            visit(section.header)
            visit(section.footer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
        return True
    except Exception:
        return False


def write_redacted_pdf(original, state, out_path):
    if original.suffix.lower() != ".pdf" or fitz is None:
        return False
    doc = None
    try:
        doc = fitz.open(str(original))
        redaction_keys = set()
        for e in accepted_entities(state):
            label = valid_label(e.get("label", "Other"))
            replacement = mask_for_label(label)
            surface = str(e.get("surface", "") or "").strip()
            for r in clean_pdf_rects(e.get("pdf_rects", [])):
                page_no = int(r["page"])
                if page_no < 0 or page_no >= len(doc):
                    continue
                key = (page_no, round(r["x0"], 2), round(r["y0"], 2), round(r["x1"], 2), round(r["y1"], 2), replacement)
                if key in redaction_keys:
                    continue
                redaction_keys.add(key)
                rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                doc[page_no].add_redact_annot(rect, text=replacement, fill=(1, 1, 1), text_color=(0, 0, 0), fontsize=6, cross_out=False)
            if surface and re.search(r"\w", surface):
                for needle in _entity_surface_variants(surface):
                    for page_no, page in enumerate(doc):
                        try:
                            hits = page.search_for(needle)
                        except Exception:
                            hits = []
                        for rect in hits:
                            key = (page_no, round(float(rect.x0), 2), round(float(rect.y0), 2), round(float(rect.x1), 2), round(float(rect.y1), 2), replacement)
                            if key in redaction_keys:
                                continue
                            redaction_keys.add(key)
                            page.add_redact_annot(rect, text=replacement, fill=(1, 1, 1), text_color=(0, 0, 0), fontsize=6, cross_out=False)
        for page in doc:
            try:
                page.apply_redactions(images=getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", 2))
            except TypeError:
                page.apply_redactions()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path), garbage=4, deflate=True, clean=True)
        doc.close()
        return True
    except Exception:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
        return False


def _verification_surfaces(state):
    seen = set()
    surfaces = []
    for e in accepted_entities(state):
        surface = str(e.get("surface", "") or "").strip()
        if len(surface) < 2 or not re.search(r"\w", surface):
            continue
        key = (surface.lower(), valid_label(e.get("label", "")))
        if key in seen:
            continue
        seen.add(key)
        surfaces.append({
            "entity_id": e.get("id", ""),
            "label": valid_label(e.get("label", "")),
            "surface": surface,
        })
    return surfaces


def verify_redacted_text_file(path, state, artifact_kind):
    rows = []
    if not path.exists():
        return rows
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return rows
    low = text.lower()
    for item in _verification_surfaces(state):
        if item["surface"].lower() in low:
            rows.append({
                "doc_id": state.get("doc_id", ""),
                "relative_path": state.get("relative_path", ""),
                "artifact_kind": artifact_kind,
                "entity_id": item["entity_id"],
                "label": item["label"],
                "surface": item["surface"],
                "location": "text",
                "hits": 1,
                "status": "possible_leak",
            })
    return rows


def _extract_docx_text(path):
    try:
        from docx import Document
        doc = Document(str(path))
        parts = []
        def visit(container):
            for p in getattr(container, "paragraphs", []):
                parts.append(p.text)
            for table in getattr(container, "tables", []):
                for row in table.rows:
                    for cell in row.cells:
                        visit(cell)
        visit(doc)
        for section in doc.sections:
            visit(section.header)
            visit(section.footer)
        return "\n".join(parts)
    except Exception:
        return ""


def verify_redacted_docx(path, state):
    rows = []
    text = _extract_docx_text(path).lower() if path.exists() else ""
    if not text:
        return rows
    for item in _verification_surfaces(state):
        if item["surface"].lower() in text:
            rows.append({
                "doc_id": state.get("doc_id", ""),
                "relative_path": state.get("relative_path", ""),
                "artifact_kind": "redacted_docx",
                "entity_id": item["entity_id"],
                "label": item["label"],
                "surface": item["surface"],
                "location": "docx_text",
                "hits": 1,
                "status": "possible_leak",
            })
    return rows


def verify_redacted_pdf(path, state):
    rows = []
    if fitz is None or not path.exists():
        return rows
    try:
        with fitz.open(str(path)) as doc:
            for item in _verification_surfaces(state):
                total_hits = 0
                locations = []
                for page_no, page in enumerate(doc):
                    try:
                        hits = page.search_for(item["surface"])
                    except Exception:
                        hits = []
                    if hits:
                        total_hits += len(hits)
                        locations.append(str(page_no + 1))
                if total_hits:
                    rows.append({
                        "doc_id": state.get("doc_id", ""),
                        "relative_path": state.get("relative_path", ""),
                        "artifact_kind": "redacted_pdf",
                        "entity_id": item["entity_id"],
                        "label": item["label"],
                        "surface": item["surface"],
                        "location": "pages " + ",".join(locations),
                        "hits": total_hits,
                        "status": "possible_leak",
                    })
    except Exception:
        return rows
    return rows


def write_csv_rows(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if fieldnames:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("")


def document_label_summary_rows(states):
    rows = []
    for state in states:
        row = {
            "doc_id": state.get("doc_id", ""),
            "relative_path": state.get("relative_path", ""),
            "file_name": state.get("file_name", ""),
            "review_status": state.get("review_status", "not reviewed"),
            "database_complete": bool(state.get("database_complete")),
            "total_accepted": 0,
            "total_manual": 0,
            "total_rejected": 0,
        }
        for label in LABEL_OPTIONS:
            row[f"accepted_{label}"] = 0
            row[f"manual_{label}"] = 0
            row[f"rejected_{label}"] = 0
        for e in state.get("entities", []):
            label = valid_label(e.get("label", "Other"))
            status = e.get("status", "accepted")
            if status in {"accepted", "manual"}:
                row["total_accepted"] += 1
                row[f"accepted_{label}"] += 1
            if e.get("manual") or status == "manual":
                row["total_manual"] += 1
                row[f"manual_{label}"] += 1
            if status == "rejected":
                row["total_rejected"] += 1
                row[f"rejected_{label}"] += 1
        rows.append(row)
    return rows


def build_review_summary(session, states):
    return {
        "sid": session.get("sid", ""),
        "documents_total": len(states),
        "documents_ready": sum(1 for s in states if s.get("processing_status") == "ready"),
        "documents_annotation_complete": sum(1 for s in states if s.get("review_status") == "annotation_complete"),
        "documents_database_complete": sum(1 for s in states if s.get("database_complete")),
        "accepted_entities_total": sum(1 for s in states for e in s.get("entities", []) if e.get("status") in {"accepted", "manual"}),
        "manual_entities_total": sum(1 for s in states for e in s.get("entities", []) if e.get("manual") or e.get("status") == "manual"),
        "rejected_entities_total": sum(1 for s in states for e in s.get("entities", []) if e.get("status") == "rejected"),
        "exported_at": time.time(),
    }


def export_session_zip(sid):
    session = load_session(sid)
    export_dir = session_dir(sid) / "export_build"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    states = [load_doc_state(sid, d["doc_id"]) for d in session.get("documents", []) if doc_state_path(sid, d["doc_id"]).exists()]
    annotation_lines = []
    audit_rows = []
    verification_rows = []
    combined_db_rows = []

    # Save raw review state so that work can always be resumed/audited from the export.
    saved_state_dir = export_dir / "saved_review_state"
    saved_state_dir.mkdir(parents=True, exist_ok=True)
    (saved_state_dir / "session.json").write_text(json.dumps(safe_obj(session), indent=2, ensure_ascii=False), encoding="utf-8")

    for state in states:
        rel = Path(state["relative_path"])
        original = Path(state["absolute_path"])
        (saved_state_dir / "doc_states").mkdir(parents=True, exist_ok=True)
        (saved_state_dir / "doc_states" / f"{state['doc_id']}.json").write_text(json.dumps(safe_obj(state), indent=2, ensure_ascii=False), encoding="utf-8")

        redacted_text_path = export_dir / "redacted_text" / rel.with_suffix(".txt")
        write_redacted_text(state, redacted_text_path)
        verification_rows.extend(verify_redacted_text_file(redacted_text_path, state, "redacted_text"))

        redacted_doc_path = export_dir / "redacted_documents" / rel
        if original.suffix.lower() == ".pdf":
            wrote_redacted = write_redacted_pdf(original, state, redacted_doc_path)
            if wrote_redacted:
                verification_rows.extend(verify_redacted_pdf(redacted_doc_path, state))
        elif original.suffix.lower() == ".docx":
            wrote_redacted = write_redacted_docx(original, state, redacted_doc_path)
            if wrote_redacted:
                verification_rows.extend(verify_redacted_docx(redacted_doc_path, state))
        else:
            wrote_redacted = False

        if not wrote_redacted and original.exists():
            t = export_dir / "original_documents" / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, t)

        for e in state.get("entities", []):
            row = {"doc_id": state["doc_id"], "relative_path": state["relative_path"], **e}
            audit_rows.append(row)
            if e.get("status") in {"accepted", "manual"}:
                annotation_lines.append(json.dumps(row, ensure_ascii=False))

        doc_db_rows = parse_csv_rows(state.get("attacker_database_csv", "")) or build_attacker_rows([state])
        if not state.get("attacker_database_csv"):
            pools = identifier_pools_for_document(state, doc_db_rows)
            doc_db_rows = generate_missing_identifiers(doc_db_rows, pools)
        for r in doc_db_rows:
            rr = dict(r)
            rr["doc_id"] = state["doc_id"]
            rr["document_relative_path"] = state["relative_path"]
            combined_db_rows.append(rr)
        db_rel_name = f"{state['doc_id']}_{slug(Path(state['relative_path']).stem)}_profiles.csv"
        (export_dir / "attacker_database" / "by_document").mkdir(parents=True, exist_ok=True)
        (export_dir / "attacker_database" / "by_document" / db_rel_name).write_text(rows_to_csv(doc_db_rows, DB_FIELDNAMES), encoding="utf-8")

    jl = export_dir / "annotations" / "entities.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("\n".join(annotation_lines), encoding="utf-8")

    if audit_rows:
        keys = sorted({k for r in audit_rows for k in r.keys()})
        write_csv_rows(export_dir / "audit" / "review_audit.csv", audit_rows, keys)
    else:
        write_csv_rows(export_dir / "audit" / "review_audit.csv", [])

    summary_rows = document_label_summary_rows(states)
    if summary_rows:
        write_csv_rows(export_dir / "audit" / "document_label_summary.csv", summary_rows, list(summary_rows[0].keys()))
    else:
        write_csv_rows(export_dir / "audit" / "document_label_summary.csv", [])

    verification_fields = ["doc_id", "relative_path", "artifact_kind", "entity_id", "label", "surface", "location", "hits", "status"]
    write_csv_rows(export_dir / "audit" / "redaction_verification.csv", verification_rows, verification_fields)

    combined_fields = ["doc_id", "document_relative_path", *DB_FIELDNAMES]
    write_csv_rows(export_dir / "attacker_database" / "profiles.csv", combined_db_rows, combined_fields)

    review_summary = build_review_summary(session, states)
    manifest = safe_obj(session)
    manifest["review_summary"] = review_summary
    manifest["export_contents"] = {
        "redacted_documents": "Redacted PDF/DOCX outputs where supported.",
        "redacted_text": "Plain-text redacted version for every document.",
        "annotations/entities.jsonl": "Accepted/manual annotations.",
        "audit/review_audit.csv": "All detections, including accepted, manual, linked and rejected.",
        "audit/document_label_summary.csv": "Per-document label and review counts.",
        "audit/redaction_verification.csv": "Surfaces still found after redaction, if any.",
        "attacker_database/profiles.csv": "Combined database with document columns.",
        "attacker_database/by_document": "Separate profile database CSV for each document.",
        "saved_review_state": "Session and document JSON states for reproducibility.",
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""RQ3 export package
==================

Session: {sid}
Exported documents: {len(states)}
Annotation-complete documents: {review_summary['documents_annotation_complete']}
Database-complete documents: {review_summary['documents_database_complete']}
Accepted/manual entities: {review_summary['accepted_entities_total']}

Contents
--------
1. redacted_documents/: redacted PDF/DOCX outputs where supported.
2. redacted_text/: plain-text redacted versions for every document.
3. annotations/entities.jsonl: accepted/manual entity annotations.
4. audit/review_audit.csv: all detections and review decisions.
5. audit/document_label_summary.csv: per-document label counts.
6. audit/redaction_verification.csv: possible remaining surfaces after redaction.
7. attacker_database/profiles.csv: combined attacker database with document references.
8. attacker_database/by_document/: one attacker database CSV per document.
9. saved_review_state/: saved session.json and every document state JSON.
10. manifest.json: settings, document list, review summary and export metadata.

Notes
-----
- The UI saves each annotation/database edit to the document state immediately.
- The per-document database is stored in each document state and exported separately.
- Generated missing identifiers reuse observed names, student numbers and course codes first.
- First-name-only rows are completed with a matching annotated full name when possible, otherwise with a generated surname.
- Synthetic fallback student numbers use the local institutional shape i + seven digits.
"""
    (export_dir / "README_RQ3_EXPORT.txt").write_text(readme, encoding="utf-8")

    zip_path = session_dir(sid) / f"rq3_export_{sid}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in export_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(export_dir).as_posix())
    return zip_path


@app.get("/export/<sid>")
def export_session(sid):
    path = export_session_zip(sid)
    return send_file(path, as_attachment=True, download_name=path.name)


if __name__ == "__main__":
    print("Open: http://127.0.0.1:7998")
    app.run(host="127.0.0.1", port=7998, debug=False, threaded=True)
