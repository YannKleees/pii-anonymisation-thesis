"""Streamlit UI for document generation and defense pipeline orchestration.

AI USAGE DISCLOSURE
-------------------
This user interface module (UI/ui.py) — including its layout, styling, ZIP
upload handling, and all visual presentation logic — was written entirely by
an AI assistant (Anthropic's Claude). No portion of this file was authored
by a human; humans reviewed and integrated the output. The underlying
document generation and defender pipelines invoked from this UI are
authored separately and are not covered by this disclosure.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

BRAND_PRIMARY = "#40b19b"
BRAND_BACKGROUND = "#fffdf3"
BRAND_TEXT = "#173f39"
BRAND_MUTED = "#64706d"
BRAND_BORDER = "#d8ebe7"
BRAND_SOFT = "#ecfaf7"
APP_BG = "#f4f6f5"

SUPPORTED_SUFFIXES = {".docx", ".pdf"}
ZIP_SUFFIXES = {".zip"}
UPLOAD_EXTENSIONS = ["docx", "pdf", "zip"]
ZIP_COPY_CHUNK = 1024 * 1024
PREVIEW_HEIGHT = 740
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MIME = "application/pdf"

def _find_repo_root() -> Path:
	current = Path(__file__).resolve().parent
	for base in [current, *current.parents]:
		if (base / "thesis_project").exists():
			return base
	raise FileNotFoundError("Repository root not found: missing thesis_project directory.")


PROJECT_ROOT = _find_repo_root()
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _load_defender():
	try:
		from thesis_project.defender.pipeline import process_many, write_audit_jsonl
		from thesis_project.shared import Config
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			f"Missing dependency: {exc.name}. Run `python -m pip install -r requirements.txt`. Full error: {exc}"
		) from exc
	except ImportError as exc:
		raise RuntimeError(
			f"Import error: {exc}"
		) from exc
	except SyntaxError as exc:
		raise RuntimeError(f"Syntax error in defender pipeline: {exc}") from exc
	return process_many, write_audit_jsonl, Config


def _load_editor_tools():
	try:
		from thesis_project.defender.parser import normalise_blocks, parse_document
		from thesis_project.defender.redact import write
		from thesis_project.shared import Config, RedactionAction
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			f"Missing dependency: {exc.name}. Run `python -m pip install -r requirements.txt`."
		) from exc
	return normalise_blocks, parse_document, write, Config, RedactionAction


def _init_state() -> None:
	st.session_state.setdefault("workspace", None)
	st.session_state.setdefault("batch", [])
	st.session_state.setdefault("selected_id", None)
	st.session_state.setdefault("results", {})
	st.session_state.setdefault("uploaded_fingerprints", set())


def _ensure_workspace() -> Path:
	current = st.session_state.get("workspace")
	if current and Path(current).exists():
		return Path(current)
	root = Path(tempfile.mkdtemp(prefix="redaction_review_ui_"))
	for name in ["batch", "outputs", "edited_outputs", "previews"]:
		(root / name).mkdir(parents=True, exist_ok=True)
	st.session_state.workspace = str(root)
	return root


def _reset_workspace() -> None:
	old = st.session_state.get("workspace")
	if old:
		shutil.rmtree(old, ignore_errors=True)
	st.session_state.workspace = None
	st.session_state.batch = []
	st.session_state.selected_id = None
	st.session_state.results = {}
	st.session_state.uploaded_fingerprints = set()
	_ensure_workspace()


def _safe_name(name: str) -> str:
	cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in str(name))
	return cleaned.strip().replace(" ", "_") or "document"


def _unique_target(directory: Path, filename: str) -> Path:
	directory.mkdir(parents=True, exist_ok=True)
	filename = _safe_name(filename)
	stem = Path(filename).stem
	suffix = Path(filename).suffix.lower()
	target = directory / f"{stem}{suffix}"
	counter = 2
	while target.exists():
		target = directory / f"{stem}_{counter}{suffix}"
		counter += 1
	return target


def _short_name(path_or_name: str, limit: int = 38) -> str:
	name = Path(path_or_name).name
	return name if len(name) <= limit else f"{name[: limit - 12]}...{name[-9:]}"


def _mime(path: Path) -> str:
	if path.suffix.lower() == ".pdf":
		return PDF_MIME
	if path.suffix.lower() == ".docx":
		return DOCX_MIME
	return "application/octet-stream"


def _asdict_list(items: Iterable[Any]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for item in items or []:
		if isinstance(item, dict):
			rows.append(dict(item))
		else:
			try:
				rows.append(asdict(item))
			except Exception:
				rows.append(dict(getattr(item, "__dict__", {})))
	return rows


def _find_office_binary() -> str | None:
	for candidate in [
		"soffice",
		"libreoffice",
		"C:/Program Files/LibreOffice/program/soffice.exe",
		"C:/Program Files (x86)/LibreOffice/program/soffice.exe",
	]:
		path = Path(candidate)
		if path.exists():
			return str(path)
		resolved = shutil.which(candidate)
		if resolved:
			return resolved
	return None


def _theme() -> None:
	st.html(
		f"""
<style>
:root {{
	--brand:{BRAND_PRIMARY}; --brand-dark:#2f8b78; --brand-soft:{BRAND_SOFT};
	--text:{BRAND_TEXT}; --text-soft:#2c5a52; --muted:{BRAND_MUTED};
	--line:{BRAND_BORDER}; --app:{APP_BG}; --panel:#ffffff; --panel-warm:#fbfefd;
	--shadow-sm:0 1px 2px rgba(15,50,40,.05);
	--shadow-md:0 4px 18px rgba(15,50,40,.08);
	--radius:14px;
}}
.stApp {{
	background:var(--app); color:var(--text);
	font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif;
}}
.main .block-container {{ max-width:1640px; padding:1.3rem 1.8rem 2.4rem; }}
section[data-testid="stSidebar"] {{
	background:linear-gradient(180deg,#fdfffe 0%,#f5fbf9 100%)!important;
	border-right:1px solid var(--line);
}}
section[data-testid="stSidebar"] .block-container {{ padding:1.2rem 1rem 1.8rem!important; }}
h1,h2,h3,h4 {{ color:var(--text); letter-spacing:-.02em; font-weight:650; }}
.topbar {{
	display:flex; align-items:center; gap:1rem;
	padding:.2rem 0 1.05rem; margin-bottom:1.2rem;
	border-bottom:1px solid var(--line);
}}
.topbar-brand {{ display:flex; align-items:center; gap:.75rem; }}
.topbar-mark {{
	width:32px; height:32px; border-radius:10px;
	background:linear-gradient(135deg,var(--brand) 0%,var(--brand-dark) 100%);
	box-shadow:0 6px 14px rgba(64,177,155,.32), inset 0 1px 0 rgba(255,255,255,.4);
	position:relative; flex-shrink:0;
}}
.topbar-mark::after {{
	content:''; position:absolute; inset:9px;
	border:1.5px solid rgba(255,255,255,.95); border-radius:3px;
	border-top-color:transparent;
}}
.topbar-meta h1 {{ margin:0; font-size:1.05rem; font-weight:650; letter-spacing:-.02em; line-height:1.2; }}
.topbar-sub {{ color:var(--muted); font-size:.76rem; margin-top:.15rem; }}
.topbar-spacer {{ flex:1; }}
.topbar-pill {{
	display:inline-flex; align-items:center; gap:.5rem;
	padding:.38rem .78rem; border-radius:999px;
	background:var(--brand-soft); color:var(--brand-dark);
	font-size:.73rem; font-weight:600;
	border:1px solid #c8e8e1; box-shadow:var(--shadow-sm);
}}
.topbar-pill::before {{
	content:''; width:7px; height:7px; border-radius:999px;
	background:var(--brand); box-shadow:0 0 0 3px rgba(64,177,155,.22);
}}
.section-label {{
	margin:1.15rem 0 .5rem; color:var(--muted);
	font-size:.66rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
}}
section[data-testid="stSidebar"] .section-label:first-of-type {{ margin-top:.2rem; }}
.empty-state {{
	background:linear-gradient(180deg,#ffffff 0%,#fbfefd 100%);
	border:1px solid var(--line); border-radius:20px;
	box-shadow:var(--shadow-md); padding:4.5rem 2rem;
	text-align:center; color:var(--muted);
}}
.empty-icon {{
	width:60px; height:60px; margin:0 auto 1.2rem; border-radius:18px;
	background:linear-gradient(135deg,var(--brand-soft) 0%,#d6f1ea 100%);
	border:1px solid #c4e6de; position:relative;
	box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
}}
.empty-icon::before {{
	content:''; position:absolute; inset:14px 16px 18px;
	background:white; border:1.5px solid var(--brand); border-radius:4px;
}}
.empty-icon::after {{
	content:''; position:absolute; left:22px; right:22px; top:24px;
	height:6px; background:repeating-linear-gradient(180deg,var(--brand) 0,var(--brand) 1.2px,transparent 1.2px,transparent 3px);
	opacity:.55;
}}
.empty-state h3 {{ margin:0 0 .45rem; font-size:1.08rem; color:var(--text); }}
.empty-state p {{ margin:0; font-size:.88rem; max-width:32rem; margin-left:auto; margin-right:auto; }}
.doc-crumb {{
	font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
	color:var(--text-soft); font-size:.76rem; margin-bottom:.9rem;
	padding:.45rem .8rem; background:var(--panel-warm);
	border:1px solid var(--line); border-radius:9px;
	display:inline-block;
}}
.review-shell, .audit-shell, .editor-shell, .export-shell {{
	background:var(--panel); border:1px solid var(--line);
	border-radius:var(--radius); box-shadow:var(--shadow-md);
	overflow:hidden; margin-bottom:1rem;
}}
.pane-header {{
	display:flex; justify-content:space-between; gap:.85rem; align-items:center;
	background:linear-gradient(180deg,#fbfefd 0%,#f4faf8 100%);
	border-bottom:1px solid var(--line); padding:.7rem .95rem;
}}
.pane-header-label {{
	color:var(--text); font-size:.72rem; font-weight:750;
	text-transform:uppercase; letter-spacing:.1em;
}}
.pane-header-meta {{
	color:var(--muted); font-size:.74rem;
	font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
	overflow:hidden; white-space:nowrap; text-overflow:ellipsis; max-width:60%;
}}
.pane-body {{ padding:.7rem; }}
.audit-shell, .editor-shell, .export-shell {{ padding:.95rem 1.05rem 1.05rem; }}
.audit-shell h4, .editor-shell h4, .export-shell h4 {{
	margin:0 0 .35rem; color:var(--text);
	font-size:.86rem; font-weight:650; letter-spacing:-.01em;
}}
.audit-shell .shell-sub, .editor-shell .shell-sub, .export-shell .shell-sub {{
	margin:0 0 .9rem; color:var(--muted); font-size:.78rem;
}}
iframe, .text-preview {{ border:0!important; border-radius:0!important; display:block; background:white; }}
div[data-testid="stFileUploader"] {{
	background:linear-gradient(180deg,#ffffff 0%,#fbfefd 100%);
	border:1.5px dashed #c4d9d4!important; border-radius:12px!important;
	padding:.4rem!important;
	transition:border-color .18s ease, background .18s ease, box-shadow .18s ease;
}}
div[data-testid="stFileUploader"]:hover {{
	border-color:var(--brand)!important;
	background:linear-gradient(180deg,#ffffff 0%,#eff9f6 100%);
	box-shadow:0 0 0 4px rgba(64,177,155,.08);
}}
div.stButton > button, div.stDownloadButton > button {{
	border-radius:10px!important; font-weight:600!important;
	transition:transform .08s ease, box-shadow .18s ease, background .18s ease, border-color .18s ease, color .18s ease;
}}
div.stButton > button[kind="primary"] {{
	background:linear-gradient(180deg,var(--brand) 0%,var(--brand-dark) 100%)!important;
	border-color:var(--brand-dark)!important; color:white!important;
	box-shadow:0 2px 8px rgba(64,177,155,.28);
}}
div.stButton > button[kind="primary"]:hover {{
	transform:translateY(-1px);
	box-shadow:0 6px 16px rgba(64,177,155,.36);
}}
div.stButton > button[kind="secondary"] {{
	background:#fff!important; border:1px solid var(--line)!important; color:var(--text)!important;
}}
div.stButton > button[kind="secondary"]:hover {{
	border-color:var(--brand)!important; color:var(--brand-dark)!important;
	background:var(--brand-soft)!important;
}}
div.stDownloadButton > button {{
	background:#fff!important; border:1px solid var(--line)!important; color:var(--text)!important;
}}
div.stDownloadButton > button:hover {{
	border-color:var(--brand)!important; color:var(--brand-dark)!important;
	background:var(--brand-soft)!important;
}}
div[data-testid="stRadio"] > label {{ display:none; }}
div[data-testid="stRadio"] label p {{
	font-family:ui-monospace, SFMono-Regular, Menlo, monospace!important;
	font-size:.78rem!important; color:var(--text-soft);
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
	padding:.32rem .5rem; border-radius:8px;
	transition:background .15s ease;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{ background:var(--brand-soft); }}
div[data-testid="stDataFrame"] {{
	border:1px solid var(--line); border-radius:10px; overflow:hidden;
	box-shadow:var(--shadow-sm);
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea {{
	border-radius:9px!important;
}}
</style>
"""
	)


def _topbar() -> None:
	batch = st.session_state.get("batch", [])
	results = st.session_state.get("results", {})
	if batch:
		pill_text = f"{len(results)} of {len(batch)} processed"
	else:
		pill_text = "Ready to ingest"
	st.markdown(
		f"""
		<div class="topbar">
			<div class="topbar-brand">
				<div class="topbar-mark"></div>
				<div class="topbar-meta">
					<h1>Document Redaction Review</h1>
					<div class="topbar-sub">Review and defend documents — edit redactions, then export.</div>
				</div>
			</div>
			<div class="topbar-spacer"></div>
			<div class="topbar-pill">{html.escape(pill_text)}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def _batch_record(path: Path, source: str) -> dict[str, str]:
	return {
		"id": uuid.uuid4().hex,
		"name": path.name,
		"path": str(path),
		"source": source,
		"added_at": datetime.now().strftime("%H:%M"),
	}


def _add_file_bytes_to_batch(filename: str, data: bytes, source: str = "Uploaded") -> None:
	if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
		return
	workspace = _ensure_workspace()
	target = _unique_target(workspace / "batch", filename)
	target.write_bytes(data)
	record = _batch_record(target, source)
	st.session_state.batch.append(record)
	st.session_state.selected_id = record["id"]


def _add_path_to_batch(path: Path, source: str = "Generated") -> None:
	if not path.exists() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
		return
	workspace = _ensure_workspace()
	target = _unique_target(workspace / "batch", path.name)
	shutil.copy2(path, target)
	record = _batch_record(target, source)
	st.session_state.batch.append(record)
	st.session_state.selected_id = record["id"]


def _add_zip_bytes_to_batch(filename: str, data: bytes) -> int:
	workspace = _ensure_workspace()
	batch_dir = workspace / "batch"
	batch_dir.mkdir(parents=True, exist_ok=True)
	added = 0
	try:
		with zipfile.ZipFile(io.BytesIO(data)) as archive:
			for info in archive.infolist():
				if info.is_dir():
					continue
				inner_name = Path(info.filename).name
				if not inner_name or inner_name.startswith(".") or "__MACOSX" in info.filename:
					continue
				if Path(inner_name).suffix.lower() not in SUPPORTED_SUFFIXES:
					continue
				target = _unique_target(batch_dir, inner_name)
				with archive.open(info) as source, target.open("wb") as sink:
					shutil.copyfileobj(source, sink, length=ZIP_COPY_CHUNK)
				record = _batch_record(target, "Zip")
				st.session_state.batch.append(record)
				st.session_state.selected_id = record["id"]
				added += 1
	except zipfile.BadZipFile:
		st.sidebar.warning(f"{filename}: not a valid zip archive.")
	return added


def _add_uploaded_files(files: Iterable[Any]) -> None:
	for uploaded in files or []:
		name = getattr(uploaded, "name", "document")
		suffix = Path(name).suffix.lower()
		if suffix not in SUPPORTED_SUFFIXES and suffix not in ZIP_SUFFIXES:
			continue
		data = uploaded.getvalue()
		fingerprint = hashlib.sha1(name.encode("utf-8") + b"\0" + data).hexdigest()
		if fingerprint in st.session_state.uploaded_fingerprints:
			continue
		if suffix in ZIP_SUFFIXES:
			_add_zip_bytes_to_batch(name, data)
		else:
			_add_file_bytes_to_batch(name, data, source="Uploaded")
		st.session_state.uploaded_fingerprints.add(fingerprint)


def _selected_record() -> dict[str, str] | None:
	batch = [record for record in st.session_state.batch if Path(record["path"]).exists()]
	st.session_state.batch = batch
	if not batch:
		st.session_state.selected_id = None
		return None
	selected_id = st.session_state.get("selected_id")
	for record in batch:
		if record["id"] == selected_id:
			return record
	st.session_state.selected_id = batch[0]["id"]
	return batch[0]


def _record_label(record: dict[str, str]) -> str:
	status = "✓" if record["id"] in st.session_state.results else "○"
	return f"{status}  {_short_name(record['name'], 32)}"


def _convert_docx_to_pdf(docx_path: Path, preview_dir: Path) -> Path | None:
	preview_dir.mkdir(parents=True, exist_ok=True)
	expected = preview_dir / f"{docx_path.stem}.pdf"
	if expected.exists():
		return expected
	office = _find_office_binary()
	if not office:
		return None
	try:
		subprocess.run(
			[office, "--headless", "--convert-to", "pdf", "--outdir", str(preview_dir), str(docx_path)],
			check=True,
			capture_output=True,
			text=True,
			timeout=90,
		)
	except Exception:
		return None
	return expected if expected.exists() else None


def _pdf_preview_path(path: Path, preview_dir: Path) -> Path | None:
	if path.suffix.lower() == ".pdf":
		return path
	if path.suffix.lower() == ".docx":
		return _convert_docx_to_pdf(path, preview_dir)
	return None


def _read_docx_text(path: Path) -> str:
	try:
		from docx import Document

		document = Document(str(path))
	except Exception as exc:
		return f"DOCX preview unavailable: {exc}"
	parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
	for table in document.tables:
		for row in table.rows:
			cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
			if any(cells):
				parts.append(" | ".join(cells))
	return "\n\n".join(parts) or "No text could be extracted."


def _read_pdf_text(path: Path) -> str:
	try:
		import fitz

		with fitz.open(str(path)) as pdf:
			text = "\n\n".join(page.get_text("text") for page in pdf).strip()
	except Exception as exc:
		return f"PDF preview unavailable: {exc}"
	return text or "No text layer found in this PDF."


def _show_pdf(path: Path) -> None:
	encoded = base64.b64encode(path.read_bytes()).decode("ascii")
	st.markdown(
		f'<iframe src="data:application/pdf;base64,{encoded}#toolbar=1&navpanes=0&view=FitH" width="100%" height="{PREVIEW_HEIGHT}"></iframe>',
		unsafe_allow_html=True,
	)


def _show_text(path: Path) -> None:
	text = _read_docx_text(path) if path.suffix.lower() == ".docx" else _read_pdf_text(path)
	st.markdown(
		f"""
		<div class="text-preview" style="height:{PREVIEW_HEIGHT}px; overflow:auto; white-space:pre-wrap;
		padding:1rem 1.1rem; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size:.82rem; line-height:1.6; color:#173f39; background:#fff;">{html.escape(text)}</div>
		""",
		unsafe_allow_html=True,
	)


def _show_document(path: Path, preview_dir: Path) -> None:
	pdf = _pdf_preview_path(path, preview_dir)
	if pdf and pdf.exists():
		_show_pdf(pdf)
	else:
		_show_text(path)


def _defender_config(output_dir: Path):
	_process_many, _write_audit_jsonl, Config = _load_defender()
	return Config(
		output=output_dir,
		quasi=bool(st.session_state.get("setting_quasi", False)),
		pdf=bool(st.session_state.get("setting_pdf", True)),
		local=bool(st.session_state.get("setting_local", False)),
		office=_find_office_binary() or "libreoffice",
	)


def _process_records(records: list[dict[str, str]]) -> None:
	paths = [Path(record["path"]) for record in records if Path(record["path"]).exists()]
	if not paths:
		return
	workspace = _ensure_workspace()
	output_dir = workspace / "outputs"
	output_dir.mkdir(parents=True, exist_ok=True)
	process_many, write_audit_jsonl, _Config = _load_defender()
	config = _defender_config(output_dir)
	with st.spinner("Running defender..."):
		results = process_many(paths, config)
		try:
			write_audit_jsonl(results, output_dir)
		except Exception:
			pass
	for record, original, result in zip(records, paths, results):
		anonymized, audit_entries = result
		metadata = anonymized.metadata or {}
		export_meta = metadata.get("export", {}) if isinstance(metadata, dict) else {}
		st.session_state.results[record["id"]] = {
			"name": original.name,
			"original": str(original),
			"redacted": str(anonymized.document_path),
			"redacted_pdf": metadata.get("pdf_path") if isinstance(metadata, dict) else None,
			"detections": metadata.get("detections", []) if isinstance(metadata, dict) else [],
			"base_redaction_plan": metadata.get("redaction_plan", []) if isinstance(metadata, dict) else [],
			"redaction_plan": metadata.get("redaction_plan", []) if isinstance(metadata, dict) else [],
			"audit": _asdict_list(audit_entries),
			"mapping_path": export_meta.get("mapping_path") if isinstance(export_meta, dict) else None,
			"edited": False,
		}


def _export_zip_bytes(record: dict[str, str], result: dict[str, Any], include_audit: bool) -> bytes:
	buffer = io.BytesIO()
	with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
		original_path = Path(result.get("original") or record["path"])
		if original_path.exists():
			archive.write(original_path, f"original/{original_path.name}")
		redacted_path = Path(result["redacted"])
		if redacted_path.exists():
			archive.write(redacted_path, f"redacted/{redacted_path.name}")
		pdf_value = result.get("redacted_pdf")
		if pdf_value:
			pdf_path = Path(pdf_value)
			if pdf_path.exists() and pdf_path != redacted_path:
				archive.write(pdf_path, f"redacted/{pdf_path.name}")
		if include_audit:
			audit_rows = _asdict_list(result.get("audit", []))
			archive.writestr("audit/audit.json", json.dumps(audit_rows, indent=2, default=str))
			detections = _asdict_list(result.get("detections", []))
			archive.writestr("audit/detections.json", json.dumps(detections, indent=2, default=str))
			plan = _asdict_list(result.get("redaction_plan", []))
			archive.writestr("audit/redaction_plan.json", json.dumps(plan, indent=2, default=str))
			mapping_value = result.get("mapping_path")
			if mapping_value:
				mapping_path = Path(mapping_value)
				if mapping_path.exists():
					archive.write(mapping_path, f"audit/{mapping_path.name}")
	return buffer.getvalue()


def _action_key(action: dict[str, Any]) -> str:
	return "|".join([str(action.get("path", "")), str(action.get("label", "")), str(int(action.get("start") or 0)), str(int(action.get("end") or 0)), str(action.get("detector") or action.get("source") or "")])


def _surface_for_action(action: dict[str, Any], detections: list[dict[str, Any]]) -> str:
	for item in detections:
		if str(item.get("path", "")) == str(action.get("path", "")) and int(item.get("start") or 0) == int(action.get("start") or 0) and int(item.get("end") or 0) == int(action.get("end") or 0) and str(item.get("label", "")) == str(action.get("label", "")):
			return str(item.get("surface", ""))
	return ""


def _redaction_options(result: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
	detections = _asdict_list(result.get("detections", []))
	plan = _asdict_list(result.get("base_redaction_plan") or result.get("redaction_plan", []))
	options: list[str] = []
	labels: dict[str, str] = {}
	for index, action in enumerate(plan):
		if action.get("action") != "mask_text":
			continue
		key = f"{index}:{_action_key(action)}"
		surface = (_surface_for_action(action, detections) or "text span").replace("\n", " ")
		labels[key] = f"{action.get('label', 'Redaction')} — {surface[:52]}"
		options.append(key)
	return options, labels


def _parse_manual_terms(text: str) -> list[str]:
	terms: list[str] = []
	seen: set[str] = set()
	for raw_line in str(text or "").splitlines():
		term = raw_line.strip()
		if term and term not in seen:
			seen.add(term)
			terms.append(term)
	return terms


def _manual_actions(document: Any, terms: list[str], RedactionAction: Any) -> list[Any]:
	actions: list[Any] = []
	for term in terms:
		for block in document.blocks:
			for match in re.finditer(re.escape(term), block.text):
				actions.append(RedactionAction(region=block.region, path=block.path, label="Manual", action="mask_text", start=match.start(), end=match.end(), replacement="[Redacted]", detector="manual_ui", metadata=dict(block.metadata or {})))
	return actions


def _custom_audit_rows(detections: list[dict[str, Any]], active_actions: list[dict[str, Any]], disabled_keys: set[str], manual_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
	active_keys = {_action_key(action) for action in active_actions}
	rows: list[dict[str, Any]] = []
	for item in detections:
		key = _action_key({"path": item.get("path", ""), "label": item.get("label", ""), "start": item.get("start", 0), "end": item.get("end", 0), "detector": item.get("source", "")})
		rows.append({"action": "kept" if key in disabled_keys else ("masked" if key in active_keys else "kept"), "label": item.get("label", ""), "surface": item.get("surface", ""), "region": item.get("region", ""), "detector": item.get("source", ""), "score": f"{float(item.get('score') or 0):.2f}"})
	for action in manual_actions:
		rows.append({"action": "masked", "label": "Manual", "surface": "manual exact text", "region": action.get("region", ""), "detector": "manual_ui", "score": "manual"})
	return rows


def _apply_redaction_edits(record: dict[str, str], result: dict[str, Any], disabled_options: list[str], manual_terms_text: str) -> None:
	normalise_blocks, parse_document, write, Config, RedactionAction = _load_editor_tools()
	source_path = Path(record["path"])
	workspace = _ensure_workspace()
	disabled_keys = {item.split(":", 1)[1] if ":" in item else item for item in disabled_options}
	detections = _asdict_list(result.get("detections", []))
	original_plan = _asdict_list(result.get("base_redaction_plan") or result.get("redaction_plan", []))
	active_actions = [action for action in original_plan if _action_key(action) not in disabled_keys]
	output_dir = workspace / "edited_outputs" / record["id"]
	output_dir.mkdir(parents=True, exist_ok=True)
	config = Config(output=output_dir, quasi=bool(st.session_state.get("setting_quasi", False)), pdf=bool(st.session_state.get("setting_pdf", True)), local=bool(st.session_state.get("setting_local", False)), office=_find_office_binary() or "libreoffice")
	document = normalise_blocks(parse_document(source_path, pdf_render_dpi=getattr(config, "pdf_render_dpi", 200), pdf_ocr_all_pages=getattr(config, "pdf_ocr_all_pages", False)))
	manual_objects = _manual_actions(document, _parse_manual_terms(manual_terms_text), RedactionAction)
	action_objects = [RedactionAction(**action) for action in active_actions] + manual_objects
	anonymized = write(document, action_objects, config)
	manual_dicts = _asdict_list(manual_objects)
	result.update({"redacted": str(anonymized.document_path), "redacted_pdf": (anonymized.metadata or {}).get("pdf_path"), "redaction_plan": active_actions + manual_dicts, "audit": _custom_audit_rows(detections, active_actions, disabled_keys, manual_dicts), "edited": True})
	st.session_state.results[record["id"]] = result


def _sidebar() -> None:
	st.sidebar.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
	uploaded = st.sidebar.file_uploader("Drag and drop DOCX, PDF, or ZIP files", type=UPLOAD_EXTENSIONS, accept_multiple_files=True, label_visibility="collapsed", key="sidebar_uploader")
	st.sidebar.caption("ZIP archives stream straight to disk; only DOCX/PDF entries are extracted.")
	_add_uploaded_files(uploaded or [])

	st.sidebar.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
	st.sidebar.checkbox("Redact quasi-identifiers", value=False, key="setting_quasi")
	st.sidebar.checkbox("Create PDF previews", value=True, key="setting_pdf")
	st.sidebar.checkbox("Use local model files only", value=False, key="setting_local")

	st.sidebar.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
	batch = st.session_state.batch
	if batch:
		ids = [record["id"] for record in batch]
		selected_id = st.session_state.get("selected_id") if st.session_state.get("selected_id") in ids else ids[0]
		chosen = st.sidebar.radio("Open document", ids, index=ids.index(selected_id), format_func=lambda item_id: _record_label(next(record for record in batch if record["id"] == item_id)), label_visibility="collapsed", key="document_picker")
		st.session_state.selected_id = chosen
		st.sidebar.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
		if st.sidebar.button("Run defender on selected", type="primary", width="stretch"):
			selected = _selected_record()
			if selected:
				_process_records([selected])
				st.rerun()
		if len(batch) > 1 and st.sidebar.button("Run defender on all", width="stretch"):
			_process_records(batch)
			st.rerun()
		if st.sidebar.button("Clear batch", width="stretch"):
			_reset_workspace()
			st.rerun()
	else:
		st.sidebar.caption("No documents yet.")


def _download_redacted_button(result: dict[str, Any]) -> None:
	redacted = Path(result["redacted"])
	if redacted.exists():
		st.download_button("Download redacted file", redacted.read_bytes(), file_name=redacted.name, mime=_mime(redacted), width="stretch", key=f"download_{redacted.name}_{redacted.stat().st_mtime_ns}")


def _audit_table(result: dict[str, Any]) -> None:
	rows = []
	for row in _asdict_list(result.get("audit", [])):
		rows.append({"action": row.get("action", ""), "label": row.get("label", ""), "surface": str(row.get("surface", "")).replace("\n", " ")[:120], "region": row.get("region", ""), "detector": row.get("detector", ""), "score": row.get("score", "")})
	if rows:
		st.dataframe(rows, width="stretch", hide_index=True)


def _export_panel(record: dict[str, str], result: dict[str, Any]) -> None:
	st.html('<div class="export-shell"><h4>Export</h4><p class="shell-sub">Bundle the original, the redacted output, and optionally the audit + JSON files into a single ZIP.</p></div>')
	left, right = st.columns([1.0, 1.6], gap="medium")
	with left:
		include_audit = st.toggle("Include audit & JSON", value=True, key=f"include_audit_{record['id']}")
	zip_bytes = _export_zip_bytes(record, result, include_audit)
	stem = Path(record["name"]).stem
	suffix = "_with_audit" if include_audit else "_documents_only"
	with right:
		st.download_button(
			"Download ZIP bundle",
			zip_bytes,
			file_name=f"{stem}{suffix}.zip",
			mime="application/zip",
			type="primary",
			width="stretch",
			key=f"export_zip_{record['id']}_{int(include_audit)}",
		)


def _redaction_editor(record: dict[str, str], result: dict[str, Any]) -> None:
	options, labels = _redaction_options(result)
	st.html('<div class="editor-shell"><h4>Edit redactions</h4><p class="shell-sub">Uncheck items to keep them visible, or add custom terms to redact. Apply to re-run the writer with your changes.</p></div>')
	left, right = st.columns([1.2, 1.0], gap="medium")
	with left:
		disabled = st.multiselect(
			f"Keep visible ({len(options)} detected)",
			options=options,
			format_func=lambda option: labels.get(option, option),
			key=f"disabled_{record['id']}",
			placeholder="Search detected items to un-redact…" if options else "No detected items.",
			disabled=not options,
		)
	with right:
		manual_terms = st.text_area(
			"Manual redactions",
			height=120,
			key=f"manual_{record['id']}",
			placeholder="One exact-match term per line",
		)
	if st.button("Apply edits and re-redact", type="primary", width="stretch", key=f"apply_edits_{record['id']}"):
		_apply_redaction_edits(record, result, disabled, manual_terms)
		st.rerun()


def _shell_start(label: str, meta: str) -> None:
	st.markdown(f'<div class="review-shell"><div class="pane-header"><span class="pane-header-label">{html.escape(label)}</span><span class="pane-header-meta">{html.escape(meta)}</span></div><div class="pane-body">', unsafe_allow_html=True)


def _shell_end() -> None:
	st.markdown('</div></div>', unsafe_allow_html=True)


def _show_unprocessed(record: dict[str, str]) -> None:
	workspace = _ensure_workspace()
	source = Path(record["path"])
	_shell_start("Original", record["name"])
	_show_document(source, workspace / "previews" / record["id"] / "original")
	_shell_end()
	if st.button("Run defender", type="primary", key=f"main_run_{record['id']}"):
		_process_records([record])
		st.rerun()


def _show_processed(record: dict[str, str], result: dict[str, Any]) -> None:
	workspace = _ensure_workspace()
	original = Path(result.get("original") or record["path"])
	redacted = Path(result["redacted"])
	redacted_preview = Path(result["redacted_pdf"]) if result.get("redacted_pdf") and Path(result["redacted_pdf"]).exists() else redacted
	preview_root = workspace / "previews" / record["id"]
	left, right = st.columns(2, gap="medium")
	with left:
		_shell_start("Original", original.name)
		_show_document(original, preview_root / "original")
		_shell_end()
	with right:
		_shell_start("Redacted", redacted.name)
		_show_document(redacted_preview, preview_root / "redacted")
		_download_redacted_button(result)
		_shell_end()
	_redaction_editor(record, result)
	st.html('<div class="audit-shell"><h4>Audit log</h4><p class="shell-sub">Every detection and the action taken on it. Use this to verify the writer kept what you intended.</p></div>')
	_audit_table(result)
	_export_panel(record, result)


def _main() -> None:
	_topbar()
	record = _selected_record()
	if not record:
		st.html('<div class="empty-state"><div class="empty-icon"></div><h3>No document selected</h3><p>Drag DOCX, PDF, or ZIP files into the sidebar. Archives stream straight to disk; only DOCX/PDF entries are extracted.</p></div>')
		return
	st.markdown(f'<p class="doc-crumb">{html.escape(_short_name(record["name"], 100))}</p>', unsafe_allow_html=True)
	result = st.session_state.results.get(record["id"])
	if result:
		_show_processed(record, result)
	else:
		_show_unprocessed(record)


def main() -> None:
	st.set_page_config(page_title="Document Redaction Review", layout="wide")
	_init_state()
	_ensure_workspace()
	_theme()
	try:
		_sidebar()
		_main()
	except Exception as exc:
		st.error(str(exc))


if __name__ == "__main__":
	main()
