"""Redaction audit trail generation and output artifact export."""
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..shared import Entry, canon, direct, quasi


def _item_key(item):
    """Return hashable key for occurrence identity."""
    return (
        str(item.path),
        canon(item.label),
        str(item.region),
        int(item.start or 0),
        int(item.end or 0),
        str(item.source or ""),
    )


def _plan_key(action):
    """Return hashable key for redaction action identity."""
    return (
        str(action.path),
        canon(action.label),
        str(action.region),
        int(action.start or 0),
        int(action.end or 0),
        str(action.detector or ""),
    )


def audit(doc, items, plan, cfg):
    """Generate audit trail entries for detections against redaction plan."""
    labels = {canon(label) for label in cfg.labels}
    redacted = {_plan_key(action) for action in plan}
    rows = []
    for item in items:
        label = canon(item.label)
        key = _item_key(item)
        redact = key in redacted or (
            direct(item.label) and cfg.direct
        ) or (
            quasi(item.label)
            and cfg.quasi
            and label in labels
        )
        action = "masked" if redact else "kept"
        score = f"score={float(item.score or 0.0):.2f}"
        if item.metadata.get("ocr_confidence") is not None:
            score += f";ocr_conf={float(item.metadata['ocr_confidence']):.1f}"
        rows.append(
            Entry(
                doc_id=doc.doc_id,
                region=item.region,
                path=item.path,
                label=item.label,
                surface=item.surface,
                detector=item.source,
                action=action,
                score=score,
            )
        )
    return rows


def _output_root(source_doc, cfg):
    """Return output directory root; use config output or document parent."""
    if getattr(cfg, "output", None):
        return Path(cfg.output)
    return source_doc.document_path.parent / "notebook_outputs"


def _copy_if_different(source_path, target_path):
    """Copy source to target only if they differ."""
    source = Path(source_path)
    target = Path(target_path)
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def unapplied_redactions(anonymized_doc, redaction_plan):
    """Identify redaction actions that don't appear in anonymized document."""
    by_path = {block.path: block.text for block in anonymized_doc.blocks}
    failed = []
    for action in redaction_plan:
        if action.action != "mask_text":
            continue
        text = by_path.get(action.path)
        if text is None:
            failed.append(f"missing_target_path:{action.path}")
            continue
        token = str(action.replacement or "").strip()
        if token and token not in text:
            failed.append(f"missing_mask_token:{action.path}:{action.label}")
    return failed


def export_redaction_artifacts(source_doc, anonymized_doc, detections, redaction_plan, audit_rows, cfg):
    """Export redacted document, audit trail, and review artifacts to output directories."""
    root = _output_root(source_doc, cfg)
    redacted_dir = root / "redacted_documents"
    mappings_dir = root / "redaction_mappings"
    review_dir = root / "review_needed"
    redacted_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    redacted_target = redacted_dir / anonymized_doc.document_path.name
    _copy_if_different(anonymized_doc.document_path, redacted_target)

    reasons = unapplied_redactions(anonymized_doc, redaction_plan)
    if detections and not redaction_plan:
        reasons.append("detections_without_redaction_plan")
    problematic = bool(reasons)

    mapping_path = mappings_dir / f"{source_doc.document_path.stem}_redaction_map.json"
    payload = {
        "doc_id": source_doc.doc_id,
        "source_document": str(source_doc.document_path),
        "redacted_document": str(redacted_target),
        "problematic": problematic,
        "problem_reasons": reasons,
        "audit_summary": dict(Counter(entry.action for entry in audit_rows)),
        "counts": {
            "detections": len(detections),
            "redaction_plan": len(redaction_plan),
            "audit_rows": len(audit_rows),
        },
        "detections": [asdict(item) for item in detections],
        "redaction_plan": [asdict(item) for item in redaction_plan],
        "audit": [asdict(item) for item in audit_rows],
    }
    mapping_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    review_case_dir = None
    if problematic:
        review_case_dir = review_dir / source_doc.document_path.stem
        review_case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_doc.document_path, review_case_dir / source_doc.document_path.name)
        shutil.copy2(redacted_target, review_case_dir / redacted_target.name)
        shutil.copy2(mapping_path, review_case_dir / mapping_path.name)

    return {
        "root": str(root),
        "redacted_document": str(redacted_target),
        "mapping": str(mapping_path),
        "problematic": problematic,
        "problem_reasons": reasons,
        "review_folder": str(review_case_dir) if review_case_dir else "",
    }