"""Detect entities, build redaction plans, and write anonymized outputs."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import thesis_project.shared as shared

from .audit import audit, export_redaction_artifacts
from .parser import SUPPORTED_INPUT_SUFFIXES, normalise_blocks, parse_document
from .redact import write


def apply_strict_zone_policy(detections, config):
    """Keep detections that pass strict-zone or body confidence thresholds.

    Direct identifiers are always kept once detected. For the experiments, the
    configured quasi identifiers are privacy targets too, so pattern detections
    for labels such as Course code, Group label, Programme, Date, and Role are
    also kept when they are explicitly enabled in config.labels. The confidence
    thresholds are still used for generic Name detections and for any future
    non-pattern quasi detector output.
    """
    kept = []
    configured_labels = {shared.canon(label) for label in getattr(config, "labels", ())}
    enabled_quasi = bool(getattr(config, "quasi", False))
    for item in detections:
        label = shared.canon(item.label)
        if shared.direct(label) and label != "Name":
            kept.append(item)
            continue
        if enabled_quasi and label in configured_labels and (
            item.source.startswith("pattern_") or item.source.startswith("ner_gliner")
        ):
            kept.append(item)
            continue
        strict = shared.is_strict_region(item.region)
        threshold = (
            shared.name_strict_threshold if strict else shared.name_body_threshold
        ) if label == "Name" else (
            shared.quasi_strict_threshold if strict else shared.quasi_body_threshold
        )
        if item.score >= threshold:
            kept.append(item)
    return kept


def merge_detections(detections):
    """Merge duplicate and overlapping detections, keeping the strongest span."""
    deduped = {}
    for item in detections:
        key = (item.path, item.start, item.end, shared.canon(item.label), item.surface)
        existing = deduped.get(key)
        rank = (item.score, shared.severity_weight_for(item.label), len(item.surface))
        if existing is None or rank > (
            existing.score,
            shared.severity_weight_for(existing.label),
            len(existing.surface),
        ):
            deduped[key] = item

    by_path = defaultdict(list)
    for item in deduped.values():
        by_path[item.path].append(item)

    merged = []
    for items in by_path.values():
        items.sort(
            key=lambda item: (
                item.start,
                item.end,
                -shared.severity_weight_for(item.label),
                -item.score,
                -len(item.surface),
            )
        )
        kept = []
        for item in items:
            if not kept or item.start >= kept[-1].end:
                kept.append(item)
                continue
            previous = kept[-1]
            prev_rank = (
                shared.severity_weight_for(previous.label),
                previous.score,
                len(previous.surface),
            )
            curr_rank = (
                shared.severity_weight_for(item.label),
                item.score,
                len(item.surface),
            )
            if curr_rank > prev_rank:
                kept[-1] = item
        merged.extend(kept)
    return sorted(merged, key=lambda item: (item.path, item.start, item.end))


def link_repeated_mentions(document, detections, config=None):
    """Propagate exact repeated direct identifiers across text blocks."""
    linked = []
    seen = set()
    for item in detections:
        normalized_surface = re.sub(r"\s+", " ", item.surface.strip().lower())
        key = (
            item.entity_key
            or f"{shared.canon(item.label)}:{normalized_surface}"
        )
        rebuilt = shared.Occurrence(
            label=shared.canon(item.label),
            surface=item.surface,
            start=item.start,
            end=item.end,
            region=shared.normalize_region(item.region),
            path=item.path,
            score=item.score,
            source=item.source,
            entity_key=key,
            metadata=dict(item.metadata or {}),
        )
        linked.append(rebuilt)
        seen.add((rebuilt.path, rebuilt.start, rebuilt.end, rebuilt.label, rebuilt.surface))

    propagatable_labels = {"Name", "Student number", "Institutional email"}
    if config is not None and getattr(config, "quasi", False):
        propagatable_labels.update(shared.canon(label) for label in getattr(config, "labels", ()))

    propagatable = [
        item for item in linked
        if shared.canon(item.label) in propagatable_labels
        and item.region != "ocr"
    ]
    for seed in propagatable:
        if not seed.surface.strip():
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(seed.surface)}(?!\w)")
        for block in document.blocks:
            for match in pattern.finditer(block.text):
                key = (block.path, match.start(), match.end(), seed.label, match.group(0))
                if key in seen:
                    continue
                seen.add(key)
                linked.append(
                    shared.Occurrence(
                        label=seed.label,
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        region=shared.normalize_region(block.region),
                        path=block.path,
                        score=max(0.70, min(seed.score, 0.92)),
                        source=f"{seed.source}_linked",
                        entity_key=seed.entity_key,
                        metadata=dict(seed.metadata or {}),
                    )
                )
    return linked


def _region_set(config):
    if not config.regions:
        return None
    return {shared.normalize_region(region) for region in config.regions}


def _enabled_label_set(config):
    labels = set()
    if config.direct:
        labels.update({"Name", "FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email"})
    if config.quasi:
        labels.update({shared.canon(label) for label in config.labels})
    return labels


def _filter_enabled_labels(detections, config):
    enabled = _enabled_label_set(config)
    return [item for item in detections if shared.canon(item.label) in enabled]


def _copy_occurrence(item, *, label=None, surface=None, start=None, end=None, metadata=None, source=None):
    """Return an Occurrence copy while preserving redaction/audit fields."""
    return shared.Occurrence(
        label=item.label if label is None else label,
        surface=item.surface if surface is None else surface,
        start=item.start if start is None else start,
        end=item.end if end is None else end,
        region=item.region,
        path=item.path,
        score=item.score,
        source=item.source if source is None else source,
        entity_key=item.entity_key,
        metadata=dict(item.metadata or {}) if metadata is None else metadata,
    )


def _expand_name_variants(detections):
    """Keep generic Name for redaction, but add gold-compatible variants for scoring.

    The detector finds a full-name span and labels it as ``Name``. The synthetic
    gold data scores ``FULL_NAME``, ``FIRST_NAME``, and ``LAST_NAME`` separately.
    The generic Name occurrence is therefore marked redaction-only so it still
    drives anonymisation without becoming a false positive during evaluation.
    """
    expanded = []
    for item in detections:
        if shared.canon(item.label) != "Name":
            expanded.append(item)
            continue

        surface = str(item.surface or "")
        if not surface.strip():
            expanded.append(item)
            continue

        redaction_metadata = dict(item.metadata or {})
        redaction_metadata["redaction_only"] = True
        expanded.append(_copy_occurrence(item, metadata=redaction_metadata))

        eval_metadata = dict(item.metadata or {})
        eval_metadata["eval_only"] = True
        expanded.append(
            _copy_occurrence(
                item,
                label="FULL_NAME",
                surface=surface,
                start=item.start,
                end=item.end,
                source=f"{item.source}_full_variant",
                metadata=dict(eval_metadata),
            )
        )

        token_matches = list(re.finditer(r"\S+", surface))
        if len(token_matches) < 2:
            continue

        first = token_matches[0]
        last = token_matches[-1]
        expanded.append(
            _copy_occurrence(
                item,
                label="FIRST_NAME",
                surface=first.group(0),
                start=item.start + first.start(),
                end=item.start + first.end(),
                source=f"{item.source}_first_variant",
                metadata=dict(eval_metadata),
            )
        )
        expanded.append(
            _copy_occurrence(
                item,
                label="LAST_NAME",
                surface=last.group(0),
                start=item.start + last.start(),
                end=item.start + last.end(),
                source=f"{item.source}_last_variant",
                metadata=dict(eval_metadata),
            )
        )
    return expanded



def _filtered_document(document, config):
    """Keep only configured regions, applying the same rule to text and image artifacts."""
    regions = _region_set(config)
    if not regions:
        return document
    return shared.DocumentRecord(
        doc_id=document.doc_id,
        document_path=document.document_path,
        template_id=document.template_id,
        split=document.split,
        annotation=document.annotation,
        blocks=[block for block in document.blocks if shared.normalize_region(block.region) in regions],
        images=[
            image
            for image in document.images
            if ("ocr" in regions) or (shared.normalize_region(image.region) in regions)
        ],
        metadata=dict(document.metadata or {}),
    )


def detect_entities(document, config):
    """Run enabled detectors and post-process detections."""
    doc = _filtered_document(document, config)
    detections = []
    if config.patterns:
        import thesis_project.defender.patterns as patterns

        detections.extend(patterns.detect(doc, config))
    if config.ner:
        import thesis_project.defender.name_model  # noqa: F401
        import thesis_project.defender.ner as ner

        detections.extend(ner.detect(doc, config))
    if config.ocr:
        import thesis_project.defender.ocr as ocr

        detections.extend(ocr.detect(doc, config))
    detections = _filter_enabled_labels(detections, config)
    if config.strict:
        detections = apply_strict_zone_policy(detections, config)
    detections = merge_detections(detections)
    if config.link:
        detections = link_repeated_mentions(doc, detections, config)
    detections = _expand_name_variants(detections)
    return detections


def build_redaction_plan(detections, config):
    """Turn detections into text replacement and image/PDF box masking actions."""
    quasi_to_redact = {shared.canon(label) for label in config.labels}
    plan = []
    image_metadata_keys = {
        "media_path",
        "bbox",
        "image_id",
        "document_type",
        "page_index",
        "page_number",
        "image_size",
        "pdf_page_rect",
        "render_dpi",
    }
    for item in detections:
        if item.metadata.get("eval_only"):
            continue
        canonical_label = shared.canon(item.label)
        if not (
            (shared.direct(item.label) and config.direct)
            or (
                shared.quasi(item.label)
                and config.quasi
                and canonical_label in quasi_to_redact
            )
        ):
            continue

        if (
            item.region == "ocr"
            and config.mask_boxes
            and item.metadata.get("media_path")
            and item.metadata.get("bbox")
        ):
            plan.append(
                shared.RedactionAction(
                    region="image",
                    path=item.path,
                    label=item.label,
                    action="mask_image_box",
                    start=0,
                    end=0,
                    detector=item.source,
                    metadata={
                        key: item.metadata.get(key)
                        for key in image_metadata_keys
                        if item.metadata.get(key) is not None
                    },
                )
            )
            continue

        plan.append(
            shared.RedactionAction(
                region=item.region,
                path=item.path,
                label=item.label,
                action="mask_text",
                start=item.start,
                end=item.end,
                replacement=shared.mask_token_for(item.label),
                detector=item.source,
                metadata={
                    key: item.metadata.get(key)
                    for key in ("document_type", "page_index", "page_number", "bbox")
                    if item.metadata.get(key) is not None
                },
            )
        )
    return plan


def process_document(document_path, config=None):
    """Process one document and return anonymized output plus audit entries."""
    cfg = config or shared.Config()
    parsed = normalise_blocks(
        parse_document(
            Path(document_path),
            pdf_render_dpi=getattr(cfg, "pdf_render_dpi", 200),
            pdf_ocr_all_pages=getattr(cfg, "pdf_ocr_all_pages", False),
        )
    )
    detections = detect_entities(parsed, cfg)
    redaction_plan = build_redaction_plan(detections, cfg)
    anonymized = write(parsed, redaction_plan, cfg)
    audit_log = audit(parsed, detections, redaction_plan, cfg)
    anonymized.metadata["detections"] = [asdict(item) for item in detections]
    anonymized.metadata["redaction_plan"] = [asdict(item) for item in redaction_plan]
    anonymized.metadata["detections_source"] = "current_document"
    anonymized.metadata["export"] = export_redaction_artifacts(
        parsed,
        anonymized,
        detections,
        redaction_plan,
        audit_log,
        cfg,
    )
    return anonymized, audit_log


def process_many(paths, config=None):
    """Process many documents with one shared config."""
    cfg = config or shared.Config()
    return [process_document(path, cfg) for path in paths]


def iter_document_inputs(path):
    """Return supported DOCX/PDF inputs from a file or directory."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES and not path.name.startswith("~$"):
        return [path]
    if path.is_dir():
        candidates = []
        for suffix in sorted(SUPPORTED_INPUT_SUFFIXES):
            for candidate in path.rglob(f"*{suffix}"):
                if not candidate.name.startswith("~$"):
                    candidates.append(candidate)
        return sorted(set(candidates))
    return []


def iter_docx_inputs(path):
    """Backward-compatible alias for older notebook/CLI code."""
    return iter_document_inputs(path)


def write_audit_jsonl(results, output_root):
    """Write audit entries to a JSONL file."""
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "audit_log.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for _, audit_entries in results:
            for entry in audit_entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return audit_path
