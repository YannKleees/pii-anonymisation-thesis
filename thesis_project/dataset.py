"""Dataset loading and annotation normalization utilities."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .defender.parser import SUPPORTED_INPUT_SUFFIXES, normalise_blocks, parse_document
from .shared import DocumentRecord, canon, normalize_region

DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _join_non_ng(values):
    """Join non-NG values into pipe-separated string."""
    return "|".join(sorted({str(v) for v in values if str(v) != "NG"}))


def _first(*values, default=""):
    """Return first non-None, non-empty value from args."""
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _as_int(value, default=0):
    """Convert value to int, returning default on failure."""
    try:
        return int(value)
    except Exception:
        return default


def _entity_identity(entity, occurrence=None):
    """Extract canonical identity fields from entity dict with optional occurrence override."""
    occurrence = occurrence or {}
    return {
        "label": canon(str(_first(occurrence.get("label"), entity.get("label")))),
        "surface": str(_first(occurrence.get("surface"), entity.get("surface"))),
        "entity_type": str(_first(occurrence.get("entity_type"), entity.get("entity_type"))),
        "entity_id": str(_first(occurrence.get("entity_id"), entity.get("entity_id"))),
        "quality": str(_first(occurrence.get("quality"), entity.get("quality"), default="clean")),
    }


def _dedupe_entities(items):
    """Remove duplicate entities by (label, surface, region, path, start, end, image_id, image_path)."""
    seen = set()
    output = []
    for item in items:
        key = (
            item.get("label"),
            item.get("surface"),
            item.get("region"),
            item.get("path"),
            item.get("start"),
            item.get("end"),
            item.get("image_id"),
            item.get("image_path"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


@lru_cache(maxsize=None)
def _load_manifest_cached(dataset_root_str):
    """Load document manifest with templates, splits, and parsed blocks; result is cached."""
    dataset_root = Path(dataset_root_str)
    documents_dir = dataset_root / "documents"
    annotations_dir = dataset_root / "annotations"

    manifest = []
    if not documents_dir.exists():
        return manifest

    document_paths = []
    for suffix in sorted(SUPPORTED_INPUT_SUFFIXES):
        document_paths.extend(documents_dir.glob(f"*{suffix}"))

    for document_path in sorted(set(document_paths)):
        if document_path.name.startswith("~$"):
            continue
        annotation_path = annotations_dir / f"{document_path.stem}.json"
        annotation_path = annotation_path if annotation_path.exists() else None
        template_id = ""
        split = ""
        if annotation_path is not None:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            template_id = str(annotation.get("template_id", "") or "")
            split = str(annotation.get("split", "") or split)
        if not template_id:
            parts = document_path.stem.split("_")
            template_id = parts[-1] if parts else document_path.stem
        if not split:
            draw = int(hashlib.sha1(document_path.stem.encode("utf-8")).hexdigest()[:8], 16) / (16**8)
            if draw < DEFAULT_SPLIT_RATIOS["train"]:
                split = "train"
            elif draw < DEFAULT_SPLIT_RATIOS["train"] + DEFAULT_SPLIT_RATIOS["val"]:
                split = "val"
            else:
                split = "test"
        parsed = normalise_blocks(parse_document(document_path))
        manifest.append(
            DocumentRecord(
                doc_id=document_path.stem,
                document_path=document_path,
                template_id=template_id,
                split=split,
                annotation=annotation_path,
                blocks=parsed.blocks,
                images=parsed.images,
                metadata={
                    "document_type": document_path.suffix.lower().lstrip("."),
                    **dict(parsed.metadata or {}),
                },
            )
        )

    return manifest


class DatasetManager:
    """Load, validate, and organise thesis datasets."""

    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.documents_dir = self.dataset_root / "documents"
        self.annotations_dir = self.dataset_root / "annotations"
        self.images_dir = self.dataset_root / "images"
        self._manifest_cache = None
        self._annotation_cache = {}

    def _stable_fraction(self, key):
        """Compute deterministic fraction in [0, 1) for a key string."""
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / (16**8)

    def _default_split(self, doc_id):
        """Assign document to train/val/test split deterministically."""
        draw = self._stable_fraction(doc_id)
        if draw < DEFAULT_SPLIT_RATIOS["train"]:
            return "train"
        if draw < DEFAULT_SPLIT_RATIOS["train"] + DEFAULT_SPLIT_RATIOS["val"]:
            return "val"
        return "test"

    def _annotation_path_for(self, document_path):
        """Return path to JSON annotation file if it exists."""
        candidate = self.annotations_dir / f"{document_path.stem}.json"
        return candidate if candidate.exists() else None

    def _read_annotation(self, path):
        """Load and cache JSON annotation file."""
        cached = self._annotation_cache.get(path)
        if cached is not None:
            return cached
        data = json.loads(path.read_text(encoding="utf-8"))
        self._annotation_cache[path] = data
        return data

    def load_manifest(self):
        """Load DOCX and PDF documents from the dataset root."""
        if self._manifest_cache is not None:
            return self._manifest_cache

        self._manifest_cache = list(_load_manifest_cached(str(self.dataset_root)))
        return self._manifest_cache

    def text_entity(self, entity, occurrence, block_by_path):
        """Normalize text entity with block-local offsets and surface validation."""
        path = str(_first(occurrence.get("path"), entity.get("path")))
        block = block_by_path.get(path)
        start = _as_int(_first(occurrence.get("start"), entity.get("start")), 0)
        end = _as_int(_first(occurrence.get("end"), entity.get("end")), start)
        surface = str(_first(occurrence.get("surface"), entity.get("surface")))

        local_start, local_end = start, end
        if block is not None:
            # Older generator annotations store document-global offsets; newer schemas may store block-local offsets.
            if int(block.start) <= start <= end <= int(block.end):
                local_start = start - int(block.start)
                local_end = end - int(block.start)
            elif 0 <= start <= end <= len(block.text):
                local_start = start
                local_end = end
            elif surface:
                found = block.text.find(surface)
                if found >= 0:
                    local_start = found
                    local_end = found + len(surface)
            if not (0 <= local_start <= local_end <= len(block.text)):
                return None
            if not surface and local_start < local_end:
                surface = block.text[local_start:local_end]

        identity = _entity_identity(entity, occurrence)
        identity["surface"] = surface or identity["surface"]
        return {
            **identity,
            "region": normalize_region(str(_first(occurrence.get("region"), entity.get("region"), default=(block.region if block else "body")))),
            "path": path,
            "start": local_start,
            "end": local_end,
        }

    def _ocr_entity(self, entity, image=None):
        """Normalize OCR entity with image metadata."""
        image = image or {}
        identity = _entity_identity(entity)
        region = normalize_region(str(_first(entity.get("region"), image.get("region"), default="ocr")))
        if region != "ocr":
            region = "ocr"
        return {
            **identity,
            "region": region,
            "image_id": str(_first(entity.get("image_id"), image.get("image_id"))),
            "image_path": str(_first(entity.get("image_path"), image.get("image_path"), image.get("path"))),
            "bbox": entity.get("bbox"),
            "path": str(_first(entity.get("path"), image.get("path"))),
        }

    def load_annotations(self, document):
        """Load and normalize text/OCR gold annotations across schema variants."""
        if document.annotation is None or not document.annotation.exists():
            return {"raw": {}, "text_entities": [], "ocr_entities": [], "text_entities_by_path": {}}

        raw = self._read_annotation(document.annotation)
        self.attach(document)
        block_by_path = {block.path: block for block in document.blocks}
        text_entities = []
        ocr_entities = []

        # Variant 1: already flattened text_entities / ocr_entities.
        for entity in raw.get("text_entities", []) or []:
            occurrence = dict(entity)
            item = self.text_entity(entity, occurrence, block_by_path)
            if item is not None:
                text_entities.append(item)

        for entity in raw.get("ocr_entities", []) or []:
            ocr_entities.append(self._ocr_entity(entity))

        # Variant 2: entities with nested occurrences, or direct flattened entity rows.
        for entity in raw.get("entities", []) or []:
            occurrences = entity.get("occurrences")
            if occurrences:
                for occurrence in occurrences:
                    item = self.text_entity(entity, occurrence, block_by_path)
                    if item is not None:
                        text_entities.append(item)
            elif str(_first(entity.get("region"))).lower() == "ocr" or entity.get("image_id") or entity.get("image_path"):
                ocr_entities.append(self._ocr_entity(entity))
            elif entity.get("path") is not None and entity.get("start") is not None and entity.get("end") is not None:
                item = self._text_entity_from_occurrence(entity, entity, block_by_path)
                if item is not None:
                    text_entities.append(item)

        # Variant 3: image artifacts with optional nested entities.
        for image in raw.get("images", []) or []:
            for entity in image.get("entities", []) or []:
                ocr_entities.append(self._ocr_entity(entity, image=image))

        text_entities = _dedupe_entities(text_entities)
        ocr_entities = _dedupe_entities(ocr_entities)
        by_path = {}
        for item in text_entities:
            by_path.setdefault(item["path"], []).append(item)

        return {
            "raw": raw,
            "text_entities": text_entities,
            "ocr_entities": ocr_entities,
            "text_entities_by_path": by_path,
        }

    def attach(self, document):
        """Attach parsed text blocks and image artifacts to one document."""
        if document.blocks or document.images:
            return document
        parsed = normalise_blocks(parse_document(document.document_path))
        document.blocks = parsed.blocks
        document.images = parsed.images
        document.metadata.update(parsed.metadata)
        return document

    def gold_coverage(self, records=None):
        """Return per-document gold coverage, split by text and OCR annotations."""
        records = records if records is not None else self.load_manifest()
        rows = []
        for record in records:
            annotation = self.load_annotations(record)
            text_entities = annotation.get("text_entities", [])
            ocr_entities = annotation.get("ocr_entities", [])
            rows.append(
                {
                    "doc_id": record.doc_id,
                    "split": record.split,
                    "template_id": record.template_id,
                    "document_type": record.metadata.get("document_type", record.document_path.suffix.lower().lstrip(".")),
                    "text_gold_count": len(text_entities),
                    "ocr_gold_count": len(ocr_entities),
                    "image_artifact_count": len(record.images),
                    "has_text_gold": bool(text_entities),
                    "has_ocr_gold": bool(ocr_entities),
                }
            )
        return rows

    def ocr_counts(self, records=None):
        """Return OCR gold counts by split and label for notebook diagnostics."""
        records = records if records is not None else self.load_manifest()
        counter = Counter()
        for record in records:
            annotation = self.load_annotations(record)
            for entity in annotation.get("ocr_entities", []):
                counter[(record.split, entity.get("label", ""))] += 1
        return [
            {"split": split, "label": label, "ocr_gold_count": count}
            for (split, label), count in sorted(counter.items())
        ]

    def unseen_holdout(self):
        """Return the template IDs reserved for robustness testing."""
        manifest = self.load_manifest()
        counts = {}
        for record in manifest:
            counts[record.template_id] = counts.get(record.template_id, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
        if not ordered:
            return []
        n = max(1, round(len(ordered) * 0.34))
        return [template for template, _count in ordered[:n]]

    def master_profiles(self):
        """Point to the single master profiles file used by attacker and generator."""
        target = self.dataset_root / "master_profiles.csv"
        if target.exists():
            try:
                existing = pd.read_csv(target).fillna("")
                if not existing.empty and any(col in existing.columns for col in ("student_number", "person_id", "record")):
                    return target
            except Exception:
                pass


        people_path = self.dataset_root / "people.csv"
        groups_path = self.dataset_root / "groups.csv"
        courses_path = self.dataset_root / "courses.csv"
        if not people_path.exists() or not groups_path.exists() or not courses_path.exists():
            raise FileNotFoundError(".csv files not found")

        people = pd.read_csv(people_path).fillna("NG")
        groups = pd.read_csv(groups_path).fillna("NG")
        courses = pd.read_csv(courses_path).fillna("NG")

        students = people.loc[people["role"] == "student"].copy()
        memberships = groups.merge(
            courses[["course_id", "course_code", "course_name"]],
            on="course_id",
            how="left",
        )
        aggregated = (
            memberships.groupby("student_number", sort=False)
            .agg(
                group_ids=("group_id", _join_non_ng),
                group_labels=("group_label", _join_non_ng),
                course_ids=("course_id", _join_non_ng),
                course_codes=("course_code", _join_non_ng),
                tutor_ids=("teaching_assistant_id", _join_non_ng),
                coordinator_ids=("coordinator_id", _join_non_ng),
            )
            .reset_index()
        )
        master = students.merge(aggregated, on="student_number", how="left").fillna("")
        master.to_csv(target, index=False)
        return target
