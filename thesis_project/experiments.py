"""Thesis experiment workflows for RQ1-RQ4 entity detection and attack evaluation."""
import copy
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .attacker import Attacker
from .dataset import DatasetManager
from .defender.pipeline import detect_entities, process_document
from .evaluation import Evaluation, _metrics as prf
from .shared import DocumentRecord, Config, canon, severity_weight_for


class Experiments:
    """High-level entry points for the thesis experiments.

    RQ1: Overall detection accuracy by region and label.
    RQ2: Component contribution (patterns, NER, OCR, zoning, linking).
    RQ3: Robustness across previously unseen templates.
    RQ4: Re-identification attack performance before/after anonymisation.
    """
    def __init__(self, dataset_root: Path, config: Config):
        self.dataset = DatasetManager(dataset_root)
        self.attacker = Attacker(config)
        self.evaluation = Evaluation()
        self.config = config

    def rq1(self):
        """Measure precision, recall, and F1 by region and label.

        RQ1 reports the complete evaluation split, including templates that contain
        embedded images/OCR gold. The template-holdout split is still used by RQ3;
        excluding it here can make the OCR rows look like detector failures even when
        the evaluated documents simply contain no OCR gold.
        """
        docs = self.rq1_documents()
        result = self._eval(docs, self.config)
        result["name"] = "Research Question 1"
        result["document_scope"] = self._scope_summary(docs)
        return result

    def rq1_documents(self):
        """Return documents used by RQ1 without dropping the OCR-bearing holdout template."""
        rows = self.dataset.load_manifest()
        for splits in (("test",), ("val", "test"), ()):  # prefer held-out records, then fall back gracefully
            docs = [row for row in rows if not splits or row.split in splits]
            if len(docs) >= 3:
                return docs
        return rows

    def _scope_summary(self, docs: Iterable[DocumentRecord]):
        docs = list(docs)
        templates = sorted({doc.template_id for doc in docs})
        splits = sorted({doc.split for doc in docs})
        return {
            "num_documents": len(docs),
            "templates": templates,
            "splits": splits,
        }

    def rq2(self):
        """Compare RQ2 component contributions on the same held-out documents as RQ1.

        The variants are intentionally split into component-isolation runs and
        incremental pipeline runs:
        - Pattern based: deterministic pattern detectors only, all text regions.
        - Named Entity Recognition: contextual name detector plus optional GLiNER zero-shot NER, all text regions.
        - Text-only baseline: pattern + NER detectors in body paragraphs only.
        - Layout-aware zoning: pattern + NER detectors across cover/header/footer/body/table text.
        - Everything: the configured full pipeline, including OCR and linking when enabled.

        This avoids the earlier misleading pattern-only result that restricted
        patterns to body paragraphs, where most emails, student numbers, dates,
        roles, and course/group metadata do not appear in the generated corpus.
        """
        docs = self.rq1_documents()
        text_regions = ("cover", "header", "footer", "body", "body_table")
        body_regions = ("body",)
        selected_ner_backend = getattr(self.config, "ner_backend", "hybrid") or "hybrid"

        def out_cfg(suffix: str, **overrides):
            return replace(self._out(self.config, suffix), **overrides)

        variants = [
            (
                "Text-only baseline",
                out_cfg(
                    "rq2_text_only_baseline",
                    patterns=True,
                    ner=True,
                    ocr=False,
                    link=False,
                    strict=True,
                    regions=body_regions,
                    ner_backend=selected_ner_backend,
                    gliner_model=getattr(self.config, "gliner_model", "urchade/gliner_multi-v2.1"),
                    gliner_threshold=getattr(self.config, "gliner_threshold", 0.35),
                ),
            ),
            (
                "Pattern based",
                out_cfg(
                    "rq2_pattern_based",
                    patterns=True,
                    ner=False,
                    ocr=False,
                    link=False,
                    strict=True,
                    regions=text_regions,
                ),
            ),
            (
                "Named Entity Recognition",
                out_cfg(
                    "rq2_named_entity_recognition",
                    patterns=False,
                    ner=True,
                    ocr=False,
                    link=False,
                    strict=True,
                    regions=text_regions,
                    ner_backend="hybrid",
                    gliner_model=getattr(self.config, "gliner_model", "urchade/gliner_multi-v2.1"),
                    gliner_threshold=getattr(self.config, "gliner_threshold", 0.35),
                ),
            ),
            (
                "Layout-aware zoning",
                out_cfg(
                    "rq2_layout_aware_zoning",
                    patterns=True,
                    ner=True,
                    ocr=False,
                    link=bool(getattr(self.config, "link", False)),
                    strict=bool(getattr(self.config, "strict", True)),
                    regions=text_regions,
                    ner_backend=selected_ner_backend,
                    gliner_model=getattr(self.config, "gliner_model", "urchade/gliner_multi-v2.1"),
                    gliner_threshold=getattr(self.config, "gliner_threshold", 0.35),
                ),
            ),
            (
                "Everything",
                out_cfg(
                    "rq2_everything",
                    patterns=True,
                    ner=True,
                    ocr=bool(getattr(self.config, "ocr", True)),
                    link=bool(getattr(self.config, "link", True)),
                    strict=bool(getattr(self.config, "strict", True)),
                    regions=(),
                    ner_backend=selected_ner_backend,
                    gliner_model=getattr(self.config, "gliner_model", "urchade/gliner_multi-v2.1"),
                    gliner_threshold=getattr(self.config, "gliner_threshold", 0.35),
                ),
            ),
        ]

        runs = []
        for name, cfg in variants:
            result = self._eval(docs, cfg)
            result["name"] = name
            result["document_scope"] = self._scope_summary(docs)
            result["config"] = {
                "patterns": bool(cfg.patterns),
                "ner": bool(cfg.ner),
                "ocr": bool(cfg.ocr),
                "link": bool(cfg.link),
                "strict": bool(cfg.strict),
                "regions": tuple(cfg.regions),
                "ner_backend": getattr(cfg, "ner_backend", "candidate"),
                "model": getattr(cfg, "model", None),
                "gliner_model": getattr(cfg, "gliner_model", None),
            }
            runs.append(result)

        baseline = runs[0]
        comparison = self.evaluation.runs(baseline, runs[1:])
        return {"baseline": baseline, "variants": runs[1:], "comparison": comparison, "all_runs": runs}

    def rq3(self):
        """Evaluate the pipeline on a template holdout set."""
        dev = self._eval(self._std(), self.config)
        hold = self._eval(self._hold(), self.config)
        return {
            "name": "Research Question 3",
            "holdout_templates": sorted(self._holdout()),
            "development": dev,
            "holdout": hold,
            "robustness_drop": self.evaluation.drop(
                dev, hold
            ),
        }

    def rq4(self):
        """Compare attacker performance before and after anonymisation."""
        docs = self._std()
        profiles = self.dataset.master_profiles()
        kb = self.attacker.load_knowledge_base(profiles)
        # Remark:
        # Closed-world: the true identity is guaranteed to be in the attacker's KB.
        # Open-world: the true identity may be absent from the attacker's KB.
        # In open-world mode we remove true IDs for some documents and allow abstention.

        before_closed = []
        after_closed = []
        before_open = []
        after_open = []

        # Run attacker before and after anonymization, in closed/open settings.
        for record in docs:
            gold_ids = self._ids(record)
            if not gold_ids:
                continue

            document = self.dataset.attach(copy.deepcopy(record))
            closed_before = self.attacker.attack_document(document, kb)
            closed_before["gold_ids"] = gold_ids
            closed_before["gold_present"] = True
            before_closed.append(closed_before)
            anonymized, _audit = process_document(document.document_path, self.config)
            anon_doc = DocumentRecord(
                doc_id=document.doc_id,
                document_path=anonymized.document_path,
                template_id=document.template_id,
                split=document.split,
            )
            closed_after = self.attacker.attack_document(anon_doc, kb)
            closed_after["gold_ids"] = gold_ids
            closed_after["gold_present"] = True
            after_closed.append(closed_after)

            gold_present = self._frac(document.doc_id) < 0.5
            excluded_ids = set(gold_ids)
            open_kb = (kb if gold_present else [candidate for candidate in kb if candidate.record not in excluded_ids])
            open_before = self.attacker.attack_document(document, open_kb)
            open_before["gold_ids"] = gold_ids
            open_before["gold_present"] = gold_present
            before_open.append(open_before)

            open_after = self.attacker.attack_document(anon_doc, open_kb)
            open_after["gold_ids"] = gold_ids
            open_after["gold_present"] = gold_present
            after_open.append(open_after)

        return {
            "name": "Research Question 4",
            "closed_world": self.evaluation.compare(before_closed, after_closed),
            "open_world": self.evaluation.compare(before_open, after_open),
            "before_closed_results": before_closed,
            "after_closed_results": after_closed,
            "before_open_results": before_open,
            "after_open_results": after_open,
        }

    def run_all(self):
        """Run all experiments in thesis order."""
        return {
            "rq1": self.rq1(),
            "rq2": self.rq2(),
            "rq3": self.rq3(),
            "rq4": self.rq4(),
        }

    def _frac(self, key: str):
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / float(16**8)

    def _holdout(self):
        return set(self.dataset.unseen_holdout())

    def _docs(self, in_hold: bool, min_n: int):
        rows = self.dataset.load_manifest()
        hold = self._holdout()

        def ok(row):
            return (row.template_id in hold) == in_hold

        def pick(splits=()):
            if not splits:
                return [row for row in rows if ok(row)]
            return [row for row in rows if ok(row) and row.split in splits]

        for splits in (("test",), ("val", "test"), ()):
            docs = pick(splits)
            if len(docs) >= min_n:
                return docs
        return docs

    def _std(self):
        return self._docs(in_hold=False, min_n=3)

    def _hold(self):
        return self._docs(in_hold=True, min_n=2)

    def _labels(self, config: Config):
        labels = set()
        if config.direct:
            labels.update({"Name", "FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email"})
        if config.quasi:
            labels.update(canon(label) for label in config.labels)
        return labels

    def _out(self, config: Config, suffix: str):
        if config.output is None:
            return config
        return replace(config, output=Path(config.output) / suffix)

    def _eval(self, documents: Iterable[DocumentRecord], config: Config):
        labels = self._labels(config)
        rows = [self._row(record, config, labels) for record in documents]
        return self._sum(rows, labels)

    def _row(self, record: DocumentRecord, config: Config, labels: set):
        # Build one evaluation row for one document.
        doc = self.dataset.attach(copy.deepcopy(record))
        gold = self.dataset.load_annotations(doc)
        pred = detect_entities(doc, config)
        return {
            "doc_id": doc.doc_id,
            "template_id": doc.template_id,
            "split": doc.split,
            "scores": self.evaluation.scores(gold, pred),
            "remaining_entities": self.evaluation.remain(gold, pred, labels),
            "predicted_entities": pred,
        }

    def _sum(self, rows: Iterable[dict], redacted_labels: set | None = None):
        total = defaultdict(int)
        by_reg: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        by_label: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        remain = []
        per_doc = []

        def add(dst, src):
            dst["tp"] += int(src["tp"])
            dst["fp"] += int(src["fp"])
            dst["fn"] += int(src["fn"])

        # Fold per-document score dicts into global counters.
        for row in rows:
            per_doc.append(row)
            score = row["scores"]
            add(total, score["overall"])
            for reg, m in score["by_region"].items():
                add(by_reg[reg], m)
            for reg, labels in score["by_region_label"].items():
                for label, m in labels.items():
                    add(by_label[reg][label], m)
            remain.extend(row["remaining_entities"])

        leakage = self.evaluation.leak(remain)
        protected_labels = {canon(label) for label in (redacted_labels or set())}
        protected_remaining = [entity for entity in remain if canon(entity.get("label", "")) in protected_labels]
        leakage["protected_remaining"] = len(protected_remaining)
        leakage["protected_weighted_total"] = sum(severity_weight_for(entity.get("label", "")) or 1.0 for entity in protected_remaining)

        return {
            "overall": prf(total["tp"], total["fp"], total["fn"]),
            "by_region": {reg: prf(c["tp"], c["fp"], c["fn"]) for reg, c in sorted(by_reg.items())},
            "by_region_label": {
                reg: {label: prf(c["tp"], c["fp"], c["fn"]) for label, c in sorted(labels.items())}
                for reg, labels in sorted(by_label.items())
            },
            "leakage": leakage,
            "per_document": per_doc,
        }

    def _ids(self, document: DocumentRecord):
        annotations = self.dataset.load_annotations(document)
        ids = set()
        for key in ("text_entities", "ocr_entities"):
            ids.update(
                entity.get("entity_id", "")
                for entity in annotations.get(key, [])
                if entity.get("entity_type") == "student" and entity.get("entity_id")
            )
        return sorted(identifier for identifier in ids if identifier)
