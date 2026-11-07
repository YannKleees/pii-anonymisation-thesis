"""Evaluation metrics and diagnostics for text and OCR entity detection."""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re

from .shared import Occurrence, _norm, canon, direct, normalize_region, quasi, severity_weight_for


def _item(entity):
    """Normalize entity dict with canonical label, region, and offsets."""
    if isinstance(entity, Occurrence):
        return {
            "label": canon(entity.label),
            "surface": entity.surface,
            "start": int(entity.start),
            "end": int(entity.end),
            "region": normalize_region(entity.region),
            "path": entity.path,
            "metadata": entity.metadata,
        }
    item = dict(entity)
    item["label"] = canon(str(item.get("label", "")))
    item["region"] = normalize_region(str(item.get("region", "")))
    item["start"] = int(item.get("start", 0) or 0)
    item["end"] = int(item.get("end", 0) or 0)
    item.setdefault("surface", "")
    item.setdefault("path", "")
    item.setdefault("metadata", {})
    return item


def _flatten_schema_dict(gold_entities):
    """Read gold entities from flattened and legacy annotation dictionary schemas."""
    text_items = [_item(item) for item in gold_entities.get("text_entities", []) or []]
    ocr_items = [_item({**item, "region": "ocr"}) for item in gold_entities.get("ocr_entities", []) or []]

    for entity in gold_entities.get("entities", []) or []:
        occurrences = entity.get("occurrences") or []
        if occurrences:
            for occurrence in occurrences:
                text_items.append(_item({**entity, **occurrence}))
        elif str(entity.get("region", "")).lower() == "ocr" or entity.get("image_id") or entity.get("image_path"):
            ocr_items.append(_item({**entity, "region": "ocr"}))
        elif entity.get("path") is not None:
            text_items.append(_item(entity))

    for image in gold_entities.get("images", []) or []:
        for entity in image.get("entities", []) or []:
            ocr_items.append(_item({**entity, "region": "ocr", "image_id": image.get("image_id", ""), "image_path": image.get("image_path", "")}))

    return text_items, ocr_items


def _split(gold_entities):
    """Split entities into text and OCR items."""
    if isinstance(gold_entities, dict):
        return _flatten_schema_dict(gold_entities)
    items = [_item(item) for item in gold_entities]
    text_items = [item for item in items if item.get("region") != "ocr"]
    ocr_items = [item for item in items if item.get("region") == "ocr"]
    return text_items, ocr_items


def _count(entities, relaxed_ocr=False):
    """Count entity occurrences, keying by (label, surface) or (label, path, start, end)."""
    counter = Counter()
    for entity in entities:
        item = _item(entity)
        if relaxed_ocr or item.get("region") == "ocr":
            key = (item.get("label"), _norm(item.get("surface", "")))
        else:
            key = (
                item.get("label"),
                item.get("path"),
                int(item.get("start", 0)),
                int(item.get("end", 0)),
            )
        counter[key] += 1
    return counter


def _metrics(tp, fp, fn):
    """Compute precision, recall, F1 from TP, FP, FN counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def zero_reason(gold_count, predicted_count, *, scope="cell"):
    """Explain why a zero-valued metric cell appears in diagnostics."""
    gold_count = int(gold_count or 0)
    predicted_count = int(predicted_count or 0)
    if gold_count == 0 and predicted_count == 0:
        return f"no gold coverage for this {scope}"
    if gold_count == 0 and predicted_count > 0:
        return f"no gold coverage; {predicted_count} prediction(s) count as false positives"
    if gold_count > 0 and predicted_count == 0:
        return f"detector produced no predictions for {gold_count} gold item(s)"
    return "gold and predictions exist; zero score means no exact/relaxed match"


def _match_stats(gold_counter, predicted_counter, debug=False):
    """Count TP/FP/FN and optionally return debug info."""
    matched_keys = set(gold_counter).intersection(predicted_counter)
    tp = sum(min(gold_counter[key], predicted_counter[key]) for key in matched_keys)
    fp = sum(predicted_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp
    result = _metrics(tp, fp, fn)
    if not debug:
        return result
    return {
        **result,
        "gold_count": sum(gold_counter.values()),
        "predicted_count": sum(predicted_counter.values()),
        "matched_keys": sorted(matched_keys),
        "gold_counter": gold_counter,
        "predicted_counter": predicted_counter,
    }


def _ocr_compact(text: str):
    """Remove non-alphanumeric chars and map OCR-common misreadings."""
    value = re.sub(r"[^a-z0-9]+", "", str(text or "").lower())
    return value.translate(str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b"}))


def _ocr_tokens(text: str):
    """Extract alphabetic tokens from text."""
    return re.findall(r"[a-z]+", str(text or "").lower())


def _ocr_initials(text: str):
    """Extract first letter of each token."""
    return "".join(token[:1] for token in _ocr_tokens(text))


def _ocr_similarity(a: str, b: str):
    """Compute string similarity accounting for OCR errors."""
    left = _ocr_compact(a)
    right = _ocr_compact(b)
    if not left or not right:
        return 0.0
    score = SequenceMatcher(None, left, right).ratio()
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 6 and shorter in longer and len(shorter) / max(1, len(longer)) >= 0.55:
        score = max(score, 0.93)
    return score


def _ocr_surfaces_match(gold_surface: str, predicted_surface: str, threshold: float = 0.84):
    """Determine if OCR gold and predicted surfaces match within threshold."""
    gold_norm = _norm(gold_surface)
    pred_norm = _norm(predicted_surface)
    if gold_norm and gold_norm == pred_norm:
        return True, 1.0
    score = _ocr_similarity(gold_surface, predicted_surface)
    if score < threshold:
        return False, score
    if score >= 0.92:
        return True, score
    gold_initials = _ocr_initials(gold_surface)
    pred_initials = _ocr_initials(predicted_surface)
    if len(gold_initials) >= 2 and len(pred_initials) >= 2:
        return gold_initials[0] == pred_initials[0] and gold_initials[-1] == pred_initials[-1], score
    return False, score


def _unique_ocr_items(entities):
    """OCR gold is image-level entity coverage, not per-mention occurrence coverage."""
    unique = {}
    for entity in entities:
        item = _item(entity)
        surface_key = _norm(item.get("surface", ""))
        if not item.get("label") or not surface_key:
            continue
        key = (item.get("label"), surface_key)
        unique.setdefault(key, item)
    return list(unique.values())


def _ocr_match_stats(gold_entities, predicted_entities, debug=False):
    """Match OCR items with exact then fuzzy matching."""
    gold_items = _unique_ocr_items(gold_entities)
    predicted_items = _unique_ocr_items(predicted_entities)
    used_predictions = set()
    matched_pairs = []

    # Exact normalized matches first.
    for gold_index, gold in enumerate(gold_items):
        for pred_index, pred in enumerate(predicted_items):
            if pred_index in used_predictions or gold["label"] != pred["label"]:
                continue
            if _norm(gold.get("surface", "")) == _norm(pred.get("surface", "")):
                used_predictions.add(pred_index)
                matched_pairs.append((gold_index, pred_index, 1.0))
                break

    matched_gold = {gold_index for gold_index, _pred_index, _score in matched_pairs}

    # Then OCR-tolerant fuzzy matches. This avoids scoring every minor OCR character
    # error as both an FP and an FN.
    for gold_index, gold in enumerate(gold_items):
        if gold_index in matched_gold:
            continue
        best = None
        for pred_index, pred in enumerate(predicted_items):
            if pred_index in used_predictions or gold["label"] != pred["label"]:
                continue
            ok, score = _ocr_surfaces_match(gold.get("surface", ""), pred.get("surface", ""))
            if not ok:
                continue
            if best is None or score > best[0]:
                best = (score, pred_index)
        if best is not None:
            score, pred_index = best
            used_predictions.add(pred_index)
            matched_pairs.append((gold_index, pred_index, score))
            matched_gold.add(gold_index)

    tp = len(matched_pairs)
    fp = len(predicted_items) - tp
    fn = len(gold_items) - tp
    result = _metrics(tp, fp, fn)
    if not debug:
        return result
    return {
        **result,
        "gold_count": len(gold_items),
        "predicted_count": len(predicted_items),
        "matched_pairs": [
            {
                "gold": gold_items[gold_index],
                "predicted": predicted_items[pred_index],
                "similarity": score,
            }
            for gold_index, pred_index, score in matched_pairs
        ],
        "unmatched_gold": [item for index, item in enumerate(gold_items) if index not in matched_gold],
        "unmatched_predicted": [item for index, item in enumerate(predicted_items) if index not in used_predictions],
    }


class Evaluation:
    """Compute the main thesis metrics and summaries."""

    def exact(self, gold_entities, predicted_entities, debug=False):
        """Compute exact match metrics for text entities."""
        gold_counter = _count(gold_entities)
        predicted_counter = _count([entity for entity in predicted_entities if _item(entity).get("region") != "ocr"])
        return _match_stats(gold_counter, predicted_counter, debug=debug)

    def ocr(self, gold_entities, predicted_entities, debug=False):
        """Compute fuzzy-match metrics for OCR entities."""
        # Accept either already-split OCR gold rows or a full annotation dictionary.
        # The latter keeps diagnostics robust across entities/text_entities/ocr_entities/images[*].entities schemas.
        if isinstance(gold_entities, dict):
            _gold_text, gold_ocr = _split(gold_entities)
        else:
            gold_ocr = gold_entities
        predicted_ocr = [entity for entity in predicted_entities if _item(entity).get("region") == "ocr"]
        return _ocr_match_stats(gold_ocr, predicted_ocr, debug=debug)

    def scores(self, gold_entities, predicted_entities):
        """Build precision/recall/F1 summaries across regions and labels."""
        gold_text, gold_ocr = _split(gold_entities)
        predicted_items = [
            _item(entity)
            for entity in predicted_entities
            if not _item(entity).get("metadata", {}).get("redaction_only")
        ]
        predicted_text = [entity for entity in predicted_items if entity.get("region") != "ocr"]
        predicted_ocr = [entity for entity in predicted_items if entity.get("region") == "ocr"]

        overall_exact = self.exact(gold_text, predicted_text)
        overall_ocr = self.ocr(gold_ocr, predicted_ocr)
        overall = _metrics(
            overall_exact["tp"] + overall_ocr["tp"],
            overall_exact["fp"] + overall_ocr["fp"],
            overall_exact["fn"] + overall_ocr["fn"],
        )

        by_region = {}
        regions = sorted({entity["region"] for entity in gold_text} | {entity["region"] for entity in predicted_text})
        if gold_ocr or predicted_ocr:
            regions.append("ocr")
        for region in regions:
            if region == "ocr":
                matched = self.ocr(gold_ocr, predicted_ocr)
                by_region[region] = _metrics(matched["tp"], matched["fp"], matched["fn"])
            else:
                gold_region = [entity for entity in gold_text if entity["region"] == region]
                predicted_region = [entity for entity in predicted_text if entity["region"] == region]
                matched = self.exact(gold_region, predicted_region)
                by_region[region] = _metrics(matched["tp"], matched["fp"], matched["fn"])

        by_region_label = defaultdict(dict)
        labels = sorted(
            {entity["label"] for entity in gold_text}
            | {entity["label"] for entity in predicted_text}
            | {entity["label"] for entity in gold_ocr}
            | {entity["label"] for entity in predicted_ocr}
        )
        for region in regions:
            for label in labels:
                if region == "ocr":
                    gold_subset = [entity for entity in gold_ocr if entity["label"] == label]
                    predicted_subset = [entity for entity in predicted_ocr if entity["label"] == label]
                    if not gold_subset and not predicted_subset:
                        continue
                    matched = self.ocr(gold_subset, predicted_subset)
                    by_region_label[region][label] = _metrics(matched["tp"], matched["fp"], matched["fn"])
                else:
                    gold_subset = [
                        entity
                        for entity in gold_text
                        if entity["region"] == region and entity["label"] == label
                    ]
                    predicted_subset = [
                        entity
                        for entity in predicted_text
                        if entity["region"] == region and entity["label"] == label
                    ]
                    if not gold_subset and not predicted_subset:
                        continue
                    matched = self.exact(gold_subset, predicted_subset)
                    by_region_label[region][label] = _metrics(matched["tp"], matched["fp"], matched["fn"])

        return {
            "overall": overall,
            "by_region": by_region,
            "by_region_label": dict(by_region_label),
            "exact_text_match": overall_exact,
            "relaxed_ocr_match": overall_ocr,
        }

    def remain(self, gold_entities, predicted_entities, redacted_labels):
        """List gold entities that remain visible after redaction."""
        redacted = {canon(label) for label in redacted_labels}
        gold_text, gold_ocr = _split(gold_entities)
        predicted_items = [_item(entity) for entity in predicted_entities]
        predicted_exact = _count([entity for entity in predicted_items if entity.get("region") != "ocr"])
        predicted_ocr_items = [entity for entity in predicted_items if entity.get("region") == "ocr"]

        remaining = []
        exact_usage = Counter()

        for entity in gold_text:
            item = _item(entity)
            key = (
                item.get("label"),
                item.get("path"),
                int(item.get("start", 0)),
                int(item.get("end", 0)),
            )
            if item["label"] in redacted and predicted_exact[key] > exact_usage[key]:
                exact_usage[key] += 1
                continue
            remaining.append(item)

        ocr_debug = _ocr_match_stats(gold_ocr, predicted_ocr_items, debug=True)
        matched_ocr_gold_keys = {
            (pair["gold"].get("label"), _norm(pair["gold"].get("surface", "")))
            for pair in ocr_debug.get("matched_pairs", [])
        }
        for entity in gold_ocr:
            item = _item(entity)
            key = (item.get("label"), _norm(item.get("surface", "")))
            if item["label"] in redacted and key in matched_ocr_gold_keys:
                # OCR annotations describe image-level entity coverage. A single fuzzy
                # match is enough to show that this entity was detected for masking.
                continue
            remaining.append(item)
        return remaining

    def leak(self, remaining_entities):
        """Summarise residual leakage from remaining entities."""
        items = [_item(item) for item in remaining_entities]
        total_mentions = len(items)
        weighted_total = sum(severity_weight_for(item["label"]) or 1.0 for item in items)
        by_label = Counter(item["label"] for item in items)
        by_region = Counter(item.get("region", "") for item in items)
        direct_remaining = sum(1 for item in items if direct(item["label"]))
        quasi_remaining = sum(1 for item in items if quasi(item["label"]))
        return {
            "total_mentions": total_mentions,
            "weighted_total": weighted_total,
            "direct_remaining": direct_remaining,
            "quasi_remaining": quasi_remaining,
            "by_label": dict(by_label),
            "by_region": dict(by_region),
        }

    def runs(self, baseline_result, variant_results):
        """Compare baseline leakage against ablation or variant runs.

        Both raw leakage and protected-label leakage are reported. The protected
        values are the safer default for RQ2 because component runs may intentionally
        enable only a subset of detectors while the evaluation still tracks the full
        protected identifier taxonomy.
        """
        baseline_leakage = baseline_result.get("leakage", {})
        baseline_weighted = baseline_leakage.get("weighted_total", 0.0)
        baseline_mentions = baseline_leakage.get("total_mentions", 0)
        baseline_protected_weighted = baseline_leakage.get("protected_weighted_total", baseline_weighted)
        baseline_protected_mentions = baseline_leakage.get("protected_remaining", baseline_mentions)

        rows = []
        for result in [baseline_result, *variant_results]:
            leakage = result.get("leakage", {})
            weighted = leakage.get("weighted_total", 0.0)
            mentions = leakage.get("total_mentions", 0)
            protected_weighted = leakage.get("protected_weighted_total", weighted)
            protected_mentions = leakage.get("protected_remaining", mentions)
            rows.append(
                {
                    "name": result.get("name", ""),
                    "weighted_total": weighted,
                    "total_mentions": mentions,
                    "protected_weighted_total": protected_weighted,
                    "protected_remaining": protected_mentions,
                    "weighted_reduction_vs_baseline": baseline_weighted - weighted,
                    "mention_reduction_vs_baseline": baseline_mentions - mentions,
                    "protected_weighted_reduction_vs_baseline": baseline_protected_weighted - protected_weighted,
                    "protected_mention_reduction_vs_baseline": baseline_protected_mentions - protected_mentions,
                }
            )
        return {"baseline": baseline_result.get("name", ""), "rows": rows}

    def drop(self, dev_result, holdout_result):
        """Measure generalisation drop from development to holdout data."""
        dev_overall = dev_result.get("overall", {})
        holdout_overall = holdout_result.get("overall", {})
        dev_leakage = dev_result.get("leakage", {})
        holdout_leakage = holdout_result.get("leakage", {})
        return {
            "precision_drop": dev_overall.get("precision", 0.0) - holdout_overall.get("precision", 0.0),
            "recall_drop": dev_overall.get("recall", 0.0) - holdout_overall.get("recall", 0.0),
            "f1_drop": dev_overall.get("f1", 0.0) - holdout_overall.get("f1", 0.0),
            "weighted_leakage": holdout_leakage.get("weighted_total", 0.0) - dev_leakage.get("weighted_total", 0.0),
            "mention_leakage": holdout_leakage.get("total_mentions", 0) - dev_leakage.get("total_mentions", 0),
        }

    def reid(self, attack_results):
        """Compute re-identification attack quality metrics."""
        closed_world = [result for result in attack_results if result.get("gold_present", True)]
        top1 = 0
        recall_at_3 = 0
        recall_at_5 = 0
        reciprocal_ranks = []
        for result in closed_world:
            gold_ids = set(result.get("gold_ids", []))
            ranked_ids = [candidate["record"] for candidate in result.get("ranked_candidates", [])]
            if ranked_ids and ranked_ids[0] in gold_ids:
                top1 += 1
            if any(record_id in gold_ids for record_id in ranked_ids[:3]):
                recall_at_3 += 1
            if any(record_id in gold_ids for record_id in ranked_ids[:5]):
                recall_at_5 += 1
            rank = next(
                (
                    index + 1
                    for index, record_id in enumerate(ranked_ids)
                    if record_id in gold_ids
                ),
                None,
            )
            reciprocal_ranks.append(1 / rank if rank else 0.0)

        open_world_correct = 0
        abstentions = 0
        for result in attack_results:
            prediction = result.get("predicted_open_world")
            gold_ids = set(result.get("gold_ids", []))
            gold_present = result.get("gold_present", True)
            if prediction is None:
                abstentions += 1
            if gold_present and prediction in gold_ids:
                open_world_correct += 1
            if not gold_present and prediction is None:
                open_world_correct += 1

        closed_n = max(1, len(closed_world))
        total_n = max(1, len(attack_results))
        return {
            "n_documents": len(attack_results),
            "closed_world_top1": top1 / closed_n,
            "closed_world_recall_at_3": recall_at_3 / closed_n,
            "closed_world_recall_at_5": recall_at_5 / closed_n,
            "closed_world_mrr": sum(reciprocal_ranks) / closed_n if reciprocal_ranks else 0.0,
            "open_world_accuracy": open_world_correct / total_n,
            "open_world_abstention_rate": abstentions / total_n,
        }

    def compare(self, before_results, after_results):
        """Compare attacker performance before against after anonymisation."""
        before_metrics = self.reid(before_results)
        after_metrics = self.reid(after_results)
        return {
            "before": before_metrics,
            "after": after_metrics,
            "delta": {
                "closed_world_top1": before_metrics["closed_world_top1"] - after_metrics["closed_world_top1"],
                "closed_world_recall_at_3": before_metrics["closed_world_recall_at_3"] - after_metrics["closed_world_recall_at_3"],
                "closed_world_recall_at_5": before_metrics["closed_world_recall_at_5"] - after_metrics["closed_world_recall_at_5"],
                "closed_world_mrr": before_metrics["closed_world_mrr"] - after_metrics["closed_world_mrr"],
                "open_world_accuracy": after_metrics["open_world_accuracy"] - before_metrics["open_world_accuracy"],
            },
        }
