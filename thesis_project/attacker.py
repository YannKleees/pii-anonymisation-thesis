"""Record-linkage attacker using direct hash matching and quasi-identifier pooling."""
import csv
import hashlib
import json
from pathlib import Path

from .defender.parser import normalise_blocks, parse_document
from .defender.pipeline import detect_entities
from .shared import KnowledgeBaseRecord, _norm, canon
from .shared import attack_margin_threshold, attack_score_threshold


core_fields = {
    "record",
    "entity_type",
    "full_name",
    "student_number",
    "institutional_email",
    "programme",
    "group_ids",
    "group_labels",
    "course_ids",
    "course_codes",
    "roles",
}

score_weights = {
    "full_names": 4.0,
    "student_numbers": 6.0,
    "emails": 5.0,
    "course_codes": 1.75,
    "group_labels": 1.5,
    "programmes": 1.0,
    "roles": 0.5,
}

soft_penalty_fields = {"course_codes", "group_labels", "programmes"}

direct_labels = {"FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email", "Name"}

knowledge_levels = {
    "weak": ("Programme", "Course code"),
    "medium": ("Programme", "Course code", "Group label", "Role"),
    "strong": ("Programme", "Course code", "Group label", "Role", "Date"),
}


def _normalised_row(row):
    """Strip BOM and whitespace from row keys."""
    return {str(key).lstrip("\ufeff").strip(): value for key, value in dict(row).items()}


def _first_non_empty(row, *keys, default=""):
    """Extract first non-empty value from row by key order, skipping 'NG' and NaN."""
    row = _normalised_row(row)
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text != "NG" and text.lower() != "nan":
            return text
    return default


def _pipe_tuple(value):
    """Parse pipe-separated value into deduplicated sorted tuple."""
    if value is None:
        return tuple()
    if isinstance(value, str):
        value = value.split("|")
    parts = [
        clean
        for item in value
        if (clean := str(item).strip()) and clean != "NG" and clean.lower() != "nan"
    ]
    return tuple(sorted(dict.fromkeys(parts)))


def _uniq_norm(values):
    """Normalize values and deduplicate while preserving sort order."""
    return tuple(sorted(dict.fromkeys(norm for value in values if (norm := _norm(value)))))


def _norm_set(values):
    """Return set of normalized non-empty values."""
    return {_norm(value) for value in values if _norm(value)}


def _hash_value(value):
    """Return SHA256 hash of normalized value, or empty string if value is empty."""
    norm = _norm(value)
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _entity_label_surface(entity):
    """Extract label and surface from entity dict or object."""
    if isinstance(entity, dict):
        return canon(entity.get("label", "")), entity.get("surface", "")
    return canon(getattr(entity, "label", "")), getattr(entity, "surface", "")


def _name_parts(full_name):
    """Extract first and last name from full name string."""
    parts = [p for p in str(full_name).split() if p]
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    return first, last


class Attacker:
    """Record-linkage attacker with profile-ranking and hash-filter modes."""

    def __init__(self, config):
        self.config = config

    def load_knowledge_base(self, kb_path):
        """Load knowledge base from JSON or CSV file and return list of KnowledgeBaseRecord."""
        kb_path = Path(kb_path)
        if not kb_path.exists():
            raise FileNotFoundError(kb_path)
        if kb_path.suffix.lower() == ".json":
            rows = json.loads(kb_path.read_text(encoding="utf-8-sig"))
            if isinstance(rows, dict):
                rows = rows.get("records", rows.get("rows", []))
        else:
            with kb_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        records = []
        for raw_row in rows:
            row = _normalised_row(raw_row)
            role_value = _first_non_empty(row, "role")
            records.append(
                KnowledgeBaseRecord(
                    record=_first_non_empty(row, "record", "student_number", "person_id"),
                    entity_type=_first_non_empty(row, "entity_type", "role", default="student"),
                    full_name=_first_non_empty(row, "full_name", "full_legal_name", "full_name_raw"),
                    student_number=_first_non_empty(row, "student_number"),
                    institutional_email=_first_non_empty(row, "institutional_email", "email"),
                    programme=_first_non_empty(row, "programme", "program"),
                    group_ids=_pipe_tuple(_first_non_empty(row, "group_ids")),
                    group_labels=_pipe_tuple(_first_non_empty(row, "group_labels")),
                    course_ids=_pipe_tuple(_first_non_empty(row, "course_ids")),
                    course_codes=_pipe_tuple(_first_non_empty(row, "course_codes")),
                    roles=_pipe_tuple(_first_non_empty(row, "roles", default=role_value)),
                    metadata={key: value for key, value in row.items() if key not in core_fields},
                )
            )
        return [record for record in records if record.record]

    def extract_signals(self, document):
        """Extract detected PII signals from document blocks."""
        if document.blocks:
            parsed = document
        else:
            parsed = normalise_blocks(parse_document(document.document_path))
        detections = detect_entities(parsed, self.config)
        label_map = {}
        for item in detections:
            label_map.setdefault(canon(item.label), []).append(item.surface)

        get_values = lambda label: _uniq_norm(label_map.get(label, ()))
        full_name_signals = _uniq_norm([
            *label_map.get("Name", ()),
            *label_map.get("FULL_NAME", ()),
        ])

        return {
            "doc_id": document.doc_id,
            "full_names": full_name_signals,
            "student_numbers": get_values("Student number"),
            "emails": get_values("Institutional email"),
            "course_codes": get_values("Course code"),
            "group_labels": get_values("Group label"),
            "programmes": get_values("Programme"),
            "roles": get_values("Role"),
            "dates": get_values("Date"),
            "num_detections": len(detections),
            "detections": detections,
        }

    def generate_candidates(self, signals, kb_records):
        """Filter knowledge base records matching any detected signal."""
        names = set(signals.get("full_names", ()))
        nums = set(signals.get("student_numbers", ()))
        emails = set(signals.get("emails", ()))
        groups = set(signals.get("group_labels", ()))
        courses = set(signals.get("course_codes", ()))
        progs = set(signals.get("programmes", ()))
        candidates = []
        for rec in kb_records:
            matched = (
                (nums and _norm(rec.student_number) in nums)
                or (emails and _norm(rec.institutional_email) in emails)
                or (names and _norm(rec.full_name) in names)
                or (groups and groups.intersection(_norm_set(rec.group_labels)))
                or (courses and courses.intersection(_norm_set(rec.course_codes)))
                or (progs and _norm(rec.programme) in progs)
            )
            if matched:
                candidates.append(rec)
        return candidates or list(kb_records)

    def score_candidates(self, signals, candidates):
        """Score candidate records against detected signals."""
        signal_sets = self._build_signal_sets(signals)
        scored = []
        for rec in candidates:
            candidate_sets = self._build_candidate_sets(rec)
            score, evidence = self._score_candidate(signal_sets, candidate_sets)
            scored.append({"record": rec.record, "score": round(score, 4), "candidate": rec, "evidence": evidence})
        return scored

    def _build_signal_sets(self, signals):
        """Convert signal dict to dict of sets for intersection operations."""
        return {
            "full_names": set(signals.get("full_names", ())),
            "student_numbers": set(signals.get("student_numbers", ())),
            "emails": set(signals.get("emails", ())),
            "course_codes": set(signals.get("course_codes", ())),
            "group_labels": set(signals.get("group_labels", ())),
            "programmes": set(signals.get("programmes", ())),
            "roles": set(signals.get("roles", ())),
        }

    def _build_candidate_sets(self, rec):
        """Convert knowledge base record to dict of sets for matching."""
        return {
            "full_names": {_norm(rec.full_name)} if rec.full_name else set(),
            "student_numbers": {_norm(rec.student_number)} if rec.student_number else set(),
            "emails": {_norm(rec.institutional_email)} if rec.institutional_email else set(),
            "course_codes": _norm_set(rec.course_codes),
            "group_labels": _norm_set(rec.group_labels),
            "programmes": {_norm(rec.programme)} if rec.programme else set(),
            "roles": _norm_set(rec.roles),
        }

    def _score_candidate(self, signal_sets, candidate_sets):
        """Compute weighted match score and evidence dict for candidate."""
        score = 0.0
        evidence = {}
        matched_types = 0
        for key, vals in signal_sets.items():
            if not vals:
                continue
            overlap = vals.intersection(candidate_sets.get(key, set()))
            if overlap:
                matched_types += 1
                score += score_weights[key] * len(overlap)
                evidence[key] = sorted(overlap)
            elif key in soft_penalty_fields:
                score -= 0.10
        score += 0.15 * matched_types
        return score, evidence

    def rank_candidates(self, scored):
        """Sort scored candidates by descending score."""
        return sorted(scored, key=lambda item: (-item["score"], str(item["record"])))

    def predict_closed_world(self, ranked):
        """Return top-ranked candidate record ID or None."""
        return ranked[0]["record"] if ranked else None

    def predict_open_world(self, ranked):
        """Return top candidate if score and margin exceed thresholds, else abstain."""
        if not ranked:
            return None
        top_score = ranked[0]["score"]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if top_score < attack_score_threshold:
            return None
        if margin < attack_margin_threshold:
            return None
        return ranked[0]["record"]

    def attack_document(self, document, kb_records):
        """Run end-to-end attack: extract signals, rank candidates, return predictions."""
        signals = self.extract_signals(document)
        candidates = self.generate_candidates(signals, kb_records)
        ranked = self.rank_candidates(self.score_candidates(signals, candidates))
        return {
            "doc_id": document.doc_id,
            "signals": signals,
            "candidate_count": len(candidates),
            "ranked_candidates": [
                {"record": item["record"], "score": item["score"], "evidence": item["evidence"]}
                for item in ranked
            ],
            "predicted_closed_world": self.predict_closed_world(ranked),
            "predicted_open_world": self.predict_open_world(ranked),
        }

    def profile_id(self, profile):
        """Return string ID for profile record."""
        return str(profile.record)

    def profile_values(self, profile, label):
        """Return set of normalized values for profile label."""
        label = canon(label)
        first_name, last_name = _name_parts(profile.full_name)
        values = {
            "Name": (profile.full_name,),
            "FULL_NAME": (profile.full_name,),
            "FIRST_NAME": (first_name,),
            "LAST_NAME": (last_name,),
            "Student number": (profile.student_number,),
            "Institutional email": (profile.institutional_email,),
            "Programme": (profile.programme,),
            "Course code": profile.course_codes,
            "Group label": profile.group_labels,
            "Role": profile.roles,
            "Date": _pipe_tuple(profile.metadata.get("date", "")) + _pipe_tuple(profile.metadata.get("dates", "")),
        }.get(label, tuple())
        return _norm_set(values)

    def hash_value(self, value):
        """Return SHA256 hash of normalized value."""
        return _hash_value(value)

    def direct_hash_matches(self, entity, kb_records):
        """Return set of profile IDs matching entity via direct hash."""
        label, surface = _entity_label_surface(entity)
        if label not in direct_labels or not surface:
            return set()
        surface_hash = _hash_value(surface)
        if not surface_hash:
            return set()
        matches = set()
        for profile in kb_records:
            profile_hashes = {_hash_value(value) for value in self.profile_values(profile, label)}
            if surface_hash in profile_hashes:
                matches.add(self.profile_id(profile))
        return matches

    def quasi_candidate_pool(self, entities, kb_records, knowledge_level="medium"):
        """Filter profiles by quasi-identifiers; return pool and labels used."""
        labels = knowledge_levels.get(knowledge_level, knowledge_levels["medium"])
        pool = {self.profile_id(profile) for profile in kb_records}
        used_labels = []
        for label in labels:
            surfaces = {
                _norm(surface)
                for entity in entities
                for entity_label, surface in [_entity_label_surface(entity)]
                if canon(entity_label) == label and _norm(surface)
            }
            if not surfaces:
                continue
            matches = {
                self.profile_id(profile)
                for profile in kb_records
                if self.profile_values(profile, label).intersection(surfaces)
            }
            if matches:
                pool &= matches
                used_labels.append(label)
        return pool, used_labels

    def hash_filter_attack(self, record, entities, kb_records, true_profile_ids=None, phase="", knowledge_level="medium"):
        """Evaluate hash-filter attack on one document."""
        true_profile_ids = {str(x) for x in (true_profile_ids or set())}
        direct_unique = set()
        direct_ambiguous = 0

        for entity in entities:
            matches = self.direct_hash_matches(entity, kb_records)
            if len(matches) == 1:
                direct_unique |= matches
            elif len(matches) > 1:
                direct_ambiguous += 1

        direct_revealed = direct_unique.intersection(true_profile_ids) if true_profile_ids else set()
        quasi_pool, used_quasi = self.quasi_candidate_pool(entities, kb_records, knowledge_level)
        quasi_unique = len(quasi_pool) == 1
        quasi_revealed = bool(quasi_unique and (not true_profile_ids or quasi_pool.intersection(true_profile_ids)))

        revealed_profiles = set(direct_revealed)
        if quasi_revealed:
            revealed_profiles |= quasi_pool

        reason = "none"
        if direct_revealed:
            reason = "direct hash match"
        elif quasi_revealed:
            reason = "indirect uniqueness"
        elif direct_unique:
            reason = "wrong unique direct hash"
        elif direct_ambiguous:
            reason = "ambiguous direct hash"
        elif quasi_unique:
            reason = "wrong unique indirect match"
        elif used_quasi:
            reason = "non-unique indirect match"

        return {
            "knowledge_level": knowledge_level,
            "phase": phase,
            "doc_id": record.doc_id,
            "template_id": record.template_id,
            "true_profiles_in_document": len(true_profile_ids),
            "remaining_identifier_mentions": len(entities),
            "direct_hash_revealed_profiles": len(direct_revealed),
            "direct_hash_candidate_hits": len(direct_unique),
            "ambiguous_direct_hash_hits": direct_ambiguous,
            "quasi_labels_used": ", ".join(used_quasi),
            "quasi_candidate_pool_size": len(quasi_pool),
            "quasi_unique_success": bool(quasi_revealed),
            "any_reidentification": bool(revealed_profiles),
            "revealed_profiles": len(revealed_profiles),
            "main_reason": reason,
        }

    def hash_filter_attacks(self, records, before_entities_by_doc, after_entities_by_doc, kb_records, true_ids_by_doc, levels=None):
        """Run hash-filter attack across records before/after redaction at multiple knowledge levels."""
        rows = []
        for record in records:
            before_entities = before_entities_by_doc.get(record.doc_id, [])
            after_entities = after_entities_by_doc.get(record.doc_id, [])
            true_ids = true_ids_by_doc.get(record.doc_id, set())
            for level in (levels or knowledge_levels):
                rows.append(self.hash_filter_attack(record, before_entities, kb_records, true_ids, "before", level))
                rows.append(self.hash_filter_attack(record, after_entities, kb_records, true_ids, "after", level))
        return rows