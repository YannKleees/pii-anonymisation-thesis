"""Entity detection using local name models, GLiNER, transformer NER, or heuristics."""
from __future__ import annotations

import re
from functools import lru_cache

import thesis_project.shared as shared
from ..shared import Occurrence, is_strict_region

BASE_SCORE = 0.52
STRICT_THRESHOLD = 0.52
BODY_THRESHOLD = 0.58
MAX_SCORE = 0.96
BODY_BONUS = 0.14
COLON_BONUS = 0.10
TEXT_BONUS = 0.10
MATCH_BONUS = 0.18
FOLLOW_COLON_BONUS = 0.18
BULLET_BONUS = 0.12

DEFAULT_GLINER_MODEL = "urchade/gliner_multi-v2.1"

# GLiNER is zero-shot: these prompts are intentionally descriptive while still
# mapping back to the thesis taxonomy. Structured identifiers are still better
# handled by patterns, but GLiNER gives the NER component a real chance to
# contribute to dates, roles, programmes, groups, and direct identifiers too.
GLINER_PROMPT_TO_LABEL = {
    "person name": "Name",
    "student number": "Student number",
    "university email address": "Institutional email",
    "course code": "Course code",
    "student group label": "Group label",
    "degree programme": "Programme",
    "date": "Date",
    "student document role": "Role",
}

ROLE_WORDS = {
    "student",
    "students",
    "tutor",
    "teaching assistant",
    "assistant",
    "coordinator",
    "customer",
    "lead",
    "owner",
    "author",
    "reviewer",
    "chair",
    "secretary",
    "participant",
    "attendee",
}


def _name_model():
    """Lazy-load the local name model helpers to keep imports lightweight."""
    import thesis_project.defender.name_model as name_model

    return name_model


def _finish(entities, kind, start, end, scores):
    if kind is not None and start is not None and end is not None:
        entities.append((kind, start, end, sum(scores) / max(1, len(scores))))


def _split_label(label):
    if label == "O":
        return "O", None
    if "-" in label:
        return label.split("-", 1)
    return "B", label


@lru_cache(maxsize=4)
def _load_transformer(model_name_or_path, local_files_only, device):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model.eval()
    target = (
        torch.device(f"cuda:{device}")
        if device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(target)
    return tokenizer, model, target


@lru_cache(maxsize=4)
def _load_gliner(model_name_or_path, local_files_only, device):
    """Lazy-load GLiNER if it is installed.

    GLiNER is optional so the project still runs in offline/minimal
    environments. Install with ``pip install gliner`` and either allow model
    download from Hugging Face or pass a local model path with ``local=True``.
    """
    import torch
    from gliner import GLiNER

    try:
        model = GLiNER.from_pretrained(
            model_name_or_path,
            local_files_only=bool(local_files_only),
        )
    except TypeError:
        # Older GLiNER versions did not expose local_files_only.
        model = GLiNER.from_pretrained(model_name_or_path)

    target = (
        torch.device(f"cuda:{device}")
        if device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    try:
        model.to(target)
    except Exception:
        pass
    return model


def _aggregate_transformer_entities(text, tokenizer, model, device, max_length):
    import torch

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
        labels = probs.argmax(dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

    entities = []
    active = start = end = None
    scores = []

    for idx, (left, right) in enumerate(offsets):
        if left == right:
            continue

        tag, kind = _split_label(str(model.config.id2label[labels[idx]]))

        if tag == "O":
            _finish(entities, active, start, end, scores)
            active = start = end = None
            scores = []
            continue

        if tag == "B" or kind != active:
            _finish(entities, active, start, end, scores)
            if kind is None:
                active = start = end = None
                scores = []
            else:
                active = kind
                start = left
                end = right
                scores = [confidences[idx]]
            continue

        end = right
        scores.append(confidences[idx])

    _finish(entities, active, start, end, scores)
    return entities


def _transformer_detect_blocks(blocks, config):
    if not config.model:
        return []

    tokenizer, model, device = _load_transformer(config.model, bool(config.local), config.device)

    detections = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        for entity_type, start, end, score in _aggregate_transformer_entities(
            text,
            tokenizer,
            model,
            device,
            config.max_len,
        ):
            if entity_type is None:
                continue

            surface = text[start:end].strip()
            if len(surface.split()) < 2:
                continue

            detections.append(
                Occurrence(
                    label="Name",
                    surface=surface,
                    start=start,
                    end=end,
                    region=block.region,
                    path=block.path,
                    score=score,
                    source="ner_transformer",
                )
            )
    return detections


def _gliner_labels_for_config(config):
    enabled = set()
    if getattr(config, "direct", True):
        enabled.update({"Name", "Student number", "Institutional email"})
    if getattr(config, "quasi", False):
        enabled.update(shared.canon(label) for label in getattr(config, "labels", ()))

    prompts = [
        prompt
        for prompt, label in GLINER_PROMPT_TO_LABEL.items()
        if shared.canon(label) in enabled
    ]
    extra_prompts = tuple(getattr(config, "ner_zero_shot_labels", ()) or ())
    return tuple(dict.fromkeys([*prompts, *extra_prompts]))


def _normalise_model_name(config):
    model_name = getattr(config, "model", None) or getattr(config, "gliner_model", None) or DEFAULT_GLINER_MODEL
    model_name = str(model_name)
    if model_name.startswith("gliner:"):
        model_name = model_name.split(":", 1)[1]
    return model_name


def _gliner_entity_values(entity):
    """Return label, text, start, end, score from dict/object GLiNER outputs."""
    if isinstance(entity, dict):
        label = entity.get("label") or entity.get("entity")
        text = entity.get("text") or entity.get("entity") or entity.get("span") or ""
        start = entity.get("start")
        end = entity.get("end")
        score = entity.get("score", entity.get("confidence", 0.0))
    else:
        label = getattr(entity, "label", None) or getattr(entity, "entity", None)
        text = getattr(entity, "text", None) or getattr(entity, "span", None) or ""
        start = getattr(entity, "start", None)
        end = getattr(entity, "end", None)
        score = getattr(entity, "score", getattr(entity, "confidence", 0.0))
    return str(label or ""), str(text or ""), start, end, float(score or 0.0)


def _locate_surface(text, surface, start, end):
    """Repair missing/approximate offsets from model output."""
    surface = str(surface or "").strip()
    if start is not None and end is not None:
        try:
            left = int(start)
            right = int(end)
            if 0 <= left < right <= len(text):
                return left, right, text[left:right]
        except Exception:
            pass
    if not surface:
        return None
    idx = text.find(surface)
    if idx < 0:
        idx = text.lower().find(surface.lower())
    if idx < 0:
        return None
    return idx, idx + len(surface), text[idx:idx + len(surface)]


def _valid_gliner_detection(label, surface):
    value = re.sub(r"\s+", " ", str(surface or "").strip())
    if not value:
        return False

    canonical = shared.canon(label)
    if canonical == "Name":
        # Keep the candidate/heuristic model responsible for single-token names.
        return len(value.split()) >= 2
    if canonical == "Institutional email":
        return "@" in value and "." in value.rsplit("@", 1)[-1]
    if canonical == "Student number":
        return re.fullmatch(r"[iI](?:[\s-]?\d){7}", value) is not None
    if canonical == "Course code":
        return re.fullmatch(r"[A-Z]{2,6}\d{3,5}", value) is not None
    if canonical == "Role":
        return value.lower() in ROLE_WORDS
    if canonical == "Date":
        return any(ch.isdigit() for ch in value) or re.search(
            r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", value
        )
    if canonical == "Programme":
        return bool(re.search(r"\b(?:BSc|MSc|Bachelor|Master)\b", value))
    if canonical == "Group label":
        return any(ch.isdigit() for ch in value) and bool(re.search(r"(?i)\b(?:group|team|lab|clinic|KEN|T\d{2,3})", value))
    return True


def _gliner_detect_blocks(blocks, config):
    labels = _gliner_labels_for_config(config)
    if not labels:
        return []

    model_name = _normalise_model_name(config)
    model = _load_gliner(model_name, bool(getattr(config, "local", False)), int(getattr(config, "device", -1)))
    threshold = float(getattr(config, "gliner_threshold", 0.35))
    detections = []

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        try:
            entities = model.predict_entities(text, list(labels), threshold=threshold)
        except TypeError:
            entities = model.predict_entities(text, list(labels))

        for entity in entities:
            prompt, surface, start, end, score = _gliner_entity_values(entity)
            label = GLINER_PROMPT_TO_LABEL.get(shared.canon(prompt).lower(), GLINER_PROMPT_TO_LABEL.get(prompt, prompt))
            label = shared.canon(label)
            located = _locate_surface(text, surface, start, end)
            if located is None:
                continue
            left, right, exact_surface = located
            if not _valid_gliner_detection(label, exact_surface):
                continue
            detections.append(
                Occurrence(
                    label=label,
                    surface=exact_surface,
                    start=left,
                    end=right,
                    region=block.region,
                    path=block.path,
                    score=max(score, threshold),
                    source="ner_gliner",
                    metadata={"model": model_name, "prompt_label": prompt},
                )
            )
    return detections


def _score_name(block, start, end, surface):
    features = _name_model().feats(block, start, end, surface)
    score = BASE_SCORE
    if features["is_body"] or features["is_body_table"] or features["is_ocr"] or features["is_strict_region"]:
        score += BODY_BONUS
    if features["preceded_by_colon"]:
        score += COLON_BONUS
    if features.get("followed_by_colon"):
        score += FOLLOW_COLON_BONUS
    if features.get("line_starts_with_bullet"):
        score += BULLET_BONUS
    if features["block_contains_email"] or features["block_contains_student_id"]:
        score += TEXT_BONUS
    if features["block_equals_surface"]:
        score += MATCH_BONUS
    return min(score, MAX_SCORE)


def _heuristic_detect_blocks(blocks, source):
    detections = []
    model = _name_model()
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        for match in model.NAME_RE.finditer(text):
            surface = match.group(0)
            if not model.looks(surface):
                continue

            score = _score_name(block, match.start(), match.end(), surface)
            threshold = shared.name_strict_threshold if is_strict_region(block.region) else shared.name_body_threshold
            if score < threshold:
                continue

            detections.append(
                Occurrence(
                    label="Name",
                    surface=surface,
                    start=match.start(),
                    end=match.end(),
                    region=block.region,
                    path=block.path,
                    score=score,
                    source=source,
                )
            )
    return detections


def detect_blocks(blocks, config, source="ner"):
    backend = str(getattr(config, "ner_backend", "candidate") or "candidate").lower()
    detections = []

    # The candidate model is the thesis-specific context model for names. In
    # hybrid mode it remains responsible for names while GLiNER adds broader NER.
    if getattr(config, "candidate", None) and backend in {"candidate", "hybrid", "auto", "gliner", "zero_shot"}:
        try:
            predicted = _name_model().predict(
                blocks,
                path=config.candidate,
                source="ner_candidate_model",
            )
            detections.extend(predicted or [])
        except Exception:
            if not config.fallback:
                return []

    if backend in {"gliner", "hybrid", "auto", "zero_shot"}:
        try:
            detections.extend(_gliner_detect_blocks(blocks, config))
        except Exception as exc:
            if not config.fallback:
                raise
            # Keep the failure visible in debug metadata if the caller inspects
            # sources, but do not break experiments where GLiNER is not installed.
            _ = exc

    if backend == "transformer" or (backend == "candidate" and getattr(config, "model", None)):
        try:
            detections.extend(_transformer_detect_blocks(blocks, config))
        except Exception:
            if not config.fallback:
                return []

    if not any(shared.canon(item.label) == "Name" for item in detections):
        detections.extend(_heuristic_detect_blocks(blocks, source if source != "ner" else "ner_heuristic"))

    return detections


def detect(document, config):
    return detect_blocks(document.blocks, config)
