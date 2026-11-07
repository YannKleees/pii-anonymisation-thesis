"""Train and apply logistic regression name detection models."""
import re
from functools import lru_cache

import joblib
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline

import thesis_project.shared as shared
from ..shared import Occurrence, _norm, is_strict_region

# AI-assisted regex: matches 2-3 capitalized name tokens, including hyphen/apostrophe parts.
NAME_RE = re.compile(
    r"""
    \b
    [A-Z\u00C0-\u00D6\u00D8-\u00DE][a-z\u00E0-\u00F6\u00F8-\u00FF]+
    (?:[-'][A-Z\u00C0-\u00D6\u00D8-\u00DE][a-z\u00E0-\u00F6\u00F8-\u00FF]+)?
    (?:\s+[A-Z\u00C0-\u00D6\u00D8-\u00DE][a-z\u00E0-\u00F6\u00F8-\u00FF]+(?:[-'][A-Z\u00C0-\u00D6\u00D8-\u00DE][a-z\u00E0-\u00F6\u00F8-\u00FF]+)?){1,2}
    \b
    """,
    re.VERBOSE,
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
STUDENT_ID_RE = re.compile(r"\b[iI](?:[\s-]?\d){7,8}\b")


COMMON_NON_NAME_TOKENS = {
    'group', 'members', 'member', 'meeting', 'minutes', 'agenda', 'project', 'plan',
    'maastricht', 'university', 'internal', 'reviewer', 'reviewers', 'author', 'authors',
    'completed', 'updated', 'finalised', 'prepare', 'implement', 'write', 'plan',
    'reviewed', 'mention', 'design', 'travel', 'security', 'task', 'tasks', 'customer',
    'owner', 'chair', 'secretary', 'participants', 'attendees', 'dream', 'team',
    'dissemination', 'level', 'delivery', 'date', 'matrix', 'collaborative', 'convergence',
    'reinvent', 'frictionless', 'architectures', 'incubate', 'granular', 'info'
}

@dataclass(slots=True)
class Candidate:
    path: str
    region: str
    start: int
    end: int
    surface: str
    features: dict


def looks(surface):
    tokens = [token for token in re.split(r"\s+", surface.strip()) if token]
    norm_tokens = _norm(surface).split()
    if len(norm_tokens) < 2 or len(norm_tokens) > 3:
        return False
    if any(len(token) < 2 for token in norm_tokens):
        return False
    if any(token.lower() in COMMON_NON_NAME_TOKENS for token in tokens):
        return False
    if any(token.isupper() for token in tokens):
        return False
    return all(token[:1].isupper() for token in tokens)


def feats(block, start, end, surface):
    text = block.text
    norm = _norm(surface)
    tokens = norm.split()
    prefix = text[max(0, start - 8) : start].rstrip()
    suffix = text[end : min(len(text), end + 32)].lstrip()
    left = text[max(0, start - 48) : start]
    right = text[end : min(len(text), end + 48)]
    title = 0.0
    orig_tokens = [token for token in re.split(r"\s+", surface.strip()) if token]
    if orig_tokens:
        title = sum(1 for token in orig_tokens if token[:1].isupper()) / len(orig_tokens)
    alpha = sum(char.isalpha() for char in surface) / max(1, len(surface))
    return {
        "n_tokens": len(tokens),
        "char_len": len(surface),
        "avg_token_len": sum(len(token) for token in tokens) / max(1, len(tokens)),
        "alpha_ratio": alpha,
        "title_ratio": title,
        "start_fraction": start / max(1, len(text)),
        "end_fraction": end / max(1, len(text)),
        "is_cover": int(block.region == "cover"),
        "is_header": int(block.region == "header"),
        "is_footer": int(block.region == "footer"),
        "is_body": int(block.region == "body"),
        "is_body_table": int(block.region == "body_table"),
        "is_ocr": int(block.region == "ocr"),
        "is_strict_region": int(is_strict_region(block.region)),
        "block_equals_surface": int(_norm(text) == norm),
        "starts_at_zero": int(start == 0),
        "preceded_by_colon": int(prefix.endswith(":")),
        "followed_by_colon": int(suffix.startswith(":")),
        "line_starts_with_bullet": int(text[max(0, start - 3):start].lstrip().startswith('-')),
        "block_contains_email": int(EMAIL_RE.search(text) is not None),
        "block_contains_student_id": int(STUDENT_ID_RE.search(text) is not None),
        "contains_hyphen": int("-" in surface),
        "contains_apostrophe": int("'" in surface),
        "same_line_pipe_delimited": int("|" in text),
        "surface_frequency_in_block": text.count(surface),
        "left_has_digits": int(re.search(r"\d", left) is not None),
        "right_has_digits": int(re.search(r"\d", right) is not None),
        "surface_is_all_caps": int(surface.isupper()),
    }


def candidates(blocks):
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        for match in NAME_RE.finditer(text):
            surface = match.group(0)
            if not looks(surface):
                continue
            start = match.start()
            end = match.end()
            yield Candidate(
                path=block.path,
                region=block.region,
                start=start,
                end=end,
                surface=surface,
                features=feats(block, start, end, surface),
            )


def frame(blocks, gold=None, *, doc=None):
    records = []
    for candidate in candidates(blocks):
        record = {
            "doc_id": doc or "",
            "path": candidate.path,
            "region": candidate.region,
            "start": candidate.start,
            "end": candidate.end,
            "surface": candidate.surface,
            **candidate.features,
        }
        if gold is not None:
            record["label"] = gold.get(
                (candidate.path, candidate.start, candidate.end, candidate.surface), 0
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def train(
    data,
    *,
    valid=None,
    output=None,
    seed=42,
):
    if data.empty:
        raise ValueError("Training frame is empty.")
    if "label" not in data.columns:
        raise ValueError("Training frame must contain a label column.")
    if data["label"].nunique() < 2:
        raise ValueError(
            "Training frame must contain both positive and negative examples."
        )
    features = [
        column
        for column in data.columns
        if column
        not in {"label", "doc_id", "path", "region", "surface", "start", "end"}
    ]
    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(
        data[features].to_dict(orient="records"),
        data["label"].astype(int).tolist(),
    )

    results = {}
    for split, part in [("train", data), ("validation", valid)]:
        if part is None or part.empty:
            continue
        probabilities = pipeline.predict_proba(part[features].to_dict(orient="records"))[:, 1]
        predicted = (probabilities >= 0.5).astype(int)
        results[split] = {
            "n_examples": len(part),
            "positive_rate": part["label"].mean(),
            "accuracy": accuracy_score(part["label"].astype(int).tolist(), predicted),
            "precision": precision_score(part["label"].astype(int).tolist(), predicted, zero_division=0),
            "recall": recall_score(part["label"].astype(int).tolist(), predicted, zero_division=0),
            "f1": f1_score(part["label"].astype(int).tolist(), predicted, zero_division=0),
            "average_precision": average_precision_score(part["label"].astype(int).tolist(), probabilities),
        }

    model = {
        "pipeline": pipeline,
        "feature_columns": features,
        "metrics": results,
        "model_type": "logistic_regression_name_candidate_classifier",
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output)
        model["model_output_path"] = str(Path(output))
    return model


@lru_cache(maxsize=8)
def load(path):
    return joblib.load(path)


def predict(blocks,*,path,source="ner_candidate_model"):
    model = load(path)
    pipeline = model["pipeline"]
    features = model["feature_columns"]
    detections = []

    for candidate in candidates(blocks):
        score = pipeline.predict_proba(
            [{column: candidate.features.get(column, 0.0) for column in features}]
        )[0, 1]
        threshold = (
            shared.name_strict_threshold
            if is_strict_region(candidate.region)
            else shared.name_body_threshold
        )
        if score < threshold:
            continue
        detections.append(
            Occurrence(
                label="Name",
                surface=candidate.surface,
                start=candidate.start,
                end=candidate.end,
                region=candidate.region,
                path=candidate.path,
                score=score,
                source=source,
            )
        )
    return detections