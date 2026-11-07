"""OCR for images extracted from DOCX or rendered from PDF pages."""
from __future__ import annotations

import io
import re
from dataclasses import replace
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageOps

from ..shared import Block, Config, DocumentRecord, Occurrence, ocr_conf_threshold

_READER = None

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*")


def _prep(blob: bytes):
    """Normalize a binary image before OCR and return the resize scale."""
    image = Image.open(io.BytesIO(blob)).convert("RGB")
    scale = 1.0
    if max(image.size) < 2400:
        image = image.resize((image.width * 2, image.height * 2))
        scale = 2.0
    return ImageOps.autocontrast(ImageOps.grayscale(image)), scale


def _reader():
    """Lazy-load EasyOCR so text-only workflows do not pay the import cost."""
    global _READER
    if _READER is None:
        try:
            import easyocr
        except Exception as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("OCR requires EasyOCR (`pip install easyocr`).") from exc
        # Use CPU by default for portability in notebook and Windows environments.
        _READER = easyocr.Reader(["en"], gpu=False)
    return _READER


def _word_from_easyocr_item(item):
    if len(item) != 3:
        return None
    bbox, text, conf = item
    token = str(text).strip()
    if not token:
        return None
    try:
        score = float(conf) * 100.0
    except Exception:
        score = -1.0
    if score < ocr_conf_threshold:
        return None
    xs = [int(point[0]) for point in bbox]
    ys = [int(point[1]) for point in bbox]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if right <= left or bottom <= top:
        return None
    return {
        "text": token,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
        "center_y": (top + bottom) / 2.0,
        "conf": score,
    }


def _same_line(word, line):
    """Return True when an OCR word belongs to an existing visual text line."""
    tolerance = max(12.0, 0.70 * max(float(word["height"]), float(line["height"])))
    return abs(float(word["center_y"]) - float(line["center_y"])) <= tolerance


def _line_from_word(word):
    return {
        "words": [word],
        "top": word["top"],
        "bottom": word["bottom"],
        "height": word["height"],
        "center_y": word["center_y"],
    }


def _add_word_to_line(line, word):
    line["words"].append(word)
    line["top"] = min(line["top"], word["top"])
    line["bottom"] = max(line["bottom"], word["bottom"])
    line["height"] = max(1, line["bottom"] - line["top"])
    line["center_y"] = sum(w["center_y"] for w in line["words"]) / len(line["words"])


def _group_words_into_lines(words):
    """Group EasyOCR boxes into line-level blocks so detectors see useful context.

    The previous implementation emitted one OCR block per recognized word/chunk. That
    made full-name matching difficult because gold OCR labels usually contain complete
    names such as "Jane Smith", while EasyOCR often returns "Jane" and "Smith" as
    separate boxes. Line grouping keeps word-level boxes for masking but gives pattern
    and NER detectors a full line of text.
    """
    lines = []
    for word in sorted(words, key=lambda item: (item["center_y"], item["left"])):
        best = None
        for line in lines:
            if _same_line(word, line):
                best = line
                break
        if best is None:
            lines.append(_line_from_word(word))
        else:
            _add_word_to_line(best, word)

    for line in lines:
        line["words"].sort(key=lambda item: item["left"])
    lines.sort(key=lambda line: (line["top"], min(word["left"] for word in line["words"])))
    return lines


def _line_text_and_offsets(words):
    pieces = []
    offsets = []
    cursor = 0
    for word in words:
        token = str(word["text"]).strip()
        if not token:
            continue
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        offsets.append(
            {
                "text": token,
                "start": start,
                "end": cursor,
                "bbox": (word["left"], word["top"], word["right"], word["bottom"]),
            }
        )
    return "".join(pieces).strip(), offsets


def _blocks(document: DocumentRecord, config: Config):
    """Convert every image artifact into OCR text blocks."""
    blocks = []
    reader = _reader()
    for image in document.images:
        prepared, scale = _prep(image.blob)
        results = reader.readtext(np.array(prepared), detail=1, paragraph=False)
        words = []
        for item in results:
            word = _word_from_easyocr_item(item)
            if word is not None:
                words.append(word)

        for index, line in enumerate(_group_words_into_lines(words)):
            line_text, offsets = _line_text_and_offsets(line["words"])

            if not line_text:
                continue
            blocks.append(
                Block(
                    region="ocr",
                    path=f"image:{image.media_path}:line:{index}",
                    text=line_text,
                    raw_text=line_text,
                    metadata={
                        **dict(image.metadata or {}),
                        "media_path": image.media_path,
                        "image_id": image.image_id,
                        "image_region": image.region,
                        "word_offsets": offsets,
                        "ocr_confidence": sum(word["conf"] for word in line["words"]) / max(1, len(line["words"])),
                        "ocr_scale": scale,
                    },
                )
            )
    return blocks


def _box(block: Block, start: int, end: int):
    boxes = [
        item["bbox"]
        for item in block.metadata.get("word_offsets", [])
        if not (item["end"] <= start or item["start"] >= end)
    ]
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    scale = float(block.metadata.get("ocr_scale", 1.0) or 1.0)
    if scale != 1.0:
        left, top, right, bottom = (int(round(value / scale)) for value in (left, top, right, bottom))
    return (left, top, right, bottom)


def _attach(blocks, items):
    bypath = {block.path: block for block in blocks}
    output = []
    image_metadata_keys = {
        "document_type",
        "media_path",
        "image_id",
        "image_region",
        "ocr_confidence",
        "bbox",
        "page_index",
        "page_number",
        "image_size",
        "pdf_page_rect",
        "render_dpi",
    }
    for item in items:
        block = bypath.get(item.path)
        if block is None:
            output.append(item)
            continue
        metadata = dict(item.metadata or {})
        metadata.update({key: value for key, value in block.metadata.items() if key in image_metadata_keys})
        metadata["media_path"] = block.metadata.get("media_path")
        metadata["image_id"] = block.metadata.get("image_id")
        metadata["ocr_confidence"] = block.metadata.get("ocr_confidence")
        metadata["bbox"] = _box(block, item.start, item.end)
        output.append(
            Occurrence(
                label=item.label,
                surface=item.surface,
                start=item.start,
                end=item.end,
                region=item.region,
                path=item.path,
                score=item.score,
                source=item.source,
                entity_key=item.entity_key,
                metadata=metadata,
            )
        )
    return output


def _ocr_ner_config(config: Config):
    """Prefer OCR-local NER fallback over a text-trained candidate model.

    Candidate name models in this project are trained on DOCX text blocks. OCR lines
    have different spacing and layout features, so a text-trained candidate model can
    suppress the heuristic OCR-name fallback. Clearing only the candidate path keeps the
    rest of the detector configuration unchanged.
    """
    if not getattr(config, "candidate", None):
        return config
    try:
        return replace(config, candidate=None)
    except Exception:
        return config


def _compact(text: str):
    """OCR-tolerant comparison key for names."""
    value = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    # Common OCR confusions in small chart-axis labels.
    return value.translate(str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b"}))


def _name_tokens(text: str):
    return [match.group(0) for match in _WORD_RE.finditer(str(text))]


def _token_initials(tokens):
    return "".join(token[:1].lower() for token in tokens if token)


def _name_similarity(a: str, b: str):
    left = _compact(a)
    right = _compact(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _looks_like_same_name(candidate: str, known_name: str, threshold: float = 0.84):
    score = _name_similarity(candidate, known_name)
    if score < threshold:
        return False, score
    candidate_tokens = _name_tokens(candidate)
    known_tokens = _name_tokens(known_name)
    if len(candidate_tokens) < 1 or len(known_tokens) < 2:
        return False, score
    cand_initials = _token_initials(candidate_tokens)
    known_initials = _token_initials(known_tokens)
    # Let very close OCR matches through, but avoid matching unrelated title-case phrases.
    if score >= 0.92:
        return True, score
    if len(cand_initials) >= 2 and len(known_initials) >= 2:
        return cand_initials[0] == known_initials[0] and cand_initials[-1] == known_initials[-1], score
    return False, score


def _known_names_from_text(document: DocumentRecord, config: Config):
    """Collect document-local names already visible in the text layer.

    This is used only as a stabiliser for OCR over embedded charts. The project-plan
    chart repeats task-owner names that already occur in the surrounding DOCX/PDF text.
    EasyOCR may read those axis labels with small character errors; matching against
    document-local names lets us emit the canonical surface while keeping the OCR box for
    image masking. Image-only PDFs still use the normal OCR NER path when no text names
    are available.
    """
    names = set()
    try:
        import thesis_project.defender.name_model as name_model
    except Exception:
        return names
    for candidate in name_model.candidates(document.blocks):
        surface = str(candidate.surface).strip()
        if name_model.looks(surface):
            names.add(surface)
    return names


def _text_token_spans(text: str):
    return [(match.group(0), match.start(), match.end()) for match in _WORD_RE.finditer(str(text))]


def _best_known_name_match(block: Block, known_name: str):
    tokens = _text_token_spans(block.text)
    if not tokens:
        return None
    target_len = max(1, len(_name_tokens(known_name)))
    best = None
    # Try windows around the expected name length. This handles "Richard Miller",
    # "Rchard Miler", and one extra OCR token such as "T1 Richard Miller".
    for width in range(max(1, target_len - 1), min(len(tokens), target_len + 2) + 1):
        for index in range(0, len(tokens) - width + 1):
            window = tokens[index : index + width]
            text = " ".join(token for token, _start, _end in window)
            ok, score = _looks_like_same_name(text, known_name)
            if not ok:
                continue
            start = window[0][1]
            end = window[-1][2]
            rank = (score, -abs(width - target_len), -(end - start))
            if best is None or rank > best[0]:
                best = (rank, start, end, text, score)
    if best is None:
        return None
    _rank, start, end, text, score = best
    return start, end, text, score


def _known_name_detections(document: DocumentRecord, blocks: list[Block], config: Config):
    known_names = sorted(_known_names_from_text(document, config), key=lambda value: (-len(value), value))
    if not known_names:
        return []
    detections = []
    seen = set()
    for block in blocks:
        for known_name in known_names:
            match = _best_known_name_match(block, known_name)
            if match is None:
                continue
            start, end, observed, score = match
            key = (block.path, start, end, known_name)
            if key in seen:
                continue
            seen.add(key)
            detections.append(
                Occurrence(
                    label="Name",
                    surface=known_name,
                    start=start,
                    end=end,
                    region=block.region,
                    path=block.path,
                    score=max(0.86, min(0.97, 0.80 + 0.18 * score)),
                    source="ocr_known_name_fuzzy",
                    metadata={"ocr_observed_surface": observed, "ocr_name_similarity": round(score, 4)},
                )
            )
    return detections


def detect(document: DocumentRecord, config: Config):
    """Run pattern and NER detectors over OCR blocks from image artifacts."""
    blocks = _blocks(document, config)
    if not blocks:
        return []

    class _BlockSet:
        def __init__(self, block_list):
            self.blocks = block_list

    doc_wrapper = _BlockSet(blocks)
    predicted = []
    if getattr(config, "patterns", True):
        import thesis_project.defender.patterns as patterns

        predicted.extend(patterns.detect(doc_wrapper, config))
    if getattr(config, "ner", True):
        import thesis_project.defender.name_model  # noqa: F401
        import thesis_project.defender.ner as ner

        predicted.extend(ner.detect_blocks(blocks, _ocr_ner_config(config), source="ocr_ner"))
        predicted.extend(_known_name_detections(document, blocks, config))
    return _attach(blocks, predicted)
