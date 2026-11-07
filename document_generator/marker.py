import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from faker import Faker

from .controller import DocPayload, FillValue
from .pii_tracker import EmittedPII, PIIPart, PIITracker
from .seed import make_rng, weighted_choice

NEWLINE = "\n"
_MARKER_RE = re.compile(r"<(?P<slot>[A-Za-z][A-Za-z0-9_]*)(?P<optional>\?)?>")


def normalize_text(text: str) -> str:
    """Normalize text for deterministic offsets."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")


# DOCX walking


def _iter_container_children(container) -> Iterable[tuple[str, object]]:
    for child in container._element.iterchildren():
        if child.tag == qn("w:p"):
            yield "p", child
        elif child.tag == qn("w:tbl"):
            yield "tbl", child


def _iter_table_paragraphs(
    table: Table, base_path: str, region: str
) -> Iterable[tuple[str, str, Paragraph]]:
    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            for para_index, paragraph in enumerate(cell.paragraphs):
                yield region, f"{base_path}:r{row_index}:c{col_index}:p{para_index}", paragraph


class _CoverDetector:
    """Heuristic cover/body splitter for the provided templates."""

    def __init__(self):
        self._in_cover = True

    @property
    def in_cover(self) -> bool:
        return self._in_cover

    def region_for_paragraph(self, paragraph_text: str) -> str:
        text = (paragraph_text or "").strip().lower()
        if text.startswith("contents") or text.startswith("big picture on progress"):
            self._in_cover = False
        return "cover" if self._in_cover else "body"


def iter_body_paragraphs(doc: Document) -> Iterable[tuple[str, str, Paragraph]]:
    """Yield (region, path, paragraph) for body + tables in XML order."""
    cover = _CoverDetector()
    paragraph_counter = 0

    for kind, child in _iter_container_children(doc._body):
        if kind == "p":
            paragraph = Paragraph(child, doc)
            region = cover.region_for_paragraph(paragraph.text)
            yield region, f"{region}:p{paragraph_counter}", paragraph
            paragraph_counter += 1
            continue

        table = Table(child, doc)
        region = "cover_table" if cover.in_cover else "body_table"
        yield from _iter_table_paragraphs(
            table, f"{region}:tbl{paragraph_counter}", region
        )


def iter_headers_footers(doc: Document) -> Iterable[tuple[str, str, Paragraph]]:
    """Yield (region, path, paragraph) for headers/footers and their tables."""
    for section_index, section in enumerate(doc.sections):
        parts = [
            ("header", section.header),
            ("footer", section.footer),
            ("first_page_header", section.first_page_header),
            ("first_page_footer", section.first_page_footer),
            ("even_page_header", section.even_page_header),
            ("even_page_footer", section.even_page_footer),
        ]
        for which, part in parts:
            paragraph_counter = 0
            for kind, child in _iter_container_children(part):
                if kind == "p":
                    yield which, f"sec{section_index}:{which}:p{paragraph_counter}", Paragraph(
                        child, doc
                    )
                    paragraph_counter += 1
                    continue
                table = Table(child, doc)
                yield from _iter_table_paragraphs(
                    table,
                    f"sec{section_index}:{which}:tbl{paragraph_counter}",
                    which + "_table",
                )


def iter_all_paragraphs(doc: Document) -> Iterable[tuple[str, str, Paragraph]]:
    yield from iter_headers_footers(doc)
    yield from iter_body_paragraphs(doc)


# DOCX modification


def replace_paragraph_text(
    paragraph: Paragraph, new_text: str, *, preserve_non_text: bool = False
):
    """Replace paragraph text while preserving paragraph-level formatting.

    When preserve_non_text is True, non-text run content (for example inline
    drawings/images) is kept intact by only rewriting textual run content.
    """
    runs = list(paragraph.runs)

    if not preserve_non_text:
        for run in runs:
            run._element.getparent().remove(run._element)
        paragraph.add_run(new_text)
        return

    if not runs:
        paragraph.add_run(new_text)
        return

    first_text_run_index = next(
        (index for index, run in enumerate(runs) if run.text), None
    )
    if first_text_run_index is None:
        paragraph.add_run(new_text)
        return

    for index, run in enumerate(runs):
        if index == first_text_run_index:
            run.text = new_text
        elif run.text:
            run.text = ""


def write_text(
    paragraph: Paragraph,
    text: str,
    tracker: PIITracker,
    pii_parts: Sequence[PIIPart] | None = None,
    *,
    region: str = "",
    path: str = "",
):
    """Write text into a paragraph and track any PII that is actually present."""
    if pii_parts:
        tracker.track(text, pii_parts, region=region, path=path)
    replace_paragraph_text(paragraph, text)


# Marker-based filling


def find_markers(text: str):
    """Return (slot_id, is_optional) markers found in text."""
    return [
        (match.group("slot").upper(), match.group("optional") is not None)
        for match in _MARKER_RE.finditer(text or "")
    ]


def count_markers_in_doc(doc: Document):
    """Count marker occurrences across headers, footers, body, and tables."""
    counts: dict[str, int] = {}
    for _region, _path, paragraph in iter_all_paragraphs(doc):
        for slot_id, _is_optional in find_markers(paragraph.text):
            counts[slot_id] = counts.get(slot_id, 0) + 1
    return counts


class MarkerResolver:
    """Resolve slot ids to FillValue, consuming sequential values per slot."""

    def __init__(self, slot_values: dict[str, Sequence[FillValue]]):
        self._values = {
            slot_id.upper(): list(values) for slot_id, values in slot_values.items()
        }
        self._next_index: dict[str, int] = {}

    def next_value(self, slot_id: str):
        key = slot_id.upper()
        if key not in self._values:
            raise KeyError(f"No fill value provided for marker <{slot_id}>.")

        values = self._values[key]
        if not values:
            return FillValue(text="")

        index = self._next_index.get(key, 0)
        self._next_index[key] = index + 1
        return values[min(index, len(values) - 1)]


def _replace_markers(text: str, resolver: MarkerResolver):
    pii_parts: list[PIIPart] = []

    def substitute(match: re.Match) -> str:
        fill_value = resolver.next_value(match.group("slot"))
        if match.group("optional") and not fill_value.text:
            return ""
        pii_parts.extend(fill_value.pii_parts)
        return fill_value.text

    return _MARKER_RE.sub(substitute, text), pii_parts


# Header/footer/body variation


def _clear_story_part(part):
    for paragraph in part.paragraphs:
        replace_paragraph_text(paragraph, "")


def _non_marker_paragraphs(paragraphs: Sequence[Paragraph]):
    return [
        paragraph
        for paragraph in paragraphs
        if not _MARKER_RE.search(paragraph.text or "")
    ]


def vary_headers_footers(doc: Document, seed: int, doc_id: int):
    """Apply deterministic but realistic header/footer variation."""
    rng = make_rng(seed, "doc", doc_id, "header_footer_variation")
    mode = weighted_choice(rng, ["clear", "vary", "keep"], [0.25, 0.30, 0.45])

    if mode == "clear":
        for section in doc.sections:
            for part in [
                section.header,
                section.footer,
                section.first_page_header,
                section.first_page_footer,
                section.even_page_header,
                section.even_page_footer,
            ]:
                _clear_story_part(part)
        return

    if mode != "vary":
        return

    faker = Faker()
    faker.random = rng
    for section in doc.sections:
        section.different_first_page_header_footer = rng.random() < 0.75
        non_marker_first_headers = _non_marker_paragraphs(
            section.first_page_header.paragraphs
        )
        non_marker_first_footers = _non_marker_paragraphs(
            section.first_page_footer.paragraphs
        )
        non_marker_even_headers = _non_marker_paragraphs(
            section.even_page_header.paragraphs
        )
        non_marker_even_footers = _non_marker_paragraphs(
            section.even_page_footer.paragraphs
        )

        for paragraph in non_marker_first_headers[:1]:
            replace_paragraph_text(
                paragraph, f"First page — {faker.catch_phrase().rstrip('.')}"
            )
        for paragraph in non_marker_first_footers[:1]:
            replace_paragraph_text(paragraph, f"Page 1 | {faker.company()}")

        if rng.random() < 0.55:
            for paragraph in non_marker_even_headers[:1]:
                replace_paragraph_text(
                    paragraph, f"Appendix view — {faker.bs().replace('-', ' ')}"
                )
        if rng.random() < 0.55:
            for paragraph in non_marker_even_footers[:1]:
                replace_paragraph_text(
                    paragraph, faker.sentence(nb_words=4).rstrip(".")
                )


def inject_non_pii_paragraphs(doc: Document, seed: int, doc_id: int):
    """Inject a small amount of generic narrative text into body paragraphs only."""
    rng = make_rng(seed, "doc", doc_id, "non_pii_injection")
    if rng.random() >= 0.50:
        return

    body_candidates = [
        paragraph
        for region, _path, paragraph in iter_body_paragraphs(doc)
        if region == "body"
    ]
    if len(body_candidates) < 2:
        return

    faker = Faker()
    faker.random = rng
    insertions = rng.randint(1, min(2, len(body_candidates) - 1))
    chosen_indexes = sorted(
        rng.sample(range(1, len(body_candidates)), k=insertions), reverse=True
    )
    for index in chosen_indexes:
        filler_text = (
            faker.paragraph(nb_sentences=rng.randint(3, 6)).replace("\n", " ").strip()
        )
        body_candidates[index].insert_paragraph_before(filler_text)


def fill_template(doc: Document, payload: DocPayload, tracker: PIITracker):
    """Fill a template DOCX in-place by replacing <marker_name> markers."""
    resolver = MarkerResolver(payload.slot_values)
    for region, path, paragraph in iter_all_paragraphs(doc):
        if not _MARKER_RE.search(paragraph.text or ""):
            continue
        new_text, pii_parts = _replace_markers(paragraph.text, resolver)
        write_text(
            paragraph, new_text, tracker, pii_parts or None, region=region, path=path
        )


# Serialization + annotation


@dataclass(frozen=True)
class TextBlock:
    region: str
    path: str
    start: int
    end: int
    text: str


def serialize_docx(doc_path: Path):
    """Return canonical full_text and paragraph blocks with offsets."""
    doc = Document(str(doc_path))
    parts: list[str] = []
    blocks: list[TextBlock] = []
    cursor = 0

    def add_block(region: str, path: str, text: str):
        nonlocal cursor
        normalized = normalize_text(text)
        if parts:
            parts.append(NEWLINE)
            cursor += 1
        start = cursor
        parts.append(normalized)
        cursor += len(normalized)
        blocks.append(TextBlock(region=region, path=path, start=start, end=cursor, text=normalized))
    for region, path, paragraph in iter_all_paragraphs(doc):
        add_block(region, path, paragraph.text or "")
    return "".join(parts), blocks


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    index = 0
    while True:
        index = haystack.find(needle, index)
        if index == -1:
            return starts
        starts.append(index)
        index += max(1, len(needle))


def _candidate_blocks(item: EmittedPII, blocks: Sequence[TextBlock], block_by_path: dict[str, TextBlock]):
    normalized_container = (normalize_text(item.container_text) if item.container_text else "")

    if item.path and item.path in block_by_path:
        path_block = block_by_path[item.path]
        if item.surface in path_block.text and (
            not normalized_container
            or path_block.text == normalized_container
            or normalized_container in path_block.text
        ):
            return [path_block]

    candidates = list(blocks)
    if item.region:
        regional = [block for block in candidates if block.region == item.region]
        if regional:
            candidates = regional

    if normalized_container:
        exact = [block for block in candidates if block.text == normalized_container]
        if exact:
            return exact
        contained = [
            block
            for block in candidates
            if normalized_container in block.text or block.text in normalized_container
        ]
        if contained:
            candidates = contained

    matching_surface = [block for block in candidates if item.surface in block.text]
    return matching_surface or list(candidates)


def build_annotations(*, doc_id: int, template_id: str, seed: int, doc_path: Path, emitted_pii: Sequence[EmittedPII]):
    """Build strict-offset annotations for one generated DOCX."""
    full_text, blocks = serialize_docx(doc_path)
    block_by_path = {block.path: block for block in blocks}

    entity_occurrences: dict[tuple[str, str, str, str, str], dict[tuple[int, int, str], dict]] = {}
    missing: set[str] = set()

    for item in emitted_pii:
        key = (item.label, item.surface, item.entity_type, item.entity_id, item.quality)
        bucket = entity_occurrences.setdefault(key, {})
        matched = False

        for block in _candidate_blocks(item, blocks, block_by_path):
            for local_start in _find_all(block.text, item.surface):
                global_start = block.start + local_start
                global_end = global_start + len(item.surface)
                if full_text[global_start:global_end] != item.surface:
                    raise ValueError(
                        f"Span mismatch while building annotations for {item.label}:{item.surface}"
                    )
                bucket[(global_start, global_end, block.path)] = {
                    "start": global_start,
                    "end": global_end,
                    "region": block.region,
                    "path": block.path,
                }
                matched = True

        if not matched:
            missing.add(f"{item.label}:{item.surface}")

    entities: list[dict] = []
    for (label, surface, entity_type, entity_id, quality), occurrence_map in sorted(entity_occurrences.items()):
        if not occurrence_map:
            continue
        occurrences = sorted(occurrence_map.values(), key=lambda occ: occ["start"])
        entities.append(
            {
                "label": label,
                "surface": surface,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "quality": quality,
                "occurrences": occurrences,
            }
        )

    return {
        "doc_id": doc_id,
        "template_id": template_id,
        "seed": seed,
        "text": full_text,
        "text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "blocks": [
            {
                "region": block.region,
                "path": block.path,
                "start": block.start,
                "end": block.end,
            } for block in blocks
        ],
        "entities": entities,
        "missing_expected": sorted(missing),
    }
