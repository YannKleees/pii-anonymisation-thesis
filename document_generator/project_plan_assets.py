from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from docx.document import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.shared import Inches
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from faker import Faker

from .controller import DocPayload, ProjectTask, choose_project_density, iso_to_eu_date
from .controller_common import person_name, student_number
from .marker import replace_paragraph_text, write_text
from .pii_tracker import PIIPart, PIITracker
from .seed import make_rng

_MILESTONE_HEADER = ("no.", "milestone name", "involved tasks", "date", "description")
_DELIVERABLE_HEADER = (
    "no.",
    "name",
    "lead",
    "type",
    "dissemination level",
    "delivery date",
)
_RISK_HEADER = ("risk", "prob", "impact", "mitigation plan")


@dataclass(frozen=True)
class ImageEntity:
    label: str
    surface: str
    entity_type: str
    entity_id: str
    quality: str = "clean"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_type: str
    image_path: str
    region: str
    placement_hint: str
    text: str
    text_lines: tuple[str, ...]
    entities: tuple[ImageEntity, ...]


@dataclass(frozen=True)
class TableBundle:
    density: str
    visible_tasks: tuple[ProjectTask, ...]
    overflow_tasks: tuple[ProjectTask, ...]


def enrich_project_plan_document(
    *,
    document: Document,
    payload: DocPayload,
    tracker: PIITracker,
    out_root: Path,
    doc_id: int,
    seed: int,
):
    """Fill project-plan tables and attach one image for OCR evaluation later."""
    if payload.template_id != "project_plan":
        return []

    density = choose_project_density(seed, doc_id)
    bundle = _split_tasks(payload.tasks, density)

    _populate_milestones_table(document, bundle.visible_tasks, tracker)
    _populate_deliverables_table(document, bundle.visible_tasks, tracker, seed, doc_id)
    _populate_risk_table(document, seed, doc_id, density)
    _populate_blank_single_column_tables(document, seed, doc_id, density)
    _populate_work_plan_table(document, payload.tasks, tracker, density)

    image_record = _generate_and_attach_gantt(
        document=document,
        tasks=bundle.visible_tasks,
        out_root=out_root,
        doc_id=doc_id,
        seed=seed,
        density=density,
    )
    return [asdict(image_record)] if image_record is not None else []


def _split_tasks(tasks: Sequence[ProjectTask], density: str):
    visible_count = {"light": 4, "medium": 6, "dense": 8}[density]
    visible = min(visible_count, len(tasks))
    return TableBundle(density=density, visible_tasks=tuple(tasks[:visible]), overflow_tasks=tuple(tasks[visible:]),)


def _cell_text(cell: _Cell):
    return "\n".join(paragraph.text for paragraph in cell.paragraphs).strip()

def _header_signature(table: Table):
    return tuple(_cell_text(cell).lower() for cell in table.rows[0].cells)

def _row_is_empty(table: Table, row_index: int, start_col: int = 0):
    return all( not _cell_text(cell).strip() for cell in table.rows[row_index].cells[start_col:])

def _tables_by_signature(document: Document, signature: tuple[str, ...]):
    return [table for table in document.tables if _header_signature(table) == signature]

def _find_blank_table(document: Document, signature: tuple[str, ...], *, data_start_row: int):
    matches = _tables_by_signature(document, signature)
    if not matches:
        return None
    for table in reversed(matches):
        if len(table.rows) <= data_start_row:
            return table
        if all(
            _row_is_empty(table, row_index, start_col=0)
            for row_index in range(data_start_row, len(table.rows))
        ):
            return table
    return matches[-1]

def _clear_example_tables(document: Document, signature: tuple[str, ...], *, data_start_row: int, keep: Table):
    for table in _tables_by_signature(document, signature):
        if table is not keep and len(table.rows) > data_start_row:
            _clear_rows(table, data_start_row)


def _ensure_row_count(table: Table, total_rows: int):
    while len(table.rows) < total_rows:
        table.add_row()


def _clear_rows(table: Table, start_row: int):
    for row_index in range(start_row, len(table.rows)):
        for cell in table.rows[row_index].cells:
            _set_cell_text(cell, "")


def _set_cell_text(
    cell: _Cell,
    text: str,
    tracker: PIITracker | None = None,
    pii_parts: Sequence[PIIPart] | None = None,
):
    for paragraph in cell.paragraphs[1:]:
        replace_paragraph_text(paragraph, "")
    paragraph = cell.paragraphs[0]
    if tracker is None:
        replace_paragraph_text(paragraph, text)
        return
    write_text(paragraph, text, tracker, pii_parts or ())


def _short_text(text: str, *, max_words: int):
    first_sentence = text.split(".", 1)[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return first_sentence
    return " ".join(words[:max_words]).rstrip(",;:") + "…"


def _title_like(faker: Faker, rng, *, max_words: int = 3):
    return " ".join(word.capitalize() for word in faker.words(nb=rng.randint(2, max_words), unique=True))

def _populate_milestones_table(
    document: Document, tasks: Sequence[ProjectTask], tracker: PIITracker
):
    table = _find_blank_table(document, _MILESTONE_HEADER, data_start_row=1)
    if table is None or not tasks:
        return
    _clear_example_tables(document, _MILESTONE_HEADER, data_start_row=1, keep=table)
    _ensure_row_count(table, 1 + len(tasks))
    _clear_rows(table, 1)

    for row_index, task in enumerate(tasks, start=1):
        values = (
            f"MS{row_index}",
            _short_text(task.title, max_words=5),
            task.task_id,
            iso_to_eu_date(task.end_date),
            _short_text(task.description, max_words=12),
        )
        for col_index, value in enumerate(values):
            _set_cell_text(table.rows[row_index].cells[col_index], value, tracker)


def _populate_deliverables_table(
    document: Document,
    tasks: Sequence[ProjectTask],
    tracker: PIITracker,
    seed: int,
    doc_id: int,
):
    table = _find_blank_table(document, _DELIVERABLE_HEADER, data_start_row=2)
    if table is None or not tasks:
        return
    _clear_example_tables(document, _DELIVERABLE_HEADER, data_start_row=2, keep=table)
    rng = make_rng(seed, "project_plan", doc_id, "deliverables")
    type_cycle = ["RE", "DEM", "OTH", "DEC"]
    _ensure_row_count(table, 2 + len(tasks))
    _clear_rows(table, 2)
    for row_offset, task in enumerate(tasks):
        row_index = 2 + row_offset
        lead_pii = (
            (
                "FULL_NAME",
                person_name(task.lead),
                "student",
                student_number(task.lead),
                "clean",
            ),
        )
        tag = rng.choice(["Report", "Package", "Prototype", "Note"])
        values = [
            f"D{row_offset + 1}",
            f"{tag}: {_short_text(task.title, max_words=6)}",
            person_name(task.lead),
            type_cycle[row_offset % len(type_cycle)],
            rng.choice(["PU", "CO", "PU"]),
            iso_to_eu_date(task.end_date),
        ]
        for col_index, value in enumerate(values):
            _set_cell_text(
                table.rows[row_index].cells[col_index],
                value,
                tracker,
                lead_pii if col_index == 2 else (),
            )


def _populate_risk_table(document: Document, seed: int, doc_id: int, density: str):
    table = _find_blank_table(document, _RISK_HEADER, data_start_row=2)
    if table is None:
        return
    _clear_example_tables(document, _RISK_HEADER, data_start_row=2, keep=table)

    rng = make_rng(seed, "project_plan", doc_id, "risk_table")
    faker = Faker()
    faker.random = rng
    risk_count = 4 if density == "light" else rng.randint(5, 7)

    def is_merged_row(row_index: int) -> bool:
        return len({id(cell._tc) for cell in table.rows[row_index].cells}) == 1

    while len(table.rows) < 2 + risk_count:
        table.add_row()

    _clear_rows(table, 2)
    merged_rows = [
        row_index for row_index in range(2, len(table.rows)) if is_merged_row(row_index)
    ]
    normal_rows = [
        row_index
        for row_index in range(2, len(table.rows))
        if not is_merged_row(row_index)
    ]
    while len(normal_rows) < risk_count:
        table.add_row()
        normal_rows = [
            row_index
            for row_index in range(2, len(table.rows))
            if not is_merged_row(row_index)
        ]

    for heading, row_index in zip(
        [f"{_title_like(faker, rng)} risks" for _ in range(max(1, len(merged_rows)))],
        merged_rows,
    ):
        _set_cell_text(table.rows[row_index].cells[0], heading)

    for row_index in normal_rows[:risk_count]:
        values = (
            faker.sentence(nb_words=rng.randint(5, 9)).rstrip("."),
            rng.choice(["L", "M", "H"]),
            rng.choice(["L", "M", "H"]),
            faker.sentence(nb_words=rng.randint(9, 16)).rstrip("."),
        )
        for col_index, value in enumerate(values):
            _set_cell_text(table.rows[row_index].cells[col_index], value)


def _blank_single_column_tables(document: Document):
    tables: list[Table] = []
    for table in document.tables:
        if len(table.columns) != 1 or _is_work_plan_table(table):
            continue
        if all(
            not _cell_text(cell).strip() for row in table.rows for cell in row.cells
        ):
            tables.append(table)
    return tables


def _populate_blank_single_column_tables(
    document: Document, seed: int, doc_id: int, density: str
):
    tables = _blank_single_column_tables(document)
    if not tables:
        return

    rng = make_rng(seed, "project_plan", doc_id, "single_column_tables")
    faker = Faker()
    faker.random = rng

    table_limit = 2 if density == "light" else min(3, len(tables))
    for table in tables[:table_limit]:
        _clear_rows(table, 0)
        row_limit = min(len(table.rows), 3 if density == "light" else 4)
        for row_index in range(row_limit):
            text = (
                faker.paragraph(nb_sentences=rng.randint(2, 4))
                .replace("\n", " ")
                .strip()
            )
            _set_cell_text(table.rows[row_index].cells[0], text)


def _is_work_plan_table(table: Table):
    if len(table.columns) != 1 or len(table.rows) < 4:
        return False
    head_a = _cell_text(table.rows[0].cells[0]).strip().lower()
    head_b = (
        _cell_text(table.rows[2].cells[0]).strip().lower()
        if len(table.rows) > 2
        else ""
    )
    return head_a.startswith("task") and head_b.startswith("task")


def _find_blank_work_plan_table(document: Document):
    candidates = [table for table in document.tables if _is_work_plan_table(table)]
    if not candidates:
        return None
    for table in reversed(candidates):
        if all(
            not _cell_text(table.rows[row_index].cells[0]).strip()
            for row_index in range(1, len(table.rows), 2)
        ):
            return table
    return candidates[-1]


def _populate_work_plan_table(
    document: Document,
    tasks: Sequence[ProjectTask],
    tracker: PIITracker,
    density: str,
):
    table = _find_blank_work_plan_table(document)
    if table is None or not tasks:
        return

    package_map: dict[str, list[ProjectTask]] = {}
    for task in tasks:
        package_map.setdefault(task.work_package, []).append(task)

    groups = list(package_map.items())[: {"light": 2, "medium": 3, "dense": 4}[density]]
    _ensure_row_count(table, max(4, len(groups) * 2))
    _clear_rows(table, 0)

    row_cursor = 0
    task_limit = 2 if density == "light" else 3
    for group_index, (package_name, package_tasks) in enumerate(groups, start=1):
        _set_cell_text(
            table.rows[row_cursor].cells[0], f"Task {group_index} {package_name}"
        )
        row_cursor += 1

        detail_lines: list[str] = []
        pii_parts: list[PIIPart] = []
        for task in package_tasks[:task_limit]:
            participants = ", ".join(
                person_name(person) for person in task.participants
            )
            detail_lines.append(
                f"{task.task_id}: {_short_text(task.title, max_words=8)}, lead: {person_name(task.lead)} participants: {participants} ({task.dates}) {_short_text(task.description, max_words=18)}"
            )
            pii_parts.append(
                (
                    "FULL_NAME",
                    person_name(task.lead),
                    "student",
                    student_number(task.lead),
                    "clean",
                )
            )
            for person in task.participants:
                pii_parts.append(
                    (
                        "FULL_NAME",
                        person_name(person),
                        "student",
                        student_number(person),
                        "clean",
                    )
                )
        _set_cell_text(
            table.rows[row_cursor].cells[0],
            "\n\n".join(detail_lines),
            tracker,
            pii_parts,
        )
        row_cursor += 1


def _generate_and_attach_gantt(
    *,
    document: Document,
    tasks: Sequence[ProjectTask],
    out_root: Path,
    doc_id: int,
    seed: int,
    density: str):
    if not tasks:
        return None

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"doc_{doc_id:05d}_gantt.png"
    title = "Project 3-1 — Gantt chart"
    subtitle = f"Synthetic {density} planning view"
    text_lines = [title, subtitle, "Task ID | Lead | Start | End | Title"]
    entities: list[ImageEntity] = []
    seen_entities: set[tuple[str, str]] = set()

    _draw_gantt_chart(image_path, tasks, title=title, subtitle=subtitle, seed=seed, doc_id=doc_id)
    _replace_gantt_paragraph(document, image_path)

    for task in tasks:
        text_lines.append(
            f"{task.task_id} | {person_name(task.lead)} | {task.start_date} | {task.end_date} | {_short_text(task.title, max_words=8)}"
        )
        key = (person_name(task.lead), student_number(task.lead))
        if key not in seen_entities:
            entities.append(
                ImageEntity(
                    label="FULL_NAME",
                    surface=person_name(task.lead),
                    entity_type="student",
                    entity_id=student_number(task.lead),
                )
            )
            seen_entities.add(key)

    return ImageRecord(
        image_id=f"doc_{doc_id:05d}_gantt_01",
        image_type="gantt_chart",
        image_path=str(image_path.relative_to(out_root)),
        region="body_image",
        placement_hint="implementation:gantt_chart",
        text="\n".join(text_lines),
        text_lines=tuple(text_lines),
        entities=tuple(entities),
    )


def _draw_gantt_chart(
    path: Path,
    tasks: Sequence[ProjectTask],
    *,
    title: str,
    subtitle: str,
    seed: int,
    doc_id: int,
):
    rng = make_rng(seed, "project_plan", doc_id, "gantt_chart")
    labels = [f"{task.task_id} — {person_name(task.lead)}" for task in tasks]

    fig_height = max(3.8, 1.2 + 0.72 * len(tasks))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    for index, task in enumerate(tasks):
        start_num = mdates.date2num(date.fromisoformat(task.start_date))
        end_num = mdates.date2num(date.fromisoformat(task.end_date))
        duration = max(1.0, end_num - start_num)
        ax.barh(index, duration, left=start_num, height=0.55, edgecolor="black")
        if rng.random() < 0.8:
            ax.text(
                start_num + duration / 2.0,
                index,
                _short_text(task.title, max_words=4),
                ha="center",
                va="center",
                fontsize=8,
            )

    if len(tasks) >= 3 and rng.random() < 0.7:
        for milestone_task in tasks[: min(3, len(tasks))]:
            ax.axvline(
                x=mdates.date2num(date.fromisoformat(milestone_task.end_date)),
                linewidth=0.9,
            )

    ax.set_title(f"{title}\n{subtitle}")
    ax.set_yticks(list(range(len(tasks))))
    ax.set_yticklabels(labels)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.autofmt_xdate()
    ax.invert_yaxis()
    ax.set_xlabel("Planned dates")
    ax.set_ylabel("Task owner")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _insert_paragraph_after(paragraph: Paragraph, text: str = ""):
    new_element = OxmlElement("w:p")
    paragraph._p.addnext(new_element)
    new_paragraph = Paragraph(new_element, paragraph._parent)
    if text:
        new_paragraph.add_run(text)
    return new_paragraph


def _replace_gantt_paragraph(document: Document, image_path: Path):
    target: Paragraph | None = None
    for paragraph in document.paragraphs:
        if any("graphicData" in run._element.xml for run in paragraph.runs):
            target = paragraph
            break
    if target is None:
        for paragraph in document.paragraphs:
            if paragraph.text.strip().lower() == "3.1 gantt chart":
                target = _insert_paragraph_after(paragraph)
                break
    if target is None:
        return False

    for run in list(target.runs):
        run._element.getparent().remove(run._element)
    target.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    target.add_run().add_picture(str(image_path), width=Inches(6.8))
    return True
