"""Public controller entrypoint.

The controller stays close to the original architecture, but is split into
smaller modules:
- controller_common.py: models and low-level helpers
- controller_content.py: content generation helpers
- controller.py: slot fillers and public payload builder
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .controller_common import (
    Course,
    DocContext,
    DocPayload,
    DocumentStyle,
    Filler,
    FillRequest,
    FillValue,
    Person,
    ProjectTask,
    SlotValues,
    choose_project_density,
    coordinator_id,
    course_code,
    document_style,
    duration,
    faker_with_rng,
    group_programme,
    iso_to_eu_date,
    maybe_typo_email,
    maybe_typo_student_number,
    meeting_date,
    people_fill,
    person_name,
    repeat_fill,
    row_at,
    role_part,
    staff_name_part,
    student_email,
    student_email_part,
    student_id_part,
    student_name_part,
    student_number,
    take_or_pad,
    tutor_id,
)
from .controller_content import (
    bullets,
    faker_paragraph,
    faker_references,
    faker_title,
    group_member_lines,
    layout_values,
    meeting_attendees,
    project_objectives_text,
    project_proposal_title,
    project_section_text,
    project_section_text_by_args,
    project_table_text,
    required_task_markers,
    section_assignments,
    task_count,
    task_or_last,
    tasks,
)
from .seed import make_rng


def _fill_layout_header(req: FillRequest):
    """Fill layout header slot."""
    return layout_values(req.ctx, req.slot_id, req.count, "header")


def _fill_layout_footer(req: FillRequest):
    """Fill layout footer slot."""
    return layout_values(req.ctx, req.slot_id, req.count, "footer")


def _fill_course_code(req: FillRequest):
    """Fill course code slot with PII metadata."""
    code = course_code(req.ctx.course)
    pii = (
        ("COURSE_CODE", code, "course", req.ctx.course.get("course_id", "NG"), "clean"),
    )
    return repeat_fill(FillValue(code, pii), req.count)


def _fill_meeting_date(req: FillRequest):
    """Fill meeting date slot with PII metadata."""
    pii = (
        ("DATE", req.ctx.meeting_date, "document", str(req.ctx.doc_id), "clean"),
    )
    return repeat_fill(FillValue(req.ctx.meeting_date, pii), req.count)


def _fill_doc_date(req: FillRequest):
    """Fill document date slot with PII metadata."""
    pii = (
        ("DATE", req.ctx.doc_date, "document", str(req.ctx.doc_id), "clean"),
    )
    return repeat_fill(FillValue(req.ctx.doc_date, pii), req.count)


def _fill_doc_duration(req: FillRequest):
    """Fill document duration slot."""
    return repeat_fill(FillValue(req.ctx.duration), req.count)


def _fill_group_label(req: FillRequest):
    """Fill group label slot with PII metadata."""
    pii = (
        ("GROUP_LABEL", req.ctx.group_label, "group", str(req.ctx.group_number), "clean"),
    )
    return repeat_fill(FillValue(req.ctx.group_label, pii), req.count)


def _fill_programme(req: FillRequest):
    """Fill course programme slot with PII metadata."""
    pii = (
        ("PROGRAMME", req.ctx.programme, "course", req.ctx.course.get("course_id", "NG"), "clean"),
    )
    return repeat_fill(FillValue(req.ctx.programme, pii), req.count)


def _fill_group_members(req: FillRequest):
    """Fill group members list slot."""
    block, pii = group_member_lines(req.ctx)
    return repeat_fill(FillValue(block, pii), req.count)


def _fill_attendees(req: FillRequest):
    """Fill meeting attendees slot."""
    block, pii = meeting_attendees(req.ctx)
    return repeat_fill(FillValue(block, pii), req.count)


def _fill_project_customer(req: FillRequest):
    """Fill project customer info slot."""
    text = f"Customer: {person_name(req.ctx.coordinator)}, Maastricht University"
    return repeat_fill(
        FillValue(
            text,
            (
                role_part("Customer", "coordinator", coordinator_id(req.ctx.coordinator)),
                staff_name_part(
                    req.ctx.coordinator,
                    "coordinator",
                    coordinator_id(req.ctx.coordinator),
                ),
            ),
        ),
        req.count,
    )


def _fill_project_coordinator_line(req: FillRequest):
    """Fill project coordinator and tutor info slot."""
    text = (
        f"Coordinator: {person_name(req.ctx.coordinator)}, Tutor: {person_name(req.ctx.teaching_assistant)}"
    )
    pii = (
        role_part("Coordinator", "coordinator", coordinator_id(req.ctx.coordinator)),
        staff_name_part(
            req.ctx.coordinator, "coordinator", coordinator_id(req.ctx.coordinator)
        ),
        role_part("Tutor", "tutor", tutor_id(req.ctx.teaching_assistant)),
        staff_name_part(
            req.ctx.teaching_assistant, "tutor", tutor_id(req.ctx.teaching_assistant)
        ),
    )
    return repeat_fill(FillValue(text, pii), req.count)


def _fill_student_field(req: FillRequest, field: str):
    """Fill one student-related slot field and attach matching PII metadata.

    Supported field values:
    - "name": full legal name
    - "id": student number (with optional typo simulation)
    - "email": institutional email (with optional typo simulation)
    """

    index = int(req.args.get("index", 1)) - 1
    student = row_at(req.ctx.group_members, index)
    if student is None:
        return repeat_fill(FillValue(""), req.count)

    if field == "name":
        return repeat_fill(
            FillValue(person_name(student), (student_name_part(student),)), req.count
        )
    if field == "id":
        rng = make_rng(
            req.ctx.master_seed,
            "slot",
            req.ctx.doc_id,
            f"{req.slot_id}:student_id:{index + 1}",
        )
        surface, quality = maybe_typo_student_number(
            student_number(student),
            rng,
            req.ctx.controller_config.typo_prob_student_number,
        )
        return repeat_fill(
            FillValue(surface, (student_id_part(student, surface, quality),)), req.count
        )

    rng = make_rng(
        req.ctx.master_seed,
        "slot",
        req.ctx.doc_id,
        f"{req.slot_id}:student_email:{index + 1}",
    )
    surface, quality = maybe_typo_email(
        student_email(student),
        rng,
        req.ctx.controller_config.typo_prob_email,
    )
    return repeat_fill(
        FillValue(surface, (student_email_part(student, surface, quality),)), req.count
    )


def _fill_student_name(req: FillRequest):
    """Fill student full name slot."""
    return _fill_student_field(req, "name")


def _fill_student_id(req: FillRequest):
    """Fill student number slot."""
    return _fill_student_field(req, "id")


def _fill_student_email(req: FillRequest):
    """Fill student institutional email slot."""
    return _fill_student_field(req, "email")


def _fill_faker_title(req: FillRequest):
    """Fill title slot with faker or project-specific title."""
    if req.ctx.template_id == "project_plan" and req.slot_id == "PROPOSAL_TITLE":
        return tuple(
            FillValue(project_proposal_title(req.ctx, occ=index))
            for index in range(req.count)
        )
    return tuple(
        FillValue(
            faker_title(req.ctx.master_seed, req.ctx.doc_id, req.slot_id, occ=index)
        )
        for index in range(req.count)
    )


def _fill_faker_paragraph(req: FillRequest):
    """Generate paragraph-like free text for generic and project-plan slots."""

    if req.ctx.template_id == "project_plan" and req.slot_id == "GANTT_TEXT":
        faker = faker_with_rng(req.ctx.master_seed, req.ctx.doc_id, req.slot_id, 0)
        return repeat_fill(
            FillValue(faker.sentence(nb_words=14).rstrip(".") + "."), req.count
        )

    if req.ctx.template_id == "project_plan" and req.slot_id == "SECTION_NOTE":
        values: list[FillValue] = []
        for index in range(req.count):
            faker = faker_with_rng(
                req.ctx.master_seed, req.ctx.doc_id, req.slot_id, index
            )
            values.append(
                FillValue(
                    faker.sentence(nb_words=faker.random.randint(9, 16)).rstrip(".")
                    + "."
                )
            )
        return tuple(values)

    paragraphs = int(req.args.get("paragraphs", 1))
    sentences = int(req.args.get("sentences", 6))
    return tuple(
        FillValue(
            faker_paragraph(
                master_seed=req.ctx.master_seed,
                doc_id=req.ctx.doc_id,
                slot_id=req.slot_id,
                occ=index,
                paragraphs=paragraphs,
                sentences=sentences,
            )
        )
        for index in range(req.count)
    )


def _fill_faker_bullets(req: FillRequest):
    """Fill bullet points slot from request args."""
    args = req.args
    return tuple(
        bullets(
            ctx=req.ctx,
            slot_id=req.slot_id,
            occ=index,
            min_items=int(args.get("min_items", 4)),
            max_items=int(args.get("max_items", 8)),
            per_member=bool(args.get("per_member", False)),
            mode=str(args.get("mode", "")),
            min_words=int(args.get("min_words", 10)),
            max_words=int(args.get("max_words", 20)),
            two_sentence_prob=float(args.get("two_sentence_prob", 0.35)),
        )
        for index in range(req.count)
    )


def _fill_project_objectives(req: FillRequest):
    """Fill project objectives text slot."""
    return tuple(
        FillValue(
            project_objectives_text(req.ctx, req.slot_id, occ=index, items=req.tasks)
        )
        for index in range(req.count)
    )


def _fill_project_section_text(req: FillRequest):
    """Fill a project section from either explicit args or slot defaults."""

    if req.args.get("section") is not None or req.args.get("min_words") is not None:
        texts = [
            project_section_text_by_args(req.ctx, req.slot_id, occ=index, args=req.args)
            for index in range(req.count)
        ]
    else:
        texts = [
            project_section_text(req.ctx, req.slot_id, occ=index)
            for index in range(req.count)
        ]
    return tuple(FillValue(text) for text in texts)


def _fill_project_table_text(req: FillRequest):
    """Fill project table text slot."""
    return tuple(
        project_table_text(req.ctx, req.slot_id, occ=index, items=req.tasks)
        for index in range(req.count)
    )


def _fill_faker_references(req: FillRequest):
    """Fill references list slot."""
    min_items = int(req.args.get("min_items", 5))
    max_items = int(req.args.get("max_items", 10))
    return tuple(
        FillValue(
            faker_references(
                master_seed=req.ctx.master_seed,
                doc_id=req.ctx.doc_id,
                slot_id=req.slot_id,
                occ=index,
                min_items=min_items,
                max_items=max_items,
            )
        )
        for index in range(req.count)
    )


def _fill_section_authors(req: FillRequest):
    """Fill section authors slot."""
    return take_or_pad(req.section_authors, req.count)


def _fill_section_reviewers(req: FillRequest):
    """Fill section reviewers slot."""
    return take_or_pad(req.section_reviewers, req.count)


def _fill_task_numbers(req: FillRequest):
    """Fill visible task numbers slot."""
    if not req.tasks:
        return repeat_fill(FillValue(""), req.count)
    visible_count = {"light": 2, "medium": 3, "dense": 4}[req.ctx.style.project_density]
    visible = req.tasks[: min(visible_count, len(req.tasks))]
    return repeat_fill(
        FillValue(", ".join(task.task_id for task in visible)), req.count
    )


def _task_values(req: FillRequest, extractor):
    """Map each requested task slot occurrence to a task-aware FillValue."""

    values: list[FillValue] = []
    for index in range(req.count):
        values.append(extractor(task_or_last(req.tasks, index)))
    return tuple(values)


def _fill_task_title(req: FillRequest):
    """Fill task title slot."""
    return _task_values(req, lambda task: FillValue(task.title if task else ""))


def _fill_task_lead(req: FillRequest):
    """Fill task lead name slot."""
    def extractor(task: ProjectTask | None):
        if task is None:
            return FillValue("")
        return FillValue(person_name(task.lead), (student_name_part(task.lead),))

    return _task_values(req, extractor)


def _fill_task_participants(req: FillRequest):
    """Fill task participants slot."""
    def extractor(task: ProjectTask | None):
        if task is None or not task.participants:
            return FillValue("")
        return people_fill(task.participants)

    return _task_values(req, extractor)


def _fill_task_dates(req: FillRequest):
    """Fill task dates slot."""
    return _task_values(req, lambda task: FillValue(task.dates if task else ""))


def _fill_task_description(req: FillRequest):
    """Fill task description slot."""
    return _task_values(req, lambda task: FillValue(task.description if task else ""))


_FILLERS: dict[str, Filler] = {
    "layout.header": _fill_layout_header,
    "layout.footer": _fill_layout_footer,
    "course.course_code": _fill_course_code,
    "meeting.date": _fill_meeting_date,
    "doc.date": _fill_doc_date,
    "doc.duration": _fill_doc_duration,
    "group.number": _fill_group_label,
    "course.programme": _fill_programme,
    "group.members": _fill_group_members,
    "meeting.attendees": _fill_attendees,
    "project.customer": _fill_project_customer,
    "project.coordinator_line": _fill_project_coordinator_line,
    "student.full_legal_name": _fill_student_name,
    "student.student_number": _fill_student_id,
    "student.institutional_email": _fill_student_email,
    "faker.title": _fill_faker_title,
    "faker.paragraph": _fill_faker_paragraph,
    "faker.bullets": _fill_faker_bullets,
    "project.objectives_text": _fill_project_objectives,
    "project.section_text": _fill_project_section_text,
    "project.table_text": _fill_project_table_text,
    "faker.references": _fill_faker_references,
    "project.section_authors": _fill_section_authors,
    "project.section_reviewers": _fill_section_reviewers,
    "project.task_numbers": _fill_task_numbers,
    "project.task_title": _fill_task_title,
    "project.task_lead": _fill_task_lead,
    "project.task_participants": _fill_task_participants,
    "project.task_dates": _fill_task_dates,
    "project.task_description": _fill_task_description,
}


def _fill_slot(
    *,
    ctx: DocContext,
    slot_id: str,
    fill_type: str,
    args: Mapping[str, Any],
    count: int,
    section_authors: Sequence[FillValue],
    section_reviewers: Sequence[FillValue],
    tasks: Sequence[ProjectTask],
):
    """Dispatch one slot fill request using the configured fill-type registry."""

    filler = _FILLERS.get(fill_type)
    if filler is None:
        raise ValueError(f"Unknown fill type: {fill_type}")

    request = FillRequest(
        ctx=ctx,
        slot_id=slot_id,
        args=args,
        count=count,
        section_authors=section_authors,
        section_reviewers=section_reviewers,
        tasks=tasks,
    )
    return filler(request)


def build_doc_payload(
    *,
    master_seed: int,
    doc_id: int,
    template_id: str,
    group_number: int,
    group_label: str,
    group_members: pd.DataFrame,
    course: Course,
    teaching_assistant: Person,
    coordinator: Person,
    controller_config,
    template_spec: Mapping[str, Any],
    marker_counts: Mapping[str, int],
):
    """Build the complete payload for one document render pass.

    The payload contains resolved slot values, section assignments, and
    project tasks, all derived deterministically from seed and document id.
    """

    meeting_iso = meeting_date(master_seed, doc_id)
    style = document_style(master_seed, doc_id, template_id)
    ctx = DocContext(
        template_id=template_id,
        master_seed=master_seed,
        doc_id=doc_id,
        group_number=group_number,
        group_label=group_label,
        group_members=group_members.reset_index(drop=True),
        course=course,
        teaching_assistant=teaching_assistant,
        coordinator=coordinator,
        controller_config=controller_config,
        meeting_date=meeting_iso,
        doc_date=iso_to_eu_date(meeting_iso),
        duration=duration(master_seed, doc_id, meeting_iso),
        programme=group_programme(group_members),
        style=style,
    )

    slots = list(template_spec.get("slots", []))
    if not slots:
        raise ValueError("template_spec has no 'slots' list")

    section_count = max(
        int(marker_counts.get("SECTION_AUTHORS", 0)),
        int(marker_counts.get("SECTION_REVIEWERS", 0)),
        int(marker_counts.get("SLOT_SECTION_AUTHORS", 0)),
        int(marker_counts.get("SLOT_SECTION_REVIEWERS", 0)),
    )
    section_authors, section_reviewers = section_assignments(ctx, section_count)

    planned_task_count = task_count(ctx, required_task_markers(marker_counts))
    planned_tasks = tasks(ctx, planned_task_count)

    slot_values: SlotValues = {}
    for slot in slots:
        slot_id = str(slot["id"]).upper()
        count = int(marker_counts.get(slot_id, 0))
        if count <= 0:
            continue
        slot_values[slot_id] = _fill_slot(
            ctx=ctx,
            slot_id=slot_id,
            fill_type=str(slot["fill"]),
            args=dict(slot.get("args", {})),
            count=count,
            section_authors=section_authors,
            section_reviewers=section_reviewers,
            tasks=planned_tasks,
        )
    return DocPayload(
        template_id=template_id, slot_values=slot_values, tasks=tuple(planned_tasks)
    )


__all__ = [
    "Person",
    "Course",
    "FillValue",
    "SlotValues",
    "ProjectTask",
    "DocPayload",
    "DocumentStyle",
    "DocContext",
    "FillRequest",
    "Filler",
    "iso_to_eu_date",
    "choose_project_density",
    "build_doc_payload",
]
