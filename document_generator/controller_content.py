from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from faker import Faker

from .controller_common import (
    DocContext,
    FillValue,
    Person,
    ProjectTask,
    biased_int,
    bounded,
    coordinator_id,
    course_code,
    course_name,
    faker_with_rng,
    iso_to_eu_date,
    maybe_typo_email,
    maybe_typo_student_number,
    people_fill,
    person_id,
    person_name,
    row_at,
    rows,
    shuffled_rows,
    staff_email,
    staff_email_part,
    staff_name_part,
    student_email,
    student_email_part,
    student_id_part,
    student_name_part,
    student_number,
    tutor_id,
)
from .seed import make_rng, weighted_choice

def faker_paragraph(*,master_seed: int, doc_id: int, slot_id: str, occ: int, paragraphs: int, sentences: int,):
    """Generate deterministic paragraph text for generic narrative slots."""
    faker = faker_with_rng(master_seed, doc_id, slot_id, occ)
    rng = faker.random
    blocks: list[str] = []
    for _ in range(max(1, int(paragraphs))):
        low = max(2, int(sentences) - 2)
        high = max(low, int(sentences) + 2)
        sent_count = rng.randint(low, high)
        blocks.append(faker.paragraph(nb_sentences=sent_count).replace("\n", " ").strip())
    return "\n".join(blocks)

def faker_title(master_seed: int, doc_id: int, slot_id: str, occ: int) -> str:
    return faker_with_rng(master_seed, doc_id, slot_id, occ).catch_phrase()

def faker_references(*,master_seed: int,doc_id: int,slot_id: str,occ: int,min_items: int,max_items: int):
    faker = faker_with_rng(master_seed, doc_id, slot_id, occ)
    rng = faker.random
    count = biased_int(rng, min_items, max_items, "normal")
    lines: list[str] = []
    for _ in range(count):
        author_last = faker.last_name()
        author_initial = rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        year = rng.randint(2000, 2026)
        title = faker.sentence(nb_words=rng.randint(4, 9)).rstrip(".")
        journal = faker.word().capitalize() + " " + faker.word().capitalize()
        volume = rng.randint(1, 25)
        issue = rng.randint(1, 6)
        page_a = rng.randint(1, 120)
        page_b = page_a + rng.randint(1, 20)
        lines.append(f"{author_last}, {author_initial}. ({year}). {title}. {journal}, {volume}({issue}), {page_a}-{page_b}.")
    return "\n".join(lines)

def group_member_lines(ctx: DocContext):
    """Build group member lines with density-based identifiers and PII metadata."""
    rng = make_rng(ctx.master_seed, "doc", ctx.doc_id, "group_members_block")
    members = rows(ctx.group_members)
    member_count = len(members)

    if member_count == 0:
        return "", ()
    
    if ctx.style.identifier_density == "sparse":
        id_count = bounded(biased_int(rng, 1, member_count, "sparse"), 1, member_count)
        email_count = 0 if member_count > 1 else 1

    elif ctx.style.identifier_density == "balanced":
        id_count = bounded(
            biased_int(rng, max(1, member_count // 2), member_count, "normal"),1,member_count)
        email_count = min(
            member_count, biased_int(rng, 0, min(2, member_count), "normal"))
        
    else:
        id_count = member_count
        email_count = min(member_count, biased_int(rng, 1, min(3, member_count), "dense"))

    id_indexes = set(rng.sample(range(member_count), k=id_count)) if id_count else set()
    email_indexes = (set(rng.sample(range(member_count), k=email_count)) if email_count else set())
    lines: list[str] = []
    pii_parts: list[tuple] = []
    for index, student in enumerate(members):
        member_rng = make_rng(ctx.master_seed, "doc", ctx.doc_id, f"member:{student_number(student)}")
        line = person_name(student)
        pii_parts.append(student_name_part(student))
        if index in id_indexes:
            sid_surface, sid_quality = maybe_typo_student_number(
                student_number(student),
                member_rng,
                ctx.controller_config.typo_prob_student_number,
            )
            line += f" ({sid_surface})"
            pii_parts.append(student_id_part(student, sid_surface, sid_quality))

        if index in email_indexes:
            email_surface, email_quality = maybe_typo_email(
                student_email(student),
                member_rng,
                ctx.controller_config.typo_prob_email,
            )
            separator = " / " if "(" in line else " — "
            line += f"{separator}{email_surface}"
            pii_parts.append(student_email_part(student, email_surface, email_quality))

        lines.append(line)
    return "\n".join(lines), tuple(pii_parts)


def meeting_attendees(ctx: DocContext) -> tuple[str, tuple]:
    """Create attendee block text and linked PII parts for meeting sections."""

    rng = make_rng(ctx.master_seed, "doc", ctx.doc_id, "attendees")
    students = shuffled_rows(ctx.group_members, rng)
    total_students = len(students)
    if total_students == 0:
        return "", ()

    minimum = 1 if total_students == 1 else min(2, total_students)
    desired_students = bounded(
        biased_int(rng, minimum, total_students, ctx.style.detail_density),
        1,
        total_students,
    )
    attendee_students = students[:desired_students]

    names = [person_name(student) for student in attendee_students]
    pii_parts = [student_name_part(student) for student in attendee_students]

    names.append(person_name(ctx.teaching_assistant))
    pii_parts.append(
        staff_name_part(
            ctx.teaching_assistant, "tutor", tutor_id(ctx.teaching_assistant)
        )
    )

    coordinator_prob = {"brief": 0.15, "normal": 0.30, "detailed": 0.45}[
        ctx.style.detail_density
    ]
    if rng.random() < coordinator_prob:
        names.append(person_name(ctx.coordinator))
        pii_parts.append(
            staff_name_part(
                ctx.coordinator, "coordinator", coordinator_id(ctx.coordinator)
            )
        )

    return "\n".join(names), tuple(pii_parts)


def generic_layout_values(ctx: DocContext, kind: str) -> list[FillValue]:
    """Return header/footer defaults with tracked quasi-identifiers when present.

    Some generic layout lines intentionally include course codes, group labels,
    programme names, or dates. These used to be emitted as plain strings, which
    meant the annotation builder never recorded them. The defender then detected
    real identifiers in headers/footers that the gold data treated as false
    positives. Returning FillValue objects fixes that mismatch at the source.
    """
    values = [
        FillValue("Maastricht University"),
        FillValue("Department of Advanced Computing Sciences"),
        FillValue(
            f"{course_code(ctx.course)} — {course_name(ctx.course)}",
            (("COURSE_CODE", course_code(ctx.course), "course", str(ctx.course.get("course_id", "NG")), "clean"),),
        ),
        FillValue(
            f"Group {ctx.group_number} ({ctx.group_label})",
            (("GROUP_LABEL", ctx.group_label, "group", str(ctx.group_number), "clean"),),
        ),
    ]
    if kind == "footer":
        values.append(
            FillValue(
                f"Document date: {ctx.doc_date}",
                (("DATE", ctx.doc_date, "document", str(ctx.doc_id), "clean"),),
            )
        )
    if ctx.template_id == "project_plan":
        values.extend(
            [
                FillValue(
                    f"Research Proposal — {course_code(ctx.course)}",
                    (("COURSE_CODE", course_code(ctx.course), "course", str(ctx.course.get("course_id", "NG")), "clean"),),
                ),
                FillValue(
                    f"Programme: {ctx.programme}",
                    (("PROGRAMME", ctx.programme, "course", str(ctx.course.get("course_id", "NG")), "clean"),),
                ),
                FillValue(f"Project duration: {ctx.duration}"),
            ]
        )
    return values


def _student_layout_value(ctx: DocContext, rng) -> FillValue:
    student = row_at(ctx.group_members, rng.randrange(len(ctx.group_members)))
    assert student is not None
    sid_surface, sid_quality = maybe_typo_student_number(
        student_number(student),
        rng,
        ctx.controller_config.typo_prob_student_number,
    )
    email_surface, email_quality = maybe_typo_email(
        student_email(student),
        rng,
        ctx.controller_config.typo_prob_email,
    )
    variant_weights = {
        "sparse": [0.55, 0.25, 0.10, 0.10],
        "balanced": [0.35, 0.30, 0.20, 0.15],
        "rich": [0.20, 0.35, 0.20, 0.25],
    }[ctx.style.identifier_density]
    variant = weighted_choice(
        rng, ["name", "name_id", "name_email", "contact_only"], variant_weights
    )

    if variant == "name":
        return FillValue(person_name(student), (student_name_part(student),))
    if variant == "name_id":
        return FillValue(
            f"{person_name(student)} ({sid_surface})",
            (
                student_name_part(student),
                student_id_part(student, sid_surface, sid_quality),
            ),
        )
    if variant == "name_email":
        return FillValue(
            f"{person_name(student)} / {email_surface}",
            (
                student_name_part(student),
                student_email_part(student, email_surface, email_quality),
            ),
        )
    return FillValue(
        email_surface, (student_email_part(student, email_surface, email_quality),)
    )


def _staff_layout_value(ctx: DocContext, rng, *, role: str) -> FillValue:
    person = ctx.teaching_assistant if role == "tutor" else ctx.coordinator
    entity_id = tutor_id(person) if role == "tutor" else coordinator_id(person)
    email_surface, email_quality = maybe_typo_email(
        staff_email(person),
        rng,
        ctx.controller_config.typo_prob_email,
    )
    variant = weighted_choice(rng, ["name", "name_email", "email"], [0.45, 0.35, 0.20])
    if variant == "name":
        return FillValue(
            person_name(person), (staff_name_part(person, role, entity_id),)
        )
    if variant == "name_email":
        return FillValue(
            f"{person_name(person)} / {email_surface}",
            (
                staff_name_part(person, role, entity_id),
                staff_email_part(email_surface, role, entity_id, email_quality),
            ),
        )
    return FillValue(email_surface,(staff_email_part(email_surface, role, entity_id, email_quality),))


def layout_pii_value(ctx: DocContext, rng):
    who = weighted_choice(rng, ["student", "tutor", "coordinator"], [0.75, 0.20, 0.05])
    if who == "student":
        return _student_layout_value(ctx, rng)
    if who == "tutor":
        return _staff_layout_value(ctx, rng, role="tutor")
    return _staff_layout_value(ctx, rng, role="coordinator")


def layout_values(ctx: DocContext, slot_id: str, count: int, kind: str):
    """Generate header/footer slot values with probabilistic content and PII density."""
    rng = make_rng(ctx.master_seed, "slot", ctx.doc_id, f"{kind}:{slot_id}")
    if count <= 0:
        return ()

    content_prob = (
        ctx.controller_config.header_prob
        if kind == "header"
        else ctx.controller_config.footer_prob
    )
    pii_prob = (
        ctx.controller_config.header_pii_prob
        if kind == "header"
        else ctx.controller_config.footer_pii_prob
    )

    detail_factor = {"brief": 0.85, "normal": 1.0, "detailed": 1.10}[
        ctx.style.detail_density
    ]
    identifier_factor = {"sparse": 0.35, "balanced": 1.0, "rich": 1.55}[
        ctx.style.identifier_density
    ]

    filled_count = sum(
        1 for _ in range(count) if rng.random() < min(1.0, content_prob * detail_factor)
    )
    if content_prob > 0.50 and filled_count == 0:
        filled_count = 1

    pii_cap = {"sparse": 0, "balanced": 1, "rich": min(2, count)}[
        ctx.style.identifier_density
    ]
    if ctx.template_id == "project_plan":
        pii_cap = min(pii_cap, 1)
    pii_count = min(
        filled_count,
        pii_cap,
        sum(
            1
            for _ in range(filled_count)
            if rng.random() < min(1.0, pii_prob * identifier_factor)
        ),
    )

    filled_indexes = (
        set(rng.sample(range(count), k=filled_count)) if filled_count else set()
    )
    pii_indexes = (
        set(rng.sample(sorted(filled_indexes), k=pii_count)) if pii_count else set()
    )
    generic_values = generic_layout_values(ctx, kind)

    values: list[FillValue] = []
    for index in range(count):
        if index not in filled_indexes:
            values.append(FillValue(""))
        elif index in pii_indexes:
            values.append(layout_pii_value(ctx, rng))
        else:
            values.append(rng.choice(generic_values))
    return tuple(values)


def bullet_verb(mode: str, rng) -> str:
    if mode == "completed":
        return weighted_choice(
            rng,
            ["Completed", "Reviewed", "Finalised", "Updated"],
            [0.40, 0.20, 0.20, 0.20],
        )
    if mode == "planned":
        return weighted_choice(
            rng, ["Plan", "Implement", "Write", "Prepare"], [0.35, 0.30, 0.20, 0.15]
        )
    return weighted_choice(rng, ["Discuss", "Review", "Check"], [0.40, 0.35, 0.25])


def bullets(
    *,
    ctx: DocContext,
    slot_id: str,
    occ: int,
    min_items: int,
    max_items: int,
    per_member: bool,
    mode: str,
    min_words: int,
    max_words: int,
    two_sentence_prob: float,
):
    """Create bullet-list text, optionally attributing items to group members."""

    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, slot_id, occ)
    rng = faker.random
    item_count = biased_int(rng, min_items, max_items, ctx.style.detail_density)

    def bullet_sentence() -> str:
        first = faker.sentence(nb_words=rng.randint(min_words, max_words)).rstrip(".")
        if mode == "questions":
            first = first if first.endswith("?") else f"{first}?"
        if rng.random() < two_sentence_prob:
            second = faker.sentence(
                nb_words=rng.randint(max(3, min_words // 2), max_words)
            ).rstrip(".")
            if mode == "questions":
                second = second if second.endswith("?") else f"{second}?"
                return f"{first} {second}"
            return f"{first}. {second}"
        return first

    if not per_member:
        return FillValue("\n".join(f"- {bullet_sentence()}" for _ in range(item_count)))

    members = shuffled_rows(ctx.group_members, rng)
    if ctx.style.identifier_density == "sparse":
        named_member_count = min(
            len(members), biased_int(rng, 1, max(1, min(2, len(members))), "sparse")
        )
    elif ctx.style.identifier_density == "balanced":
        named_member_count = min(
            len(members), biased_int(rng, 1, len(members), "normal")
        )
    else:
        named_member_count = len(members)
    named_members = members[:named_member_count]

    items: list[str] = []
    pii_parts: list[tuple] = []
    seen_person_ids: set[str] = set()

    for member in named_members:
        if len(items) >= item_count:
            break
        member_items = 1
        if (
            ctx.style.detail_density == "detailed"
            and len(items) + 1 < item_count
            and rng.random() < 0.40
        ):
            member_items = 2
        for _ in range(member_items):
            if len(items) >= item_count:
                break
            items.append(
                f"- {person_name(member)}: {bullet_verb(mode, rng)} {bullet_sentence()}"
            )
        pid = person_id(member)
        if pid not in seen_person_ids:
            pii_parts.append(student_name_part(member))
            seen_person_ids.add(pid)

    while len(items) < item_count:
        prefix = (
            f"{bullet_verb(mode, rng)} "
            if mode in {"completed", "planned"} and rng.random() < 0.45
            else ""
        )
        items.append(f"- {prefix}{bullet_sentence()}")

    return FillValue("\n".join(items), tuple(pii_parts))


def section_assignments(ctx: DocContext, section_count: int):
    """Assign section authors and reviewers with deterministic rotation rules."""
    if section_count <= 0 or ctx.group_members.empty:
        return [], []

    rng = make_rng(ctx.master_seed, "doc", ctx.doc_id, "section_assignments")
    ordered_members = shuffled_rows(ctx.group_members, rng)
    authors: list[FillValue] = []
    reviewers: list[FillValue] = []

    for index in range(section_count):
        author_count = 1
        if (
            len(ordered_members) > 2
            and ctx.style.detail_density == "detailed"
            and rng.random() < 0.40
        ):
            author_count = 2
        author_indices = [
            (index + offset) % len(ordered_members) for offset in range(author_count)
        ]
        chosen_authors = tuple(ordered_members[position] for position in author_indices)
        chosen_author_ids = {person_id(person) for person in chosen_authors}
        reviewer_pool = [
            member
            for member in ordered_members
            if person_id(member) not in chosen_author_ids
        ]
        if not reviewer_pool:
            reviewer_pool = [
                ordered_members[(index + author_count) % len(ordered_members)]
            ]
        reviewer_count = 1
        if (
            len(reviewer_pool) > 1
            and ctx.style.identifier_density == "rich"
            and rng.random() < 0.35
        ):
            reviewer_count = 2
        chosen_reviewers = tuple(reviewer_pool[:reviewer_count])

        authors.append(people_fill(chosen_authors))
        reviewers.append(people_fill(chosen_reviewers))
    return authors, reviewers


# Task generation

TASK_MARKER_KEYS = (
    "TASK_LEAD",
    "TASK_TITLE",
    "TASK_PARTICIPANTS",
    "TASK_DATES",
    "DESCRIPTION",
    "SLOT_TASK_LEAD",
)


def required_task_markers(marker_counts: Mapping[str, int]):
    return max(int(marker_counts.get(key, 0)) for key in TASK_MARKER_KEYS)


def task_count(ctx: DocContext, required_count: int):
    """This function computes the planned task count from required markers and style density."""
    if required_count <= 0:
        return 0
    if ctx.template_id != "project_plan":
        return required_count

    minimum_by_density = {"light": 4, "medium": 5, "dense": 7}
    slack_by_density = {"light": 1, "medium": 2, "dense": 3}
    minimum = minimum_by_density[ctx.style.project_density]
    slack = slack_by_density[ctx.style.project_density]
    return max(required_count + slack, minimum)


def tasks(ctx: DocContext, total_tasks: int):
    """This function generates deterministic project tasks including leads, participants, and dates."""
    if total_tasks <= 0 or ctx.group_members.empty:
        return []

    rng = make_rng(ctx.master_seed, "project_plan", ctx.doc_id, "tasks")
    faker = Faker()
    faker.random = rng

    members = shuffled_rows(ctx.group_members, rng)
    package_names = [
        "Preparation",
        "Requirements",
        "Data generation",
        "Implementation",
        "Validation",
        "Reporting",
    ]
    title_prefixes = ["Task", "Study", "Prototype", "Review", "Workflow", "Package"]

    anchor = date.fromisoformat(ctx.meeting_date) + timedelta(days=rng.randint(2, 8))
    next_start = anchor
    created: list[ProjectTask] = []

    for index in range(total_tasks):
        lead = members[index % len(members)]
        lead_id = person_id(lead)
        participant_pool = [
            member for member in members if person_id(member) != lead_id
        ]
        rng.shuffle(participant_pool)

        max_participants = min(
            len(participant_pool),
            {"light": 1, "medium": 2, "dense": 3}[ctx.style.project_density],
        )
        if ctx.style.identifier_density == "sparse":
            max_participants = min(max_participants, 1)

        if (
            max_participants <= 0
            or rng.random()
            < {"light": 0.45, "medium": 0.20, "dense": 0.08}[ctx.style.project_density]
        ):
            participants: tuple[Person, ...] = ()
        else:
            participant_count = biased_int(
                rng, 1, max_participants, ctx.style.project_density
            )
            participants = tuple(participant_pool[:participant_count])

        package_name = package_names[index % len(package_names)]
        work_package = f"WP{(index % 4) + 1}: {package_name}"

        title_core = faker.bs().replace("-", " ").strip()
        if rng.random() < 0.35:
            title_core = faker.catch_phrase().rstrip(".")
        title = f"{title_prefixes[index % len(title_prefixes)]}: {title_core}"

        duration_days = biased_int(rng, 4, 15, ctx.style.project_density)
        overlap_days = (
            0
            if index == 0
            else rng.randint(
                0, {"light": 2, "medium": 4, "dense": 6}[ctx.style.project_density]
            )
        )
        start_date = next_start - timedelta(days=overlap_days)
        end_date = start_date + timedelta(days=duration_days)
        next_start = start_date + timedelta(days=max(3, duration_days // 2))

        paragraph_blocks = (
            1 if ctx.style.detail_density != "detailed" or rng.random() < 0.70 else 2
        )
        description_parts: list[str] = []
        for block_index in range(paragraph_blocks):
            sentence_count = (
                rng.randint(2, 5) if block_index == 0 else rng.randint(2, 4)
            )
            description_parts.append(
                faker.paragraph(nb_sentences=sentence_count).replace("\n", " ").strip()
            )

        created.append(
            ProjectTask(
                task_id=f"T{index + 1}",
                work_package=work_package,
                title=title,
                lead=lead,
                participants=participants,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                dates=f"{start_date.strftime('%d/%m/%Y')} – {end_date.strftime('%d/%m/%Y')}",
                description=" ".join(description_parts),
            )
        )
    return created


def task_or_last(items: Sequence[ProjectTask], index: int):
    if not items:
        return None
    return items[min(index, len(items) - 1)]


# Section G: project-plan long-form helpers


def long_form_text(faker: Faker, target_words: int, min_paragraphs: int, max_paragraphs: int):
    rng = faker.random
    paragraphs: list[str] = []
    words = 0

    target_words = max(40, int(target_words))
    min_paragraphs = max(1, int(min_paragraphs))
    max_paragraphs = max(min_paragraphs, int(max_paragraphs))

    while words < target_words or len(paragraphs) < min_paragraphs:
        paragraph = (
            faker.paragraph(nb_sentences=rng.randint(7, 14)).replace("\n", " ").strip()
        )
        paragraphs.append(paragraph)
        words += len(paragraph.split())
        if len(paragraphs) >= max_paragraphs and words >= target_words:
            break
    return "\n".join(paragraphs)


def project_proposal_title(ctx: DocContext, occ: int):
    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, "proposal_title", occ)
    rng = faker.random
    course_hint = course_name(ctx.course).split(":", 1)[0].strip()
    if rng.random() < 0.45:
        return faker.catch_phrase().rstrip(".")
    return f"{course_hint}: {faker.bs().replace('-', ' ').title()}"


def faker_words_title(rng, faker: Faker, min_words: int = 2, max_words: int = 4):
    words = [word.capitalize() for word in faker.words(nb=rng.randint(min_words, max_words), unique=True)]
    return " ".join(words)


def paragraph_bank(faker: Faker, *, count: int, min_sentences: int, max_sentences: int):
    rng = faker.random
    return [
        faker.paragraph(nb_sentences=rng.randint(min_sentences, max_sentences))
        .replace("\n", " ")
        .strip()
        for _ in range(count)
    ]


def gibberish_requirement_block(faker: Faker, *, target_words: int):
    rng = faker.random
    category_count = 2 if target_words < 800 else 3
    lines: list[str] = []
    for category_index in range(1, category_count + 1):
        lines.append(f"R{category_index} {faker_words_title(rng, faker)}")
        for item_index in range(1, rng.randint(2, 4) + 1):
            sentence = faker.sentence(nb_words=rng.randint(10, 20)).rstrip(".")
            priority = rng.choice([1, 1, 2, 3])
            lines.append(f"R{category_index}.{item_index} {sentence}. (priority {priority})")
        lines.append("")
    tail = paragraph_bank(faker, count=rng.randint(1, 3), min_sentences=3, max_sentences=6)
    return "\n".join(lines).strip() + "\n\n" + "\n\n".join(tail)


def gibberish_related_work_block(faker: Faker, *, target_words: int):
    rng = faker.random
    row_count = 3 if target_words < 1000 else 4
    rows_out = ["Projects & publications | Summary | Connection / difference"]
    for _ in range(row_count):
        author = faker.last_name()
        year = rng.randint(2001, 2026)
        title = faker_words_title(rng, faker, 2, 5)
        summary = faker.sentence(nb_words=rng.randint(8, 16)).rstrip(".")
        relation = faker.sentence(nb_words=rng.randint(10, 18)).rstrip(".")
        rows_out.append(f"{author} et al. ({year}) — {title} | {summary} | {relation}")
    tail = paragraph_bank(faker, count=rng.randint(1, 3), min_sentences=3, max_sentences=5)
    return "\n".join(rows_out) + "\n\n" + "\n\n".join(tail)


def gibberish_bullets(faker: Faker, *, min_items: int, max_items: int):
    rng = faker.random
    return "\n".join(
        f"- {faker.sentence(nb_words=rng.randint(10, 20)).rstrip('.')} .".replace(" .", ".")
        for _ in range(rng.randint(min_items, max_items))
    )


def project_objectives_text(ctx: DocContext, slot_id: str, occ: int, items: Sequence[ProjectTask]):
    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, slot_id, occ)
    rng = faker.random
    objective_count = biased_int(rng, 2, 4, ctx.style.project_density)
    task_ids = [task.task_id for task in items] or [f"T{number}" for number in range(1, 5)]
    lines: list[str] = []
    for index in range(1, objective_count + 1):
        lines.append(f"Objective O{index}: {faker.sentence(nb_words=rng.randint(5, 9)).rstrip('.') }.")
        reference_count = min(
            len(task_ids),
            biased_int(rng, 1, min(3, len(task_ids)), ctx.style.project_density),
        )
        referenced = sorted(rng.sample(task_ids, k=reference_count))
        lines.append(f"Addressed in Task {', '.join(referenced)}")
        lines.append("S.M.A.R.T. Justification")
        lines.append("\n".join(
                paragraph_bank(
                    faker, count=rng.randint(1, 2), min_sentences=2, max_sentences=4
                )))
        lines.append("KPIs")
        lines.append(f"KPI O{index}.1: {faker.sentence(nb_words=rng.randint(7, 12)).rstrip('.') }.")
        lines.append(f"KPI O{index}.2: {faker.sentence(nb_words=rng.randint(7, 12)).rstrip('.') }.")
        lines.append("")
    return "\n".join(lines).strip()


PROJECT_SECTION_PLANS: tuple[tuple[str, int, int], ...] = (
    ("motivation", 650, 950),
    ("requirements", 900, 1300),
    ("approach", 1100, 1700),
    ("work_so_far", 700, 1100),
    ("prototypes", 350, 650),
    ("ambition", 180, 320),
    ("related_work", 750, 1200),
    ("impact_societal", 280, 520),
    ("impact_economic", 260, 480),
    ("impact_research", 260, 480),
    ("impact_other", 260, 480),
    ("work_plan", 700, 1200),
    ("conclusion", 320, 600),
)


def project_section_text(ctx: DocContext, slot_id: str, occ: int):
    """Generate long-form section text using predefined section plans by index."""

    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, slot_id, occ)
    rng = faker.random
    kind, min_words, max_words = (
        PROJECT_SECTION_PLANS[occ]
        if occ < len(PROJECT_SECTION_PLANS)
        else ("generic", 300, 500)
    )
    target_words = rng.randint(min_words, max_words)
    return project_section_text_for_kind(ctx, faker, kind=kind, target_words=target_words)


def project_section_text_by_args(ctx: DocContext, slot_id: str, occ: int, args: Mapping[str, Any]):
    """Generate section text from explicit section and word-budget arguments."""

    section = str(args.get("section", "")).strip().lower()
    min_words = int(args.get("min_words", 0) or 0)
    max_words_raw = args.get("max_words")
    max_words = int(max_words_raw) if max_words_raw is not None else None

    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, slot_id, occ)
    rng = faker.random
    if max_words is not None and max_words >= min_words and min_words > 0:
        target_words = rng.randint(min_words, max_words)
    elif min_words > 0:
        target_words = min_words
    else:
        target_words = 500

    normalized = section.replace("/", " ").replace("-", " ")
    section_key = "_".join(normalized.split())
    kind = {
        "motivation": "motivation",
        "motivation_context_problem_statement": "motivation",
        "problem_statement": "motivation",
        "requirements": "requirements",
        "concept_and_approach": "approach",
        "approach": "approach",
        "methodology": "approach",
        "work_so_far": "work_so_far",
        "envisioned_additional_prototypes": "prototypes",
        "prototypes": "prototypes",
        "ambition": "ambition",
        "related_work": "related_work",
        "societal_impact": "impact_societal",
        "economic_impact": "impact_economic",
        "research_impact": "impact_research",
        "other_impact": "impact_other",
        "work_plan": "work_plan",
        "implementation": "work_plan",
        "concluding_remarks": "conclusion",
        "conclusion": "conclusion",
    }.get(section_key, "generic")
    return project_section_text_for_kind(
        ctx, faker, kind=kind, target_words=target_words
    )


def project_section_text_for_kind(ctx: DocContext, faker: Faker, *, kind: str, target_words: int):
    """Render section text according to semantic kind and style density."""

    density = ctx.style.project_density
    if kind == "ambition":
        return gibberish_bullets(
            faker,
            min_items=3 if density == "light" else 4,
            max_items=5 if density != "dense" else 7,
        )
    if kind == "prototypes":
        return gibberish_bullets(
            faker,
            min_items=3 if density == "light" else 4,
            max_items=6 if density != "dense" else 8,
        )
    if kind == "requirements":
        return gibberish_requirement_block(faker, target_words=target_words)
    if kind == "related_work":
        return gibberish_related_work_block(faker, target_words=target_words)

    paragraph_targets = {"light": (2, 4), "medium": (3, 5), "dense": (4, 7)}
    min_paragraphs, max_paragraphs = paragraph_targets[density]
    if kind.startswith("impact_") and density == "light":
        max_paragraphs = max(min_paragraphs, max_paragraphs - 1)
    return long_form_text(
        faker,
        target_words=target_words,
        min_paragraphs=min_paragraphs,
        max_paragraphs=max_paragraphs,
    )


def project_table_text(ctx: DocContext, slot_id: str, occ: int, items: Sequence[ProjectTask]):
    """Build project-table style blocks (milestones, deliverables, or risks)."""

    faker = faker_with_rng(ctx.master_seed, ctx.doc_id, slot_id, occ)
    rng = faker.random

    if ctx.template_id == "project_plan":
        visible_tasks = list(items[: min(4, len(items))])
        if occ == 0:
            heading = "Milestone overview"
            bullets = [
                f"- {task.task_id}: milestone near {iso_to_eu_date(task.end_date)}"
                for task in visible_tasks
            ] or ["- Milestones defined in the planning section."]
            return FillValue(heading + "\n" + "\n".join(bullets))
        if occ == 1:
            heading = "Deliverable overview"
            bullets = [
                f"- {task.task_id}: {faker_words_title(rng, faker, 2, 4)} planned for {iso_to_eu_date(task.end_date)}"
                for task in visible_tasks
            ] or ["- Deliverables aligned with the work packages."]
            return FillValue(heading + "\n" + "\n".join(bullets))
        heading = "Risk overview"
        bullets = [
            f"- {faker.sentence(nb_words=rng.randint(8, 14)).rstrip('.')} .".replace(
                " .", "."
            )
            for _ in range(rng.randint(2, 4))
        ]
        return FillValue(heading + "\n" + "\n".join(bullets))

    if occ == 0:
        lines = ["Milestone | Date | Description"]
        for index in range(rng.randint(2, 4)):
            lines.append(
                f"MS{index + 1} | {faker.date_between(start_date='+5d', end_date='+120d').isoformat()} | {faker.sentence(nb_words=rng.randint(6, 12)).rstrip('.')}"
            )
        return FillValue("\n".join(lines))

    if occ == 1:
        lead = row_at(ctx.group_members, 0)
        if lead is None:
            return FillValue("")
        lines = ["No. | Name | Lead | Type | Dissemination | Delivery Date"]
        for index in range(1, rng.randint(3, 5) + 1):
            lines.append(
                f"D{index} | {faker.sentence(nb_words=rng.randint(3, 7)).rstrip('.')} | {person_name(lead)} | {rng.choice(['RE', 'DEM', 'DEC', 'OTH'])} | {rng.choice(['PU', 'CO'])} | {faker.date_between(start_date='+30d', end_date='+220d').isoformat()}"
            )
        return FillValue("\n".join(lines), (student_name_part(lead),))

    lines = ["Risk | Prob | Impact | Mitigation Plan"]
    for _ in range(rng.randint(4, 7)):
        lines.append(
            f"{faker.sentence(nb_words=rng.randint(5, 9)).rstrip('.')} | {rng.choice(['L', 'M', 'H'])} | {rng.choice(['L', 'M', 'H'])} | {faker.sentence(nb_words=rng.randint(8, 14)).rstrip('.')}"
        )
    return FillValue("\n".join(lines))


__all__ = [
    "faker_paragraph",
    "faker_title",
    "faker_references",
    "group_member_lines",
    "meeting_attendees",
    "layout_values",
    "bullets",
    "section_assignments",
    "required_task_markers",
    "task_count",
    "tasks",
    "task_or_last",
    "project_proposal_title",
    "project_objectives_text",
    "project_section_text",
    "project_section_text_by_args",
    "project_table_text",
]
