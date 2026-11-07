"""Shared models and helper utilities used by the document controller"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
import pandas as pd
from faker import Faker
from .config import ControllerConfig
from .pii_tracker import PIIPart
from .seed import make_rng, weighted_choice

Person = pd.Series
Course = pd.Series


@dataclass(frozen=True)
class FillValue:
    """This class represents a generated text fragment plus the PII entities it represents."""
    text: str
    pii_parts: tuple[PIIPart, ...] = ()


SlotValues = dict[str, tuple[FillValue, ...]]


@dataclass(frozen=True)
class ProjectTask:
    """This class represents the task payload used in project-plan templates."""

    task_id: str
    work_package: str
    title: str
    lead: Person
    participants: tuple[Person, ...]
    start_date: str
    end_date: str
    dates: str
    description: str


@dataclass(frozen=True)
class DocPayload:
    """This class represents the final payload consumed by the template rendering layer."""

    template_id: str
    slot_values: SlotValues
    tasks: tuple[ProjectTask, ...] = ()


@dataclass(frozen=True)
class DocumentStyle:
    """This class represents the density profile that controls how much detail and PII appears."""

    identifier_density: str
    detail_density: str
    project_density: str


@dataclass(frozen=True)
class DocContext:
    """Deterministic context shared by all slot fillers for one document."""

    template_id: str
    master_seed: int
    doc_id: int
    group_number: int
    group_label: str
    group_members: pd.DataFrame
    course: Course
    teaching_assistant: Person
    coordinator: Person
    controller_config: ControllerConfig
    meeting_date: str
    doc_date: str
    duration: str
    programme: str
    style: DocumentStyle


@dataclass(frozen=True)
class FillRequest:
    """Input bundle provided to a filler for one slot render call."""

    ctx: DocContext
    slot_id: str
    args: Mapping[str, Any]
    count: int
    section_authors: Sequence[FillValue]
    section_reviewers: Sequence[FillValue]
    tasks: Sequence[ProjectTask]



Filler = Callable[[FillRequest], tuple[FillValue, ...]]

# Row accessor helpers

def row_value(row: Mapping[str, Any] | pd.Series, key: str, default: str = "NG"):
    """This function reads a field from a row and normalizes missing/NaN values to the default."""
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text or default


def person_value(person: Person, key: str, default: str = "NG"):
    return row_value(person, key, default)


def course_value(course: Course, key: str, default: str = "NG"):
    return row_value(course, key, default)


def person_id(person: Person):
    return person_value(person, "person_id")


def person_name(person: Person):
    return person_value(person, "full_legal_name", person_value(person, "full_name"))


def person_first_name(person: Person):
    return person_value(person, "first_name")


def person_last_name(person: Person):
    return person_value(person, "last_name")


def person_role(person: Person):
    return person_value(person, "role")


def student_number(person: Person):
    return person_value(person, "student_number")


def student_email(person: Person):
    return person_value(person, "institutional_email")


def staff_email(person: Person):
    return person_value(person, "email")


def tutor_id(person: Person):
    return person_value(person, "tutor_id")


def coordinator_id(person: Person):
    return person_value(person, "coordinator_id")


def course_id(course: Course):
    return course_value(course, "course_id")


def course_code(course: Course):
    return course_value(course, "course_code")


def course_name(course: Course):
    return course_value(course, "course_name")


def rows(frame: pd.DataFrame):
    return [frame.iloc[index] for index in range(len(frame))]


def row_at(frame: pd.DataFrame, index: int):
    if index < 0 or index >= len(frame):
        return None
    return frame.iloc[index]


def shuffled_rows(frame: pd.DataFrame, rng) -> list[Person]:
    indexes = list(range(len(frame)))
    rng.shuffle(indexes)
    return [frame.iloc[index] for index in indexes]


# Context helpers


def iso_to_eu_date(iso_date: str):
    """Convert YYYY-MM-DD into DD/MM/YYYY."""
    year, month, day = iso_date.split("-")
    return f"{day}/{month}/{year}"


def choose_project_density(master_seed: int, doc_id: int):
    rng = make_rng(master_seed, "project_plan", doc_id, "density")
    draw = rng.random()
    if draw < 0.35:
        return "light"
    if draw < 0.75:
        return "medium"
    return "dense"


def meeting_date(master_seed: int, doc_id: int):
    rng = make_rng(master_seed, "doc", doc_id, "meeting_date")
    year = 2025
    month = 9 + rng.randint(0, 3)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def duration(master_seed: int, doc_id: int, start_iso: str):
    rng = make_rng(master_seed, "doc", doc_id, "duration")
    start_date = date.fromisoformat(start_iso)
    end_date = start_date + timedelta(days=rng.randint(90, 170))
    return f"{start_iso} to {end_date.isoformat()}"


def group_programme(group_members: pd.DataFrame):
    default_programme = "BSc Data Science and Artificial Intelligence"
    if group_members.empty or "programme" not in group_members.columns:
        return default_programme
    programme_rows = group_members.loc[
        group_members["programme"].fillna("NG") != "NG",
        ["programme"],
    ]
    programme_counts = (
        programme_rows.groupby("programme")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "programme"], ascending=[False, True], kind="stable")
    )
    if programme_counts.empty:
        return default_programme
    return str(programme_counts.iloc[0]["programme"])


def document_style(master_seed: int, doc_id: int, template_id: str):
    """Choose deterministic style densities for the current document."""
    identifier_rng = make_rng(master_seed, "doc", doc_id, "identifier_density")
    detail_rng = make_rng(master_seed, "doc", doc_id, "detail_density")
    if template_id == "project_plan":
        identifier_weights = [0.30, 0.50, 0.20]
        detail_weights = [0.25, 0.50, 0.25]
    else:
        identifier_weights = [0.45, 0.40, 0.15]
        detail_weights = [0.30, 0.50, 0.20]
    return DocumentStyle(
        identifier_density=weighted_choice(
            identifier_rng, ["sparse", "balanced", "rich"], identifier_weights
        ),
        detail_density=weighted_choice(
            detail_rng, ["brief", "normal", "detailed"], detail_weights
        ),
        project_density=choose_project_density(master_seed, doc_id),
    )


def faker_with_rng(master_seed: int, doc_id: int, slot_id: str, occ: int):
    faker = Faker()
    faker.random = make_rng(master_seed, "slot", doc_id, f"{slot_id}:{occ}")
    return faker


def bounded(value: int, minimum: int, maximum: int):
    return max(minimum, min(maximum, value))


def biased_int(rng, minimum: int, maximum: int, bias: str):
    minimum = int(minimum)
    maximum = int(maximum)
    if maximum <= minimum:
        return minimum

    span = maximum - minimum
    if bias in {"sparse", "brief", "light"}:
        return rng.randint(minimum, minimum + max(0, span // 2))
    if bias in {"rich", "detailed", "dense"}:
        return rng.randint(maximum - max(0, span // 2), maximum)

    center_low = minimum + span // 4
    center_high = maximum - span // 4
    if center_low > center_high:
        center_low, center_high = minimum, maximum
    return rng.randint(center_low, center_high)


def repeat_fill(value: FillValue, count: int):
    return tuple(value for _ in range(max(0, count)))


def take_or_pad(values: Sequence[FillValue], count: int):
    selected = list(values[:count])
    while len(selected) < count:
        selected.append(FillValue(""))
    return tuple(selected)


# PII tuple helpers standardize annotation format across all fillers.
def pii_part(label: str, surface: str, entity_type: str, entity_id: str, quality: str = "clean"):
    return (label, surface, entity_type, entity_id, quality)


def role_part(surface: str, entity_type: str, entity_id: str):
    return pii_part("ROLE", surface, entity_type, entity_id)


def student_name_part(student: Person):
    return pii_part("FULL_NAME", person_name(student), "student", student_number(student))


def student_id_part(student: Person, surface: str, quality: str):
    return pii_part("STUDENT_NUMBER", surface, "student", student_number(student), quality)



def student_email_part(student: Person, surface: str, quality: str):
    return pii_part("INSTITUTIONAL_EMAIL", surface, "student", student_number(student), quality)


def staff_name_part(person: Person, entity_type: str, entity_id: str):
    return pii_part("FULL_NAME", person_name(person), entity_type, entity_id)


def staff_email_part(surface: str, entity_type: str, entity_id: str, quality: str):
    return pii_part("INSTITUTIONAL_EMAIL", surface, entity_type, entity_id, quality)


def people_fill(people: Sequence[Person]):
    names = [person_name(person) for person in people]
    pii = tuple(student_name_part(person) for person in people)
    return FillValue(", ".join(names), pii)


# Typo simulation helpers inject realistic noise for robustness testing.
def maybe_typo_student_number(value: str, rng, prob: float):
    """Inject a realistic student-number typo with probability prob."""
    if rng.random() >= prob:
        return value, "clean"
    if len(value) != 8 or not value.startswith("i"):
        return value, "typo"
    digits = list(value[1:])
    if rng.random() < 0.5 and digits:
        digits.pop(rng.randrange(len(digits)))
        return "i" + "".join(digits), "typo"
    left = rng.randrange(len(digits))
    right = rng.randrange(len(digits))
    if left != right:
        digits[left], digits[right] = digits[right], digits[left]
    return "i" + "".join(digits), "typo"


def maybe_typo_email(value: str, rng, prob: float):
    """Inject a realistic email typo with probability prob."""
    if rng.random() >= prob:
        return value, "clean"
    if "@" not in value:
        return value, "typo"
    local, domain = value.split("@", 1)
    variants: list[str] = []
    at_domain = "@" + domain
    if "." in local:
        variants.append(local.replace(".", "", 1) + at_domain)
    if len(local) > 3:
        index = rng.randrange(1, len(local))
        variants.append(local[:index] + local[index + 1 :] + at_domain)
    if len(local) > 2:
        index = rng.randrange(0, len(local) - 1)
        swapped = list(local)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        variants.append("".join(swapped) + at_domain)
    if "maastrichtuniversity" in domain:
        variants.append(
            local + "@" + domain.replace("maastrichtuniversity", "maastrichuniversity")
        )
    return (rng.choice(variants) if variants else value), "typo"


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
    "row_value",
    "person_value",
    "course_value",
    "person_id",
    "person_name",
    "person_first_name",
    "person_last_name",
    "person_role",
    "student_number",
    "student_email",
    "staff_email",
    "tutor_id",
    "coordinator_id",
    "course_id",
    "course_code",
    "course_name",
    "rows",
    "row_at",
    "shuffled_rows",
    "iso_to_eu_date",
    "choose_project_density",
    "meeting_date",
    "duration",
    "group_programme",
    "document_style",
    "faker_with_rng",
    "bounded",
    "biased_int",
    "repeat_fill",
    "take_or_pad",
    "pii_part",
    "student_name_part",
    "student_id_part",
    "student_email_part",
    "staff_name_part",
    "staff_email_part",
    "people_fill",
    "maybe_typo_student_number",
    "maybe_typo_email",
]
