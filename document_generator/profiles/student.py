"""Generate student person profiles as a DataFrame."""
from faker import Faker
import pandas as pd

from ..seed import make_rng

_PROGRAMMES = [
    "BSc Data Science and Artificial Intelligence",
    "BSc Computer Science",
    "BSc Artificial Intelligence",
    "MSc Data Science for Decision Making",
]

_NATIONALITIES = [
    "Dutch",
    "German",
    "French",
    "Italian",
    "Spanish",
    "Belgian",
    "Luxembourgish",
    "Polish",
    "Romanian",
    "Greek",
    "Turkish",
    "Portuguese",
]

_COLUMNS = [
    "person_id",
    "role",
    "gender",
    "first_name",
    "last_name",
    "full_name",
    "full_legal_name",
    "student_number",
    "tutor_id",
    "coordinator_id",
    "object_id",
    "date_of_birth",
    "nationality",
    "institutional_email",
    "email",
    "private_email",
    "current_residential_address",
    "programme",
]


def _name_fragment(value):
    """Extract alphabetic characters from value, lowercase."""
    cleaned = "".join(char for char in value.lower() if char.isalpha())
    return cleaned or "x"


def _student_email(first_name, last_name, used):
    """Generate unique student email from name, appending suffix if needed."""
    local = f"{_name_fragment(first_name)[:1]}.{_name_fragment(last_name)}"
    candidate = f"{local}@student.maastrichtuniversity.nl"
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while True:
        candidate = f"{local}{suffix}@student.maastrichtuniversity.nl"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def _gendered_name(faker: Faker, gender):
    """Return gendered first name from faker."""
    return faker.first_name_male() if gender == "male" else faker.first_name_female()


def generate_students(master_seed, n):
    """Generate n student records with deterministic profiles."""
    base_rng = make_rng(master_seed, "population", "students", "base_student_number")
    base = 6_000_000 + base_rng.randint(0, 500_000)
    faker = Faker()
    rows: list[dict] = []
    used_emails: set[str] = set()
    used_object_ids: set[str] = set()
    for index in range(1, n + 1):
        name_rng = make_rng(master_seed, "student", index, "name")
        dob_rng = make_rng(master_seed, "student", index, "dob")
        nat_rng = make_rng(master_seed, "student", index, "nationality")
        addr_rng = make_rng(master_seed, "student", index, "address")
        private_rng = make_rng(master_seed, "student", index, "private_email")
        obj_rng = make_rng(master_seed, "student", index, "object_id")
        programme_rng = make_rng(master_seed, "student", index, "programme")
        gender_rng = make_rng(master_seed, "student", index, "gender")

        gender = "male" if gender_rng.random() < 0.5 else "female"
        faker.random = name_rng
        first_name = _gendered_name(faker, gender)
        last_name = faker.last_name()
        full_name = f"{first_name} {last_name}"
        faker.random = dob_rng
        date_of_birth = faker.date_of_birth(minimum_age=18, maximum_age=28).isoformat()
        faker.random = addr_rng
        current_residential_address = faker.address().replace("\n", ", ")
        faker.random = private_rng
        private_email = faker.free_email()
        number = base + (index - 1)
        digits = f"{number:07d}"
        if digits[0] != "6":
            digits = "6" + digits[1:]
        student_number = f"i{digits}"

        object_id = str(10_000_000_000 + obj_rng.randint(0, 9_999_999_999))
        while object_id in used_object_ids:
            object_id = str(int(object_id) + 1)
        used_object_ids.add(object_id)

        institutional_email = _student_email(first_name, last_name, used_emails)

        rows.append(
            {
                "person_id": f"student_{index:05d}",
                "role": "student",
                "gender": gender,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "full_legal_name": full_name,
                "student_number": student_number,
                "tutor_id": "NG",
                "coordinator_id": "NG",
                "object_id": object_id,
                "date_of_birth": date_of_birth,
                "nationality": _NATIONALITIES[nat_rng.randrange(len(_NATIONALITIES))],
                "institutional_email": institutional_email,
                "email": institutional_email,
                "private_email": private_email,
                "current_residential_address": current_residential_address,
                "programme": _PROGRAMMES[programme_rng.randrange(len(_PROGRAMMES))],
            }
        )

    return pd.DataFrame(rows, columns=_COLUMNS)
