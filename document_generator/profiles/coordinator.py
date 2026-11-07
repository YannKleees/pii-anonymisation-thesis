"""Generate coordinator person profiles as a DataFrame."""
from faker import Faker
import pandas as pd

from ..seed import make_rng

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


def _gendered_name(faker: Faker, gender):
    """Return gendered first name from faker."""
    return faker.first_name_male() if gender == "male" else faker.first_name_female()


def _staff_email(first_name, last_name, used: set[str]):
    """Generate unique staff email from name, appending suffix if needed."""
    local = f"{_name_fragment(first_name)[:1]}.{_name_fragment(last_name)}"
    candidate = f"{local}@maastrichtuniversity.nl"
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while True:
        candidate = f"{local}{suffix}@maastrichtuniversity.nl"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def generate_coordinators(master_seed, n: int):
    """Generate n coordinator records with deterministic names and emails."""
    faker = Faker()
    rows: list[dict] = []
    used_emails: set[str] = set()

    for index in range(1, n + 1):
        name_rng = make_rng(master_seed, "coordinator", index, "name")
        gender_rng = make_rng(master_seed, "coordinator", index, "gender")
        gender = "male" if gender_rng.random() < 0.5 else "female"

        faker.random = name_rng
        first_name = _gendered_name(faker, gender)
        last_name = faker.last_name()
        full_name = f"{first_name} {last_name}"
        email = _staff_email(first_name, last_name, used_emails)
        coordinator_id = f"co_{index:05d}"
        rows.append(
            {
                "person_id": coordinator_id,
                "role": "coordinator",
                "gender": gender,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "full_legal_name": full_name,
                "student_number": "NG",
                "tutor_id": "NG",
                "coordinator_id": coordinator_id,
                "object_id": "NG",
                "date_of_birth": "NG",
                "nationality": "NG",
                "institutional_email": email,
                "email": email,
                "private_email": "NG",
                "current_residential_address": "NG",
                "programme": "NG",
            }
        )

    return pd.DataFrame(rows, columns=_COLUMNS)
