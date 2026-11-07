"""Generate course records as a DataFrame."""
from collections.abc import Sequence

from faker import Faker
import pandas as pd

from ..seed import make_rng


_COURSE_CATALOG: Sequence[tuple[str, str, str]] = (
    ("KEN2300", "Project 2-1", "2024-2025"),
    ("KEN2600", "Project 2-2", "2024-2025"),
    ("KEN3300", "Project 3-1", "2025-2026"),
    ("KEN1300", "Project 1-1", "2023-2024"),
    ("KEN1600", "Project 1-2", "2023-2024"),
)


def generate_courses(master_seed, n, academic_year="2025-2026"):
    """Generate n course records, shuffling from catalog and adding synthetic ones."""
    faker = Faker()
    rng = make_rng(master_seed, "population", "courses", "shuffle")
    catalog = list(_COURSE_CATALOG)
    rng.shuffle(catalog)
    rows: list[dict] = []
    used_codes = set()

    for index, (code, name, year) in enumerate(catalog[: min(n, len(catalog))], start=1):
        used_codes.add(code)
        rows.append(
            {
                "course_id": f"course_{index:04d}",
                "course_code": code,
                "course_name": name,
                "academic_year": year or academic_year,
            }
        )
    dept_codes = ["DSAI", "CS"]

    for index in range(len(rows) + 1, n + 1):
        code_rng = make_rng(master_seed, "course", index, "code")
        dept = dept_codes[code_rng.randrange(len(dept_codes))]
        number = 1000 + code_rng.randint(0, 8999)
        code = f"{dept}{number}"
        while code in used_codes:
            number = 1000 + code_rng.randint(0, 8999)
            code = f"{dept}{number}"
        used_codes.add(code)
        faker.random = make_rng(master_seed, "course", index, "name")
        rows.append({
                "course_id": f"course_{index:04d}",
                "course_code": code,
                "course_name": faker.catch_phrase(),
                "academic_year": academic_year,
            })
    return pd.DataFrame(rows, columns=["course_id", "course_code", "course_name", "academic_year"])
