"""Assemble dataset from population and template configs, rendering documents and annotations."""
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document

from .config import GeneratorConfig, TemplateConfig, load_config
from .controller import build_doc_payload
from .marker import (
    build_annotations,
    count_markers_in_doc,
    fill_template,
    inject_non_pii_paragraphs,
    vary_headers_footers,
)
from .pii_tracker import PIITracker
from .profiles.coordinator import generate_coordinators
from .profiles.courses import generate_courses
from .profiles.student import generate_students
from .profiles.teaching_assistant import generate_teaching_assistants
from .project_plan_assets import enrich_project_plan_document
from .seed import make_rng, weighted_choice


def write_json(path: Path, data):
    """Write data to JSON file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_template_markers(template_spec: Mapping[str, Any], marker_counts: dict[str, int], template_id):
    """Raise if template spec slot IDs don't match document marker counts."""
    slots = list(template_spec.get("slots", []))
    spec_ids = {str(slot["id"]).upper() for slot in slots}
    marker_ids = {marker.upper() for marker in marker_counts}
    missing_in_spec = marker_ids - spec_ids
    missing_in_template = spec_ids - marker_ids
    if missing_in_spec or missing_in_template:
        raise ValueError("Template markers and spec do not match.")


def _write_frame(path: Path, frame: pd.DataFrame):
    """Write dataframe to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.fillna("NG").to_csv(path, index=False)


def _dedupe_people_emails(people: pd.DataFrame):
    """Deduplicate emails, appending numeric suffixes to collisions."""
    people = people.copy()
    used: dict[str, int] = {}
    for index, email in people["institutional_email"].items():
        if not isinstance(email, str) or email == "NG":
            continue
        local, separator, domain = email.partition("@")
        if not separator:
            continue
        seen_count = used.get(email, 0)
        if seen_count == 0:
            used[email] = 1
            continue
        next_email = f"{local}{seen_count + 1}@{domain}"
        while next_email in used:
            seen_count += 1
            next_email = f"{local}{seen_count + 1}@{domain}"
        people.at[index, "institutional_email"] = next_email
        people.at[index, "email"] = next_email
        used[next_email] = 1
        used[email] = seen_count + 1
    return people


def write_people_csv(path: Path, people: pd.DataFrame):
    """Write people dataframe to CSV."""
    _write_frame(path, people)


def write_courses_csv(path: Path, courses: pd.DataFrame):
    """Write courses dataframe to CSV."""
    _write_frame(path, courses)


def write_groups_csv(path: Path, groups: pd.DataFrame):
    """Write groups dataframe to CSV."""
    _write_frame(path, groups)


def write_groups_with_members_csv(path: Path, groups: pd.DataFrame, group_members: pd.DataFrame):
    """Write merged groups and members dataframe to CSV."""
    merged = group_members.merge(groups, on="group_id", how="left")
    ordered_columns = [
        "group_id",
        "group_number",
        "group_label",
        "course_id",
        "teaching_assistant_id",
        "coordinator_id",
        "student_number",
    ]
    remaining = [column for column in merged.columns if column not in ordered_columns]
    _write_frame(path, merged[ordered_columns + remaining])


_GROUP_LABEL_CATALOG: Sequence[str] = (
    "KEN4",
    "Group 09",
    "Labs 1",
    "T47",
    "Project Groups 13",
    "Group 04",
    "Assignment group 9",
    "Group 7",
    "Dream Team 2",
    "Clinics 4",
)


def _group_label_for(seed, group_id, group_number):
    """Get deterministic group label from catalog or generate one."""
    rng = make_rng(seed, "group", group_id, "label")
    if _GROUP_LABEL_CATALOG and rng.random() < 0.75:
        return _GROUP_LABEL_CATALOG[rng.randrange(len(_GROUP_LABEL_CATALOG))]
    return f"Group {group_number:02d}"


def _group_size_bounds(num_students, total_groups):
    """Compute min/max group size based on average group density."""
    average = num_students / max(1, total_groups)
    if average >= 6.0:
        return 6, 8
    if average >= 4.5:
        return 4, 6
    if average >= 3.5:
        return 3, 5
    if average >= 2.5:
        return 2, 4
    return 2, 3


def _take_indices(pool: list[int], offsets: dict[str, int], key, desired, disallow: set[int]):
    """Draw up to desired indices from pool, skipping disallowed indices."""
    if desired <= 0 or not pool:
        return []
    selected: list[int] = []
    attempts = 0
    max_attempts = max(len(pool) * 3, desired * 4)
    while len(selected) < desired and attempts < max_attempts:
        offset = offsets.get(key, 0)
        index = pool[offset % len(pool)]
        offsets[key] = offset + 1
        attempts += 1
        if index in disallow or index in selected:
            continue
        selected.append(index)
    return selected


def build_groups(*, seed, students: pd.DataFrame, courses: pd.DataFrame, teaching_assistants: pd.DataFrame, coordinators: pd.DataFrame, groups_per_course):
    """Build group and membership records, distributing students across groups."""
    total_groups = len(courses) * groups_per_course
    group_size_min, group_size_max = _group_size_bounds(len(students), total_groups)

    all_student_indices = list(students.index)
    make_rng(seed, "population", "groups", "all_students_shuffle").shuffle(all_student_indices)
    programme_pools: dict[str, list[int]] = {}
    for programme, frame in students.groupby("programme", sort=True):
        pool = list(frame.index)
        make_rng(seed, "population", str(programme), "programme_shuffle").shuffle(pool)
        programme_pools[str(programme)] = pool
    programme_names = sorted(programme_pools)
    offsets: dict[str, int] = {"__all__": 0}
    group_records: list[dict] = []
    membership_records: list[dict] = []
    group_id_counter = 1

    for _, course in courses.reset_index(drop=True).iterrows():
        course_rng = make_rng(seed, "course", course["course_id"], "group_plan")
        coordinator = coordinators.iloc[course_rng.randrange(len(coordinators))]

        ta_indices = list(range(len(teaching_assistants)))
        course_rng.shuffle(ta_indices)
        ta_pool = [
            teaching_assistants.iloc[index]
            for index in ta_indices[: max(1, min(3, len(ta_indices)))]
        ]

        dominant_programme = programme_names[course_rng.randrange(len(programme_names))]
        for group_number in range(1, groups_per_course + 1):
            group_id = f"group_{group_id_counter:04d}"
            group_id_counter += 1
            teaching_assistant = ta_pool[(group_number - 1) % len(ta_pool)]
            group_size = course_rng.randint(group_size_min, group_size_max)

            target_programme_count = max(1, min(group_size, group_size - 1 if len(programme_names) > 1 else group_size))
            selected_indices = _take_indices(
                programme_pools.get(dominant_programme, []),
                offsets,
                dominant_programme,
                target_programme_count,
                set(),
            )
            if len(selected_indices) < group_size:
                selected_indices.extend(
                    _take_indices(
                        all_student_indices,
                        offsets,
                        "__all__",
                        group_size - len(selected_indices),
                        set(selected_indices),
                    )
                )

            group_records.append(
                {
                    "group_id": group_id,
                    "course_id": course["course_id"],
                    "group_number": group_number,
                    "group_label": _group_label_for(seed, group_id, group_number),
                    "teaching_assistant_id": teaching_assistant["tutor_id"],
                    "coordinator_id": coordinator["coordinator_id"],
                }
            )
            for student_index in selected_indices:
                membership_records.append({
                        "group_id": group_id,
                        "student_number": students.loc[student_index, "student_number"],
                    })

    return pd.DataFrame(group_records), pd.DataFrame(membership_records)


def select_template(seed, doc_id, templates: Sequence[TemplateConfig]):
    """Select template weighted by template.weight."""
    rng = make_rng(seed, "doc", doc_id, "template_choice")
    return weighted_choice(rng, templates, [float(template.weight) for template in templates])


def _select_pdf_doc_ids(seed, num_documents, pdf_ratio):
    """Select document IDs to render as PDF based on pdf_ratio."""
    clamped_ratio = max(0.0, min(1.0, pdf_ratio))
    target_pdf_count = int(round(num_documents * clamped_ratio))
    doc_ids = list(range(1, num_documents + 1))
    rng = make_rng(seed, "documents", "all", "format_split")
    rng.shuffle(doc_ids)
    return set(doc_ids[:target_pdf_count])


def _convert_docx_to_pdf(docx_path: Path, converter_binary):
    """Convert DOCX to PDF using LibreOffice, with Windows fallback paths."""
    converter_cmd = converter_binary
    if shutil.which(converter_cmd) is None:
        windows_fallbacks = [
            Path("C:/Program Files/LibreOffice/program/soffice.exe"),
            Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        ]
        fallback = next((path for path in windows_fallbacks if path.exists()), None)
        if fallback is not None:
            converter_cmd = str(fallback)
    try:
        subprocess.run(
            [
                converter_cmd,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(docx_path.parent),
                str(docx_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"PDF conversion binary not found: {converter_cmd!r}. "
            "Set documents.pdf_converter_binary in the config to your LibreOffice/soffice executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"PDF conversion failed for {docx_path.name}: {error_output}"
        ) from exc

    pdf_path = docx_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"PDF conversion did not produce output file: {pdf_path}")
    return pdf_path


def _resolve_config(config_or_path: GeneratorConfig | Path | str):
    """Load config, returning (config, config_path) where path is None if passed a GeneratorConfig."""
    if isinstance(config_or_path, GeneratorConfig):
        return config_or_path, None
    config_path = Path(config_or_path)
    return load_config(config_path), config_path


def _output_root(config: GeneratorConfig, config_path: Path | None):
    """Resolve output directory, handling relative paths from config location."""
    output_dir = config.output_dir
    if not output_dir.is_absolute() and config_path is not None:
        output_dir = (config_path.parent / output_dir).resolve()
    elif not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    return output_dir / config.dataset_name


def generate_dataset(config_or_path: GeneratorConfig | Path | str):
    """Generate complete dataset: render documents, extract annotations, optionally convert to PDF."""
    config, config_path = _resolve_config(config_or_path)
    out_root = _output_root(config, config_path)

    documents_dir = out_root / "documents"
    annotations_dir = out_root / "annotations"
    images_dir = out_root / "images"
    documents_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    people_df = _dedupe_people_emails(pd.concat([
                generate_students(config.seed, config.population.num_students),
                generate_teaching_assistants(
                    config.seed, config.population.num_teaching_assistants),
                generate_coordinators(config.seed, config.population.num_coordinators)],
            ignore_index=True,
        ).fillna("NG"))

    students_df = people_df.loc[people_df["role"] == "student"].reset_index(drop=True)
    ta_df = people_df.loc[people_df["role"] == "teaching_assistant"].reset_index(drop=True)
    coordinator_df = people_df.loc[people_df["role"] == "coordinator"].reset_index(drop=True)
    courses_df = (generate_courses(config.seed, config.population.num_courses).fillna("NG").reset_index(drop=True))

    groups_df, group_members_df = build_groups(
        seed=config.seed,
        students=students_df,
        courses=courses_df,
        teaching_assistants=ta_df,
        coordinators=coordinator_df,
        groups_per_course=config.population.groups_per_course,
    )

    members_enriched_df = group_members_df.merge(students_df, on="student_number", how="left")
    members_by_group = {
        group_id: frame.reset_index(drop=True)
        for group_id, frame in members_enriched_df.groupby("group_id", sort=False)
    }
    courses_by_id = courses_df.set_index("course_id", drop=False)
    tas_by_id = ta_df.set_index("tutor_id", drop=False)
    coordinators_by_id = coordinator_df.set_index("coordinator_id", drop=False)
    spec_cache: dict[Path, dict] = {}

    total_images = 0
    pdf_doc_ids = _select_pdf_doc_ids(config.seed, config.documents.num_documents, config.documents.pdf_ratio)
    for doc_id in range(1, config.documents.num_documents + 1):
        group_rng = make_rng(config.seed, "doc", doc_id, "group_choice")
        group = groups_df.iloc[group_rng.randrange(len(groups_df))]
        group_id = str(group["group_id"])

        course = courses_by_id.loc[str(group["course_id"])]
        teaching_assistant = tas_by_id.loc[str(group["teaching_assistant_id"])]
        coordinator = coordinators_by_id.loc[str(group["coordinator_id"])]
        members = members_by_group[group_id]
        template = select_template(config.seed, doc_id, config.templates)

        template_spec = spec_cache.get(template.spec_path)
        if template_spec is None:
            template_spec = json.loads(template.spec_path.read_text(encoding="utf-8"))
            spec_cache[template.spec_path] = template_spec

        document = Document(str(template.path))
        marker_counts = count_markers_in_doc(document)
        _validate_template_markers(template_spec, marker_counts, template.template_id)

        if template.template_id != "project_plan":
            vary_headers_footers(document, config.seed, doc_id)

        payload = build_doc_payload(
            master_seed=config.seed,
            doc_id=doc_id,
            template_id=template.template_id,
            group_number=int(group["group_number"]),
            group_label=str(group["group_label"]),
            group_members=members,
            course=course,
            teaching_assistant=teaching_assistant,
            coordinator=coordinator,
            controller_config=config.controller,
            template_spec=template_spec,
            marker_counts=marker_counts,
        )
        tracker = PIITracker()
        fill_template(document, payload, tracker)

        image_records = enrich_project_plan_document(
            document=document,
            payload=payload,
            tracker=tracker,
            out_root=out_root,
            doc_id=doc_id,
            seed=config.seed,
        )

        if template.template_id != "project_plan":
            inject_non_pii_paragraphs(document, config.seed, doc_id)

        doc_stem = f"doc_{doc_id:05d}_{template.template_id}"
        docx_out = documents_dir / f"{doc_stem}.docx"
        ann_out = annotations_dir / f"{doc_stem}.json"
        document.save(str(docx_out))

        annotations = build_annotations(
            doc_id=doc_id,
            template_id=template.template_id,
            seed=config.seed,
            doc_path=docx_out,
            emitted_pii=tracker.emissions(),
        )
        if image_records:
            annotations["images"] = image_records
            total_images += len(image_records)
        write_json(ann_out, annotations)

        if doc_id in pdf_doc_ids:
            _convert_docx_to_pdf(docx_out, config.documents.pdf_converter_binary)
            docx_out.unlink(missing_ok=True)

    write_people_csv(out_root / "people.csv", people_df)
    write_courses_csv(out_root / "courses.csv", courses_df)
    write_groups_with_members_csv(out_root / "groups.csv", groups_df, group_members_df)

    write_json(
        out_root / "summary.json",
        {
            "dataset_name": config.dataset_name,
            "seed": config.seed,
            "num_people": len(people_df),
            "num_students": int((people_df["role"] == "student").sum()),
            "num_tutors": int((people_df["role"] != "student").sum()),
            "num_courses": len(courses_df),
            "num_groups": len(groups_df),
            "num_documents": config.documents.num_documents,
            "num_docx_documents": config.documents.num_documents - len(pdf_doc_ids),
            "num_pdf_documents": len(pdf_doc_ids),
            "pdf_ratio": config.documents.pdf_ratio,
            "num_images": total_images,
            "templates": [template.template_id for template in config.templates],
            "output_dir": str(out_root),
        },
    )
    return out_root
