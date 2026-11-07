"""Load and validate typed configuration for dataset generation."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemplateConfig:
    """Configuration for one template and its slot specification file."""
    template_id: str
    path: Path
    spec_path: Path
    weight: float = 1.0


@dataclass(frozen=True)
class PopulationConfig:
    """Population sizes used to synthesize people, courses, and groups."""
    num_students: int
    num_teaching_assistants: int
    num_coordinators: int
    num_courses: int
    groups_per_course: int


@dataclass(frozen=True)
class DocumentsConfig:
    """Number of documents to generate and output format settings."""
    num_documents: int
    pdf_ratio: float = 0.0
    pdf_converter_binary: str = "soffice"


@dataclass(frozen=True)
class SplitsConfig:
    """Output split ratios for train, validation, and test datasets."""
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


@dataclass(frozen=True)
class ControllerConfig:
    """Probabilities controlling noise injection and header/footer enrichment."""
    typo_prob_student_number: float = 0.03
    typo_prob_email: float = 0.08
    header_prob: float = 0.75
    header_pii_prob: float = 0.30
    footer_prob: float = 0.85
    footer_pii_prob: float = 0.25


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level typed representation of all generator settings."""
    dataset_name: str
    seed: int
    output_dir: Path
    templates: list[TemplateConfig]
    population: PopulationConfig
    documents: DocumentsConfig
    splits: SplitsConfig
    controller: ControllerConfig


def _resolve_path(value: str | Path, base_dir: Path):
    """Resolve a path from config, interpreting relative paths from config location."""
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path):
    """Load JSON config and coerce values into typed generator settings."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    directory = config_path.parent
    templates = [
        TemplateConfig(
            template_id=str(item["template_id"]),
            path=_resolve_path(item["path"], directory),
            spec_path=_resolve_path(
                item.get("spec")
                or item.get("spec_path")
                or Path(item["path"]).with_suffix(".spec.json"),
                directory,
            ),
            weight=float(item.get("weight", 1.0)),
        )
        for item in raw.get("templates", [])
    ]
    population_raw = raw.get("population", {})
    documents_raw = raw.get("documents", {})
    splits_raw = raw.get("splits", {})
    controller_raw = raw.get("controller", {})
    return GeneratorConfig(
        dataset_name=str(raw.get("dataset_name", "dataset")).strip() or "dataset",
        seed=int(raw.get("seed", 0)),
        output_dir=Path(raw.get("output_dir", "out")),
        templates=templates,
        population=PopulationConfig(
            num_students=int(population_raw.get("num_students", 100)),
            num_teaching_assistants=int(
                population_raw.get("num_teaching_assistants", 10)
            ),
            num_coordinators=int(population_raw.get("num_coordinators", 5)),
            num_courses=int(population_raw.get("num_courses", 10)),
            groups_per_course=int(population_raw.get("groups_per_course", 6)),
        ),
        documents=DocumentsConfig(
            num_documents=int(documents_raw.get("num_documents", 25)),
            pdf_ratio=float(documents_raw.get("pdf_ratio", 0.0)),
            pdf_converter_binary=str(
                documents_raw.get("pdf_converter_binary", "soffice")
            ),
        ),
        splits=SplitsConfig(
            train=float(splits_raw.get("train", 0.8)),
            val=float(splits_raw.get("val", 0.1)),
            test=float(splits_raw.get("test", 0.1)),
        ),
        controller=ControllerConfig(
            typo_prob_student_number=float(
                controller_raw.get(
                    "typo_prob_student_number",
                    controller_raw.get("type_studentid", 0.03),
                )
            ),
            typo_prob_email=float(
                controller_raw.get(
                    "typo_prob_email", controller_raw.get("typo_mail", 0.08)
                )
            ),
            header_prob=float(controller_raw.get("header_prob", 0.75)),
            header_pii_prob=float(controller_raw.get("header_pii_prob", 0.30)),
            footer_prob=float(controller_raw.get("footer_prob", 0.85)),
            footer_pii_prob=float(controller_raw.get("footer_pii_prob", 0.25)),
        ),
    )
