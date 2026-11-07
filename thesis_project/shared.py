"""PII entity taxonomy, data models, and configuration for defender/attacker workflows."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

direct_labels = {
    "FULL_NAME",
    "FIRST_NAME",
    "LAST_NAME",
    "Name",
    "Student number",
    "Institutional email",
}

quasi_labels = {
    "Course code",
    "Group label",
    "Programme",
    "Date",
    "Role",
}

strict_regions = {"cover", "header", "footer"}

region_aliases = {
    "first_page_header": "header",
    "even_page_header": "header",
    "header_table": "header",
    "first_page_header_table": "header",
    "even_page_header_table": "header",

    "first_page_footer": "footer",
    "even_page_footer": "footer",
    "footer_table": "footer",
    "first_page_footer_table": "footer",
    "even_page_footer_table": "footer",

    "cover_table": "body_table",

    "body_image": "image",
}

label_masks = {
    "FULL_NAME": "[Full name]",
    "FIRST_NAME": "[First name]",
    "LAST_NAME": "[Last name]",
    "Name": "[Name]",
    "Student number": "[Student number]",
    "Institutional email": "[Institutional email]",
    "Course code": "[Course code]",
    "Group label": "[Group label]",
    "Programme": "[Programme]",
    "Date": "[Date]",
    "Role": "[Role]",
}

label_weights = {
    "FULL_NAME": 3.0,
    "FIRST_NAME": 2.0,
    "LAST_NAME": 2.0,
    "Name": 3.0,
    "Student number": 3.0,
    "Institutional email": 3.0,
    "Course code": 2.0,
    "Group label": 2.0,
    "Programme": 2.0,
    "Date": 1.0,
    "Role": 1.0,
}

label_lookup = {label.lower(): label for label in (direct_labels | quasi_labels)}
label_lookup.update({
    'full name': 'FULL_NAME',
    'full_name': 'FULL_NAME',
    'first name': 'FIRST_NAME',
    'first_name': 'FIRST_NAME',
    'last name': 'LAST_NAME',
    'last_name': 'LAST_NAME',
    'student name': 'Name',
    'name': 'Name',
    'student number': 'Student number',
    'student_number': 'Student number',
    'student id': 'Student number',
    'student_id': 'Student number',
    'institutional email': 'Institutional email',
    'institutional_email': 'Institutional email',
    'email': 'Institutional email',
    'course code': 'Course code',
    'course_code': 'Course code',
    'group label': 'Group label',
    'group_label': 'Group label',
    'group': 'Group label',
    'programme': 'Programme',
    'program': 'Programme',
    'date': 'Date',
    'role': 'Role',
})

student_number_digits = 7
name_body_threshold = 0.80
name_strict_threshold = 0.70
quasi_body_threshold = 0.80
quasi_strict_threshold = 0.70
ocr_conf_threshold = 40.0
ocr_psm = 6
attack_score_threshold = 1.20
attack_margin_threshold = 0.20


@dataclass
class Block:
    """One text segment extracted from a document."""
    region: str
    path: str
    text: str
    raw_text: str = ""
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageArtifact:
    """Binary image payload used for OCR-driven detection."""
    image_id: str
    region: str
    path: str
    media_path: str
    blob: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Occurrence:
    """Detected entity with offsets and detector metadata."""
    label: str
    surface: str
    start: int
    end: int
    region: str
    path: str
    score: float = 1.0
    source: str = ""
    entity_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedactionAction:
    """Single redaction command for text or image output."""
    region: str
    path: str
    label: str
    action: str
    start: int
    end: int
    replacement: str = ""
    detector: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entry:
    """Audit trail row describing one redaction decision."""
    doc_id: str
    region: str
    path: str
    label: str
    surface: str
    detector: str
    action: str
    score: str = ""


@dataclass
class DocumentRecord:
    """Parsed document and all extracted artifacts."""
    doc_id: str
    document_path: Path
    template_id: str = ""
    split: str = ""
    annotation: Path = None
    blocks: list[Block] = field(default_factory=list)
    images: list[ImageArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBaseRecord:
    """Knowledge-base identity profile used by attacker matching."""
    record: str
    entity_type: str
    full_name: str = ""
    student_number: str = ""
    institutional_email: str = ""
    programme: str = ""
    group_ids: tuple[str, ...] = ()
    group_labels: tuple[str, ...] = ()
    course_ids: tuple[str, ...] = ()
    course_codes: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Runtime toggles and paths for defender/attacker flows."""
    output: Path = None
    patterns: bool = True
    ner: bool = True
    ocr: bool = True
    strict: bool = True
    link: bool = True
    direct: bool = True
    quasi: bool = False
    labels: tuple[str, ...] = ("Course code", "Group label", "Programme", "Date", "Role")
    regions: tuple[str, ...] = ()
    mask_boxes: bool = True
    pdf: bool = True
    office: str = "libreoffice"
    tesseract: str = None
    model: str = None
    candidate: str = None
    ner_backend: str = "hybrid"
    gliner_model: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = 0.35
    ner_zero_shot_labels: tuple[str, ...] = ()
    local: bool = False
    fallback: bool = True
    device: int = -1
    max_len: int = 256
    pdf_render_dpi: int = 200
    pdf_ocr_all_pages: bool = False


def _norm(text):
    """Normalize text for case-insensitive matching."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_region(region):
    """Map region aliases to canonical region names."""
    return region_aliases.get(region, region)


def canon(label):
    """Normalize user-provided labels to canonical taxonomy values."""
    text = str(label or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip().lower()
    return label_lookup.get(normalized, re.sub(r"\s+", " ", text).strip())


def severity_weight_for(label):
    """Return label weight used by ranking/scoring paths."""
    return float(label_weights.get(canon(label), 0.0))


def direct(label):
    """True when a label is a direct identifier."""
    return canon(label) in direct_labels


def quasi(label):
    """True when a label is a quasi identifier."""
    return canon(label) in quasi_labels


def is_strict_region(region):
    """True when region belongs to strict policy zones."""
    return normalize_region(region) in strict_regions


def mask_token_for(label):
    """Mask token to inject for a given label."""
    return label_masks.get(canon(label), "[Redacted]")


def validate_taxonomy():
    """Guard against overlapping label groups."""
    if direct_labels & quasi_labels:
        raise ValueError("Identifier label taxonomy overlaps.")
