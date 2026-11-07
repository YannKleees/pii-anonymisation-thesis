"""Pattern-based extraction of student IDs, emails, dates, and course codes."""
import re

from ..shared import Occurrence, canon, normalize_region, student_number_digits

email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
student = re.compile(rf"\b[iI](?:[\s-]?\d){{{student_number_digits}}}\b")

# Course codes in the synthetic UM-style data, e.g. KEN2300, DSAI7833, CS4811.
course = re.compile(r"\b[A-Z]{2,6}\d{3,5}\b")

# Dates seen in the generated documents. Keep these explicit to avoid matching
# every short number as a privacy target.
date_patterns = (
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/20\d{2}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-20\d{2}\b"),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.20\d{2}\b"),
    re.compile(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},\s+20\d{2}\b"
    ),
)

# Programmes often occur as a bare value on the project-plan cover page, not
# necessarily after a literal "Programme:" prefix. Keep whitespace line-local so
# a value such as "BSc Computer Science" does not absorb the next field label
# on the following line, for example "Date:".
PROGRAMME_CORE = (
    r"(?:BSc|MSc|Bachelor|Master)(?:[ \t]+(?:of|in))?[ \t]+"
    r"[A-Z][A-Za-z&/-]+(?:[ \t]+[A-Z][A-Za-z&/-]+){0,6}"
)
programme_value = re.compile(r"\b" + PROGRAMME_CORE + r"\b")
programme_field = re.compile(
    r"(?i)(?:programme|program(?:me)?)[ \t]*[:\-]?[ \t]*(" + PROGRAMME_CORE + r")"
)

# Group labels are not one stable format in the generated population. Examples:
# Group 04, Group 7, Assignment group 9, Project Groups 13, Labs 1, Clinics 4,
# Dream Team 2, KEN4, T47.
group_field = re.compile(
    r"(?i)(?:group(?:\s+(?:numbers?|number|label|id|ids))?|assignment\s+group|project\s+groups?|lab(?:s)?|clinic(?:s)?|dream\s+team)\s*[:#-]\s*"
    r"([A-Za-z0-9][A-Za-z0-9 -]{0,30}?\d{1,3})\b"
)
group_parenthetical = re.compile(r"(?i)group\s+\d+\s*\(([A-Za-z0-9-]{1,16})\)")
group_plain = re.compile(
    r"\b(?:Group\s+\d{1,3}|Assignment\s+group\s+\d{1,3}|Project\s+Groups?\s+\d{1,3}|Labs?\s+\d{1,3}|Clinics?\s+\d{1,3}|Dream\s+Team\s+\d{1,3})\b",
    re.IGNORECASE,
)
# Keep short-code support conservative to avoid catching generic tokens
# such as T1/T2 or MS1 used for tasks/milestones in templates.
group_short_code = re.compile(r"\b(?:[A-Z]{3,5}\d{1,2}|[A-Z]{2}\d{2,3}|T\d{2,3})\b")

group_context_hint = re.compile(
    # Use explicit group-like cues only. A generic word such as "project" is
    # too broad in project-plan bodies and caused task IDs such as T1/T2/MS1
    # to be treated as group labels.
    r"(?i)\b(?:group|team|lab(?:s)?|clinic(?:s)?|assignment\s+group|project\s+groups?|dream\s+team)\b"
)

task_code = re.compile(r"(?i)^(?:T|MS)\d{1,2}$")
task_context_hint = re.compile(
    r"(?i)\b(?:task|milestone|work\s*package|gantt|deliverable|phase|sprint|timeline|planning)\b"
)
explicit_group_left_context = re.compile(
    r"(?i)(?:group|team|lab(?:s)?|clinic(?:s)?|assignment\s+group|project\s+groups?|dream\s+team)"
    r"(?:\s+(?:label|id|ids|number|numbers?))?\s*[:#-]?\s*$"
)

ROLE_VALUE = re.compile(
    r"(?i)(?:(?:role|roles|position|positions)\s*:\s*)"
    r"(student|students|tutor|teaching assistant|assistant|coordinator|customer|lead|owner|author|reviewer|chair|secretary)"
)

# In the generated project-plan cover pages, role annotations mark the field
# labels themselves, for example "Customer", "Coordinator", and "Tutor" in
# "Customer: Alice Smith". The old detector only looked for values after a
# literal "Role:" prefix, which made the Role label score 0 recall.
ROLE_LABEL = re.compile(r"(?i)\b(customer|coordinator|tutor)\s*:")

# Conservative name patterns used only when direct-name redaction is enabled.
name_field = re.compile(
    r"(?i)(?:name|author|owner|coordinator|chair|secretary|participant|attendee)\s*:\s*"
    r"([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})"
)
name_titled = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3}\b"
)


def _configured_labels(cfg):
    return {canon(label) for label in getattr(cfg, "labels", ())}


def _enabled(cfg, label):
    label = canon(label)
    if label in {"Name", "FULL_NAME", "FIRST_NAME", "LAST_NAME", "Student number", "Institutional email"}:
        return bool(getattr(cfg, "direct", True))
    return bool(getattr(cfg, "quasi", False)) and label in _configured_labels(cfg)


def _yield(block, pattern, label, source, score):
    for match in pattern.finditer(block.text):
        yield Occurrence(
            label=label,
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            region=block.region,
            path=block.path,
            score=score,
            source=source,
        )


def _yield_group(block, pattern, label, source, score, group=1):
    for match in pattern.finditer(block.text):
        yield Occurrence(
            label=label,
            surface=match.group(group),
            start=match.start(group),
            end=match.end(group),
            region=block.region,
            path=block.path,
            score=score,
            source=source,
        )


def _strictish(block):
    return normalize_region(block.region) in {"cover", "header", "footer"}


def _date_context(text, region):
    return (
        re.search(r"(?i)\b(?:date|deadline|due|meeting|created|updated|version)\b", text) is not None
        or region in {"cover", "header", "footer", "body_table", "ocr"}
    )


def _group_short_code_has_context(text, start, end, window=28):
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return bool(group_context_hint.search(snippet))


def _valid_group_candidate(text, surface, start, end):
    """Reject task/milestone IDs that resemble compact group labels."""
    value = str(surface or "").strip()
    if not value:
        return False
    if task_code.fullmatch(value):
        left = text[max(0, start - 36):start]
        window = text[max(0, start - 48):min(len(text), end + 48)]
        if task_context_hint.search(window) and not explicit_group_left_context.search(left):
            return False
    return True


def detect(doc, cfg):
    detections = []
    for block in doc.blocks:
        region = normalize_region(block.region)
        text = block.text

        if _enabled(cfg, "Institutional email"):
            detections.extend(_yield(block, email, "Institutional email", "pattern_email", 0.99))

        if _enabled(cfg, "Student number"):
            for match in student.finditer(text):
                if len(re.sub(r"\D", "", match.group(0))) == student_number_digits:
                    detections.append(
                        Occurrence(
                            label="Student number",
                            surface=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            region=block.region,
                            path=block.path,
                            score=0.995,
                            source="pattern_student_number",
                        )
                    )

        if _enabled(cfg, "Course code"):
            detections.extend(_yield(block, course, "Course code", "pattern_course", 0.95))

        if _enabled(cfg, "Programme"):
            for match in programme_field.finditer(text):
                detections.append(
                    Occurrence(
                        label="Programme",
                        surface=match.group(1),
                        start=match.start(1),
                        end=match.end(1),
                        region=block.region,
                        path=block.path,
                        score=0.94 if _strictish(block) else 0.88,
                        source="pattern_programme",
                    )
                )
            detections.extend(_yield(block, programme_value, "Programme", "pattern_programme", 0.92 if _strictish(block) else 0.86))

        if _enabled(cfg, "Group label"):
            for occ in _yield_group(block, group_field, "Group label", "pattern_group", 0.90 if _strictish(block) else 0.84):
                if _valid_group_candidate(text, occ.surface, occ.start, occ.end):
                    detections.append(occ)
            for occ in _yield_group(block, group_parenthetical, "Group label", "pattern_group", 0.86 if _strictish(block) else 0.80):
                if _valid_group_candidate(text, occ.surface, occ.start, occ.end):
                    detections.append(occ)
            for occ in _yield(block, group_plain, "Group label", "pattern_group", 0.90 if _strictish(block) else 0.84):
                if _valid_group_candidate(text, occ.surface, occ.start, occ.end):
                    detections.append(occ)
            for match in group_short_code.finditer(text):
                # Keep short-code matching conservative: only accept when a nearby
                # group-like cue exists to avoid broad alphanumeric overmatching.
                if not _group_short_code_has_context(text, match.start(), match.end()):
                    continue
                if not _valid_group_candidate(text, match.group(0), match.start(), match.end()):
                    continue
                detections.append(
                    Occurrence(
                        label="Group label",
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        region=block.region,
                        path=block.path,
                        score=0.86 if _strictish(block) else 0.80,
                        source="pattern_group",
                    )
                )

        if _enabled(cfg, "Date") and _date_context(text, region):
            for pattern in date_patterns:
                detections.extend(_yield(block, pattern, "Date", "pattern_date", 0.90 if _strictish(block) else 0.86))

        if _enabled(cfg, "Role"):
            for match in ROLE_VALUE.finditer(text):
                detections.append(
                    Occurrence(
                        label="Role",
                        surface=match.group(1),
                        start=match.start(1),
                        end=match.end(1),
                        region=block.region,
                        path=block.path,
                        score=0.88 if _strictish(block) else 0.82,
                        source="pattern_role",
                    )
                )
            for match in ROLE_LABEL.finditer(text):
                detections.append(
                    Occurrence(
                        label="Role",
                        surface=match.group(1),
                        start=match.start(1),
                        end=match.end(1),
                        region=block.region,
                        path=block.path,
                        score=0.92 if _strictish(block) else 0.86,
                        source="pattern_role_label",
                    )
                )

    return detections
