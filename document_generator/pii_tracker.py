"""PII tracking.

The generator records PII at write time, not while payload strings are being
assembled. Location hints are optional but allow the annotation builder to map
surfaces back to the exact paragraph that was written.
"""

from collections.abc import Sequence
from dataclasses import dataclass

PIIPart = tuple[str, str, str, str, str]
@dataclass(frozen=True)
class EmittedPII:
    """A piece of sensitive text that was actually written to the document."""

    label: str
    surface: str
    entity_type: str
    entity_id: str
    quality: str
    region: str = ""
    path: str = ""
    container_text: str = ""


class PIITracker:
    """Tracks PII emissions for one generated document."""

    def __init__(self):
        self._emissions: list[EmittedPII] = []

    def emit(
        self,
        label: str,
        surface: str,
        entity_type: str,
        entity_id: str,
        quality: str = "clean",
        *,
        region: str = "",
        path: str = "",
        container_text: str = "",
    ):
        self._emissions.append(
            EmittedPII(
                label=label,
                surface=surface,
                entity_type=entity_type,
                entity_id=entity_id,
                quality=quality,
                region=region,
                path=path,
                container_text=container_text,
            )
        )

    def track(self, text: str, pii_parts: Sequence[PIIPart], *, region: str = "", path: str = ""):
        """Emit all PII parts that are actually present in the provided text."""
        for label, surface, entity_type, entity_id, quality in pii_parts:
            if surface and surface in text:
                self.emit(
                    label,
                    surface,
                    entity_type,
                    entity_id,
                    quality,
                    region=region,
                    path=path,
                    container_text=text,
                )
                if label == "FULL_NAME":
                    parts = [part for part in str(surface).split() if part]
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_name = parts[-1]
                        if first_name and first_name in text:
                            self.emit(
                                "FIRST_NAME",
                                first_name,
                                entity_type,
                                entity_id,
                                quality,
                                region=region,
                                path=path,
                                container_text=text,
                            )
                        if last_name and last_name != first_name and last_name in text:
                            self.emit(
                                "LAST_NAME",
                                last_name,
                                entity_type,
                                entity_id,
                                quality,
                                region=region,
                                path=path,
                                container_text=text,
                            )

    def emissions(self):
        """Return a copy of the recorded emissions."""
        return list(self._emissions)
