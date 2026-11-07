"""Defender package exports with lazy pipeline imports."""


class DefenderPipeline:
    """Lightweight wrapper around process_document/process_many for compatibility."""

    def __init__(self, config=None):
        """Initialize with optional config."""
        from ..shared import Config
        self.config = config or Config()

    def process_document(self, document_path):
        """Process one document, returning redacted result."""
        from .pipeline import process_document
        return process_document(document_path, self.config)

    def process_many(self, paths):
        """Process multiple documents."""
        from .pipeline import process_many
        return process_many(paths, self.config)


__all__ = [
    "DefenderPipeline", "iter_docx_inputs", "iter_document_inputs",
    "process_document", "process_many", "write_audit_jsonl",
]


def __getattr__(name):
    """Lazily import and return exports from pipeline module."""
    if name in {"iter_docx_inputs", "iter_document_inputs", "process_document", "process_many", "write_audit_jsonl"}:
        import importlib
        pipeline = importlib.import_module(".pipeline", __name__)
        return getattr(pipeline, name)
    raise AttributeError(name)

