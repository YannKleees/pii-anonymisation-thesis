"""Top-level package exports with lazy imports."""

__all__ = [
    "DatasetManager",
    "DefenderPipeline",
    "Config",
    "Experiments",
    "frame",
    "train",
    "iter_docx_inputs",
    "iter_document_inputs",
    "process_document",
    "process_many",
    "write_audit_jsonl",
]


def __getattr__(name):
    """Lazily import and return exports from submodules."""
    if name == "DatasetManager":
        from .dataset import DatasetManager

        return DatasetManager
    if name == "Config":
        from .shared import Config

        return Config
    if name == "Experiments":
        from .experiments import Experiments

        return Experiments
    if name in {"DefenderPipeline", "iter_docx_inputs", "iter_document_inputs", "process_document", "process_many", "write_audit_jsonl"}:
        from . import defender

        return getattr(defender, name)
    if name in {"frame", "train"}:
        from . import training

        return getattr(training, name)
    raise AttributeError(name)

