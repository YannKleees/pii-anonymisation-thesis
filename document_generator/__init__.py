"""Document generator package exports with lazy imports."""

__all__ = ["generate_dataset"]


def __getattr__(name):
    """Lazily import and return generate_dataset."""
    if name == "generate_dataset":
        from .dataset import generate_dataset
        return generate_dataset
    raise AttributeError(name)

