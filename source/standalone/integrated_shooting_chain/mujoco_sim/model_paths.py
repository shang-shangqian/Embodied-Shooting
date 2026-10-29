"""Resolve model assets separately from the checkout and result directories."""
from pathlib import Path


def resolve_model_path(value, *, model_root, project_root):
    """Rebase relative and checkout-local paths; preserve other absolute paths.

    Never fall back to a second copy when an asset is absent from model_root.
    This also handles absolute paths embedded in older training_settings.json.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.relative_to(Path(project_root))
        except ValueError:
            return path.resolve()
    return (Path(model_root) / path).resolve()
