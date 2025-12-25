"""Directory and path utilities for consistent file handling.

Provides centralized functions for:
- Creating directories on-demand
- Validating input paths
- Initializing output directory structure
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import Settings


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        The same path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    """Create parent directory for a file path.

    Args:
        path: File path whose parent should exist

    Returns:
        The same path (for chaining)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_input_dir(path: Path, name: str = "Input") -> None:
    """Validate that an input directory exists.

    Args:
        path: Directory path to validate
        name: Descriptive name for error messages

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {path}")


def validate_input_file(path: Path, name: str = "Input") -> None:
    """Validate that an input file exists.

    Args:
        path: File path to validate
        name: Descriptive name for error messages

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{name} path is not a file: {path}")


def init_output_dirs(settings: "Settings") -> None:
    """Initialize all output directories from settings.

    Creates the following directories:
    - models_path (data/output/models)
    - metrics_path (data/output/metrics)
    - figures_path (data/output/figures)
    - logs_path (logs)

    Args:
        settings: Settings instance with path properties
    """
    ensure_dir(settings.models_path)
    ensure_dir(settings.metrics_path)
    ensure_dir(settings.figures_path)
    ensure_dir(settings.logs_path)


def clean_output_dirs(settings: "Settings", keep_structure: bool = True) -> None:
    """Remove contents of output directories.

    Args:
        settings: Settings instance with path properties
        keep_structure: If True, recreate empty directories after cleaning
    """
    import shutil

    for path in [settings.models_path, settings.metrics_path, settings.figures_path]:
        if path.exists():
            shutil.rmtree(path)
        if keep_structure:
            ensure_dir(path)
