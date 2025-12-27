"""Configuration loading utilities.

Handles YAML config file loading and dataclass conversion.
"""

from pathlib import Path
from typing import Optional, Type, TypeVar

import yaml

T = TypeVar("T")


def dict_to_dataclass(cls: Type[T], data: dict) -> T:
    """Recursively convert dict to dataclass.

    Args:
        cls: Dataclass type to convert to
        data: Dictionary with field values

    Returns:
        Instance of cls with values from data
    """
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def find_config_path(config_path: Optional[Path] = None) -> Optional[Path]:
    """Find config.yml file.

    Args:
        config_path: Explicit path to config file, or None to search

    Returns:
        Path to config file, or None if not found
    """
    if config_path is not None:
        config_path = Path(config_path)  # Handle both str and Path
        return config_path if config_path.exists() else None

    possible_paths = [
        Path("config.yml"),
        Path(__file__).parent.parent.parent.parent / "config.yml",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def load_yaml(config_path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Dictionary with configuration data
    """
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: Optional[Path] = None):
    """Load settings from config file.

    This is a convenience wrapper that imports and calls load_config
    to avoid circular imports.

    Args:
        config_path: Optional path to config file

    Returns:
        Settings instance
    """
    from smiles2spec.core.config import load_config
    return load_config(config_path)
