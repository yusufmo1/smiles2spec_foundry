"""Custom exceptions for SPEC2SMILES pipeline."""


class Spec2SmilesError(Exception):
    """Base exception for SPEC2SMILES pipeline."""

    pass


class ConfigurationError(Spec2SmilesError):
    """Raised when configuration is invalid."""

    pass


class DataError(Spec2SmilesError):
    """Raised when data loading or processing fails."""

    pass


class ModelError(Spec2SmilesError):
    """Raised when model operations fail."""

    pass


class InferenceError(Spec2SmilesError):
    """Raised when inference fails."""

    pass
