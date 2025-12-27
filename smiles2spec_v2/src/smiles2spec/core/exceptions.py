"""Custom exceptions for SMILES2SPEC pipeline.

Exception hierarchy:
    Smiles2SpecError (base)
    ├── ConfigurationError - Invalid configuration
    ├── DataError - Data loading/processing issues
    ├── FeatureError - Feature extraction failures
    ├── ModelError - Model training/loading issues
    └── InferenceError - Prediction failures
"""


class Smiles2SpecError(Exception):
    """Base exception for SMILES2SPEC pipeline."""

    pass


class ConfigurationError(Smiles2SpecError):
    """Raised when configuration is invalid or missing."""

    pass


class DataError(Smiles2SpecError):
    """Raised when data loading or processing fails."""

    pass


class FeatureError(Smiles2SpecError):
    """Raised when feature extraction fails."""

    pass


class ModelError(Smiles2SpecError):
    """Raised when model training or loading fails."""

    pass


class InferenceError(Smiles2SpecError):
    """Raised when prediction/inference fails."""

    pass
