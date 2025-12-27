"""SMILES2SPEC: SMILES to Mass Spectrum Prediction Pipeline.

A production-grade pipeline for predicting mass spectra from molecular structures.

Best Performance:
    - Bin-by-bin Ensemble: 0.8164 cosine similarity
    - Simple Weighted Ensemble: 0.8037
    - Random Forest: 0.7837
    - ModularNet: 0.7691
"""

from smiles2spec.core.config import load_config, reload_config, settings
from smiles2spec.core.exceptions import (
    ConfigurationError,
    DataError,
    FeatureError,
    InferenceError,
    ModelError,
    Smiles2SpecError,
)

__version__ = "0.1.0"
__all__ = [
    "settings",
    "load_config",
    "reload_config",
    "Smiles2SpecError",
    "ConfigurationError",
    "DataError",
    "FeatureError",
    "ModelError",
    "InferenceError",
]
