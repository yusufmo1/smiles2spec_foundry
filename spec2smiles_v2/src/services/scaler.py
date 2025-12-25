"""Unified descriptor scaling service.

Ensures Part A and Part B use identical scaling, fixing the dual-scaler bug
where different StandardScaler instances caused inference failures.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import ModelError


class ScalerService:
    """Central scaler management for Part A and Part B.

    Provides singleton access to ensure both pipeline components use
    identical descriptor scaling during training and inference.

    Example:
        # During training (fit and save)
        scaler = ScalerService()
        scaler.fit(descriptors)
        scaler.save(output_dir / "scaler.pkl")

        # During inference (load and transform)
        scaler = ScalerService()
        scaler.load(model_dir / "scaler.pkl")
        scaled = scaler.transform(descriptors)
    """

    _instance: Optional["ScalerService"] = None

    def __init__(self):
        """Initialize scaler service."""
        self._scaler: Optional[StandardScaler] = None
        self._scaler_path: Optional[Path] = None

    @classmethod
    def get_instance(cls) -> "ScalerService":
        """Get singleton instance.

        Returns:
            Shared ScalerService instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        return self._scaler is not None

    def fit(self, descriptors: np.ndarray) -> "ScalerService":
        """Fit scaler on descriptor data.

        Args:
            descriptors: Array of shape (n_samples, n_descriptors)

        Returns:
            Self for method chaining
        """
        self._scaler = StandardScaler()
        self._scaler.fit(descriptors)
        return self

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Transform descriptors using fitted scaler.

        Args:
            descriptors: Array of shape (n_samples, n_descriptors)

        Returns:
            Scaled descriptors

        Raises:
            ModelError: If scaler not fitted
        """
        if self._scaler is None:
            raise ModelError("Scaler not fitted. Call fit() or load() first.")
        return self._scaler.transform(descriptors)

    def inverse_transform(self, scaled_descriptors: np.ndarray) -> np.ndarray:
        """Inverse transform scaled descriptors.

        Args:
            scaled_descriptors: Scaled array of shape (n_samples, n_descriptors)

        Returns:
            Original-scale descriptors

        Raises:
            ModelError: If scaler not fitted
        """
        if self._scaler is None:
            raise ModelError("Scaler not fitted. Call fit() or load() first.")
        return self._scaler.inverse_transform(scaled_descriptors)

    def fit_transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Fit scaler and transform in one step.

        Args:
            descriptors: Array of shape (n_samples, n_descriptors)

        Returns:
            Scaled descriptors
        """
        self.fit(descriptors)
        return self.transform(descriptors)

    def save(self, path: Path) -> None:
        """Save fitted scaler to disk.

        Args:
            path: Output path for scaler pickle

        Raises:
            ModelError: If scaler not fitted
        """
        if self._scaler is None:
            raise ModelError("Cannot save unfitted scaler.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._scaler, f)

        self._scaler_path = path

    def load(self, path: Path) -> "ScalerService":
        """Load fitted scaler from disk.

        Args:
            path: Path to scaler pickle

        Returns:
            Self for method chaining

        Raises:
            ModelError: If file not found
        """
        path = Path(path)
        if not path.exists():
            raise ModelError(f"Scaler not found: {path}")

        with open(path, "rb") as f:
            self._scaler = pickle.load(f)

        self._scaler_path = path
        return self

    def get_state(self) -> dict:
        """Get scaler state for serialization.

        Returns:
            Dictionary with scaler parameters
        """
        if self._scaler is None:
            return {}

        return {
            "mean_": self._scaler.mean_.tolist(),
            "scale_": self._scaler.scale_.tolist(),
            "var_": self._scaler.var_.tolist(),
            "n_features_in_": self._scaler.n_features_in_,
            "n_samples_seen_": int(self._scaler.n_samples_seen_),
        }

    def from_state(self, state: dict) -> "ScalerService":
        """Restore scaler from serialized state.

        Args:
            state: Dictionary from get_state()

        Returns:
            Self for method chaining
        """
        if not state:
            return self

        self._scaler = StandardScaler()
        self._scaler.mean_ = np.array(state["mean_"])
        self._scaler.scale_ = np.array(state["scale_"])
        self._scaler.var_ = np.array(state["var_"])
        self._scaler.n_features_in_ = state["n_features_in_"]
        self._scaler.n_samples_seen_ = state["n_samples_seen_"]

        return self
