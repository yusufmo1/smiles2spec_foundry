"""Spectrum preprocessing service."""

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.domain.spectrum import process_spectrum
from src.utils.exceptions import ModelError


class PreprocessorService:
    """Process and transform spectral data."""

    def __init__(self):
        """Initialize preprocessor with settings."""
        self.n_bins = settings.n_bins
        self.bin_width = settings.bin_width
        self.max_mz = settings.max_mz
        self.transform = settings.transform
        self.normalize = settings.normalize
        self._scaler = StandardScaler()
        self._scaler_fitted = False

    def process_peaks(self, peaks: List[Tuple[float, float]]) -> np.ndarray:
        """Convert peak list to processed spectrum.

        Args:
            peaks: List of (m/z, intensity) tuples

        Returns:
            Processed spectrum array of shape (n_bins,)
        """
        return process_spectrum(
            peaks,
            n_bins=self.n_bins,
            bin_width=self.bin_width,
            max_mz=self.max_mz,
            transform=self.transform,
            normalize=self.normalize,
        )

    def process_batch(
        self, peak_lists: List[List[Tuple[float, float]]]
    ) -> np.ndarray:
        """Process multiple spectra.

        Args:
            peak_lists: List of peak lists

        Returns:
            Array of shape (n_samples, n_bins)
        """
        return np.array([self.process_peaks(peaks) for peaks in peak_lists])

    # ===========================================
    # Descriptor Scaling
    # ===========================================
    @property
    def is_scaler_fitted(self) -> bool:
        """Check if descriptor scaler is fitted."""
        return self._scaler_fitted

    def fit_scaler(self, descriptors: np.ndarray) -> "PreprocessorService":
        """Fit descriptor scaler on training data.

        Args:
            descriptors: Training descriptor array of shape (n_samples, n_descriptors)

        Returns:
            Self for method chaining
        """
        self._scaler.fit(descriptors)
        self._scaler_fitted = True
        return self

    def transform_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Transform descriptors to standardized values.

        Args:
            descriptors: Descriptor array of shape (n_samples, n_descriptors)

        Returns:
            Standardized descriptors with zero mean and unit variance

        Raises:
            ModelError: If scaler has not been fitted
        """
        if not self._scaler_fitted:
            raise ModelError("Scaler must be fitted before transform")
        return self._scaler.transform(descriptors)

    def fit_transform_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            descriptors: Training descriptor array

        Returns:
            Standardized descriptors
        """
        return self.fit_scaler(descriptors).transform_descriptors(descriptors)

    def inverse_transform_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert standardized descriptors back to original scale.

        Args:
            descriptors: Standardized descriptor array

        Returns:
            Descriptors in original scale

        Raises:
            ModelError: If scaler has not been fitted
        """
        if not self._scaler_fitted:
            raise ModelError("Scaler must be fitted before inverse_transform")
        return self._scaler.inverse_transform(descriptors)

    # ===========================================
    # Persistence
    # ===========================================
    def save_scaler(self, path: Path) -> None:
        """Save fitted scaler to disk.

        Args:
            path: Output path for pickle file
        """
        if not self._scaler_fitted:
            raise ModelError("Cannot save unfitted scaler")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)

    def load_scaler(self, path: Path) -> "PreprocessorService":
        """Load fitted scaler from disk.

        Args:
            path: Path to pickle file

        Returns:
            Self for method chaining
        """
        path = Path(path)
        if not path.exists():
            raise ModelError(f"Scaler file not found: {path}")

        self._scaler = joblib.load(path)
        self._scaler_fitted = True
        return self
