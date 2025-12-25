"""Part A service - Spectrum to Descriptors using LightGBM ensemble."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import settings
from src.models.lgbm_ensemble import LGBMEnsemble
from src.services.preprocessor import PreprocessorService
from src.utils.exceptions import ModelError


class PartAService:
    """Service for training and inference of Spectrum -> Descriptors model."""

    def __init__(self, preprocessor: Optional[PreprocessorService] = None):
        """Initialize Part A service.

        Args:
            preprocessor: Preprocessor service (created if not provided)
        """
        self.preprocessor = preprocessor or PreprocessorService()
        self.model: Optional[LGBMEnsemble] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Train LightGBM ensemble on spectral data.

        Args:
            X_train: Training spectra of shape (n_samples, n_bins)
            y_train: Training descriptors of shape (n_samples, n_descriptors)
            X_val: Optional validation spectra
            y_val: Optional validation descriptors
            verbose: Whether to show progress

        Returns:
            Dictionary of per-descriptor metrics
        """
        # Fit descriptor scaler on training data
        self.preprocessor.fit_scaler(y_train)

        # Create and train model
        self.model = LGBMEnsemble()
        self.model.fit(X_train, y_train, X_val, y_val, verbose=verbose)
        self._trained = True

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            metrics = self.model.evaluate(X_val, y_val)
        else:
            metrics = self.model.evaluate(X_train, y_train)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict molecular descriptors from spectra.

        Args:
            X: Input spectra of shape (n_samples, n_bins)

        Returns:
            Predicted descriptors of shape (n_samples, n_descriptors)

        Raises:
            ModelError: If model is not trained
        """
        if not self._trained:
            raise ModelError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        """Predict and scale descriptors for Part B input.

        Args:
            X: Input spectra of shape (n_samples, n_bins)

        Returns:
            Scaled predicted descriptors
        """
        predictions = self.predict(X)
        if self.preprocessor.is_scaler_fitted:
            return self.preprocessor.transform_descriptors(predictions)
        return predictions

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model on test data.

        Args:
            X: Test spectra
            y_true: True descriptors

        Returns:
            Dictionary mapping descriptor names to metric dictionaries
        """
        if not self._trained:
            raise ModelError("Model must be trained before evaluation")
        return self.model.evaluate(X, y_true)

    def get_summary_metrics(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Get aggregate performance metrics.

        Args:
            X: Test spectra
            y_true: True descriptors

        Returns:
            Dictionary with mean/median R2 and best/worst descriptors
        """
        if not self._trained:
            raise ModelError("Model must be trained before evaluation")
        return self.model.get_summary_metrics(X, y_true)

    # ===========================================
    # Persistence
    # ===========================================
    def save(self, output_dir: Path) -> None:
        """Save trained model and scaler.

        Args:
            output_dir: Output directory
        """
        if not self._trained:
            raise ModelError("Cannot save untrained model")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(output_dir / "lgbm_ensemble.pkl")

        # Save scaler
        self.preprocessor.save_scaler(output_dir / "descriptor_scaler.pkl")

    def load(self, model_dir: Path) -> "PartAService":
        """Load trained model and scaler.

        Args:
            model_dir: Directory containing model files

        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)

        # Load model
        model_path = model_dir / "lgbm_ensemble.pkl"
        if not model_path.exists():
            raise ModelError(f"Model not found: {model_path}")
        self.model = LGBMEnsemble.load(model_path)

        # Load scaler if exists
        scaler_path = model_dir / "descriptor_scaler.pkl"
        if scaler_path.exists():
            self.preprocessor.load_scaler(scaler_path)

        self._trained = True
        return self
