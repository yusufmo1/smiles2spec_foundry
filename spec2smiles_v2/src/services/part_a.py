"""Part A service - Spectrum to Descriptors.

Uses the Hybrid CNN-Transformer model for spectrum-to-descriptor prediction,
combining local (CNN) and global (Transformer) pattern recognition.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.config import settings
from src.models.registry import ModelRegistry
from src.services.scaler import ScalerService
from src.utils.exceptions import ModelError

# Import model to trigger registration (required for decorator to run)
import src.models.hybrid_wrapper  # noqa: F401

# Model filename
MODEL_FILE = "hybrid.pt"


class PartAService:
    """Service for Spectrum -> Descriptors prediction.

    Uses Hybrid CNN-Transformer model combining local (CNN) and global
    (Transformer) pattern recognition for spectrum analysis.
    """

    def __init__(
        self,
        scaler: Optional[ScalerService] = None,
    ):
        """Initialize Part A service.

        Args:
            scaler: Scaler service (uses singleton if not provided)
        """
        self.scaler = scaler or ScalerService.get_instance()
        self.model: Optional[Any] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained

    def _create_model(self) -> Any:
        """Create Hybrid CNN-Transformer model instance using registry."""
        model_class = ModelRegistry.get("hybrid")
        return model_class()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        log_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train model on spectral data.

        Args:
            X_train: Training spectra of shape (n_samples, n_bins)
            y_train: Training descriptors of shape (n_samples, n_descriptors)
            X_val: Optional validation spectra
            y_val: Optional validation descriptors
            verbose: Whether to show progress
            log_dir: Optional directory for live epoch logging

        Returns:
            Dictionary of per-descriptor metrics
        """
        if verbose:
            print("Training Part A with Hybrid CNN-Transformer model")

        # Fit scaler on training descriptors
        self.scaler.fit(y_train)

        # Create model via registry
        self.model = self._create_model()

        # Train with appropriate parameters based on model type
        if hasattr(self.model, "fit"):
            fit_params = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "verbose": verbose,
            }
            # Add log_dir for neural models
            if log_dir is not None:
                fit_params["log_dir"] = log_dir

            self.model.fit(**fit_params)

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
        if self.scaler.is_fitted:
            return self.scaler.transform(predictions)
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

        # Save model type metadata
        metadata = {"model_type": "hybrid"}
        with open(output_dir / "model_type.json", "w") as f:
            json.dump(metadata, f)

        # Save model
        self.model.save(output_dir / MODEL_FILE)

        # Save scaler (shared with Part B)
        self.scaler.save(output_dir / "descriptor_scaler.pkl")

    def load(self, model_dir: Path) -> "PartAService":
        """Load trained model and scaler.

        Args:
            model_dir: Directory containing model files

        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILE

        if not model_path.exists():
            raise ModelError(f"Model not found: {model_path}")

        # Load model using registry
        model_class = ModelRegistry.get("hybrid")
        self.model = model_class.load(model_path)

        # Load scaler
        scaler_path = model_dir / "descriptor_scaler.pkl"
        if scaler_path.exists():
            self.scaler.load(scaler_path)

        self._trained = True
        return self
