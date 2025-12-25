"""Part A service - Spectrum to Descriptors (model-agnostic).

Supports multiple model backends via registry pattern:
- lgbm: LightGBM ensemble (fast, lightweight)
- transformer: SpectrumTransformer (deep learning)
- hybrid: CNN-Transformer hybrid (local + global patterns)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.config import settings
from src.models.registry import ModelRegistry
from src.services.scaler import ScalerService
from src.utils.exceptions import ModelError

# Import models to trigger registration (required for decorator to run)
import src.models.lgbm_ensemble  # noqa: F401
import src.models.transformer_wrapper  # noqa: F401

# Model file mapping for save/load
MODEL_FILES = {
    "lgbm": "lgbm_ensemble.pkl",
    "transformer": "transformer.pt",
    "hybrid": "hybrid.pt",
}


class PartAService:
    """Model-agnostic service for Spectrum -> Descriptors prediction.

    Uses registry pattern for clean model switching. All models share
    the same interface (fit, predict, evaluate, save, load).
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        scaler: Optional[ScalerService] = None,
    ):
        """Initialize Part A service.

        Args:
            model_type: Model to use (lgbm, transformer, hybrid). Defaults to settings.
            scaler: Scaler service (uses singleton if not provided)
        """
        self.model_type = model_type or settings.part_a_model
        self.scaler = scaler or ScalerService.get_instance()
        self.model: Optional[Any] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained

    def _create_model(self) -> Any:
        """Create model instance using registry."""
        model_class = ModelRegistry.get(self.model_type)
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
            print(f"Training Part A with model: {self.model_type}")
            print(f"  Available models: {ModelRegistry.available()}")

        # Fit scaler on training descriptors
        self.scaler.fit(y_train)

        # Create model via registry
        self.model = self._create_model()

        # Train with appropriate parameters based on model type
        # Check if model supports log_dir (neural models do, LGBM doesn't)
        if hasattr(self.model, "fit"):
            fit_params = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "verbose": verbose,
            }
            # Add log_dir for models that support it
            if log_dir is not None and self.model_type in ("transformer", "hybrid"):
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
        metadata = {"model_type": self.model_type}
        with open(output_dir / "model_type.json", "w") as f:
            json.dump(metadata, f)

        # Save model using type-specific filename
        model_file = MODEL_FILES.get(self.model_type, f"{self.model_type}_model.pkl")
        self.model.save(output_dir / model_file)

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

        # Detect model type from metadata or files
        metadata_path = model_dir / "model_type.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.model_type = metadata["model_type"]
        else:
            # Fallback: detect from file existence
            for model_name, model_file in MODEL_FILES.items():
                if (model_dir / model_file).exists():
                    self.model_type = model_name
                    break
            else:
                raise ModelError(f"No model found in: {model_dir}")

        # Load model using registry
        model_class = ModelRegistry.get(self.model_type)
        model_file = MODEL_FILES.get(self.model_type)
        model_path = model_dir / model_file

        if not model_path.exists():
            raise ModelError(f"Model not found: {model_path}")

        self.model = model_class.load(model_path)

        # Load scaler
        scaler_path = model_dir / "descriptor_scaler.pkl"
        if scaler_path.exists():
            self.scaler.load(scaler_path)

        self._trained = True
        return self
