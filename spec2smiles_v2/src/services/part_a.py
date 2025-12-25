"""Part A service - Spectrum to Descriptors (model-agnostic).

Supports multiple model backends:
- lgbm: LightGBM ensemble (fast, lightweight)
- transformer: SpectrumTransformer (deep learning)
"""

import json
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np

from src.config import settings
from src.models.lgbm_ensemble import LGBMEnsemble
from src.models.transformer_wrapper import TransformerWrapper
from src.services.preprocessor import PreprocessorService
from src.utils.exceptions import ModelError

ModelType = Literal["lgbm", "transformer"]


class PartAService:
    """Model-agnostic service for Spectrum -> Descriptors prediction.

    Supports switching between LightGBM and Transformer models via
    model_type parameter or SPEC2SMILES_PART_A_MODEL env variable.
    """

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        preprocessor: Optional[PreprocessorService] = None,
    ):
        """Initialize Part A service.

        Args:
            model_type: Model to use ("lgbm" or "transformer"). Defaults to settings.
            preprocessor: Preprocessor service (created if not provided)
        """
        self.model_type: ModelType = model_type or settings.part_a_model
        self.preprocessor = preprocessor or PreprocessorService()
        self.model: Optional[Union[LGBMEnsemble, TransformerWrapper]] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained

    def _create_model(self) -> Union[LGBMEnsemble, TransformerWrapper]:
        """Factory method to create model based on model_type."""
        if self.model_type == "lgbm":
            return LGBMEnsemble()
        elif self.model_type == "transformer":
            return TransformerWrapper()
        else:
            raise ModelError(f"Unknown model type: {self.model_type}")

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
            log_dir: Optional directory for live epoch logging (Transformer only)

        Returns:
            Dictionary of per-descriptor metrics
        """
        if verbose:
            print(f"Training Part A with model: {self.model_type}")

        # Fit descriptor scaler on training data
        self.preprocessor.fit_scaler(y_train)

        # Create and train model using factory
        self.model = self._create_model()

        # TransformerWrapper supports log_dir, LGBMEnsemble does not
        if self.model_type == "transformer" and log_dir is not None:
            self.model.fit(X_train, y_train, X_val, y_val, verbose=verbose, log_dir=log_dir)
        else:
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

        # Save model type metadata
        metadata = {"model_type": self.model_type}
        with open(output_dir / "model_type.json", "w") as f:
            json.dump(metadata, f)

        # Save model based on type
        if self.model_type == "lgbm":
            self.model.save(output_dir / "lgbm_ensemble.pkl")
        else:  # transformer
            self.model.save(output_dir / "transformer.pt")

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

        # Detect model type from metadata or files
        metadata_path = model_dir / "model_type.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.model_type = metadata["model_type"]
        elif (model_dir / "lgbm_ensemble.pkl").exists():
            self.model_type = "lgbm"
        elif (model_dir / "transformer.pt").exists():
            self.model_type = "transformer"
        else:
            raise ModelError(f"No model found in: {model_dir}")

        # Load model based on type
        if self.model_type == "lgbm":
            model_path = model_dir / "lgbm_ensemble.pkl"
            if not model_path.exists():
                raise ModelError(f"Model not found: {model_path}")
            self.model = LGBMEnsemble.load(model_path)
        else:  # transformer
            model_path = model_dir / "transformer.pt"
            if not model_path.exists():
                raise ModelError(f"Model not found: {model_path}")
            self.model = TransformerWrapper.load(model_path)

        # Load scaler if exists
        scaler_path = model_dir / "descriptor_scaler.pkl"
        if scaler_path.exists():
            self.preprocessor.load_scaler(scaler_path)

        self._trained = True
        return self
