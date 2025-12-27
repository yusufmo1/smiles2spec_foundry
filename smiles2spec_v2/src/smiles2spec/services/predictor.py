"""Prediction service for SMILES to spectrum.

Handles inference with trained models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from smiles2spec.core.exceptions import InferenceError
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.domain.spectrum import SpectrumBinner, inverse_transform
from smiles2spec.models.base import BaseModel
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.services.featurizer import FeaturizationService


class PredictionService:
    """Service for generating spectrum predictions from SMILES."""

    def __init__(
        self,
        model: BaseModel,
        featurizer: FeaturizationService,
        preprocessor: Optional[FeaturePreprocessor] = None,
        binner: Optional[SpectrumBinner] = None,
    ):
        """Initialize prediction service.

        Args:
            model: Trained prediction model
            featurizer: Feature extraction service
            preprocessor: Optional feature preprocessor
            binner: Optional spectrum binner for peak conversion
        """
        self.model = model
        self.featurizer = featurizer
        self.preprocessor = preprocessor
        self.binner = binner or SpectrumBinner()

    @classmethod
    def from_directory(
        cls,
        model_dir: Union[str, Path],
        model_name: str = "random_forest",
        featurizer: Optional[FeaturizationService] = None,
    ) -> "PredictionService":
        """Create service from saved model directory.

        Args:
            model_dir: Directory containing saved models
            model_name: Name of model to load
            featurizer: Feature extraction service

        Returns:
            Initialized prediction service
        """
        model_dir = Path(model_dir)

        # Load model
        if model_name in ("random_forest", "rf"):
            model_path = model_dir / "random_forest.pkl"
            model = ModelRegistry.get_class("random_forest").load(model_path)
        elif model_name.endswith("_net"):
            model_path = model_dir / "neural" / f"{model_name}.pth"
            model = ModelRegistry.get_class(model_name).load(model_path)
        elif "ensemble" in model_name:
            model_path = model_dir / f"ensemble_{model_name.replace('ensemble_', '')}.pkl"
            model = ModelRegistry.get_class(model_name).load(model_path)
        else:
            raise InferenceError(f"Unknown model: {model_name}")

        # Load preprocessor
        preprocessor = None
        preprocessor_path = model_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            preprocessor = FeaturePreprocessor.load(preprocessor_path)

        return cls(
            model=model,
            featurizer=featurizer or FeaturizationService(),
            preprocessor=preprocessor,
        )

    def predict_from_smiles(
        self,
        smiles_list: Union[str, List[str]],
        return_peaks: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[List[Tuple[float, float]]]]]:
        """Predict spectra from SMILES.

        Args:
            smiles_list: Single SMILES or list of SMILES
            return_peaks: If True, also return peak lists

        Returns:
            Predicted spectra, optionally with peak lists
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        # Extract features
        features, _, failed = self.featurizer.extract(smiles_list)

        if len(failed) > 0:
            raise InferenceError(f"Failed to extract features for {len(failed)} molecules")

        # Preprocess
        if self.preprocessor:
            features = self.preprocessor.transform(features)

        # Predict
        predictions = self.model.predict(features)

        if return_peaks:
            peaks_list = [
                self.binner.to_peaks(pred) for pred in predictions
            ]
            return predictions, peaks_list

        return predictions

    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 1000,
    ) -> np.ndarray:
        """Predict spectra in batches for large datasets.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing

        Returns:
            All predictions
        """
        predictions = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            batch_preds = self.predict_from_smiles(batch)
            predictions.append(batch_preds)

        return np.vstack(predictions)

    def predict_to_msp(
        self,
        smiles_list: List[str],
        output_path: Union[str, Path],
        names: Optional[List[str]] = None,
        threshold: float = 0.01,
    ) -> None:
        """Predict spectra and save as MSP file.

        Args:
            smiles_list: List of SMILES strings
            output_path: Output MSP file path
            names: Optional compound names
            threshold: Minimum intensity for peaks
        """
        predictions, peaks_list = self.predict_from_smiles(smiles_list, return_peaks=True)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, (smiles, peaks) in enumerate(zip(smiles_list, peaks_list)):
                name = names[i] if names else f"Compound_{i + 1}"

                # Filter by threshold
                peaks = [(mz, inten) for mz, inten in peaks if inten > threshold]

                f.write(f"Name: {name}\n")
                f.write(f"SMILES: {smiles}\n")
                f.write(f"Num Peaks: {len(peaks)}\n")

                for mz, intensity in sorted(peaks):
                    f.write(f"{mz:.4f} {intensity:.4f}\n")

                f.write("\n")
