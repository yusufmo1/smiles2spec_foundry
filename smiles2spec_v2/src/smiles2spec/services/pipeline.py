"""End-to-end pipeline service for SMILES to spectrum prediction.

Orchestrates the complete workflow from data loading to prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from smiles2spec.core.config import Settings
from smiles2spec.core.loader import load_settings
from smiles2spec.data.loader import DataLoader
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.data.splitter import DataSplitter
from smiles2spec.domain.spectrum import SpectrumBinner
from smiles2spec.models.base import BaseModel
from smiles2spec.services.featurizer import FeaturizationService
from smiles2spec.services.predictor import PredictionService
from smiles2spec.services.training import TrainingService


class PipelineService:
    """End-to-end SMILES to spectrum prediction pipeline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize pipeline.

        Args:
            settings: Pipeline settings
            config_path: Path to config.yml
        """
        if settings is None and config_path is not None:
            settings = load_settings(config_path)
        elif settings is None:
            settings = Settings()

        self.settings = settings
        self._models: Dict[str, BaseModel] = {}
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._featurizer: Optional[FeaturizationService] = None

    def setup(self) -> "PipelineService":
        """Set up pipeline components.

        Returns:
            Self for method chaining
        """
        # Initialize featurizer
        self._featurizer = FeaturizationService(
            config=self.settings.features,
            cache_dir=self.settings.cache_path,
            use_cache=True,
        )

        return self

    def load_data(self) -> Tuple[List[str], List[List[Tuple[float, float]]]]:
        """Load raw data from configured dataset.

        Returns:
            Tuple of (smiles_list, peaks_list)
        """
        loader = DataLoader(
            data_dir=self.settings.data_path.parent,
            dataset=self.settings.dataset,
        )
        return loader.extract_smiles_and_peaks()

    def featurize(
        self,
        smiles_list: List[str],
        peaks_list: List[List[Tuple[float, float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and bin spectra.

        Args:
            smiles_list: List of SMILES strings
            peaks_list: List of peak lists

        Returns:
            Tuple of (features, spectra)
        """
        if self._featurizer is None:
            self.setup()

        # Extract features
        features, _, failed = self._featurizer.extract(smiles_list)

        # Bin spectra
        binner = SpectrumBinner(
            n_bins=self.settings.spectrum.n_bins,
            bin_width=self.settings.spectrum.bin_width,
            max_mz=self.settings.spectrum.max_mz,
            transform=self.settings.spectrum.transform,
        )

        # Filter failed molecules
        valid_peaks = [p for i, p in enumerate(peaks_list) if i not in failed]
        spectra = binner.batch_bin(valid_peaks)

        return features, spectra

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_neural: bool = False,
        neural_models: Optional[List[str]] = None,
    ) -> Dict[str, BaseModel]:
        """Train models on featurized data.

        Args:
            X: Feature matrix
            y: Target spectra
            train_neural: Whether to train neural networks
            neural_models: List of neural model names

        Returns:
            Dict of trained models
        """
        training_service = TrainingService(
            settings=self.settings,
            output_dir=self.settings.models_path,
        )

        self._models = training_service.train_full_pipeline(
            X, y,
            train_neural=train_neural,
            neural_models=neural_models,
        )

        # Load preprocessor
        preprocessor_path = self.settings.models_path / "preprocessor.pkl"
        if preprocessor_path.exists():
            self._preprocessor = FeaturePreprocessor.load(preprocessor_path)

        return self._models

    def predict(
        self,
        smiles_list: Union[str, List[str]],
        model_name: str = "random_forest",
    ) -> np.ndarray:
        """Generate predictions for SMILES.

        Args:
            smiles_list: SMILES string(s)
            model_name: Model to use for prediction

        Returns:
            Predicted spectra
        """
        if model_name not in self._models:
            # Try loading from disk
            predictor = PredictionService.from_directory(
                self.settings.models_path,
                model_name=model_name,
                featurizer=self._featurizer,
            )
            return predictor.predict_from_smiles(smiles_list)

        predictor = PredictionService(
            model=self._models[model_name],
            featurizer=self._featurizer,
            preprocessor=self._preprocessor,
        )
        return predictor.predict_from_smiles(smiles_list)

    def run_full_pipeline(
        self,
        train_neural: bool = False,
        evaluate: bool = True,
    ) -> Dict:
        """Run complete pipeline from data loading to evaluation.

        Args:
            train_neural: Whether to train neural networks
            evaluate: Whether to evaluate on test set

        Returns:
            Results dictionary
        """
        # Setup
        self.setup()

        # Load data
        smiles_list, peaks_list = self.load_data()

        # Featurize
        X, y = self.featurize(smiles_list, peaks_list)

        # Train
        models = self.train(X, y, train_neural=train_neural)

        results = {
            "n_samples": len(smiles_list),
            "n_features": X.shape[1],
            "n_bins": y.shape[1],
            "models_trained": list(models.keys()),
        }

        return results

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PipelineService":
        """Create pipeline from config file.

        Args:
            config_path: Path to config.yml

        Returns:
            Initialized pipeline
        """
        settings = load_settings(config_path)
        return cls(settings=settings)
