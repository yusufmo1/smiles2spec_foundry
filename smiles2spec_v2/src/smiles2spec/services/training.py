"""Training service for SMILES to spectrum models.

Handles model training, validation, and checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from smiles2spec.core.config import Settings
from smiles2spec.data.dataset import create_loaders_from_arrays
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.data.splitter import DataSplitter, SplitData
from smiles2spec.models.base import BaseModel
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.models.neural.trainer import NeuralTrainer


class TrainingService:
    """Service for training spectrum prediction models."""

    def __init__(
        self,
        settings: Settings,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        """Initialize training service.

        Args:
            settings: Pipeline settings
            output_dir: Directory for saving models
            verbose: Print training progress
        """
        self.settings = settings
        self.output_dir = Path(output_dir) if output_dir else settings.models_path
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save: bool = True,
    ) -> BaseModel:
        """Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (unused for RF)
            y_val: Validation targets (unused for RF)
            save: Whether to save model

        Returns:
            Trained model
        """
        if self.verbose:
            print("Training Random Forest...")

        rf_config = self.settings.random_forest
        model = ModelRegistry.create(
            "random_forest",
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            max_features=rf_config.max_features,
            n_jobs=rf_config.n_jobs,
        )

        model.fit(X_train, y_train)

        if save:
            model_path = self.output_dir / "random_forest.pkl"
            model.save(model_path)
            if self.verbose:
                print(f"Model saved to {model_path}")

        return model

    def train_neural(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save: bool = True,
    ) -> BaseModel:
        """Train neural network model.

        Args:
            model_name: Model name (modular_net, hierarchical_net, etc.)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            save: Whether to save model

        Returns:
            Trained model
        """
        if self.verbose:
            print(f"Training {model_name}...")

        nn_config = self.settings.neural_network
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        # Create model
        model = ModelRegistry.create(
            model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            num_modules=nn_config.num_modules,
            dropout=nn_config.dropout,
        )

        # Create data loaders
        loaders = create_loaders_from_arrays(
            X_train, y_train, X_val, y_val,
            batch_size=nn_config.batch_size,
        )

        # Train
        checkpoint_dir = self.output_dir / "neural" if save else None
        trainer = NeuralTrainer(
            model,
            learning_rate=nn_config.learning_rate,
            max_epochs=nn_config.max_epochs,
            patience=nn_config.patience,
            checkpoint_dir=checkpoint_dir,
            verbose=self.verbose,
        )

        trainer.train(loaders["train"], loaders["val"])

        if save:
            model_path = self.output_dir / "neural" / f"{model_name}.pth"
            model.save(model_path)
            if self.verbose:
                print(f"Model saved to {model_path}")

        return model

    def train_ensemble(
        self,
        models: Dict[str, BaseModel],
        X_val: np.ndarray,
        y_val: np.ndarray,
        ensemble_type: str = "bin_by_bin",
        save: bool = True,
    ) -> BaseModel:
        """Train ensemble model.

        Args:
            models: Dict of trained models
            X_val: Validation features for weight optimization
            y_val: Validation targets
            ensemble_type: 'weighted' or 'bin_by_bin'
            save: Whether to save ensemble

        Returns:
            Trained ensemble
        """
        if self.verbose:
            print(f"Training {ensemble_type} ensemble...")

        ensemble = ModelRegistry.create(ensemble_type, models=models)
        ensemble.fit(None, None, X_val, y_val)

        if save:
            ensemble_path = self.output_dir / f"ensemble_{ensemble_type}.pkl"
            ensemble.save(ensemble_path)
            if self.verbose:
                print(f"Ensemble saved to {ensemble_path}")

        return ensemble

    def train_full_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_neural: bool = True,
        neural_models: Optional[list] = None,
    ) -> Dict[str, BaseModel]:
        """Train complete model pipeline.

        Args:
            X: Feature matrix
            y: Target spectra
            train_neural: Whether to train neural networks
            neural_models: List of neural model names to train

        Returns:
            Dict of trained models
        """
        # Split data
        splitter = DataSplitter(
            train_size=self.settings.split.train_size,
            val_size=self.settings.split.val_size,
            test_size=self.settings.split.test_size,
            random_seed=self.settings.split.random_seed,
        )
        split = splitter.split_arrays(X, y)

        # Preprocess
        preprocessor = FeaturePreprocessor(scaling="robust")
        X_train = preprocessor.fit_transform(split.X_train)
        X_val = preprocessor.transform(split.X_val)
        X_test = preprocessor.transform(split.X_test)

        # Save preprocessor
        preprocessor.save(self.output_dir / "preprocessor.pkl")

        models = {}

        # Train Random Forest
        models["random_forest"] = self.train_random_forest(
            X_train, split.y_train, X_val, split.y_val
        )

        # Train neural networks
        if train_neural:
            neural_models = neural_models or ["modular_net"]
            for model_name in neural_models:
                models[model_name] = self.train_neural(
                    model_name, X_train, split.y_train, X_val, split.y_val
                )

        # Train ensemble
        if len(models) > 1:
            ensemble_type = self.settings.ensemble.method
            models["ensemble"] = self.train_ensemble(
                models, X_val, split.y_val, ensemble_type
            )

        return models
