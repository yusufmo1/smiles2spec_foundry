#!/usr/bin/env python
"""Train neural network models using pre-split data with binned spectra.

This script is designed for datasets with established train/val/test splits
(e.g., GNPS) where spectra are already binned.

Usage:
    poetry run python train_nn_presplit.py --config config_gnps.yml
    poetry run python train_nn_presplit.py --config config_gnps.yml --model sparse_gated_net
    poetry run python train_nn_presplit.py --config config_gnps.yml --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from smiles2spec.core.loader import load_settings
from smiles2spec.data.loader import DataLoader
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.data.dataset import create_loaders_from_arrays
from smiles2spec.evaluation.metrics import compute_all_metrics
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.models.neural.trainer import NeuralTrainer
from smiles2spec.services.featurizer import FeaturizationService

# Import to register models
import smiles2spec.models.neural.modular_net  # noqa
import smiles2spec.models.neural.hierarchical_net  # noqa
import smiles2spec.models.neural.sparse_gated_net  # noqa
import smiles2spec.models.neural.regional_expert  # noqa


AVAILABLE_MODELS = [
    "modular_net",
    "hierarchical_net",
    "sparse_gated_net",
    "regional_expert_net",
]


def featurize_split(
    smiles_list: list,
    featurizer: FeaturizationService,
    split_name: str,
) -> tuple:
    """Featurize a split and return valid features and indices.

    Args:
        smiles_list: List of SMILES strings
        featurizer: FeaturizationService instance
        split_name: Name of split for logging

    Returns:
        Tuple of (features, failed_indices)
    """
    print(f"  Featurizing {split_name}: {len(smiles_list)} samples...")
    X, _, failed = featurizer.extract(smiles_list)
    if failed:
        print(f"  Warning: {len(failed)} failed featurizations in {split_name}")
    return X, failed


def main():
    parser = argparse.ArgumentParser(
        description="Train neural networks using pre-split data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_gnps.yml",
        help="Config file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="modular_net",
        choices=AVAILABLE_MODELS,
        help="Model to train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models",
    )
    args = parser.parse_args()

    # Load settings
    config_path = Path(__file__).parent / args.config
    settings = load_settings(config_path)

    models_to_train = AVAILABLE_MODELS if args.all else [args.model]

    print("=" * 60)
    print("SMILES2SPEC: Neural Network Training (Pre-split Data)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Dataset: {settings.dataset}")
    print(f"Models: {models_to_train}")

    # Load pre-split data
    loader = DataLoader(
        data_dir=settings.input_path.parent.parent,
        dataset=settings.dataset,
    )

    if not loader.has_presplit():
        print(f"\nError: No pre-split data found for dataset '{settings.dataset}'")
        print("Expected files: train_data.jsonl, val_data.jsonl, test_data.jsonl")
        sys.exit(1)

    print("\nLoading pre-split data...")
    train_entries, val_entries, test_entries = loader.load_presplit()
    print(f"  Train: {len(train_entries)} samples")
    print(f"  Val:   {len(val_entries)} samples")
    print(f"  Test:  {len(test_entries)} samples")

    # Extract SMILES and pre-binned spectra
    train_smiles, y_train = loader.extract_smiles_and_spectra_presplit(train_entries)
    val_smiles, y_val = loader.extract_smiles_and_spectra_presplit(val_entries)
    test_smiles, y_test = loader.extract_smiles_and_spectra_presplit(test_entries)

    print(f"\nSpectrum shape: {y_train.shape[1]} bins")

    # Featurize SMILES for each split
    print("\nFeaturizing SMILES...")
    featurizer = FeaturizationService(
        config=settings.features,
        cache_dir=settings.cache_path,
        use_cache=False,  # Disabled - duplicate column names in cache
        n_jobs=-1,  # Use all cores for parallel extraction
    )

    X_train_raw, train_failed = featurize_split(train_smiles, featurizer, "train")
    X_val_raw, val_failed = featurize_split(val_smiles, featurizer, "val")
    X_test_raw, test_failed = featurize_split(test_smiles, featurizer, "test")

    # Remove failed samples from spectra
    if train_failed:
        mask = np.ones(len(train_smiles), dtype=bool)
        mask[list(train_failed)] = False
        y_train = y_train[mask]
    if val_failed:
        mask = np.ones(len(val_smiles), dtype=bool)
        mask[list(val_failed)] = False
        y_val = y_val[mask]
    if test_failed:
        mask = np.ones(len(test_smiles), dtype=bool)
        mask[list(test_failed)] = False
        y_test = y_test[mask]

    print(f"\nAfter featurization:")
    print(f"  Train: {X_train_raw.shape[0]} samples, {X_train_raw.shape[1]} features")
    print(f"  Val:   {X_val_raw.shape[0]} samples")
    print(f"  Test:  {X_test_raw.shape[0]} samples")

    # Preprocess features (fit on train only)
    print("\nPreprocessing features...")
    preprocessor = FeaturePreprocessor(scaling="standard")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Save preprocessor for later use
    preprocessor_path = settings.models_path / "preprocessor_gnps.pkl"
    preprocessor.save(preprocessor_path)
    print(f"  Saved preprocessor to {preprocessor_path}")

    # Create data loaders
    nn_config = settings.neural_network
    print(f"\nNeural network config:")
    print(f"  Batch size: {nn_config.batch_size}")
    print(f"  Learning rate: {nn_config.learning_rate}")
    print(f"  Weight decay: {nn_config.weight_decay}")
    print(f"  Dropout: {nn_config.dropout}")

    loaders = create_loaders_from_arrays(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=nn_config.batch_size,
    )

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    print(f"\nModel dimensions:")
    print(f"  Input:  {input_dim}")
    print(f"  Output: {output_dim}")

    # Train models
    results = {}
    (settings.models_path / "neural").mkdir(parents=True, exist_ok=True)

    for model_name in models_to_train:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}")
        print("=" * 60)

        model = ModelRegistry.create(
            model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            num_modules=nn_config.num_modules,
            dropout=nn_config.dropout,
        )
        print(f"Parameters: {model.count_parameters():,}")

        trainer = NeuralTrainer(
            model,
            learning_rate=nn_config.learning_rate,
            weight_decay=nn_config.weight_decay,
            max_epochs=nn_config.max_epochs,
            patience=nn_config.patience,
            checkpoint_dir=settings.models_path / "neural",
        )

        history = trainer.train(loaders["train"], loaders["val"])

        # Evaluate on test set
        predictions = model.predict(X_test)
        metrics = compute_all_metrics(predictions, y_test)

        cos = metrics["cosine_similarity"]
        print(f"\nTest Results:")
        print(f"  Cosine: {cos['mean']:.4f} +/- {cos['std']:.4f}")
        print(f"  Best epoch: {history.best_epoch}")

        results[model_name] = metrics

        # Save model
        model_path = settings.models_path / "neural" / f"{model_name}_gnps.pth"
        model.save(model_path)
        print(f"  Saved model to {model_path}")

    # Save evaluation results
    metrics_dir = settings.output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for model_name, metrics in results.items():
        metrics_path = metrics_dir / f"{model_name}_gnps_evaluation.json"
        with open(metrics_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    serializable[k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()
                    }
                elif isinstance(v, (np.floating, float)):
                    serializable[k] = float(v)
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Dataset: {settings.dataset} ({len(train_entries)} train samples)")
    print("\nResults (sorted by cosine similarity):")
    for name, metrics in sorted(
        results.items(),
        key=lambda x: x[1]["cosine_similarity"]["mean"],
        reverse=True,
    ):
        cos = metrics["cosine_similarity"]["mean"]
        std = metrics["cosine_similarity"]["std"]
        print(f"  {name}: {cos:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
