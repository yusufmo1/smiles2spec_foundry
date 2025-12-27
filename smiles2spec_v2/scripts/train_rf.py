#!/usr/bin/env python
"""Train Random Forest model for spectrum prediction.

Usage:
    poetry run python scripts/train_rf.py
    poetry run python scripts/train_rf.py --config config.yml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smiles2spec.core.loader import load_settings
from smiles2spec.data.loader import DataLoader
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.data.splitter import DataSplitter
from smiles2spec.domain.spectrum import SpectrumBinner
from smiles2spec.evaluation.metrics import compute_all_metrics
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.services.featurizer import FeaturizationService

# Import to register model
import smiles2spec.models.random_forest  # noqa


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Config file path"
    )
    args = parser.parse_args()

    # Load settings
    config_path = Path(__file__).parent.parent / args.config
    settings = load_settings(config_path)

    print("=" * 60)
    print("SMILES2SPEC: Random Forest Training")
    print("=" * 60)
    print(f"Dataset: {settings.dataset}")

    # Load and featurize data (input_path.parent.parent goes from data/input/hpj -> data)
    loader = DataLoader(
        data_dir=settings.input_path.parent.parent,
        dataset=settings.dataset,
    )
    smiles_list, peaks_list = loader.extract_smiles_and_peaks()

    featurizer = FeaturizationService(
        config=settings.features,
        cache_dir=settings.cache_path,
        use_cache=False,  # Disable caching for now
    )
    X, _, failed = featurizer.extract(smiles_list)

    binner = SpectrumBinner(
        n_bins=settings.spectrum.n_bins,
        transform=settings.spectrum.transform,
    )
    valid_peaks = [p for i, p in enumerate(peaks_list) if i not in failed]
    y = binner.batch_bin(valid_peaks)

    print(f"Features: {X.shape}, Targets: {y.shape}")

    # Split data
    splitter = DataSplitter(
        train_size=settings.split.train,
        val_size=settings.split.val,
        test_size=settings.split.test,
        random_seed=settings.split.seed,
    )
    split = splitter.split_arrays(X, y)

    # Preprocess
    preprocessor = FeaturePreprocessor(scaling="robust")
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Train model
    print("\nTraining Random Forest...")
    rf_config = settings.random_forest
    model = ModelRegistry.create(
        "random_forest",
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        max_features=rf_config.max_features,
        n_jobs=rf_config.n_jobs,
    )
    model.fit(X_train, split.y_train)

    # Evaluate
    print("\nEvaluating on test set...")
    predictions = model.predict(X_test)
    metrics = compute_all_metrics(predictions, split.y_test, compute_bootstrap=True)

    cos = metrics["cosine_similarity"]
    print(f"\nResults:")
    print(f"  Cosine Similarity: {cos['mean']:.4f} ± {cos['std']:.4f}")
    print(f"  95% CI: [{cos['ci_low']:.4f}, {cos['ci_high']:.4f}]")
    print(f"  OOB Score: {model.oob_score:.4f}")

    # Save
    settings.models_path.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_path / "random_forest.pkl"
    model.save(model_path)
    preprocessor.save(settings.models_path / "preprocessor.pkl")

    print(f"\nModel saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
