#!/usr/bin/env python
"""Evaluate trained models on test set.

Usage:
    poetry run python scripts/evaluate.py
    poetry run python scripts/evaluate.py --model random_forest
    poetry run python scripts/evaluate.py --visualize
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
from smiles2spec.evaluation.evaluator import ModelEvaluator
from smiles2spec.evaluation.metrics import compute_all_metrics
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.services.featurizer import FeaturizationService
from smiles2spec.visualization.diagnostics import plot_2x2_diagnostic

# Import to register models
import smiles2spec.models.random_forest  # noqa
import smiles2spec.models.neural.modular_net  # noqa


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Config file path"
    )
    parser.add_argument(
        "--model", type=str, help="Specific model to evaluate"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate diagnostic plots"
    )
    args = parser.parse_args()

    # Load settings
    config_path = Path(__file__).parent.parent / args.config
    settings = load_settings(config_path)

    print("=" * 60)
    print("SMILES2SPEC: Model Evaluation")
    print("=" * 60)

    # Load and process data (input_path.parent.parent goes from data/input/hpj -> data)
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

    # Split
    splitter = DataSplitter(
        train_size=settings.split.train,
        val_size=settings.split.val,
        test_size=settings.split.test,
        random_seed=settings.split.seed,
    )
    split = splitter.split_arrays(X, y)

    # Load preprocessors (RF uses robust, NN uses standard)
    preprocessor_rf_path = settings.models_path / "preprocessor.pkl"
    preprocessor_nn_path = settings.models_path / "preprocessor_nn.pkl"

    preprocessor_rf = None
    preprocessor_nn = None

    if preprocessor_rf_path.exists():
        preprocessor_rf = FeaturePreprocessor.load(preprocessor_rf_path)
    if preprocessor_nn_path.exists():
        preprocessor_nn = FeaturePreprocessor.load(preprocessor_nn_path)

    # Prepare test data with appropriate preprocessors
    X_test_rf = preprocessor_rf.transform(split.X_test) if preprocessor_rf else split.X_test
    X_test_nn = preprocessor_nn.transform(split.X_test) if preprocessor_nn else split.X_test

    # Find models to evaluate
    models = {}
    model_to_X_test = {}  # Map model name to appropriate test data

    # Random Forest
    rf_path = settings.models_path / "random_forest.pkl"
    if rf_path.exists() and (args.model is None or args.model == "random_forest"):
        models["random_forest"] = ModelRegistry.get_class("random_forest").load(rf_path)
        model_to_X_test["random_forest"] = X_test_rf

    # Neural networks
    neural_dir = settings.models_path / "neural"
    if neural_dir.exists():
        for model_path in neural_dir.glob("*.pth"):
            model_name = model_path.stem
            if args.model is None or args.model == model_name:
                try:
                    models[model_name] = ModelRegistry.get_class(model_name).load(model_path)
                    model_to_X_test[model_name] = X_test_nn
                except Exception as e:
                    print(f"Could not load {model_name}: {e}")

    if not models:
        print("No trained models found!")
        return

    print(f"Found {len(models)} models: {list(models.keys())}")

    # Evaluate each model with its appropriate test data
    evaluator = ModelEvaluator(
        output_dir=settings.metrics_path,
    )

    # Build per-model results since different models use different preprocessors
    all_results = {}
    for name, model in models.items():
        X_test_model = model_to_X_test[name]
        predictions = model.predict(X_test_model)
        metrics = compute_all_metrics(predictions, split.y_test)
        all_results[name] = metrics

    # Create comparison structure
    rankings = []
    for name, metrics in all_results.items():
        cos = metrics["cosine_similarity"]
        rankings.append({
            "name": name,
            "cosine_mean": cos["mean"],
            "cosine_std": cos["std"],
        })
    rankings.sort(key=lambda x: x["cosine_mean"], reverse=True)

    comparison = {
        "rankings": rankings,
        "best_model": rankings[0]["name"] if rankings else None,
        "results": all_results,
    }

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for rank in comparison["rankings"]:
        print(f"  {rank['name']}: {rank['cosine_mean']:.4f} ± {rank['cosine_std']:.4f}")

    print(f"\nBest model: {comparison['best_model']}")

    # Visualize
    if args.visualize:
        print("\nGenerating diagnostic plots...")
        settings.figures_path.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            X_test_model = model_to_X_test[name]
            predictions = model.predict(X_test_model)
            save_path = settings.figures_path / f"{name}_diagnostic.png"
            plot_2x2_diagnostic(predictions, split.y_test, name, save_path)
            print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
