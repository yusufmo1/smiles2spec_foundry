#!/usr/bin/env python
"""Train neural network models for spectrum prediction.

Usage:
    poetry run python scripts/train_nn.py
    poetry run python scripts/train_nn.py --model modular_net
    poetry run python scripts/train_nn.py --all
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
from smiles2spec.data.dataset import create_loaders_from_arrays
from smiles2spec.domain.spectrum import SpectrumBinner
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


def main():
    parser = argparse.ArgumentParser(description="Train neural network model")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Config file path"
    )
    parser.add_argument(
        "--model", type=str, default="modular_net",
        choices=AVAILABLE_MODELS, help="Model to train"
    )
    parser.add_argument(
        "--all", action="store_true", help="Train all models"
    )
    args = parser.parse_args()

    # Load settings
    config_path = Path(__file__).parent.parent / args.config
    settings = load_settings(config_path)

    models_to_train = AVAILABLE_MODELS if args.all else [args.model]

    print("=" * 60)
    print("SMILES2SPEC: Neural Network Training")
    print("=" * 60)
    print(f"Models: {models_to_train}")

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

    # Split and preprocess
    splitter = DataSplitter(
        train_size=settings.split.train,
        val_size=settings.split.val,
        test_size=settings.split.test,
        random_seed=settings.split.seed,
    )
    split = splitter.split_arrays(X, y)

    preprocessor = FeaturePreprocessor(scaling="standard")
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)

    # Create data loaders
    nn_config = settings.neural_network
    loaders = create_loaders_from_arrays(
        X_train, split.y_train, X_val, split.y_val, X_test, split.y_test,
        batch_size=nn_config.batch_size,
    )

    input_dim = X_train.shape[1]
    output_dim = y.shape[1]

    # Save NN preprocessor (uses standard scaler)
    preprocessor.save(settings.models_path / "preprocessor_nn.pkl")

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
            max_epochs=nn_config.max_epochs,
            patience=nn_config.patience,
            checkpoint_dir=settings.models_path / "neural",
        )

        history = trainer.train(loaders["train"], loaders["val"])

        # Evaluate
        predictions = model.predict(X_test)
        metrics = compute_all_metrics(predictions, split.y_test)

        cos = metrics["cosine_similarity"]
        print(f"\nTest Results:")
        print(f"  Cosine: {cos['mean']:.4f} ± {cos['std']:.4f}")
        print(f"  Best epoch: {history.best_epoch}")

        results[model_name] = metrics

        # Save
        model_path = settings.models_path / "neural" / f"{model_name}.pth"
        model.save(model_path)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, metrics in sorted(
        results.items(),
        key=lambda x: x[1]["cosine_similarity"]["mean"],
        reverse=True,
    ):
        cos = metrics["cosine_similarity"]["mean"]
        print(f"  {name}: {cos:.4f}")


if __name__ == "__main__":
    main()
