#!/usr/bin/env python
"""Train Part A (Spectrum -> Descriptors) model.

Usage:
    python scripts/train_part_a.py [--model lgbm|transformer] [--config config.yml]

Or via Makefile:
    make train-part-a-lgbm
    make train-part-a-transformer
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors
from src.services.data_loader import DataLoaderService
from src.services.part_a import PartAService


def main():
    parser = argparse.ArgumentParser(description="Train Part A model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "transformer"],
        default=None,
        help="Model type (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show training progress"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Use command line override or config default
    model_type = args.model or settings.part_a_model

    print("=" * 60)
    print(f"Training Part A (Spectrum -> Descriptors)")
    print("=" * 60)
    print(f"Model:   {model_type}")
    print(f"Dataset: {settings.dataset}")
    print(f"Device:  {settings.torch_device}")
    print()

    # Load data
    data_loader = DataLoaderService(data_dir=Path(settings.data_input_dir) / settings.dataset)

    # Check for pre-processed splits
    processed_dir = Path(settings.data_input_dir) / settings.dataset
    train_path = processed_dir / "train_data.jsonl"

    if train_path.exists():
        print("Loading preprocessed data splits...")
        train_data, val_data, test_data, metadata = data_loader.load_processed_splits()

        X_train, y_train, _ = data_loader.extract_features_and_targets(train_data)
        X_val, y_val, _ = data_loader.extract_features_and_targets(val_data)
        X_test, y_test, _ = data_loader.extract_features_and_targets(test_data)
    else:
        print("Loading and preprocessing raw data...")
        raw_data, total = data_loader.load_raw_data()
        print(f"Loaded {len(raw_data)}/{total} valid samples")

        # Process spectra and calculate descriptors
        import numpy as np
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split

        spectra = []
        descriptors = []
        valid_smiles = []

        for sample in tqdm(raw_data, desc="Processing samples"):
            # Process spectrum
            spectrum = process_spectrum(
                sample["peaks"],
                n_bins=settings.n_bins,
                bin_width=settings.bin_width,
                max_mz=settings.max_mz,
                transform=settings.transform,
                normalize=settings.normalize,
            )

            # Calculate descriptors
            desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)
            if desc is not None:
                spectra.append(spectrum)
                descriptors.append(desc)
                valid_smiles.append(sample["smiles"])

        X = np.array(spectra)
        y = np.array(descriptors)

        print(f"Valid samples after processing: {len(X)}")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=settings.val_ratio + settings.test_ratio,
            random_state=settings.random_seed
        )

        test_fraction = settings.test_ratio / (settings.val_ratio + settings.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_fraction,
            random_state=settings.random_seed
        )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print()

    # Train model
    print(f"Training {model_type.upper()} model...")
    service = PartAService(model_type=model_type)

    # Setup log directory for live epoch logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics = service.train(X_train, y_train, X_val, y_val, verbose=args.verbose, log_dir=log_dir)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    summary = service.get_summary_metrics(X_test, y_test)

    print(f"\nTest Results:")
    print(f"  Mean R²:   {summary['mean_r2']:.4f}")
    print(f"  Median R²: {summary['median_r2']:.4f}")
    print(f"  Best:      {summary['best_descriptor']} (R² = {summary['best_r2']:.4f})")
    print(f"  Worst:     {summary['worst_descriptor']} (R² = {summary['worst_r2']:.4f})")

    # Save model
    output_dir = settings.models_path / "part_a"
    print(f"\nSaving model to {output_dir}...")
    service.save(output_dir)

    # Save metrics
    metrics_path = settings.metrics_path / f"part_a_{model_type}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    full_metrics = {
        "model_type": model_type,
        "summary": summary,
        "per_descriptor": metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

    # Auto-generate visualizations
    print("\nGenerating visualizations...")
    from scripts.visualize import generate_part_a_plots
    generate_part_a_plots(settings.figures_path, skip_missing=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
