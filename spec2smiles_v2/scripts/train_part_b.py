#!/usr/bin/env python
"""Train Part B (Descriptors -> SMILES) model.

Usage:
    python scripts/train_part_b.py [--model vae|direct] [--augment] [--n-augment N]

Or via Makefile:
    make train-part-b-vae
    make train-part-b-direct
    make train-part-b-augmented
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.descriptors import calculate_descriptors
from src.services.data_loader import DataLoaderService
from src.services.part_b import PartBService
from src.utils.paths import validate_input_dir


def main():
    parser = argparse.ArgumentParser(description="Train Part B model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vae", "direct"],
        default=None,
        help="Model type: 'vae' (ConditionalVAE) or 'direct' (DirectDecoder)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable SMILES augmentation (6x data via random atom ordering)"
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=5,
        help="Number of augmented SMILES per molecule (default: 5 = 6x total)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show training progress"
    )
    # Descriptor augmentation (noise injection)
    parser.add_argument(
        "--desc-augment",
        action="store_true",
        default=False,
        help="Enable descriptor noise augmentation (simulates LightGBM errors)"
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.5,
        help="Probability of adding noise per sample (default: 0.5)"
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Noise scale relative to RMSE (default: 1.0 = full error)"
    )
    parser.add_argument(
        "--rmse-path",
        type=Path,
        default=None,
        help="Path to Part A metrics JSON (auto-detected if not provided)"
    )
    # Option 1: Train on predicted descriptors (recommended for E2E)
    parser.add_argument(
        "--use-predicted-descriptors",
        action="store_true",
        default=False,
        help="Train on Part A predicted descriptors instead of true descriptors"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Validate input directory exists
    validate_input_dir(settings.input_path, "Dataset")

    # Determine RMSE path for descriptor augmentation
    rmse_path = args.rmse_path
    if args.desc_augment and rmse_path is None:
        # Auto-detect Part A metrics
        rmse_path = settings.metrics_path / "part_a_lgbm_metrics.json"
        if not rmse_path.exists():
            print(f"Warning: RMSE metrics not found at {rmse_path}")
            print("  Run Part A evaluation first or provide --rmse-path")
            rmse_path = None

    print("=" * 60)
    print(f"Training Part B (Descriptors -> SMILES)")
    print("=" * 60)
    print(f"Dataset:    {settings.dataset}")
    print(f"Model:      direct (DirectDecoder)")
    print(f"SMILES Aug: {'Yes (' + str(args.n_augment) + 'x)' if args.augment else 'No'}")
    if args.use_predicted_descriptors:
        print(f"Desc Mode:  PREDICTED (Option 1 - train on Part A predictions)")
    elif args.desc_augment:
        print(f"Desc Aug:   Yes (p={args.noise_prob}, scale={args.noise_scale})")
    else:
        print(f"Desc Aug:   No (using TRUE descriptors)")
    print(f"Device:     {settings.torch_device}")
    print()

    # Load data
    data_loader = DataLoaderService(data_dir=Path(settings.data_input_dir) / settings.dataset)

    # Check for pre-processed splits
    processed_dir = Path(settings.data_input_dir) / settings.dataset
    train_path = processed_dir / "train_data.jsonl"

    if train_path.exists():
        print("Loading preprocessed data splits...")
        train_data, val_data, test_data, metadata = data_loader.load_processed_splits()

        # Extract SMILES and descriptors
        train_smiles = [s["smiles"] for s in train_data]
        train_descriptors = [s["descriptors"] for s in train_data]

        val_smiles = [s["smiles"] for s in val_data]
        val_descriptors = [s["descriptors"] for s in val_data]

        import numpy as np
        train_descriptors = np.array(train_descriptors, dtype=np.float32)
        val_descriptors = np.array(val_descriptors, dtype=np.float32)
    else:
        print("Loading and preprocessing raw data...")
        raw_data, total = data_loader.load_raw_data()
        print(f"Loaded {len(raw_data)}/{total} valid samples")

        # Calculate descriptors for all samples
        import numpy as np
        from sklearn.model_selection import train_test_split

        all_smiles = []
        all_descriptors = []

        for sample in raw_data:
            desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)
            if desc is not None:
                all_smiles.append(sample["smiles"])
                all_descriptors.append(desc)

        all_descriptors = np.array(all_descriptors)

        # Split data
        train_smiles, temp_smiles, train_descriptors, temp_descriptors = train_test_split(
            all_smiles, all_descriptors,
            test_size=settings.val_ratio + settings.test_ratio,
            random_state=settings.random_seed
        )

        test_fraction = settings.test_ratio / (settings.val_ratio + settings.test_ratio)
        val_smiles, _, val_descriptors, _ = train_test_split(
            temp_smiles, temp_descriptors,
            test_size=test_fraction,
            random_state=settings.random_seed
        )

    print(f"\nData split:")
    print(f"  Train: {len(train_smiles)} samples")
    print(f"  Val:   {len(val_smiles)} samples")
    print()

    # Option 1: Load predicted descriptors if requested
    if args.use_predicted_descriptors:
        pred_train_path = processed_dir / "train_predicted_descriptors.npy"
        pred_val_path = processed_dir / "val_predicted_descriptors.npy"

        if not pred_train_path.exists():
            print(f"Error: Predicted descriptors not found at {pred_train_path}")
            print("Run generate_predicted_descriptors.py first")
            sys.exit(1)

        print("Loading predicted descriptors (Option 1)...")
        # Keep TRUE descriptors for scaler fitting
        true_train_descriptors = train_descriptors.copy()

        # Load predicted descriptors for training
        train_descriptors = np.load(pred_train_path)
        val_descriptors = np.load(pred_val_path)

        print(f"  Loaded train predictions: {train_descriptors.shape}")
        print(f"  Loaded val predictions: {val_descriptors.shape}")

        # Calculate prediction error for diagnostics
        pred_error = np.sqrt(np.mean((train_descriptors - true_train_descriptors) ** 2))
        print(f"  Train RMSE (pred vs true): {pred_error:.4f}")

    # Initialize Part B service with descriptor augmentation
    service = PartBService(
        desc_augment=args.desc_augment,
        noise_prob=args.noise_prob,
        noise_scale=args.noise_scale,
        rmse_path=rmse_path,
        descriptor_names=settings.descriptor_names,
    )

    # Scale descriptors FIRST (before encoding/filtering) - matching pkg behavior
    # This ensures scaler sees full distribution, not just filtered subset
    print("Scaling descriptors...")

    if args.use_predicted_descriptors:
        # CRITICAL: Fit scaler on TRUE descriptors, then transform PREDICTED
        # This maintains consistency with inference where scaler was fitted on true values
        print("  Fitting scaler on TRUE descriptors (for consistency with inference)...")
        service.scaler.fit(true_train_descriptors)
        scaled_train_descriptors = service.scaler.transform(train_descriptors)
        scaled_val_descriptors = service.scaler.transform(val_descriptors)
    else:
        scaled_train_descriptors = service.scaler.fit_transform(train_descriptors)
        scaled_val_descriptors = service.scaler.transform(val_descriptors)

    # Prepare data (build vocabulary, encode, and optionally augment)
    # Pass pre-scaled descriptors
    print("Preparing training data...")
    encoded_train, scaled_train_desc, train_indices = service.prepare_data(
        train_smiles, scaled_train_descriptors, verbose=args.verbose,
        augment=args.augment, n_augment=args.n_augment if args.augment else 0,
    )

    print(f"Valid training samples: {len(train_indices)}/{len(train_smiles)}")
    print(f"Vocabulary size: {service.encoder.vocab_size}")

    # Prepare validation data (with pre-scaled descriptors)
    encoded_val, scaled_val_desc, val_indices = service.encoder.batch_encode(
        val_smiles, verbose=False
    )
    # Filter scaled val descriptors to valid SELFIES indices
    valid_val_indices = [i for i, s in enumerate(val_smiles)
                         if service.encoder.smiles_to_selfies(s) is not None]
    scaled_val_desc = scaled_val_descriptors[valid_val_indices] if valid_val_indices else None

    # Setup log directory for live epoch logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    print(f"\nTraining DirectDecoder...")
    history = service.train(
        encoded_train,
        scaled_train_desc,
        encoded_val if len(encoded_val) > 0 else None,
        scaled_val_desc,
        verbose=args.verbose,
        log_dir=log_dir,
    )

    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

    # Save model
    output_dir = settings.models_path / "part_b"
    print(f"\nSaving model to {output_dir}...")
    service.save(output_dir)

    # Save metrics
    metrics_path = settings.metrics_path / "part_b_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": "direct",
            "smiles_augmented": args.augment,
            "n_smiles_augment": args.n_augment if args.augment else 0,
            "use_predicted_descriptors": args.use_predicted_descriptors,
            "desc_augmented": args.desc_augment,
            "noise_prob": args.noise_prob if args.desc_augment else None,
            "noise_scale": args.noise_scale if args.desc_augment else None,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "n_epochs": len(history["train_loss"]),
            "vocab_size": service.encoder.vocab_size,
        }, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

    # Auto-generate visualizations
    print("\nGenerating visualizations...")
    from scripts.visualize import generate_part_b_plots
    generate_part_b_plots(settings.figures_path, skip_missing=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
