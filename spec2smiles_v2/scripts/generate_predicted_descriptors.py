#!/usr/bin/env python
"""Generate Part A predicted descriptors for train/val splits.

Uses trained LightGBM models to predict descriptors from spectra.
This creates the training data for Part B (Option 1 approach).

Usage:
    python scripts/generate_predicted_descriptors.py --config config_gnps_unique28.yml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config


def load_lgbm_models(model_dir: Path) -> dict:
    """Load LightGBM models for all descriptors."""
    model_file = model_dir / "models.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"LightGBM models not found at {model_file}")

    with open(model_file, "rb") as f:
        models = pickle.load(f)

    print(f"Loaded {len(models)} LightGBM models")
    return models


def predict_descriptors(models: dict, spectra: np.ndarray, descriptor_names: list) -> np.ndarray:
    """Predict descriptors using LightGBM ensemble."""
    n_samples = len(spectra)
    predictions = np.zeros((n_samples, len(descriptor_names)), dtype=np.float32)

    for i, name in enumerate(tqdm(descriptor_names, desc="Predicting descriptors")):
        if name in models:
            predictions[:, i] = models[name].predict(spectra)
        else:
            print(f"Warning: No model for descriptor {name}")

    return predictions


def load_split_data(data_path: Path) -> tuple:
    """Load split data and extract spectra and SMILES."""
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    spectra = np.array([s["spectrum"] for s in samples], dtype=np.float32)
    smiles = [s["smiles"] for s in samples]
    true_descriptors = np.array([s["descriptors"] for s in samples], dtype=np.float32)

    return spectra, smiles, true_descriptors


def main():
    parser = argparse.ArgumentParser(description="Generate predicted descriptors for train/val")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    data_dir = Path(settings.data_input_dir) / settings.dataset
    model_dir = settings.models_path / "part_a_lgbm"

    print("=" * 60)
    print("Generating Predicted Descriptors")
    print("=" * 60)
    print(f"Dataset:    {settings.dataset}")
    print(f"Model dir:  {model_dir}")
    print(f"Descriptors: {len(settings.descriptor_names)}")
    print()

    # Check for split files
    train_path = data_dir / "train_data.jsonl"
    val_path = data_dir / "val_data.jsonl"

    if not train_path.exists():
        print(f"Error: train_data.jsonl not found at {train_path}")
        print("Run preprocess_splits.py first")
        sys.exit(1)

    # Load LightGBM models
    print("Loading LightGBM models...")
    models = load_lgbm_models(model_dir)

    # Process train split
    print("\n--- Processing Train Split ---")
    train_spectra, train_smiles, train_true_desc = load_split_data(train_path)
    print(f"Loaded {len(train_spectra)} train samples")

    train_pred_desc = predict_descriptors(
        models, train_spectra, list(settings.descriptor_names)
    )

    # Calculate in-sample error (for diagnostics)
    train_errors = train_pred_desc - train_true_desc
    train_rmse = np.sqrt(np.mean(train_errors ** 2, axis=0))
    print(f"Train RMSE (in-sample): mean={np.mean(train_rmse):.4f}, max={np.max(train_rmse):.4f}")

    # Process val split
    print("\n--- Processing Validation Split ---")
    val_spectra, val_smiles, val_true_desc = load_split_data(val_path)
    print(f"Loaded {len(val_spectra)} val samples")

    val_pred_desc = predict_descriptors(
        models, val_spectra, list(settings.descriptor_names)
    )

    # Calculate out-of-sample error
    val_errors = val_pred_desc - val_true_desc
    val_rmse = np.sqrt(np.mean(val_errors ** 2, axis=0))
    print(f"Val RMSE (out-of-sample): mean={np.mean(val_rmse):.4f}, max={np.max(val_rmse):.4f}")

    # Save predicted descriptors
    print("\n--- Saving Predictions ---")

    np.save(data_dir / "train_predicted_descriptors.npy", train_pred_desc)
    np.save(data_dir / "val_predicted_descriptors.npy", val_pred_desc)

    # Also save SMILES lists for reference
    with open(data_dir / "train_smiles.json", "w") as f:
        json.dump(train_smiles, f)
    with open(data_dir / "val_smiles.json", "w") as f:
        json.dump(val_smiles, f)

    # Save prediction metadata
    pred_metadata = {
        "train_samples": len(train_spectra),
        "val_samples": len(val_spectra),
        "n_descriptors": len(settings.descriptor_names),
        "descriptor_names": list(settings.descriptor_names),
        "train_rmse_mean": float(np.mean(train_rmse)),
        "val_rmse_mean": float(np.mean(val_rmse)),
        "per_descriptor_val_rmse": {
            name: float(rmse)
            for name, rmse in zip(settings.descriptor_names, val_rmse)
        },
    }

    with open(data_dir / "predicted_descriptors_metadata.json", "w") as f:
        json.dump(pred_metadata, f, indent=2)

    print(f"\nFiles saved to {data_dir}:")
    print(f"  - train_predicted_descriptors.npy ({train_pred_desc.shape})")
    print(f"  - val_predicted_descriptors.npy ({val_pred_desc.shape})")
    print(f"  - train_smiles.json")
    print(f"  - val_smiles.json")
    print(f"  - predicted_descriptors_metadata.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
