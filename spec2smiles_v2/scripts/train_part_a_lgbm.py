#!/usr/bin/env python
"""Train Part A (Spectrum -> Descriptors) using LightGBM.

Trains one LightGBM model per descriptor for better performance
than the neural network approach.

Usage:
    python scripts/train_part_a_lgbm.py --config config_gnps_optimal.yml
"""

import argparse
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors_batch


def train_single_descriptor(args):
    """Train LightGBM for a single descriptor."""
    idx, name, X_train, y_train, X_val, y_val, X_test, y_test = args

    # LightGBM parameters optimized for regression
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': 1,  # Single thread per model since we parallelize across descriptors
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train[:, idx])
    val_data = lgb.Dataset(X_val, label=y_val[:, idx], reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_true = y_test[:, idx]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'name': name,
        'idx': idx,
        'model': model,
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'best_iteration': model.best_iteration
    }


def main():
    parser = argparse.ArgumentParser(description="Train Part A with LightGBM")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all CPUs)"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()

    print("=" * 60)
    print("Training Part A (Spectrum -> Descriptors) with LightGBM")
    print("=" * 60)
    print(f"Dataset:     {settings.dataset}")
    print(f"Descriptors: {len(settings.descriptor_names)}")
    print(f"Workers:     {n_jobs}")
    print()

    # Load and preprocess data
    from src.services.data_loader import DataLoaderService

    data_loader = DataLoaderService(data_dir=Path(settings.data_input_dir) / settings.dataset)

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

        # Extract SMILES and calculate descriptors in parallel
        smiles_list = [sample["smiles"] for sample in raw_data]

        print(f"Calculating {len(settings.descriptor_names)} descriptors in parallel...")
        all_descriptors, valid_mask = calculate_descriptors_batch(
            smiles_list,
            settings.descriptor_names,
            return_valid_mask=True,
            n_jobs=n_jobs
        )

        # Process spectra
        spectra = []
        descriptors = []
        desc_idx = 0

        for i, sample in enumerate(tqdm(raw_data, desc="Processing spectra")):
            if valid_mask[i]:
                spectrum = process_spectrum(
                    sample["peaks"],
                    n_bins=settings.n_bins,
                    bin_width=settings.bin_width,
                    max_mz=settings.max_mz,
                    transform=settings.transform,
                    normalize=settings.normalize,
                )
                spectra.append(spectrum)
                descriptors.append(all_descriptors[desc_idx])
                desc_idx += 1

        X = np.array(spectra)
        y = np.array(descriptors)

        print(f"Valid samples: {len(X)}")

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

    # Train one model per descriptor
    print(f"Training {len(settings.descriptor_names)} LightGBM models...")

    descriptor_names = list(settings.descriptor_names)

    # Prepare arguments for parallel training
    train_args = [
        (idx, name, X_train, y_train, X_val, y_val, X_test, y_test)
        for idx, name in enumerate(descriptor_names)
    ]

    # Train in parallel
    results = []
    models = {}

    # Use fewer workers to avoid memory issues with large data copies
    effective_workers = min(n_jobs, 8)

    with Pool(effective_workers) as pool:
        for result in tqdm(
            pool.imap(train_single_descriptor, train_args),
            total=len(descriptor_names),
            desc=f"Training LightGBM ({effective_workers} workers)"
        ):
            results.append(result)
            models[result['name']] = result['model']

    # Sort by R2
    results.sort(key=lambda x: x['R2'], reverse=True)

    # Print results
    print("\n" + "=" * 60)
    print("Results (sorted by R²)")
    print("=" * 60)
    print(f"{'Rank':<5} {'Descriptor':<30} {'R²':>10} {'MAE':>10}")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        marker = "🟢" if r['R2'] >= 0.7 else ("🟡" if r['R2'] >= 0.5 else "🟠")
        print(f"{i:<5} {r['name']:<30} {r['R2']:>10.4f} {r['MAE']:>10.4f} {marker}")

    # Summary statistics
    r2_values = [r['R2'] for r in results]
    print("-" * 60)
    print(f"Mean R²:   {np.mean(r2_values):.4f}")
    print(f"Median R²: {np.median(r2_values):.4f}")
    print(f"Max R²:    {max(r2_values):.4f}")
    print(f"Min R²:    {min(r2_values):.4f}")
    print(f"R² >= 0.7: {sum(1 for r in r2_values if r >= 0.7)} descriptors")
    print(f"R² >= 0.8: {sum(1 for r in r2_values if r >= 0.8)} descriptors")

    # Save models
    import pickle
    output_dir = settings.models_path / "part_a_lgbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Save metrics
    metrics_path = settings.metrics_path / "part_a_lgbm_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    per_descriptor = {r['name']: {'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2']} for r in results}

    full_metrics = {
        "model_type": "lightgbm",
        "summary": {
            "mean_r2": float(np.mean(r2_values)),
            "median_r2": float(np.median(r2_values)),
            "best_descriptor": results[0]['name'],
            "best_r2": float(results[0]['R2']),
            "worst_descriptor": results[-1]['name'],
            "worst_r2": float(results[-1]['R2']),
        },
        "per_descriptor": per_descriptor,
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\nModels saved to {output_dir}")
    print(f"Metrics saved to {metrics_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
