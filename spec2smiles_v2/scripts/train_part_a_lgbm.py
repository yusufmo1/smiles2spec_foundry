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
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


class TeeOutput:
    """Redirect stdout to both console and file."""
    def __init__(self, log_file: Path):
        self.file = open(log_file, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def setup_logging(log_dir: Path, script_name: str) -> Path:
    """Setup logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    # Redirect stdout to both console and file
    sys.stdout = TeeOutput(log_file)
    sys.stderr = sys.stdout  # Also capture stderr

    return log_file

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors_batch, get_descriptor_type


MAX_CLASSES_FOR_CLASSIFICATION = 0  # Use pure regression for ALL descriptors


def train_single_descriptor(args):
    """Train LightGBM for a single descriptor.

    Uses classification for low-cardinality discrete descriptors (< 10 classes),
    regression for high-cardinality discrete and continuous descriptors.
    """
    idx, name, X_train, y_train, X_val, y_val, X_test, y_test, desc_type = args

    # Extract target column
    y_train_col = y_train[:, idx]
    y_val_col = y_val[:, idx]
    y_test_col = y_test[:, idx]

    if desc_type == "discrete":
        # Convert to integer labels
        y_train_int = y_train_col.astype(int)
        y_val_int = y_val_col.astype(int)
        y_test_int = y_test_col.astype(int)

        # Determine number of classes from all data
        all_vals = np.concatenate([y_train_int, y_val_int, y_test_int])
        n_classes = int(np.max(all_vals)) + 1

        # Use classification only for low-cardinality targets
        if n_classes < MAX_CLASSES_FOR_CLASSIFICATION:
            # Classification for low-cardinality discrete descriptors
            params = {
                'objective': 'multiclass',
                'num_class': n_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': 1,
            }

            # Create datasets with integer labels
            train_data = lgb.Dataset(X_train, label=y_train_int)
            val_data = lgb.Dataset(X_val, label=y_val_int, reference=train_data)

            # Train with early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Predict: get class with highest probability
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Metrics for classification
            accuracy = float((y_pred == y_test_int).mean())
            mae = float(mean_absolute_error(y_test_int, y_pred))

            return {
                'name': name,
                'idx': idx,
                'model': model,
                'type': 'discrete_class',  # Classification
                'accuracy': accuracy,
                'MAE': mae,
                'n_classes': n_classes,
                'best_iteration': model.best_iteration
            }
        else:
            # Regression for high-cardinality discrete descriptors
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
                'n_jobs': 1,
            }

            train_data = lgb.Dataset(X_train, label=y_train_col)
            val_data = lgb.Dataset(X_val, label=y_val_col, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Predict and round to nearest integer
            y_pred = model.predict(X_test)
            y_pred_rounded = np.round(y_pred).astype(int)
            y_pred_rounded = np.clip(y_pred_rounded, 0, None)  # Ensure non-negative

            mae = float(mean_absolute_error(y_test_int, y_pred_rounded))
            rmse = float(np.sqrt(mean_squared_error(y_test_col, y_pred)))
            r2 = float(r2_score(y_test_col, y_pred))

            return {
                'name': name,
                'idx': idx,
                'model': model,
                'type': 'discrete_reg',  # Regression for high-cardinality
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'n_classes': n_classes,
                'best_iteration': model.best_iteration
            }
    else:
        # Regression for continuous and bounded descriptors
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
            'n_jobs': 1,
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train_col)
        val_data = lgb.Dataset(X_val, label=y_val_col, reference=train_data)

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

        # Clip bounded descriptors
        if desc_type == "bounded":
            y_pred = np.clip(y_pred, 0.0, 1.0)

        mae = float(mean_absolute_error(y_test_col, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test_col, y_pred)))
        r2 = float(r2_score(y_test_col, y_pred))

        return {
            'name': name,
            'idx': idx,
            'model': model,
            'type': desc_type,  # 'continuous' or 'bounded'
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
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

    # Setup logging to file and console
    log_dir = Path(__file__).parent.parent / "logs"
    log_file = setup_logging(log_dir, "train_part_a_lgbm")
    print(f"Logging to: {log_file}")

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
    descriptor_names = list(settings.descriptor_names)

    # Count descriptor types
    discrete_count = sum(1 for name in descriptor_names if get_descriptor_type(name) == "discrete")
    bounded_count = sum(1 for name in descriptor_names if get_descriptor_type(name) == "bounded")
    continuous_count = len(descriptor_names) - discrete_count - bounded_count

    print(f"Training {len(descriptor_names)} LightGBM models...")
    print(f"  Discrete (classification): {discrete_count}")
    print(f"  Bounded (regression+clip): {bounded_count}")
    print(f"  Continuous (regression):   {continuous_count}")
    print()

    # Prepare arguments for parallel training (now includes descriptor type)
    train_args = [
        (idx, name, X_train, y_train, X_val, y_val, X_test, y_test, get_descriptor_type(name))
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

    # Separate results by type
    discrete_class_results = [r for r in results if r['type'] == 'discrete_class']
    discrete_reg_results = [r for r in results if r['type'] == 'discrete_reg']
    continuous_results = [r for r in results if r['type'] in ('continuous', 'bounded')]

    # Sort each by their respective primary metric
    discrete_class_results.sort(key=lambda x: x['accuracy'], reverse=True)
    discrete_reg_results.sort(key=lambda x: x['R2'], reverse=True)
    continuous_results.sort(key=lambda x: x['R2'], reverse=True)

    # Print discrete classification results
    if discrete_class_results:
        print("\n" + "=" * 75)
        print(f"DISCRETE (Classification, <{MAX_CLASSES_FOR_CLASSIFICATION} classes) - sorted by Accuracy")
        print("=" * 75)
        print(f"{'Rank':<5} {'Descriptor':<30} {'Acc':>8} {'MAE':>8} {'Classes':>8}")
        print("-" * 75)

        for i, r in enumerate(discrete_class_results, 1):
            marker = "🟢" if r['accuracy'] >= 0.8 else ("🟡" if r['accuracy'] >= 0.6 else "🟠")
            print(f"{i:<5} {r['name']:<30} {r['accuracy']:>8.3f} {r['MAE']:>8.3f} {r['n_classes']:>8} {marker}")

        acc_values = [r['accuracy'] for r in discrete_class_results]
        print("-" * 75)
        print(f"Mean Accuracy:   {np.mean(acc_values):.4f}")
        print(f"Acc >= 0.8: {sum(1 for a in acc_values if a >= 0.8)} descriptors")

    # Print discrete regression results (high-cardinality)
    if discrete_reg_results:
        print("\n" + "=" * 75)
        print(f"DISCRETE (Regression+Round, >={MAX_CLASSES_FOR_CLASSIFICATION} classes) - sorted by R²")
        print("=" * 75)
        print(f"{'Rank':<5} {'Descriptor':<30} {'R²':>8} {'MAE':>8} {'Classes':>8}")
        print("-" * 75)

        for i, r in enumerate(discrete_reg_results, 1):
            marker = "🟢" if r['R2'] >= 0.7 else ("🟡" if r['R2'] >= 0.5 else "🟠")
            print(f"{i:<5} {r['name']:<30} {r['R2']:>8.4f} {r['MAE']:>8.3f} {r['n_classes']:>8} {marker}")

        r2_values = [r['R2'] for r in discrete_reg_results]
        print("-" * 75)
        print(f"Mean R²: {np.mean(r2_values):.4f}")

    # Print continuous results
    if continuous_results:
        print("\n" + "=" * 75)
        print("CONTINUOUS (Regression) - sorted by R²")
        print("=" * 75)
        print(f"{'Rank':<5} {'Descriptor':<30} {'R²':>10} {'MAE':>10} {'Type':>10}")
        print("-" * 75)

        for i, r in enumerate(continuous_results, 1):
            marker = "🟢" if r['R2'] >= 0.7 else ("🟡" if r['R2'] >= 0.5 else "🟠")
            print(f"{i:<5} {r['name']:<30} {r['R2']:>10.4f} {r['MAE']:>10.4f} {r['type']:>10} {marker}")

        r2_values = [r['R2'] for r in continuous_results]
        print("-" * 75)
        print(f"Mean R²:   {np.mean(r2_values):.4f}")
        print(f"R² >= 0.7: {sum(1 for r in r2_values if r >= 0.7)} descriptors")

    # Overall summary
    print("\n" + "=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)
    print(f"Total descriptors:        {len(results)}")
    print(f"  Discrete (classify):    {len(discrete_class_results)} (<{MAX_CLASSES_FOR_CLASSIFICATION} classes)")
    print(f"  Discrete (regress):     {len(discrete_reg_results)} (>={MAX_CLASSES_FOR_CLASSIFICATION} classes)")
    print(f"  Continuous:             {len(continuous_results)}")
    if discrete_class_results:
        print(f"Mean accuracy (classify): {np.mean([r['accuracy'] for r in discrete_class_results]):.4f}")
    if discrete_reg_results:
        print(f"Mean R² (discrete reg):   {np.mean([r['R2'] for r in discrete_reg_results]):.4f}")
    if continuous_results:
        print(f"Mean R² (continuous):     {np.mean([r['R2'] for r in continuous_results]):.4f}")

    # Save models and metadata
    import pickle
    output_dir = settings.models_path / "part_a_lgbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Save descriptor metadata (types and class info) for inference
    descriptor_meta = {}
    for r in results:
        if r['type'] == 'discrete_class':
            descriptor_meta[r['name']] = {
                'type': 'discrete_class',
                'n_classes': r['n_classes']
            }
        elif r['type'] == 'discrete_reg':
            descriptor_meta[r['name']] = {
                'type': 'discrete_reg',
                'n_classes': r['n_classes']
            }
        else:
            descriptor_meta[r['name']] = {
                'type': r['type']
            }

    with open(output_dir / "descriptor_meta.json", "w") as f:
        json.dump(descriptor_meta, f, indent=2)

    # Generate and save test set predictions for visualization
    print("\nGenerating test predictions for visualization...")
    y_pred_test = np.zeros_like(y_test)

    # Create lookup for result info by name
    result_lookup = {r['name']: r for r in results}

    for idx, name in enumerate(descriptor_names):
        model = models[name]
        result_type = result_lookup[name]['type']

        if result_type == "discrete_class":
            # Classification: get probabilities and take argmax
            y_pred_proba = model.predict(X_test)
            y_pred_test[:, idx] = np.argmax(y_pred_proba, axis=1)
        elif result_type == "discrete_reg":
            # Regression + round for high-cardinality discrete
            y_pred = model.predict(X_test)
            y_pred_test[:, idx] = np.clip(np.round(y_pred), 0, None)
        elif result_type == "bounded":
            # Regression + clip for bounded [0, 1]
            y_pred_test[:, idx] = np.clip(model.predict(X_test), 0.0, 1.0)
        else:
            # Continuous: direct regression
            y_pred_test[:, idx] = model.predict(X_test)

    # Save predictions as numpy arrays
    predictions_dir = settings.metrics_path
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir / "part_a_lgbm_y_true.npy", y_test)
    np.save(predictions_dir / "part_a_lgbm_y_pred.npy", y_pred_test)
    print(f"  Saved test predictions: {y_test.shape}")

    # Save metrics
    metrics_path = settings.metrics_path / "part_a_lgbm_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-descriptor metrics based on type
    per_descriptor = {}
    for r in results:
        if r['type'] == 'discrete_class':
            per_descriptor[r['name']] = {
                'type': 'discrete_class',
                'accuracy': r['accuracy'],
                'MAE': r['MAE'],
                'n_classes': r['n_classes']
            }
        elif r['type'] == 'discrete_reg':
            per_descriptor[r['name']] = {
                'type': 'discrete_reg',
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'R2': r['R2'],
                'n_classes': r['n_classes']
            }
        else:
            per_descriptor[r['name']] = {
                'type': r['type'],
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'R2': r['R2']
            }

    # Compute summary statistics
    summary = {
        "n_descriptors": len(results),
        "n_discrete_class": len(discrete_class_results),
        "n_discrete_reg": len(discrete_reg_results),
        "n_continuous": len(continuous_results),
        "max_classes_for_classification": MAX_CLASSES_FOR_CLASSIFICATION,
    }

    if discrete_class_results:
        acc_values = [r['accuracy'] for r in discrete_class_results]
        summary["mean_accuracy_discrete_class"] = float(np.mean(acc_values))
        summary["best_discrete_class"] = discrete_class_results[0]['name']
        summary["best_discrete_class_accuracy"] = float(discrete_class_results[0]['accuracy'])

    if discrete_reg_results:
        r2_values = [r['R2'] for r in discrete_reg_results]
        summary["mean_r2_discrete_reg"] = float(np.mean(r2_values))

    if continuous_results:
        r2_values = [r['R2'] for r in continuous_results]
        summary["mean_r2_continuous"] = float(np.mean(r2_values))
        summary["best_continuous"] = continuous_results[0]['name']
        summary["best_continuous_r2"] = float(continuous_results[0]['R2'])

    full_metrics = {
        "model_type": "lightgbm_mixed",
        "summary": summary,
        "per_descriptor": per_descriptor,
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\nModels saved to {output_dir}")
    print(f"Metrics saved to {metrics_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
