#!/usr/bin/env python
"""Train Part A (Spectrum -> Descriptors) using Multi-Head Transformer.

Uses shared CNN-Transformer backbone with:
- Classification heads for low-cardinality discrete descriptors (< 10 classes)
- Regression heads for continuous + bounded + high-cardinality discrete descriptors

Usage:
    python scripts/train_part_a_multihead.py --config config.yml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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
    sys.stdout = TeeOutput(log_file)
    sys.stderr = sys.stdout
    return log_file


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors_batch, get_descriptor_type
from src.models.multihead_transformer import MultiHeadTransformer, MultiTaskLoss


MAX_CLASSES_FOR_CLASSIFICATION = 10


def get_class_descriptor_info(y_train: np.ndarray, descriptor_names: list) -> dict:
    """Determine which descriptors should use classification heads.

    Returns dict of {name: n_classes} for classification descriptors.
    """
    class_info = {}

    for idx, name in enumerate(descriptor_names):
        desc_type = get_descriptor_type(name)
        if desc_type == "discrete":
            # Get max value to determine n_classes
            n_classes = int(y_train[:, idx].max()) + 1
            if n_classes < MAX_CLASSES_FOR_CLASSIFICATION:
                class_info[name] = n_classes

    return class_info


def prepare_targets(y: np.ndarray, descriptor_names: list, class_info: dict, device: torch.device):
    """Prepare classification and regression targets.

    Returns:
        class_targets: Dict[name, tensor] for classification
        reg_targets: tensor (batch, n_reg) for regression
        reg_names: list of regression descriptor names
    """
    class_names = list(class_info.keys())
    reg_names = [n for n in descriptor_names if n not in class_names]

    # Classification targets
    class_targets = {}
    for name in class_names:
        idx = descriptor_names.index(name)
        class_targets[name] = torch.tensor(y[:, idx], dtype=torch.long, device=device)

    # Regression targets (ensure contiguous for MPS compatibility)
    reg_indices = [descriptor_names.index(n) for n in reg_names]
    reg_targets = torch.tensor(y[:, reg_indices].copy(), dtype=torch.float32, device=device)

    return class_targets, reg_targets, reg_names


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    class_names: list,
    reg_names: list,
    descriptor_names: list,
    class_info: dict,
    device: torch.device,
) -> tuple:
    """Evaluate model on dataset.

    Returns:
        (loss, class_acc_dict, reg_r2_dict, class_preds, reg_preds, class_true, reg_true)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Collect predictions
    all_class_preds = {name: [] for name in class_names}
    all_class_true = {name: [] for name in class_names}
    all_reg_preds = []
    all_reg_true = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            class_logits, reg_preds = model(batch_x)

            # Prepare targets
            class_targets, reg_targets, _ = prepare_targets(
                batch_y.cpu().numpy(), descriptor_names, class_info, device
            )

            # Compute loss
            loss, _, _ = criterion(class_logits, reg_preds, class_targets, reg_targets)
            total_loss += loss.item()
            n_batches += 1

            # Store predictions
            for name in class_names:
                preds = class_logits[name].argmax(dim=-1).cpu().numpy()
                true = class_targets[name].cpu().numpy()
                all_class_preds[name].extend(preds)
                all_class_true[name].extend(true)

            all_reg_preds.append(reg_preds.cpu().numpy())
            all_reg_true.append(reg_targets.cpu().numpy())

    # Compute metrics
    class_acc = {}
    for name in class_names:
        class_acc[name] = accuracy_score(all_class_true[name], all_class_preds[name])

    # Regression R²
    reg_preds_all = np.vstack(all_reg_preds)
    reg_true_all = np.vstack(all_reg_true)

    reg_r2 = {}
    for i, name in enumerate(reg_names):
        reg_r2[name] = r2_score(reg_true_all[:, i], reg_preds_all[:, i])

    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, class_acc, reg_r2, all_class_preds, reg_preds_all, all_class_true, reg_true_all


def main():
    parser = argparse.ArgumentParser(description="Train Part A with Multi-Head Transformer")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yml file")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Workers for data loading")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(__file__).parent.parent / "logs"
    log_file = setup_logging(log_dir, "train_part_a_multihead")
    print(f"Logging to: {log_file}")

    # Reload config if provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("=" * 70)
    print("Training Part A (Spectrum -> Descriptors) with Multi-Head Transformer")
    print("=" * 70)
    print(f"Dataset:     {settings.dataset}")
    print(f"Descriptors: {len(settings.descriptor_names)}")
    print(f"Device:      {device}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print()

    # Load data
    from src.services.data_loader import DataLoaderService
    from multiprocessing import cpu_count

    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()

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

        smiles_list = [sample["smiles"] for sample in raw_data]

        print(f"Calculating {len(settings.descriptor_names)} descriptors in parallel...")
        all_descriptors, valid_mask = calculate_descriptors_batch(
            smiles_list,
            settings.descriptor_names,
            return_valid_mask=True,
            n_jobs=n_jobs
        )

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

    # Determine classification vs regression descriptors
    descriptor_names = list(settings.descriptor_names)
    class_info = get_class_descriptor_info(y_train, descriptor_names)
    class_names = list(class_info.keys())
    reg_names = [n for n in descriptor_names if n not in class_names]

    print(f"\nDescriptor split (threshold: {MAX_CLASSES_FOR_CLASSIFICATION} classes):")
    print(f"  Classification: {len(class_names)} descriptors")
    for name, n_classes in class_info.items():
        print(f"    - {name}: {n_classes} classes")
    print(f"  Regression: {len(reg_names)} descriptors")
    print()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = MultiHeadTransformer(
        n_bins=settings.n_bins,
        class_descriptor_info=class_info,
        reg_descriptor_names=reg_names,
        cnn_hidden=256,
        transformer_dim=256,
        n_heads=8,
        n_transformer_layers=4,
        d_ff=1024,
        dropout=0.2,
        class_head_hidden=256,
        reg_head_hidden=128,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = MultiTaskLoss(class_names, reg_names, class_weight=1.0, reg_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_class_acc": [], "val_reg_r2": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            class_logits, reg_preds = model(batch_x)

            # Prepare targets
            class_targets, reg_targets, _ = prepare_targets(
                batch_y.cpu().numpy(), descriptor_names, class_info, device
            )

            # Loss
            loss, class_loss, reg_loss = criterion(class_logits, reg_preds, class_targets, reg_targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / n_batches

        # Validation
        val_loss, val_class_acc, val_reg_r2, _, _, _, _ = evaluate(
            model, val_loader, criterion, class_names, reg_names,
            descriptor_names, class_info, device
        )

        # Track history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_class_acc"].append(np.mean(list(val_class_acc.values())) if val_class_acc else 0)
        history["val_reg_r2"].append(np.mean(list(val_reg_r2.values())) if val_reg_r2 else 0)

        # Print progress
        mean_class_acc = np.mean(list(val_class_acc.values())) if val_class_acc else 0
        mean_reg_r2 = np.mean(list(val_reg_r2.values())) if val_reg_r2 else 0

        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {mean_class_acc:.4f} | Val R²: {mean_reg_r2:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_class_acc, test_reg_r2, test_class_preds, test_reg_preds, test_class_true, test_reg_true = evaluate(
        model, test_loader, criterion, class_names, reg_names,
        descriptor_names, class_info, device
    )

    # Print classification results
    if class_names:
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION RESULTS (<{MAX_CLASSES_FOR_CLASSIFICATION} classes)")
        print(f"{'='*70}")
        print(f"{'Descriptor':<30} {'Accuracy':>10} {'Classes':>10}")
        print("-" * 70)

        for name in sorted(class_names, key=lambda n: test_class_acc[n], reverse=True):
            n_classes = class_info[name]
            acc = test_class_acc[name]
            marker = "🟢" if acc >= 0.8 else ("🟡" if acc >= 0.6 else "🟠")
            print(f"{name:<30} {acc:>10.4f} {n_classes:>10} {marker}")

        print("-" * 70)
        mean_acc = np.mean(list(test_class_acc.values()))
        print(f"Mean Accuracy: {mean_acc:.4f}")

    # Print regression results
    if reg_names:
        print(f"\n{'='*70}")
        print("REGRESSION RESULTS")
        print(f"{'='*70}")
        print(f"{'Descriptor':<30} {'R²':>10} {'Type':>15}")
        print("-" * 70)

        sorted_reg = sorted(reg_names, key=lambda n: test_reg_r2[n], reverse=True)
        for name in sorted_reg:
            r2 = test_reg_r2[name]
            desc_type = get_descriptor_type(name)
            marker = "🟢" if r2 >= 0.7 else ("🟡" if r2 >= 0.5 else "🟠")
            print(f"{name:<30} {r2:>10.4f} {desc_type:>15} {marker}")

        print("-" * 70)
        mean_r2 = np.mean(list(test_reg_r2.values()))
        print(f"Mean R²: {mean_r2:.4f}")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total descriptors:      {len(descriptor_names)}")
    print(f"  Classification heads: {len(class_names)}")
    print(f"  Regression heads:     {len(reg_names)}")
    print(f"Test loss:              {test_loss:.4f}")
    if class_names:
        print(f"Mean accuracy (class):  {np.mean(list(test_class_acc.values())):.4f}")
    if reg_names:
        print(f"Mean R² (regression):   {np.mean(list(test_reg_r2.values())):.4f}")

    # Save model
    output_dir = settings.models_path / "part_a_multihead"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_descriptor_info": class_info,
        "reg_descriptor_names": reg_names,
        "descriptor_names": descriptor_names,
        "n_bins": settings.n_bins,
        "config": {
            "cnn_hidden": 256,
            "transformer_dim": 256,
            "n_heads": 8,
            "n_transformer_layers": 4,
            "d_ff": 1024,
            "dropout": 0.2,
        }
    }, output_dir / "model.pt")

    # Save metrics
    metrics = {
        "model_type": "multihead_transformer",
        "test_loss": test_loss,
        "n_params": n_params,
        "classification": {
            "n_descriptors": len(class_names),
            "mean_accuracy": float(np.mean(list(test_class_acc.values()))) if class_names else None,
            "per_descriptor": {name: float(acc) for name, acc in test_class_acc.items()},
        },
        "regression": {
            "n_descriptors": len(reg_names),
            "mean_r2": float(np.mean(list(test_reg_r2.values()))) if reg_names else None,
            "per_descriptor": {name: float(r2) for name, r2 in test_reg_r2.items()},
        },
        "training": {
            "epochs": len(history["train_loss"]),
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        }
    }

    metrics_path = settings.metrics_path / "part_a_multihead_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")

    # Comparison with LightGBM
    lgbm_metrics_path = settings.metrics_path / "part_a_lgbm_metrics.json"
    if lgbm_metrics_path.exists():
        print(f"\n{'='*70}")
        print("COMPARISON WITH LightGBM")
        print(f"{'='*70}")

        with open(lgbm_metrics_path) as f:
            lgbm_metrics = json.load(f)

        lgbm_summary = lgbm_metrics.get("summary", {})
        lgbm_class_acc = lgbm_summary.get("mean_accuracy_discrete_class", 0)
        lgbm_discrete_r2 = lgbm_summary.get("mean_r2_discrete_reg", 0)
        lgbm_continuous_r2 = lgbm_summary.get("mean_r2_continuous", 0)

        mh_class_acc = np.mean(list(test_class_acc.values())) if class_names else 0
        mh_reg_r2 = np.mean(list(test_reg_r2.values())) if reg_names else 0

        print(f"{'Metric':<30} {'LightGBM':>12} {'MultiHead':>12} {'Delta':>10}")
        print("-" * 70)
        print(f"{'Classification Accuracy':<30} {lgbm_class_acc:>12.4f} {mh_class_acc:>12.4f} {mh_class_acc - lgbm_class_acc:>+10.4f}")
        print(f"{'Discrete Reg R²':<30} {lgbm_discrete_r2:>12.4f} {'-':>12} {'-':>10}")
        print(f"{'Continuous Reg R²':<30} {lgbm_continuous_r2:>12.4f} {'-':>12} {'-':>10}")
        print(f"{'Overall Reg R²':<30} {'-':>12} {mh_reg_r2:>12.4f} {'-':>10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
