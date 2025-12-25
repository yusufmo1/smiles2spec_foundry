#!/usr/bin/env python
"""Generate visualization figures for SPEC2SMILES pipeline.

Usage:
    python scripts/visualize.py [--config config.yml] [--output-dir figures]
    python scripts/visualize.py --part-a-only
    python scripts/visualize.py --part-b-only
    python scripts/visualize.py --hybrid-only
    python scripts/visualize.py --skip-missing

Or via Makefile:
    make visualize
    make visualize-part-a
    make visualize-part-b
    make visualize-hybrid
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import settings, reload_config


# =============================================================================
# Part A Visualizations
# =============================================================================


def plot_part_a_parity(metrics_path: Path, output_dir: Path) -> bool:
    """Plot parity plots for Part A descriptor predictions.

    Args:
        metrics_path: Path to part_a_metrics.json
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if metrics missing
    """
    if not metrics_path.exists():
        print(f"  [SKIP] Part A metrics not found: {metrics_path}")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    per_descriptor = metrics.get("per_descriptor", metrics)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (name, data) in enumerate(per_descriptor.items()):
        if i >= 12:
            break
        ax = axes[i]
        r2 = data.get("R2", 0)
        rmse = data.get("RMSE", 0)

        ax.set_title(f"{name}\nR² = {r2:.3f}, RMSE = {rmse:.2f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "part_a_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def plot_transformer_training(log_path: Path, output_dir: Path) -> bool:
    """Plot transformer training curves from epoch log CSV.

    Args:
        log_path: Path to epoch log CSV (e.g., train_transformer_*_epochs.csv)
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if log missing
    """
    if not log_path.exists():
        print(f"  [SKIP] Transformer epoch log not found: {log_path}")
        return False

    # Read CSV epoch log
    epochs, train_loss, val_loss, train_r2, val_r2, lr = [], [], [], [], [], []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_r2.append(float(row["train_r2"]))
            val_r2.append(float(row["val_r2"]))
            lr.append(float(row["lr"]))

    if not epochs:
        print(f"  [SKIP] Empty epoch log: {log_path}")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² curves
    ax = axes[1]
    ax.plot(epochs, train_r2, label="Train R²", color="#2ca02c")
    ax.plot(epochs, val_r2, label="Val R²", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("R² Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 3: Learning rate schedule
    ax = axes[2]
    ax.plot(epochs, lr, color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    output_path = output_dir / "part_a_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def find_latest_epoch_log(log_dir: Path, model_name: str) -> Optional[Path]:
    """Find the most recent epoch log file for a model.

    Args:
        log_dir: Directory containing log files
        model_name: Model name (e.g., "transformer", "vae")

    Returns:
        Path to most recent log file, or None if not found
    """
    pattern = f"train_{model_name}_*_epochs.csv"
    logs = sorted(log_dir.glob(pattern), reverse=True)
    return logs[0] if logs else None


def plot_hybrid_training(log_path: Path, output_dir: Path) -> bool:
    """Plot hybrid CNN-Transformer training curves from epoch log CSV.

    Args:
        log_path: Path to epoch log CSV (e.g., train_hybrid_*_epochs.csv)
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if log missing
    """
    if not log_path.exists():
        print(f"  [SKIP] Hybrid epoch log not found: {log_path}")
        return False

    # Read CSV epoch log
    epochs, train_loss, val_loss, train_r2, val_r2, lr = [], [], [], [], [], []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_r2.append(float(row["train_r2"]))
            val_r2.append(float(row["val_r2"]))
            lr.append(float(row["lr"]))

    if not epochs:
        print(f"  [SKIP] Empty epoch log: {log_path}")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Hybrid CNN-Transformer: Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² curves
    ax = axes[1]
    ax.plot(epochs, train_r2, label="Train R²", color="#2ca02c")
    ax.plot(epochs, val_r2, label="Val R²", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("R² Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 3: Learning rate schedule (OneCycleLR)
    ax = axes[2]
    ax.plot(epochs, lr, color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("OneCycleLR Schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    output_path = output_dir / "part_a_hybrid_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def generate_part_a_plots(output_dir: Optional[Path] = None, skip_missing: bool = False) -> None:
    """Generate all Part A visualizations.

    Args:
        output_dir: Output directory (defaults to settings.figures_path)
        skip_missing: If True, silently skip missing files
    """
    output_dir = output_dir or settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Part A visualizations...")

    # Metrics plot
    metrics_path = settings.metrics_path / "part_a_metrics.json"
    plot_part_a_parity(metrics_path, output_dir)

    # Training curves for neural models
    transformer_log = find_latest_epoch_log(settings.logs_path, "transformer")
    hybrid_log = find_latest_epoch_log(settings.logs_path, "hybrid")

    if transformer_log:
        plot_transformer_training(transformer_log, output_dir)

    if hybrid_log:
        plot_hybrid_training(hybrid_log, output_dir)

    if not transformer_log and not hybrid_log and not skip_missing:
        print("  [SKIP] No neural model epoch logs found")


# =============================================================================
# Part B Visualizations
# =============================================================================


def plot_training_history(metrics_path: Path, output_dir: Path) -> bool:
    """Plot training loss curves for Part B.

    Args:
        metrics_path: Path to part_b_metrics.json
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if metrics missing
    """
    if not metrics_path.exists():
        print(f"  [SKIP] Part B metrics not found: {metrics_path}")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Part B Training History")

    final_loss = metrics.get("final_train_loss", 0)
    n_epochs = metrics.get("n_epochs", 0)

    ax.text(0.5, 0.5, f"Final Train Loss: {final_loss:.4f}\nEpochs: {n_epochs}",
            transform=ax.transAxes, ha="center", va="center", fontsize=14)

    output_path = output_dir / "part_b_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def plot_vae_loss_components(log_path: Path, output_dir: Path) -> bool:
    """Plot VAE loss components (reconstruction vs KL divergence).

    Args:
        log_path: Path to epoch log CSV (e.g., train_vae_*_epochs.csv)
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if log missing
    """
    if not log_path.exists():
        print(f"  [SKIP] VAE epoch log not found: {log_path}")
        return False

    # Read CSV epoch log
    epochs, train_loss, val_loss, recon_loss, kl_loss, beta = [], [], [], [], [], []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            recon_loss.append(float(row["recon_loss"]))
            kl_loss.append(float(row["kl_loss"]))
            beta.append(float(row["beta"]))

    if not epochs:
        print(f"  [SKIP] Empty epoch log: {log_path}")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Overall loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss components
    ax = axes[1]
    ax.plot(epochs, recon_loss, label="Reconstruction", color="#2ca02c")
    ax.plot(epochs, kl_loss, label="KL Divergence", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Beta annealing schedule
    ax = axes[2]
    ax.plot(epochs, beta, color="#9467bd", linewidth=2)
    ax.fill_between(epochs, 0, beta, alpha=0.3, color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Beta (KL Weight)")
    ax.set_title("Cyclical KL Annealing")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_path = output_dir / "part_b_loss_components.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def generate_part_b_plots(output_dir: Optional[Path] = None, skip_missing: bool = False) -> None:
    """Generate all Part B visualizations.

    Args:
        output_dir: Output directory (defaults to settings.figures_path)
        skip_missing: If True, silently skip missing files
    """
    output_dir = output_dir or settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Part B visualizations...")

    # Basic training summary
    metrics_path = settings.metrics_path / "part_b_metrics.json"
    plot_training_history(metrics_path, output_dir)

    # Detailed loss components from epoch log
    log_path = find_latest_epoch_log(settings.logs_path, "vae")
    if log_path:
        plot_vae_loss_components(log_path, output_dir)
    elif not skip_missing:
        print("  [SKIP] No VAE epoch log found")


# =============================================================================
# Pipeline Visualizations
# =============================================================================


def plot_pipeline_results(results_path: Path, output_dir: Path) -> bool:
    """Plot pipeline evaluation results.

    Args:
        results_path: Path to evaluation_results.json
        output_dir: Directory to save figures

    Returns:
        True if plot was generated, False if results missing
    """
    if not results_path.exists():
        print(f"  [SKIP] Evaluation results not found: {results_path}")
        return False

    with open(results_path) as f:
        results = json.load(f)

    pipeline = results.get("pipeline", {})

    # Hit@K bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Hit@K
    ax = axes[0]
    k_values = [1, 5, 10]
    hits = [
        pipeline.get("hit_at_1", 0),
        pipeline.get("hit_at_5", 0),
        pipeline.get("hit_at_10", 0),
    ]

    bars = ax.bar(k_values, hits, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_xlabel("K")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit@K Performance")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1)

    for bar, hit in zip(bars, hits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{hit:.1%}", ha="center", va="bottom", fontsize=10)

    # Right: Other metrics
    ax = axes[1]
    metric_names = ["Best Tanimoto", "Validity", "Uniqueness"]
    metric_values = [
        pipeline.get("mean_best_tanimoto", 0),
        pipeline.get("validity_rate", 0),
        pipeline.get("uniqueness", 0),
    ]

    bars = ax.bar(metric_names, metric_values, color=["#d62728", "#9467bd", "#8c564b"])
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Quality Metrics")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "pipeline_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {output_path}")
    return True


def generate_pipeline_plots(output_dir: Optional[Path] = None, skip_missing: bool = False) -> None:
    """Generate pipeline evaluation visualizations.

    Args:
        output_dir: Output directory (defaults to settings.figures_path)
        skip_missing: If True, silently skip missing files
    """
    output_dir = output_dir or settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating pipeline visualizations...")

    results_path = settings.metrics_path / "evaluation_results.json"
    plot_pipeline_results(results_path, output_dir)


# =============================================================================
# Hybrid Model Visualizations
# =============================================================================


def generate_hybrid_plots(output_dir: Optional[Path] = None, skip_missing: bool = False) -> None:
    """Generate hybrid model-specific visualizations.

    Args:
        output_dir: Output directory (defaults to settings.figures_path)
        skip_missing: If True, silently skip missing files
    """
    output_dir = output_dir or settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Hybrid model visualizations...")

    # Training curves
    hybrid_log = find_latest_epoch_log(settings.logs_path, "hybrid")
    if hybrid_log:
        plot_hybrid_training(hybrid_log, output_dir)
    elif not skip_missing:
        print("  [SKIP] No hybrid epoch log found")


# =============================================================================
# Combined Generation Functions (for training scripts)
# =============================================================================


def generate_all_plots(output_dir: Optional[Path] = None, skip_missing: bool = True) -> None:
    """Generate all pipeline visualizations.

    Args:
        output_dir: Output directory (defaults to settings.figures_path)
        skip_missing: If True, silently skip missing files
    """
    output_dir = output_dir or settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_part_a_plots(output_dir, skip_missing)
    generate_part_b_plots(output_dir, skip_missing)
    generate_pipeline_plots(output_dir, skip_missing)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPEC2SMILES visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualize.py                    # Generate all plots
    python scripts/visualize.py --part-a-only      # Part A only
    python scripts/visualize.py --part-b-only      # Part B only
    python scripts/visualize.py --skip-missing     # Skip missing files silently
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default from config)"
    )
    parser.add_argument(
        "--part-a-only",
        action="store_true",
        help="Generate Part A plots only"
    )
    parser.add_argument(
        "--part-b-only",
        action="store_true",
        help="Generate Part B plots only"
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Generate pipeline results plots only"
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Generate hybrid model plots only"
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip if metrics/log files are missing"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    output_dir = Path(args.output_dir) if args.output_dir else settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SPEC2SMILES Visualization Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    # Determine which plots to generate
    if args.part_a_only:
        generate_part_a_plots(output_dir, args.skip_missing)
    elif args.part_b_only:
        generate_part_b_plots(output_dir, args.skip_missing)
    elif args.pipeline_only:
        generate_pipeline_plots(output_dir, args.skip_missing)
    elif args.hybrid_only:
        generate_hybrid_plots(output_dir, args.skip_missing)
    else:
        generate_all_plots(output_dir, args.skip_missing)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
