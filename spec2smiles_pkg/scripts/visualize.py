#!/usr/bin/env python
"""Generate visualization figures for SPEC2SMILES pipeline.

Usage:
    python scripts/visualize.py [--config config.yml] [--output-dir figures]

Or via Makefile:
    make visualize
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import settings, reload_config


def plot_part_a_parity(metrics_path: Path, output_dir: Path):
    """Plot parity plots for Part A descriptor predictions."""
    if not metrics_path.exists():
        print(f"Part A metrics not found: {metrics_path}")
        return

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
    plt.savefig(output_dir / "part_a_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'part_a_metrics.png'}")


def plot_training_history(metrics_path: Path, output_dir: Path):
    """Plot training loss curves for Part B."""
    if not metrics_path.exists():
        print(f"Part B metrics not found: {metrics_path}")
        return

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

    plt.savefig(output_dir / "part_b_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'part_b_training.png'}")


def plot_pipeline_results(results_path: Path, output_dir: Path):
    """Plot pipeline evaluation results."""
    if not results_path.exists():
        print(f"Evaluation results not found: {results_path}")
        return

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
    plt.savefig(output_dir / "pipeline_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'pipeline_results.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
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
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    output_dir = Path(args.output_dir) if args.output_dir else settings.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating SPEC2SMILES Visualizations")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    # Generate plots
    print("Generating Part A metrics plot...")
    plot_part_a_parity(settings.metrics_path / "part_a_metrics.json", output_dir)

    print("Generating Part B training plot...")
    plot_training_history(settings.metrics_path / "part_b_metrics.json", output_dir)

    print("Generating pipeline results plot...")
    plot_pipeline_results(settings.metrics_path / "evaluation_results.json", output_dir)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
