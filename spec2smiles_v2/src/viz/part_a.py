"""Part A visualization: Spectrum → Descriptors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import COLORS
from .loaders import find_latest_log, read_epoch_log


def plot_metrics(metrics: dict, output_dir: Path, metrics_dir: Path = None) -> bool:
    """Plot true vs predicted scatter plots for all descriptors."""
    if "part_a_lgbm" not in metrics:
        print("  [SKIP] Part A LightGBM metrics not found")
        return False

    # Try to load prediction data
    if metrics_dir is None:
        metrics_dir = output_dir.parent / "metrics"

    y_true_path = metrics_dir / "part_a_lgbm_y_true.npy"
    y_pred_path = metrics_dir / "part_a_lgbm_y_pred.npy"

    if not y_true_path.exists() or not y_pred_path.exists():
        print("  [SKIP] Prediction data not found - run train_part_a_lgbm.py first")
        return False

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    per_desc = metrics["part_a_lgbm"].get("per_descriptor", {})
    descriptor_names = list(per_desc.keys())

    # Sort by R² (best first)
    r2_values = [per_desc[name].get("R2", per_desc[name].get("r2", 0)) for name in descriptor_names]
    sorted_idx = np.argsort(r2_values)[::-1]
    descriptor_names = [descriptor_names[i] for i in sorted_idx]

    # Create 4x4 grid for top 16 descriptors (or less)
    n_plots = min(16, len(descriptor_names))
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_plots):
        ax = axes[i]
        name = descriptor_names[i]

        # Find original index for this descriptor
        orig_idx = list(per_desc.keys()).index(name)

        y_t = y_true[:, orig_idx]
        y_p = y_pred[:, orig_idx]

        r2 = per_desc[name].get("R2", per_desc[name].get("r2", 0))
        rmse = per_desc[name].get("RMSE", per_desc[name].get("rmse", 0))

        # Scatter plot
        ax.scatter(y_t, y_p, alpha=0.5, s=10, color=COLORS["lgbm"])

        # Perfect prediction line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect")

        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{name}\nR² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Part A (LightGBM): True vs Predicted Descriptors", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "part_a_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_a_metrics.png")
    return True


def plot_training(log_dir: Path, output_dir: Path) -> bool:
    """Plot training curves from epoch log."""
    log_path = find_latest_log(log_dir, "hybrid")
    if not log_path:
        print("  [SKIP] No hybrid epoch log found")
        return False

    data = read_epoch_log(log_path)
    if not data["epoch"]:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(data["epoch"], data["train_loss"], label="Train", color="#1f77b4")
    axes[0].plot(data["epoch"], data["val_loss"], label="Val", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R² curves
    axes[1].plot(data["epoch"], data["train_r2"], label="Train R²", color="#2ca02c")
    axes[1].plot(data["epoch"], data["val_r2"], label="Val R²", color="#d62728")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² Performance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Learning rate
    axes[2].plot(data["epoch"], data["lr"], color="#9467bd")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "part_a_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_a_training.png")
    return True


def plot_comparison(metrics: dict, output_dir: Path) -> bool:
    """Compare LightGBM vs Hybrid model performance."""
    if "part_a_lgbm" not in metrics or "part_a_hybrid" not in metrics:
        print("  [SKIP] Need both LightGBM and Hybrid metrics for comparison")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall R² comparison
    models = ["LightGBM", "Hybrid\nCNN-Transformer"]
    r2_scores = [
        metrics["part_a_lgbm"]["summary"]["mean_r2"],
        metrics["part_a_hybrid"]["summary"]["mean_r2"],
    ]
    colors = [COLORS["lgbm"], COLORS["hybrid"]]

    bars = axes[0].bar(models, r2_scores, color=colors, edgecolor="black", width=0.5)
    axes[0].set_ylabel("Mean R² Score")
    axes[0].set_title("Part A: Spectrum → Descriptors", fontweight="bold")
    axes[0].set_ylim(0, 1)

    for bar, val in zip(bars, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha="center", fontweight="bold")

    # Per-descriptor comparison (top 10)
    lgbm_desc = metrics["part_a_lgbm"]["per_descriptor"]
    hybrid_desc = metrics["part_a_hybrid"]["per_descriptor"]

    def get_r2(d, key):
        val = d.get(key, 0)
        return val.get("R2", val.get("r2", 0)) if isinstance(val, dict) else val

    common = sorted(lgbm_desc.keys(), key=lambda x: get_r2(lgbm_desc, x), reverse=True)[:10]
    x = np.arange(len(common))
    width = 0.35

    lgbm_vals = [get_r2(lgbm_desc, d) for d in common]
    hybrid_vals = [get_r2(hybrid_desc, d) for d in common]

    axes[1].barh(x - width / 2, lgbm_vals, width, label="LightGBM", color=COLORS["lgbm"])
    axes[1].barh(x + width / 2, hybrid_vals, width, label="Hybrid", color=COLORS["hybrid"])
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("Top 10 Descriptors by R²", fontweight="bold")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(common, fontsize=9)
    axes[1].set_xlim(0, 1)
    axes[1].legend(loc="lower right")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "part_a_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_a_comparison.png")
    return True


def generate_part_a(metrics: dict, output_dir: Path, log_dir: Path = None, metrics_dir: Path = None) -> None:
    """Generate all Part A visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or Path("logs")
    metrics_dir = metrics_dir or output_dir.parent / "metrics"

    print("Generating Part A visualizations...")
    plot_metrics(metrics, output_dir, metrics_dir)
    plot_training(log_dir, output_dir)
    plot_comparison(metrics, output_dir)
