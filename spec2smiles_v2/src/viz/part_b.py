"""Part B visualization: Descriptors → SMILES."""

from pathlib import Path

import matplotlib.pyplot as plt

from .constants import COLORS
from .loaders import find_latest_log, read_epoch_log


def plot_training(metrics: dict, output_dir: Path) -> bool:
    """Plot Part B training summary."""
    if "part_b" not in metrics:
        print("  [SKIP] Part B metrics not found")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Part B Training History")

    final_loss = metrics["part_b"].get("final_train_loss", 0)
    n_epochs = metrics["part_b"].get("n_epochs", 0)
    ax.text(0.5, 0.5, f"Final Train Loss: {final_loss:.4f}\nEpochs: {n_epochs}",
            transform=ax.transAxes, ha="center", va="center", fontsize=14)

    plt.savefig(output_dir / "part_b_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_b_training.png")
    return True


def plot_loss_components(log_dir: Path, output_dir: Path) -> bool:
    """Plot VAE loss components (reconstruction vs KL)."""
    log_path = find_latest_log(log_dir, "vae")
    if not log_path:
        print("  [SKIP] No VAE epoch log found")
        return False

    data = read_epoch_log(log_path)
    if not data["epoch"]:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Overall loss
    axes[0].plot(data["epoch"], data["train_loss"], label="Train", color="#1f77b4")
    axes[0].plot(data["epoch"], data["val_loss"], label="Val", color="#ff7f0e", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss components
    if data["recon_loss"] and data["kl_loss"]:
        axes[1].plot(data["epoch"], data["recon_loss"], label="Reconstruction", color="#2ca02c")
        axes[1].plot(data["epoch"], data["kl_loss"], label="KL Divergence", color="#d62728")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Components")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Beta annealing
    if data["beta"]:
        axes[2].plot(data["epoch"], data["beta"], color="#9467bd", linewidth=2)
        axes[2].fill_between(data["epoch"], 0, data["beta"], alpha=0.3, color="#9467bd")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Beta (KL Weight)")
        axes[2].set_title("Cyclical KL Annealing")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / "part_b_loss_components.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_b_loss_components.png")
    return True


def plot_hitatk(metrics: dict, output_dir: Path) -> bool:
    """Plot Hit@K performance bars."""
    if "oracle" not in metrics:
        print("  [SKIP] Oracle evaluation not found")
        return False

    oracle = metrics["oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hit@K
    k_values = [1, 5, 10]
    hits = [
        oracle.get("hit_at_1", oracle.get("exact_match", 0)),
        oracle.get("hit_at_5", 0),
        oracle.get("hit_at_10", 0),
    ]

    bars = axes[0].bar(k_values, hits, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("Hit@K Performance (Oracle)")
    axes[0].set_xticks(k_values)
    axes[0].set_ylim(0, 1)

    for bar, hit in zip(bars, hits):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{hit:.1%}", ha="center", fontsize=10)

    # Quality metrics
    metric_names = ["Tanimoto", "Validity", "Uniqueness"]
    metric_values = [
        oracle.get("mean_best_tanimoto", 0),
        oracle.get("validity", 0),
        oracle.get("uniqueness", 0),
    ]

    bars = axes[1].bar(metric_names, metric_values, color=["#d62728", "#9467bd", "#8c564b"])
    axes[1].set_ylabel("Score")
    axes[1].set_title("Quality Metrics (Oracle)")
    axes[1].set_ylim(0, 1)

    for bar, val in zip(bars, metric_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "part_b_hitatk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] part_b_hitatk.png")
    return True


def generate_part_b(metrics: dict, output_dir: Path, log_dir: Path = None) -> None:
    """Generate all Part B visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or Path("logs")

    print("Generating Part B visualizations...")
    plot_training(metrics, output_dir)
    plot_loss_components(log_dir, output_dir)
    plot_hitatk(metrics, output_dir)
