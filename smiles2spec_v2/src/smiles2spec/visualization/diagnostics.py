"""Diagnostic plots for model evaluation."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from smiles2spec.evaluation.metrics import cosine_similarity
from smiles2spec.visualization.constants import COLORS, STYLES, apply_style


def plot_predicted_vs_actual(
    pred: np.ndarray,
    target: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Optional[Union[str, Path]] = None,
    sample_size: int = 5000,
) -> plt.Figure:
    """Plot predicted vs actual intensities.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)
        title: Plot title
        save_path: Path to save figure
        sample_size: Number of points to sample

    Returns:
        Matplotlib figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=STYLES["figure_size_small"])

    # Flatten and sample
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if len(pred_flat) > sample_size:
        idx = np.random.choice(len(pred_flat), sample_size, replace=False)
        pred_flat = pred_flat[idx]
        target_flat = target_flat[idx]

    ax.scatter(target_flat, pred_flat, alpha=0.3, s=1, c=COLORS["primary"])

    # Perfect prediction line
    max_val = max(target_flat.max(), pred_flat.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1, label="Perfect prediction")

    ax.set_xlabel("Actual Intensity")
    ax.set_ylabel("Predicted Intensity")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")

    return fig


def plot_residual_distribution(
    pred: np.ndarray,
    target: np.ndarray,
    title: str = "Residual Distribution",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot residual distribution.

    Args:
        pred: Predictions
        target: Targets
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=STYLES["figure_size_small"])

    residuals = (pred - target).flatten()

    ax.hist(residuals, bins=100, alpha=0.7, color=COLORS["residual"], edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.axvline(residuals.mean(), color=COLORS["primary"], linestyle="-", lw=2,
               label=f"Mean: {residuals.mean():.4f}")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")

    return fig


def plot_sample_spectra(
    pred: np.ndarray,
    target: np.ndarray,
    indices: Optional[List[int]] = None,
    n_samples: int = 4,
    title: str = "Sample Spectra Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot sample spectrum comparisons.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)
        indices: Specific indices to plot
        n_samples: Number of samples if indices not provided
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    apply_style()

    if indices is None:
        indices = np.random.choice(len(pred), min(n_samples, len(pred)), replace=False)

    n_plots = len(indices)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, idx in zip(axes, indices):
        cos_sim = float(cosine_similarity(pred[idx:idx + 1], target[idx:idx + 1]))

        mz = np.arange(len(target[idx]))

        ax.bar(mz, target[idx], alpha=0.5, color=COLORS["target"], label="Target", width=1)
        ax.plot(mz, pred[idx], color=COLORS["predicted"], lw=1, label="Predicted")

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Sample {idx} (cos={cos_sim:.4f})")
        ax.legend(loc="upper right")

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=STYLES["title_size"])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")

    return fig


def plot_2x2_diagnostic(
    pred: np.ndarray,
    target: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create 2x2 diagnostic plot grid.

    Args:
        pred: Predictions
        target: Targets
        model_name: Model name for titles
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    idx = np.random.choice(len(pred_flat), min(5000, len(pred_flat)), replace=False)
    ax.scatter(target_flat[idx], pred_flat[idx], alpha=0.3, s=1, c=COLORS["primary"])
    max_val = max(target_flat.max(), pred_flat.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")

    # 2. Residual Distribution
    ax = axes[0, 1]
    residuals = (pred - target).flatten()
    ax.hist(residuals, bins=100, alpha=0.7, color=COLORS["residual"], edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")

    # 3. Cosine similarity distribution
    ax = axes[1, 0]
    cos_sims = cosine_similarity(pred, target)
    ax.hist(cos_sims, bins=50, alpha=0.7, color=COLORS["success"], edgecolor="white")
    ax.axvline(np.mean(cos_sims), color=COLORS["primary"], linestyle="-", lw=2,
               label=f"Mean: {np.mean(cos_sims):.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Cosine Similarity Distribution")
    ax.legend()

    # 4. Per-bin MAE
    ax = axes[1, 1]
    mae_per_bin = np.mean(np.abs(pred - target), axis=0)
    ax.plot(mae_per_bin, color=COLORS["primary"], lw=1)
    ax.set_xlabel("m/z bin")
    ax.set_ylabel("MAE")
    ax.set_title("Per-bin MAE")

    fig.suptitle(f"{model_name} Diagnostic", fontsize=STYLES["title_size"] + 2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")

    return fig
