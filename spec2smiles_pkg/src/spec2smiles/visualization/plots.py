"""Visualization functions for SPEC2SMILES pipeline.

Replicates all visualizations from the original notebooks:
- 00_data_preperation.ipynb: Data quality diagnostics
- 01_spectra_to_descriptors.ipynb: Part A regression analysis
- 02_descriptors_to_smiles.ipynb: Part B training dynamics
- 03_spectra_to_smiles.ipynb: Integrated pipeline performance
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .styles import PALETTE, set_style, get_performance_color


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save figure to file with consistent settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Data Preparation Visualizations (from 00_data_preperation.ipynb)
# =============================================================================

def plot_data_preparation_summary(
    spectra: np.ndarray,
    descriptors: np.ndarray,
    descriptor_names: List[str],
    save_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """Generate 2x2 data preparation summary figure.

    Plots:
    - Average spectrum profile
    - Molecular weight distribution
    - Heavy atom count distribution
    - Descriptor correlation heatmap

    Args:
        spectra: Processed spectra array (N, 500)
        descriptors: Descriptor array (N, 12)
        descriptor_names: List of descriptor names
        save_path: Optional path to save figure
        dataset_name: Name of dataset for title

    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Average Spectrum Profile
    mean_spectrum = spectra.mean(axis=0)
    axes[0, 0].plot(mean_spectrum, color=PALETTE["blue"], linewidth=0.8)
    axes[0, 0].fill_between(range(len(mean_spectrum)), mean_spectrum, alpha=0.3, color=PALETTE["blue"])
    axes[0, 0].set_xlabel("m/z bin")
    axes[0, 0].set_ylabel("Mean Intensity")
    axes[0, 0].set_title("Average Spectrum Profile")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Molecular Weight Distribution
    mw_idx = descriptor_names.index("MolWt") if "MolWt" in descriptor_names else 0
    mw_values = descriptors[:, mw_idx]
    axes[0, 1].hist(mw_values, bins=50, edgecolor="black", alpha=0.7, color=PALETTE["green"])
    axes[0, 1].axvline(mw_values.mean(), color=PALETTE["vermillion"], linestyle="--",
                       linewidth=2, label=f"Mean: {mw_values.mean():.1f}")
    axes[0, 1].set_xlabel("Molecular Weight (Da)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("MW Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Heavy Atom Count Distribution
    ha_idx = descriptor_names.index("HeavyAtomCount") if "HeavyAtomCount" in descriptor_names else 1
    ha_values = descriptors[:, ha_idx]
    axes[1, 0].hist(ha_values, bins=30, edgecolor="black", alpha=0.7, color=PALETTE["orange"])
    axes[1, 0].set_xlabel("Heavy Atom Count")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Heavy Atom Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Descriptor Correlation Heatmap
    descriptor_df = pd.DataFrame(descriptors, columns=descriptor_names)
    corr_matrix = descriptor_df.corr()
    sns.heatmap(corr_matrix, ax=axes[1, 1], cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True)
    axes[1, 1].set_title("Descriptor Correlations")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].tick_params(axis='y', rotation=0)

    plt.suptitle(f"Data Processing Summary - {dataset_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Part A Visualizations (from 01_spectra_to_descriptors.ipynb)
# =============================================================================

def plot_part_a_regression_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    descriptor_names: List[str],
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    key_descriptors: Optional[List[str]] = None,
) -> plt.Figure:
    """Generate regression diagnostic parity plots for Part A.

    Args:
        y_true: True descriptor values (N, 12)
        y_pred: Predicted descriptor values (N, 12)
        descriptor_names: List of descriptor names
        metrics: Dict of {descriptor_name: {R2, RMSE, MAE}}
        save_path: Optional path to save figure
        key_descriptors: Optional list of descriptors to plot (default: 6 key ones)

    Returns:
        matplotlib Figure object
    """
    set_style()

    if key_descriptors is None:
        key_descriptors = ["MolWt", "HeavyAtomCount", "NumHeteroatoms",
                          "NumAromaticRings", "NOCount", "TPSA"]

    # Filter to available descriptors
    key_descriptors = [d for d in key_descriptors if d in descriptor_names]
    n_plots = min(6, len(key_descriptors))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, desc in enumerate(key_descriptors[:n_plots]):
        i = descriptor_names.index(desc)
        ax = axes[idx]

        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10, color=PALETTE["blue"])

        # Identity line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--",
                label="Identity", alpha=0.7, linewidth=2)

        # Annotations
        ax.set_xlabel(f"Observed {desc}")
        ax.set_ylabel(f"Predicted {desc}")

        r2 = metrics.get(desc, {}).get("R2", 0)
        mae = metrics.get(desc, {}).get("MAE", 0)
        ax.set_title(f"{desc}\nR² = {r2:.3f}, MAE = {mae:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, 6):
        axes[idx].axis("off")

    plt.suptitle("Test Set Regression Analysis: Observed vs Predicted Values",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_part_a_performance_summary(
    metrics: Dict[str, Dict[str, float]],
    descriptor_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate R² performance bar chart for Part A.

    Args:
        metrics: Dict of {descriptor_name: {R2, RMSE, MAE}}
        descriptor_names: List of descriptor names
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    r2_scores = [metrics.get(desc, {}).get("R2", 0) for desc in descriptor_names]
    colors = [get_performance_color(r2) for r2 in r2_scores]

    bars = ax.bar(range(len(descriptor_names)), r2_scores, color=colors)

    ax.set_xlabel("Molecular Descriptor")
    ax.set_ylabel("Coefficient of Determination (R²)")
    ax.set_title("Descriptor Prediction Performance: Test Set R² Scores")
    ax.set_xticks(range(len(descriptor_names)))
    ax.set_xticklabels(descriptor_names, rotation=45, ha="right")

    # Performance thresholds
    ax.axhline(y=0.7, color=PALETTE["green"], linestyle="--", alpha=0.5,
               label="High Performance (R² > 0.7)")
    ax.axhline(y=0.5, color=PALETTE["orange"], linestyle="--", alpha=0.5,
               label="Moderate Performance (R² > 0.5)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])

    # Add value annotations
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{r2:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_part_a_feature_importance(
    feature_importances: Dict[str, np.ndarray],
    descriptors_to_compare: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate feature importance comparison plots.

    Args:
        feature_importances: Dict of {descriptor_name: importance_array}
        descriptors_to_compare: Optional list of descriptors to plot
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_style()

    if descriptors_to_compare is None:
        descriptors_to_compare = ["MolWt", "HeavyAtomCount", "NumAromaticRings", "TPSA"]

    # Filter to available descriptors
    descriptors_to_compare = [d for d in descriptors_to_compare if d in feature_importances]
    n_plots = min(4, len(descriptors_to_compare))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, descriptor in enumerate(descriptors_to_compare[:n_plots]):
        ax = axes[idx]

        importances = feature_importances[descriptor]
        importances_normalized = importances / importances.sum()

        # Plot importance distribution
        ax.plot(importances_normalized, alpha=0.7, linewidth=0.5, color=PALETTE["blue"])
        ax.fill_between(range(len(importances_normalized)), importances_normalized,
                       alpha=0.3, color=PALETTE["blue"])
        ax.set_xlabel("m/z bin index")
        ax.set_ylabel("Normalized Importance")
        ax.set_title(f"{descriptor}")
        ax.grid(True, alpha=0.3)

        # Highlight top features
        top_idx = np.argsort(importances_normalized)[-5:]
        for ti in top_idx:
            ax.axvline(x=ti, color=PALETTE["vermillion"], alpha=0.3, linestyle="--")

    # Hide unused axes
    for idx in range(n_plots, 4):
        axes[idx].axis("off")

    plt.suptitle("Comparative Feature Importance Patterns Across Descriptors", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Part B Visualizations (from 02_descriptors_to_smiles.ipynb)
# =============================================================================

def plot_part_b_training_dynamics(
    history: Dict[str, List[float]],
    tanimoto_scores: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate training dynamics dashboard for Part B.

    Args:
        history: Training history dict with keys:
            - train_loss, val_loss
            - train_recon, val_recon
            - train_kl, val_kl
            - beta, lr
        tanimoto_scores: Optional list of Tanimoto similarity scores
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(history["train_loss"], label="Train", alpha=0.7, color=PALETTE["blue"])
    axes[0, 0].plot(history["val_loss"], label="Validation", alpha=0.7, color=PALETTE["orange"])
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss (ELBO)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(history["train_recon"], label="Train", alpha=0.7, color=PALETTE["blue"])
    axes[0, 1].plot(history["val_recon"], label="Validation", alpha=0.7, color=PALETTE["orange"])
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reconstruction Loss")
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence
    axes[0, 2].plot(history["train_kl"], label="Train", alpha=0.7, color=PALETTE["blue"])
    axes[0, 2].plot(history["val_kl"], label="Validation", alpha=0.7, color=PALETTE["orange"])
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("KL Divergence")
    axes[0, 2].set_title("KL Divergence")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Beta schedule
    axes[1, 0].plot(history["beta"], alpha=0.7, color=PALETTE["green"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("β")
    axes[1, 0].set_title("Cyclical β Schedule")
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history["lr"], alpha=0.7, color=PALETTE["orange"])
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    # Tanimoto distribution
    if tanimoto_scores and len(tanimoto_scores) > 0:
        axes[1, 2].hist(tanimoto_scores, bins=30, edgecolor="black", alpha=0.7,
                       color=PALETTE["purple"])
        mean_tan = np.mean(tanimoto_scores)
        axes[1, 2].axvline(mean_tan, color=PALETTE["vermillion"], linestyle="--",
                          label=f"Mean: {mean_tan:.3f}")
        axes[1, 2].set_xlabel("Tanimoto Similarity")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].set_title("Tanimoto Similarity Distribution")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(0.5, 0.5, "No Tanimoto scores available",
                       ha="center", va="center", transform=axes[1, 2].transAxes)

    plt.suptitle("Training Dynamics and Performance Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Pipeline Visualizations (from 03_spectra_to_smiles.ipynb)
# =============================================================================

def plot_pipeline_performance(
    results: Dict[str, Any],
    part_a_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    part_b_results: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate comprehensive pipeline performance visualization.

    Args:
        results: Pipeline evaluation results with keys:
            - hit_at_k: {k: rate}
            - exact_match_rate, formula_match_rate
            - mean_tanimoto, median_tanimoto
            - raw_metrics: {tanimoto_scores, processing_times, n_candidates}
        part_a_metrics: Optional Part A test metrics
        part_b_results: Optional Part B results
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Hit@K curve
    hit_at_k = results.get("hit_at_k", {})
    if hit_at_k:
        k_values = list(hit_at_k.keys())
        hit_rates = list(hit_at_k.values())

        axes[0, 0].plot(k_values, hit_rates, "o-", linewidth=2, markersize=8,
                       color=PALETTE["blue"])
        axes[0, 0].set_xlabel("K")
        axes[0, 0].set_ylabel("Hit Rate")
        axes[0, 0].set_title("Hit@K Performance")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, max(hit_rates) * 1.1 if hit_rates else 1])

        for k, rate in zip(k_values, hit_rates):
            axes[0, 0].text(k, rate + 0.01, f"{rate:.1%}", ha="center", fontsize=8)

    # Tanimoto distribution
    raw_metrics = results.get("raw_metrics", {})
    tanimoto_scores = raw_metrics.get("tanimoto_scores", [])
    if tanimoto_scores:
        axes[0, 1].hist(tanimoto_scores, bins=30, edgecolor="black", alpha=0.7,
                       color=PALETTE["green"])
        mean_tan = np.mean(tanimoto_scores)
        median_tan = np.median(tanimoto_scores)
        axes[0, 1].axvline(mean_tan, color=PALETTE["vermillion"], linestyle="--",
                          label=f"Mean: {mean_tan:.3f}")
        axes[0, 1].axvline(median_tan, color=PALETTE["orange"], linestyle="--",
                          label=f"Median: {median_tan:.3f}")
        axes[0, 1].set_xlabel("Tanimoto Similarity")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Structural Similarity Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Processing time distribution
    processing_times = raw_metrics.get("processing_times", [])
    if processing_times:
        axes[0, 2].hist(processing_times, bins=30, edgecolor="black", alpha=0.7,
                       color=PALETTE["orange"])
        mean_time = np.mean(processing_times)
        axes[0, 2].axvline(mean_time, color=PALETTE["vermillion"], linestyle="--",
                          label=f"Mean: {mean_time:.3f}s")
        axes[0, 2].set_xlabel("Time (seconds)")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].set_title("Processing Time Distribution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Key performance metrics bar chart
    metrics_names = ["Exact\nMatch", "Formula\nMatch", "Mean\nTanimoto"]
    metrics_values = [
        results.get("exact_match_rate", 0),
        results.get("formula_match_rate", 0),
        results.get("mean_tanimoto", 0)
    ]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]

    bars = axes[1, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Key Performance Metrics")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, metrics_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", fontsize=9)

    # Candidates generated distribution
    n_candidates = raw_metrics.get("n_candidates", [])
    if n_candidates:
        axes[1, 1].hist(n_candidates, bins=20, edgecolor="black", alpha=0.7,
                       color=PALETTE["purple"])
        mean_cand = np.mean(n_candidates)
        axes[1, 1].axvline(mean_cand, color=PALETTE["vermillion"], linestyle="--",
                          label=f"Mean: {mean_cand:.1f}")
        axes[1, 1].set_xlabel("Number of Candidates")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Candidates Generated per Molecule")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Component performance comparison
    if part_a_metrics or part_b_results:
        component_names = ["Component A\n(Spectra→Desc)",
                          "Component B\n(Desc→SMILES)",
                          "End-to-End\n(Full Pipeline)"]

        # Part A: average R²
        if part_a_metrics:
            part_a_r2 = np.mean([m.get("R2", 0) for m in part_a_metrics.values()])
        else:
            part_a_r2 = 0

        # Part B: exact match rate
        if part_b_results:
            part_b_exact = part_b_results.get("exact_match_rate", 0)
        else:
            part_b_exact = 0

        # End-to-end: exact match rate
        e2e_exact = results.get("exact_match_rate", 0)

        component_scores = [part_a_r2, part_b_exact, e2e_exact]
        comp_colors = [PALETTE["vermillion"], PALETTE["blue"], PALETTE["green"]]

        bars = axes[1, 2].bar(component_names, component_scores, color=comp_colors, alpha=0.7)
        axes[1, 2].set_ylabel("Performance Score")
        axes[1, 2].set_title("Component vs End-to-End Performance")
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, component_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{val:.3f}", ha="center", fontsize=9)
    else:
        axes[1, 2].axis("off")

    plt.suptitle("Integrated Pipeline Performance Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
