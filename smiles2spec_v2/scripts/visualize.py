#!/usr/bin/env python
"""Generate all visualizations for SMILES2SPEC pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from smiles2spec.core.loader import load_settings
from smiles2spec.data.loader import DataLoader
from smiles2spec.data.preprocessor import FeaturePreprocessor
from smiles2spec.data.splitter import DataSplitter
from smiles2spec.domain.spectrum import SpectrumBinner
from smiles2spec.evaluation.metrics import compute_all_metrics, cosine_similarity
from smiles2spec.models.registry import ModelRegistry
from smiles2spec.services.featurizer import FeaturizationService
from smiles2spec.visualization.constants import COLORS, STYLES, apply_style
from smiles2spec.visualization.diagnostics import plot_2x2_diagnostic, plot_sample_spectra

# Import to register models
import smiles2spec.models.random_forest  # noqa
import smiles2spec.models.neural.modular_net  # noqa


def plot_model_comparison(results: dict, save_path: Path) -> None:
    """Bar chart comparing all models."""
    apply_style()

    models = list(results.keys())
    means = [results[m]["cosine_similarity"]["mean"] for m in models]
    stds = [results[m]["cosine_similarity"]["std"] for m in models]

    # Sort by performance
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS["success"] if i == 0 else COLORS["primary"] for i in range(len(models))]
    bars = ax.barh(models, means, xerr=stds, color=colors, edgecolor="white", capsize=3)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                f"{mean:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Model Performance Comparison")
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_performance_by_mz(pred: np.ndarray, target: np.ndarray, save_path: Path) -> None:
    """Performance breakdown by m/z range."""
    apply_style()

    # Define m/z regions
    regions = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-bin performance
    ax = axes[0]
    cos_per_sample = cosine_similarity(pred, target)
    mae_per_bin = np.mean(np.abs(pred - target), axis=0)

    ax.plot(mae_per_bin, color=COLORS["primary"], lw=1.5)
    ax.set_xlabel("m/z bin")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error by m/z")

    # Regional performance
    ax = axes[1]
    region_labels = []
    region_cos = []

    for start, end in regions:
        region_pred = pred[:, start:end]
        region_target = target[:, start:end]

        # Only compute if region has non-zero values
        mask = region_target.sum(axis=1) > 0
        if mask.sum() > 0:
            cos = cosine_similarity(region_pred[mask], region_target[mask])
            region_cos.append(np.mean(cos))
        else:
            region_cos.append(0)
        region_labels.append(f"{start}-{end}")

    bars = ax.bar(region_labels, region_cos, color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("m/z Range")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Performance by m/z Region")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, region_cos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_best_worst_samples(pred: np.ndarray, target: np.ndarray, save_path: Path) -> None:
    """Plot best and worst predicted samples."""
    apply_style()

    cos_sims = cosine_similarity(pred, target)

    # Get best and worst indices
    best_idx = np.argsort(cos_sims)[-4:][::-1]
    worst_idx = np.argsort(cos_sims)[:4]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Best samples
    for i, idx in enumerate(best_idx):
        ax = axes[0, i]
        mz = np.arange(len(target[idx]))
        ax.bar(mz, target[idx], alpha=0.5, color=COLORS["target"], width=1)
        ax.plot(mz, pred[idx], color=COLORS["predicted"], lw=1)
        ax.set_title(f"Best #{i+1} (cos={cos_sims[idx]:.4f})")
        ax.set_xlabel("m/z")
        if i == 0:
            ax.set_ylabel("Intensity")

    # Worst samples
    for i, idx in enumerate(worst_idx):
        ax = axes[1, i]
        mz = np.arange(len(target[idx]))
        ax.bar(mz, target[idx], alpha=0.5, color=COLORS["target"], width=1)
        ax.plot(mz, pred[idx], color=COLORS["predicted"], lw=1)
        ax.set_title(f"Worst #{i+1} (cos={cos_sims[idx]:.4f})")
        ax.set_xlabel("m/z")
        if i == 0:
            ax.set_ylabel("Intensity")

    axes[0, 0].legend(["Predicted", "Target"], loc="upper right")

    fig.suptitle("Best and Worst Predictions", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_summary_dashboard(results: dict, best_pred: np.ndarray, target: np.ndarray, save_path: Path) -> None:
    """Create summary dashboard with key metrics."""
    apply_style()

    fig = plt.figure(figsize=(14, 10))

    # Model comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    models = list(results.keys())
    means = [results[m]["cosine_similarity"]["mean"] for m in models]
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]

    colors = [COLORS["success"] if i == 0 else COLORS["primary"] for i in range(len(models))]
    ax1.barh(models, means, color=colors, edgecolor="white")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_title("Model Comparison")
    ax1.set_xlim(0.7, 0.85)

    # Cosine distribution (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    cos_sims = cosine_similarity(best_pred, target)
    ax2.hist(cos_sims, bins=50, color=COLORS["success"], edgecolor="white", alpha=0.7)
    ax2.axvline(np.mean(cos_sims), color="red", linestyle="--", lw=2,
                label=f"Mean: {np.mean(cos_sims):.4f}")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Best Model: Similarity Distribution")
    ax2.legend()

    # Per-bin MAE (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    mae_per_bin = np.mean(np.abs(best_pred - target), axis=0)
    ax3.plot(mae_per_bin, color=COLORS["primary"], lw=1)
    ax3.fill_between(range(len(mae_per_bin)), mae_per_bin, alpha=0.3, color=COLORS["primary"])
    ax3.set_xlabel("m/z bin")
    ax3.set_ylabel("MAE")
    ax3.set_title("Error by m/z Position")

    # Sample spectrum (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    # Pick a good sample (median performance)
    median_idx = np.argsort(cos_sims)[len(cos_sims)//2]
    mz = np.arange(len(target[median_idx]))
    ax4.bar(mz, target[median_idx], alpha=0.5, color=COLORS["target"], width=1, label="Target")
    ax4.plot(mz, best_pred[median_idx], color=COLORS["predicted"], lw=1, label="Predicted")
    ax4.set_xlabel("m/z")
    ax4.set_ylabel("Intensity")
    ax4.set_title(f"Sample Prediction (cos={cos_sims[median_idx]:.4f})")
    ax4.legend()

    plt.suptitle("SMILES2SPEC Performance Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=STYLES["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate SMILES2SPEC visualizations")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--diagnostics", action="store_true", help="Diagnostic plots only")
    parser.add_argument("--performance", action="store_true", help="Performance plots only")
    parser.add_argument("--config", type=str, default="config.yml", help="Config file path")
    args = parser.parse_args()

    # Default to all if no specific option
    if not any([args.all, args.diagnostics, args.performance]):
        args.all = True

    # Load settings
    config_path = Path(__file__).parent.parent / args.config
    settings = load_settings(config_path)

    print("=" * 60)
    print("SMILES2SPEC Visualization Generator")
    print("=" * 60)

    # Ensure output directory exists
    figures_path = settings.figures_path
    figures_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {figures_path}")

    # Load data
    print("\nLoading data...")
    loader = DataLoader(
        data_dir=settings.input_path.parent.parent,
        dataset=settings.dataset,
    )
    smiles_list, peaks_list = loader.extract_smiles_and_peaks()

    # Featurize
    featurizer = FeaturizationService(
        config=settings.features,
        cache_dir=settings.cache_path,
        use_cache=False,
    )
    X, _, failed = featurizer.extract(smiles_list)

    # Bin spectra
    binner = SpectrumBinner(
        n_bins=settings.spectrum.n_bins,
        transform=settings.spectrum.transform,
    )
    valid_peaks = [p for i, p in enumerate(peaks_list) if i not in failed]
    y = binner.batch_bin(valid_peaks)

    # Split
    splitter = DataSplitter(
        train_size=settings.split.train,
        val_size=settings.split.val,
        test_size=settings.split.test,
        random_seed=settings.split.seed,
    )
    split = splitter.split_arrays(X, y)

    # Load preprocessors
    preprocessor_rf_path = settings.models_path / "preprocessor.pkl"
    preprocessor_nn_path = settings.models_path / "preprocessor_nn.pkl"

    preprocessor_rf = FeaturePreprocessor.load(preprocessor_rf_path) if preprocessor_rf_path.exists() else None
    preprocessor_nn = FeaturePreprocessor.load(preprocessor_nn_path) if preprocessor_nn_path.exists() else None

    X_test_rf = preprocessor_rf.transform(split.X_test) if preprocessor_rf else split.X_test
    X_test_nn = preprocessor_nn.transform(split.X_test) if preprocessor_nn else split.X_test

    # Load models
    print("Loading models...")
    models = {}
    model_to_X_test = {}

    # Random Forest
    rf_path = settings.models_path / "random_forest.pkl"
    if rf_path.exists():
        models["random_forest"] = ModelRegistry.get_class("random_forest").load(rf_path)
        model_to_X_test["random_forest"] = X_test_rf

    # Neural networks
    neural_dir = settings.models_path / "neural"
    if neural_dir.exists():
        for model_path in neural_dir.glob("*.pth"):
            model_name = model_path.stem
            try:
                models[model_name] = ModelRegistry.get_class(model_name).load(model_path)
                model_to_X_test[model_name] = X_test_nn
            except Exception as e:
                print(f"  Could not load {model_name}: {e}")

    print(f"Loaded {len(models)} models")

    # Compute predictions and metrics
    print("\nComputing predictions...")
    results = {}
    predictions = {}

    for name, model in models.items():
        X_test = model_to_X_test[name]
        pred = model.predict(X_test)
        predictions[name] = pred
        results[name] = compute_all_metrics(pred, split.y_test)

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]["cosine_similarity"]["mean"])
    best_pred = predictions[best_model]

    print(f"Best model: {best_model} ({results[best_model]['cosine_similarity']['mean']:.4f})")

    # Generate visualizations
    print("\nGenerating visualizations...")

    if args.all or args.performance:
        # Model comparison
        plot_model_comparison(results, figures_path / "model_comparison.png")

        # Summary dashboard
        plot_summary_dashboard(results, best_pred, split.y_test, figures_path / "summary_dashboard.png")

        # Performance by m/z
        plot_performance_by_mz(best_pred, split.y_test, figures_path / "performance_by_mz.png")

        # Best/worst samples
        plot_best_worst_samples(best_pred, split.y_test, figures_path / "best_worst_samples.png")

    if args.all or args.diagnostics:
        # Diagnostic plots for each model
        for name, pred in predictions.items():
            save_path = figures_path / f"{name}_diagnostic.png"
            plot_2x2_diagnostic(pred, split.y_test, name, save_path)
            print(f"  [OK] {save_path.name}")

        # Sample spectra for best model
        plot_sample_spectra(best_pred, split.y_test, n_samples=6,
                           title=f"{best_model}: Sample Predictions",
                           save_path=figures_path / "sample_spectra.png")
        print(f"  [OK] sample_spectra.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
