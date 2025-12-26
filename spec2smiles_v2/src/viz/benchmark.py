"""MassSpecGym benchmark comparison."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .constants import COLORS, MASSSPECGYM_BASELINES
from .tables import generate_csv_table, generate_markdown_table


def plot_comparison(metrics: dict, output_dir: Path) -> bool:
    """Combined side-by-side comparison at k=10."""
    if "e2e" not in metrics or "oracle" not in metrics:
        print("  [SKIP] Need E2E and Oracle metrics for benchmark comparison")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["Random", "SMILES\nTransformer", "SELFIES\nTransformer", "Ours\n(E2E)", "Ours\n(Oracle)"]
    colors = [COLORS["baseline"]] * 3 + [COLORS["e2e"], COLORS["oracle"]]

    # Accuracy
    accuracies = [
        MASSSPECGYM_BASELINES["Random"]["top10_accuracy"],
        MASSSPECGYM_BASELINES["SMILES Transformer"]["top10_accuracy"],
        MASSSPECGYM_BASELINES["SELFIES Transformer"]["top10_accuracy"],
        metrics["e2e"]["end_to_end"]["exact_match"] * 100,
        metrics["oracle"]["exact_match"] * 100,
    ]
    bars1 = axes[0].bar(methods, accuracies, color=colors, edgecolor="black")
    for bar, acc in zip(bars1, accuracies):
        axes[0].annotate(f"{acc:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, max(acc, 2)),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    axes[0].set_ylabel("Exact Match (%)")
    axes[0].set_title("(a) Exact Match Accuracy (k=10)", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    # Tanimoto
    tanimotos = [
        MASSSPECGYM_BASELINES["Random"]["top10_tanimoto"],
        MASSSPECGYM_BASELINES["SMILES Transformer"]["top10_tanimoto"],
        MASSSPECGYM_BASELINES["SELFIES Transformer"]["top10_tanimoto"],
        metrics["e2e"]["end_to_end"]["mean_best_tanimoto"],
        metrics["oracle"]["mean_best_tanimoto"],
    ]
    bars2 = axes[1].bar(methods, tanimotos, color=colors, edgecolor="black")
    for bar, tan in zip(bars2, tanimotos):
        axes[1].annotate(f"{tan:.3f}", xy=(bar.get_x() + bar.get_width() / 2, tan),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    axes[1].set_ylabel("Mean Best Tanimoto")
    axes[1].set_title("(b) Tanimoto Similarity (k=10)", fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["baseline"], label="MassSpecGym Baselines"),
        mpatches.Patch(color=COLORS["e2e"], label="Ours E2E"),
        mpatches.Patch(color=COLORS["oracle"], label="Ours Oracle"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_dir / "massspecgym_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] massspecgym_comparison.png")
    return True


def plot_improvement(metrics: dict, output_dir: Path) -> bool:
    """Summary plot showing improvement percentages."""
    if "e2e" not in metrics or "oracle" not in metrics:
        print("  [SKIP] Need E2E and Oracle metrics for improvement summary")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ["Exact Match\n(vs 0%)", "Tanimoto\n(vs 0.17)", "Validity"]
    our_e2e = [
        metrics["e2e"]["end_to_end"]["exact_match"] * 100,
        metrics["e2e"]["end_to_end"]["mean_best_tanimoto"] * 100,
        metrics["e2e"]["end_to_end"]["validity"] * 100,
    ]
    our_oracle = [
        metrics["oracle"]["exact_match"] * 100,
        metrics["oracle"]["mean_best_tanimoto"] * 100,
        metrics["oracle"]["validity"] * 100,
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, our_e2e, width, label="E2E", color=COLORS["e2e"])
    bars2 = ax.bar(x + width / 2, our_oracle, width, label="Oracle", color=COLORS["oracle"])

    # Improvement annotations
    ax.annotate("∞× vs\nbaseline", xy=(0, our_e2e[0] + 3), ha="center", fontsize=9, color="#1565C0")
    improvement = our_e2e[1] / 17  # 0.17 baseline
    ax.annotate(f"{improvement:.1f}× vs\nbest", xy=(1, our_e2e[1] + 3), ha="center", fontsize=9, color="#1565C0")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center")

    ax.set_ylabel("Performance (%)")
    ax.set_title("SPEC2SMILES Performance Summary\nvs MassSpecGym Baselines", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] improvement_summary.png")
    return True


def generate_benchmark(metrics: dict, output_dir: Path) -> None:
    """Generate all benchmark visualizations and tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Benchmark visualizations...")
    plot_comparison(metrics, output_dir)
    plot_improvement(metrics, output_dir)
    generate_csv_table(metrics, output_dir)
    generate_markdown_table(metrics, output_dir)
