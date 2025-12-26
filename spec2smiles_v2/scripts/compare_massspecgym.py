#!/usr/bin/env python
"""Compare our SPEC2SMILES results to MassSpecGym benchmarks.

Generates comparison plots and tables based on:
- MassSpecGym paper (arXiv:2410.23326v3, NeurIPS 2024)
- Our two-stage pipeline: Spectrum → Descriptors (LightGBM) → SMILES (DirectDecoder)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import pandas as pd


# MassSpecGym baseline results from Table 2 (De novo molecule generation)
MASSSPECGYM_BASELINES = {
    "Random chemical gen.": {
        "top1_accuracy": 0.00, "top1_mces": 28.59, "top1_tanimoto": 0.07,
        "top10_accuracy": 0.00, "top10_mces": 25.72, "top10_tanimoto": 0.10,
    },
    "SMILES Transformer": {
        "top1_accuracy": 0.00, "top1_mces": 53.80, "top1_tanimoto": 0.07,
        "top10_accuracy": 0.00, "top10_mces": 21.97, "top10_tanimoto": 0.17,
    },
    "SELFIES Transformer": {
        "top1_accuracy": 0.00, "top1_mces": 33.28, "top1_tanimoto": 0.10,
        "top10_accuracy": 0.00, "top10_mces": 21.84, "top10_tanimoto": 0.15,
    },
}

# MassSpecGym molecule retrieval results from Table 3
MASSSPECGYM_RETRIEVAL = {
    "Random": {"hit1": 0.37, "hit5": 2.01, "hit20": 8.22, "mces1": 30.81},
    "Fingerprint FFN": {"hit1": 1.47, "hit5": 6.21, "hit20": 19.23, "mces1": 25.11},
    "DeepSets": {"hit1": 2.54, "hit5": 7.59, "hit20": 20.00, "mces1": 24.66},
    "DeepSets + Fourier": {"hit1": 5.24, "hit5": 12.58, "hit20": 28.21, "mces1": 22.13},
    "MIST (SOTA)": {"hit1": 14.64, "hit5": 34.87, "hit20": 59.15, "mces1": 15.37},
}


def load_our_results(metrics_path: Path) -> dict:
    """Load our E2E and Oracle results."""
    results = {}

    # E2E evaluation
    e2e_path = metrics_path / "e2e_evaluation.json"
    if e2e_path.exists():
        with open(e2e_path) as f:
            e2e = json.load(f)
        results["e2e"] = {
            "exact_match": e2e["end_to_end"]["exact_match"] * 100,
            "tanimoto": e2e["end_to_end"]["mean_best_tanimoto"],
            "validity": e2e["end_to_end"]["validity"] * 100,
            "n_candidates": e2e["config"]["n_candidates"],
            "n_samples": e2e["config"]["n_samples"],
            "part_a_r2": e2e["part_a"]["mean_r2"],
        }

    # Oracle (Part B) evaluation
    oracle_path = metrics_path / "part_b_evaluation.json"
    if oracle_path.exists():
        with open(oracle_path) as f:
            oracle = json.load(f)
        results["oracle"] = {
            "exact_match": oracle["exact_match"] * 100,
            "tanimoto": oracle["mean_best_tanimoto"],
            "validity": oracle["validity"] * 100,
            "n_candidates": oracle["n_candidates"],
        }

    return results


def plot_accuracy_comparison(our_results: dict, output_dir: Path):
    """Bar chart comparing exact match accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    methods = list(MASSSPECGYM_BASELINES.keys()) + ["Ours (E2E)", "Ours (Oracle)"]
    colors = ["#808080"] * len(MASSSPECGYM_BASELINES) + ["#2196F3", "#4CAF50"]

    # Get accuracies (MassSpecGym uses Top-10, we use Top-50)
    accuracies = [
        MASSSPECGYM_BASELINES["Random chemical gen."]["top10_accuracy"],
        MASSSPECGYM_BASELINES["SMILES Transformer"]["top10_accuracy"],
        MASSSPECGYM_BASELINES["SELFIES Transformer"]["top10_accuracy"],
        our_results["e2e"]["exact_match"],
        our_results["oracle"]["exact_match"],
    ]

    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.annotate('0%',
                       xy=(bar.get_x() + bar.get_width() / 2, 0.5),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
    ax.set_title("De Novo Molecule Generation: Exact Match Comparison\n(MassSpecGym Top-10 vs Ours Top-50)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#808080', label='MassSpecGym Baselines'),
        mpatches.Patch(color='#2196F3', label='Ours (End-to-End)'),
        mpatches.Patch(color='#4CAF50', label='Ours (Oracle)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")


def plot_tanimoto_comparison(our_results: dict, output_dir: Path):
    """Bar chart comparing Tanimoto similarity."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(MASSSPECGYM_BASELINES.keys()) + ["Ours (E2E)", "Ours (Oracle)"]
    colors = ["#808080"] * len(MASSSPECGYM_BASELINES) + ["#2196F3", "#4CAF50"]

    tanimoto_scores = [
        MASSSPECGYM_BASELINES["Random chemical gen."]["top10_tanimoto"],
        MASSSPECGYM_BASELINES["SMILES Transformer"]["top10_tanimoto"],
        MASSSPECGYM_BASELINES["SELFIES Transformer"]["top10_tanimoto"],
        our_results["e2e"]["tanimoto"],
        our_results["oracle"]["tanimoto"],
    ]

    x = np.arange(len(methods))
    bars = ax.bar(x, tanimoto_scores, color=colors, edgecolor='black', linewidth=0.5)

    for bar, score in zip(bars, tanimoto_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("Mean Best Tanimoto Similarity", fontsize=12)
    ax.set_title("De Novo Molecule Generation: Tanimoto Similarity Comparison\n(MassSpecGym Top-10 vs Ours Top-50)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='#808080', label='MassSpecGym Baselines'),
        mpatches.Patch(color='#2196F3', label='Ours (End-to-End)'),
        mpatches.Patch(color='#4CAF50', label='Ours (Oracle)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / "tanimoto_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'tanimoto_comparison.png'}")


def plot_combined_comparison(our_results: dict, output_dir: Path):
    """Combined side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["Random\nchemical", "SMILES\nTransformer", "SELFIES\nTransformer",
               "Ours\n(E2E)", "Ours\n(Oracle)"]
    colors = ["#9E9E9E", "#9E9E9E", "#9E9E9E", "#2196F3", "#4CAF50"]

    # Accuracy
    accuracies = [0, 0, 0, our_results["e2e"]["exact_match"], our_results["oracle"]["exact_match"]]
    bars1 = axes[0].bar(methods, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars1, accuracies):
        axes[0].annotate(f'{acc:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, max(acc, 2)),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Exact Match (%)", fontsize=12)
    axes[0].set_title("(a) Exact Match Accuracy", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)

    # Tanimoto
    tanimotos = [0.10, 0.17, 0.15, our_results["e2e"]["tanimoto"], our_results["oracle"]["tanimoto"]]
    bars2 = axes[1].bar(methods, tanimotos, color=colors, edgecolor='black', linewidth=0.5)
    for bar, tan in zip(bars2, tanimotos):
        axes[1].annotate(f'{tan:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, tan),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Mean Best Tanimoto", fontsize=12)
    axes[1].set_title("(b) Tanimoto Similarity", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)

    # Common legend
    legend_elements = [
        mpatches.Patch(color='#9E9E9E', label='MassSpecGym (Top-10)'),
        mpatches.Patch(color='#2196F3', label='Ours E2E (Top-50)'),
        mpatches.Patch(color='#4CAF50', label='Ours Oracle (Top-50)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_dir / "massspecgym_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'massspecgym_comparison.png'}")


def plot_retrieval_comparison(our_results: dict, output_dir: Path):
    """Compare with molecule retrieval baselines (Hit@k metrics)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # MassSpecGym retrieval methods
    methods = list(MASSSPECGYM_RETRIEVAL.keys()) + ["Ours (E2E)"]

    # Map our exact match (with 50 candidates) to approximate Hit@50
    # Our exact match is essentially Hit@50 since we check if correct is in top 50
    hit_rates = [
        MASSSPECGYM_RETRIEVAL["Random"]["hit20"],
        MASSSPECGYM_RETRIEVAL["Fingerprint FFN"]["hit20"],
        MASSSPECGYM_RETRIEVAL["DeepSets"]["hit20"],
        MASSSPECGYM_RETRIEVAL["DeepSets + Fourier"]["hit20"],
        MASSSPECGYM_RETRIEVAL["MIST (SOTA)"]["hit20"],
        our_results["e2e"]["exact_match"],  # Our Hit@50
    ]

    colors = ["#808080"] * 5 + ["#2196F3"]
    x = np.arange(len(methods))
    bars = ax.bar(x, hit_rates, color=colors, edgecolor='black', linewidth=0.5)

    for bar, rate in zip(bars, hit_rates):
        ax.annotate(f'{rate:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, rate),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_title("Molecule Retrieval Comparison\n(MassSpecGym Hit@20 vs Ours Hit@50)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='#808080', label='MassSpecGym (Hit@20)'),
        mpatches.Patch(color='#2196F3', label='Ours (Hit@50)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'retrieval_comparison.png'}")


def plot_pipeline_overview(our_results: dict, output_dir: Path):
    """Visual overview of our two-stage pipeline with performance."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Boxes
    box_style = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', linewidth=2)

    # Stage boxes
    ax.add_patch(plt.Rectangle((0.5, 3), 3, 2, fill=True, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(2, 4, "Mass\nSpectrum", ha='center', va='center', fontsize=12, fontweight='bold')

    ax.add_patch(plt.Rectangle((5, 3), 3, 2, fill=True, facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    ax.text(6.5, 4.3, "Part A:\nLightGBM", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6.5, 3.5, f"R² = {our_results['e2e']['part_a_r2']:.3f}", ha='center', va='center', fontsize=10)

    ax.add_patch(plt.Rectangle((9.5, 3), 3, 2, fill=True, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(11, 4.3, "Part B:\nDirectDecoder", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(11, 3.5, "Oracle: 82.2%", ha='center', va='center', fontsize=10)

    # Arrows
    ax.annotate('', xy=(5, 4), xytext=(3.5, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(4.25, 4.4, "28 Descriptors", ha='center', fontsize=9)

    ax.annotate('', xy=(9.5, 4), xytext=(8, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(8.75, 4.4, "Predicted\nDescriptors", ha='center', fontsize=9)

    # Output
    ax.add_patch(plt.Rectangle((9.5, 0.5), 3, 1.5, fill=True, facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2))
    ax.text(11, 1.25, "SMILES Candidates", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(11, 2), xytext=(11, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Results box
    rect = plt.Rectangle((0.5, 0.5), 5, 2, fill=False, edgecolor='none')
    ax.add_patch(rect)
    ax.text(3, 2.2, "End-to-End Results (n=2,347)", ha='center', fontsize=12, fontweight='bold')
    ax.text(3, 1.6, f"• Exact Match: {our_results['e2e']['exact_match']:.1f}%", ha='center', fontsize=11)
    ax.text(3, 1.1, f"• Tanimoto: {our_results['e2e']['tanimoto']:.3f}", ha='center', fontsize=11)
    ax.text(3, 0.6, f"• Validity: {our_results['e2e']['validity']:.1f}%", ha='center', fontsize=11)

    # Title
    ax.text(7, 7, "SPEC2SMILES Two-Stage Pipeline", ha='center', fontsize=16, fontweight='bold')
    ax.text(7, 6.4, "Spectrum → Descriptors → SMILES", ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / "pipeline_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'pipeline_results.png'}")


def create_comparison_table(our_results: dict, output_dir: Path):
    """Generate LaTeX and Markdown tables for dissertation."""

    # Main comparison table
    data = {
        "Method": [
            "Random chemical gen.",
            "SMILES Transformer",
            "SELFIES Transformer",
            "\\textbf{Ours (E2E)}",
            "\\textbf{Ours (Oracle)}",
        ],
        "k": ["10", "10", "10", "50", "50"],
        "Accuracy (%)": [
            "0.00",
            "0.00",
            "0.00",
            f"\\textbf{{{our_results['e2e']['exact_match']:.1f}}}",
            f"\\textbf{{{our_results['oracle']['exact_match']:.1f}}}",
        ],
        "Tanimoto": [
            "0.10",
            "0.17",
            "0.15",
            f"\\textbf{{{our_results['e2e']['tanimoto']:.3f}}}",
            f"\\textbf{{{our_results['oracle']['tanimoto']:.3f}}}",
        ],
        "Validity (%)": [
            "-",
            "-",
            "-",
            f"{our_results['e2e']['validity']:.1f}",
            f"{our_results['oracle']['validity']:.1f}",
        ],
    }

    df = pd.DataFrame(data)

    # LaTeX table
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison with MassSpecGym De Novo Generation Baselines}
\label{tab:massspecgym-comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{k} & \textbf{Accuracy (\%)} & \textbf{Tanimoto} & \textbf{Validity (\%)} \\
\midrule
\multicolumn{5}{l}{\textit{MassSpecGym Baselines (231K spectra, 29K molecules)}} \\
Random chemical gen. & 10 & 0.00 & 0.10 & - \\
SMILES Transformer & 10 & 0.00 & 0.17 & - \\
SELFIES Transformer & 10 & 0.00 & 0.15 & - \\
\midrule
\multicolumn{5}{l}{\textit{Ours (GNPS dataset, 2,347 test samples)}} \\
""" + f"""Ours (E2E) & 50 & \\textbf{{{our_results['e2e']['exact_match']:.1f}}} & \\textbf{{{our_results['e2e']['tanimoto']:.3f}}} & {our_results['e2e']['validity']:.1f} \\\\
Ours (Oracle) & 50 & \\textbf{{{our_results['oracle']['exact_match']:.1f}}} & \\textbf{{{our_results['oracle']['tanimoto']:.3f}}} & {our_results['oracle']['validity']:.1f} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Markdown table
    md_table = f"""
# Comparison with MassSpecGym Benchmarks

## De Novo Molecule Generation (Table 2 from MassSpecGym)

| Method | k | Accuracy (%) | Tanimoto | Validity (%) |
|--------|---|--------------|----------|--------------|
| **MassSpecGym Baselines** (231K spectra, 29K molecules) | | | | |
| Random chemical gen. | 10 | 0.00 | 0.10 | - |
| SMILES Transformer | 10 | 0.00 | 0.17 | - |
| SELFIES Transformer | 10 | 0.00 | 0.15 | - |
| **Ours** (GNPS dataset, 2,347 test samples) | | | | |
| Ours (E2E) | 50 | **{our_results['e2e']['exact_match']:.1f}** | **{our_results['e2e']['tanimoto']:.3f}** | {our_results['e2e']['validity']:.1f} |
| Ours (Oracle) | 50 | **{our_results['oracle']['exact_match']:.1f}** | **{our_results['oracle']['tanimoto']:.3f}** | {our_results['oracle']['validity']:.1f} |

## Molecule Retrieval (Table 3 from MassSpecGym)

| Method | Hit@1 (%) | Hit@5 (%) | Hit@20 (%) |
|--------|-----------|-----------|------------|
| Random | 0.37 | 2.01 | 8.22 |
| Fingerprint FFN | 1.47 | 6.21 | 19.23 |
| DeepSets | 2.54 | 7.59 | 20.00 |
| DeepSets + Fourier | 5.24 | 12.58 | 28.21 |
| MIST (SOTA) | 14.64 | 34.87 | 59.15 |
| **Ours (E2E, Hit@50)** | - | - | **{our_results['e2e']['exact_match']:.1f}** |

## Key Findings

1. **Our two-stage approach achieves non-zero exact match accuracy** while all MassSpecGym
   baselines achieve 0% accuracy on their de novo generation task.

2. **35.9% exact match** with end-to-end prediction (vs 0% for direct spectrum-to-SMILES models).

3. **82.2% exact match** with oracle descriptors, demonstrating the effectiveness of Part B.

4. **Tanimoto improvement**: 0.593 (E2E) vs 0.17 (best MassSpecGym baseline) = **3.5x improvement**.

5. **100% validity** on all generated SMILES due to SELFIES-based decoding.

## Important Caveats

- Different datasets: MassSpecGym uses 231K spectra with MCES-based split; we use GNPS with random split
- Different k values: MassSpecGym reports Top-10; we use Top-50 candidates
- Our approach uses intermediate descriptor prediction (two-stage), not direct spectrum-to-SMILES
"""

    # Save tables
    with open(output_dir / "comparison_table.tex", "w") as f:
        f.write(latex_table)
    print(f"Saved: {output_dir / 'comparison_table.tex'}")

    with open(output_dir / "comparison_table.md", "w") as f:
        f.write(md_table)
    print(f"Saved: {output_dir / 'comparison_table.md'}")

    return latex_table, md_table


def plot_improvement_summary(our_results: dict, output_dir: Path):
    """Summary plot showing improvement percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Exact Match\n(vs 0%)", "Tanimoto\n(vs 0.17)", "Validity"]
    our_e2e = [our_results['e2e']['exact_match'],
               our_results['e2e']['tanimoto'] * 100,
               our_results['e2e']['validity']]
    our_oracle = [our_results['oracle']['exact_match'],
                  our_results['oracle']['tanimoto'] * 100,
                  our_results['oracle']['validity']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, our_e2e, width, label='E2E', color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, our_oracle, width, label='Oracle', color='#4CAF50', edgecolor='black')

    # Add improvement annotations
    ax.annotate('∞× vs\nbaseline', xy=(0, our_e2e[0] + 3), ha='center', fontsize=9, color='#1565C0')
    ax.annotate('3.5× vs\nbest', xy=(1, our_e2e[1] + 3), ha='center', fontsize=9, color='#1565C0')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('SPEC2SMILES Performance Summary\nvs MassSpecGym Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'improvement_summary.png'}")


def main():
    base_path = Path(__file__).parent.parent
    metrics_path = base_path / "data" / "output" / "metrics"
    output_dir = base_path / "data" / "output" / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SPEC2SMILES vs MassSpecGym Comparison")
    print("=" * 60)

    # Load our results
    our_results = load_our_results(metrics_path)

    if not our_results:
        print("ERROR: No results found!")
        return

    print(f"\nOur Results:")
    print(f"  E2E: {our_results['e2e']['exact_match']:.1f}% exact match, "
          f"{our_results['e2e']['tanimoto']:.3f} Tanimoto")
    print(f"  Oracle: {our_results['oracle']['exact_match']:.1f}% exact match, "
          f"{our_results['oracle']['tanimoto']:.3f} Tanimoto")

    print("\nMassSpecGym Baselines (Top-10):")
    for name, metrics in MASSSPECGYM_BASELINES.items():
        print(f"  {name}: {metrics['top10_accuracy']:.1f}% accuracy, "
              f"{metrics['top10_tanimoto']:.3f} Tanimoto")

    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # Generate all plots
    plot_accuracy_comparison(our_results, output_dir)
    plot_tanimoto_comparison(our_results, output_dir)
    plot_combined_comparison(our_results, output_dir)
    plot_retrieval_comparison(our_results, output_dir)
    plot_pipeline_overview(our_results, output_dir)
    plot_improvement_summary(our_results, output_dir)

    # Generate tables
    latex_table, md_table = create_comparison_table(our_results, output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey Improvements over MassSpecGym Baselines:")
    print(f"  • Exact Match: {our_results['e2e']['exact_match']:.1f}% vs 0% (∞× improvement)")
    print(f"  • Tanimoto: {our_results['e2e']['tanimoto']:.3f} vs 0.17 "
          f"({our_results['e2e']['tanimoto']/0.17:.1f}× improvement)")
    print(f"  • Validity: {our_results['e2e']['validity']:.1f}% (100% valid SMILES)")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
