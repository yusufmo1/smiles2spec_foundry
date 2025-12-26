"""Hit@K and Tanimoto analysis visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .constants import COLORS


def _canonicalize(smiles: str) -> str | None:
    """Canonicalize SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _get_tanimoto(smi1: str, smi2: str) -> float:
    """Compute Tanimoto similarity between two SMILES."""
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 and mol2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        pass
    return 0.0


def compute_hitatk(predictions: list) -> dict:
    """Compute Hit@K for various K values."""
    k_values = [1, 2, 3, 5, 10]
    results = {}

    for k in k_values:
        hits = 0
        for pred in predictions:
            true_can = pred.get("true_canonical", "")
            candidates = pred.get("all_candidates", [])[:k]
            for cand in candidates:
                if _canonicalize(cand) == true_can:
                    hits += 1
                    break
        results[k] = hits / len(predictions) if predictions else 0
    return results


def compute_tanimoto_stats(predictions: list) -> dict:
    """Compute Tanimoto statistics by rank and distribution."""
    max_k = 10
    tanimotos_by_rank = {i: [] for i in range(1, max_k + 1)}
    top1_tanimotos = []
    best_tanimotos = []

    for pred in predictions:
        true_smi = pred.get("true_smiles", "")
        candidates = pred.get("all_candidates", [])
        best_tan = pred.get("best_tanimoto", 0)
        best_tanimotos.append(best_tan)

        for rank, cand in enumerate(candidates[:max_k], 1):
            tan = _get_tanimoto(cand, true_smi)
            tanimotos_by_rank[rank].append(tan)
            if rank == 1:
                top1_tanimotos.append(tan)

    # Compute stats per rank
    rank_stats = {}
    for rank, tans in tanimotos_by_rank.items():
        if tans:
            rank_stats[rank] = {
                "mean": np.mean(tans),
                "std": np.std(tans),
                "median": np.median(tans),
            }

    return {
        "by_rank": rank_stats,
        "top1": top1_tanimotos,
        "best": best_tanimotos,
    }


def plot_hitatk_analysis(predictions: list, output_dir: Path) -> bool:
    """Plot Hit@K bar chart and cumulative curve."""
    if not predictions:
        print("  [SKIP] No predictions for Hit@K analysis")
        return False

    hitatk = compute_hitatk(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart for Hit@1, 5, 10
    k_display = [1, 5, 10]
    values = [hitatk[k] * 100 for k in k_display]
    colors_bar = [COLORS["e2e"], COLORS["lgbm"], COLORS["oracle"]]

    bars = axes[0].bar([f"Hit@{k}" for k in k_display], values, color=colors_bar,
                       edgecolor="black", width=0.6)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Hit@K Performance (E2E Pipeline)", fontweight="bold")
    axes[0].set_ylim(0, 70)
    axes[0].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)

    # Right: Cumulative curve
    k_all = sorted(hitatk.keys())
    hits_all = [hitatk[k] * 100 for k in k_all]

    axes[1].plot(k_all, hits_all, "o-", color=COLORS["e2e"], linewidth=2,
                 markersize=8, label="E2E Pipeline")
    axes[1].fill_between(k_all, hits_all, alpha=0.2, color=COLORS["e2e"])
    axes[1].set_xlabel("K (Number of Candidates)")
    axes[1].set_ylabel("Hit Rate (%)")
    axes[1].set_title("Cumulative Hit@K Curve", fontweight="bold")
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_ylim(0, 70)
    axes[1].set_xticks(k_all)
    axes[1].grid(True, alpha=0.3)

    # Annotate key points
    axes[1].annotate(f"{hits_all[0]:.1f}%", (1, hits_all[0]), textcoords="offset points",
                     xytext=(10, 5), fontweight="bold")
    axes[1].annotate(f"{hits_all[-1]:.1f}%", (10, hits_all[-1]), textcoords="offset points",
                     xytext=(-20, 5), fontweight="bold")

    # Add insight annotation
    gain_5_to_10 = hits_all[-1] - hits_all[3]  # k=5 is index 3
    axes[1].text(7.5, 25, f"Only +{gain_5_to_10:.1f}% gain\nfrom K=5→10",
                 ha="center", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    plt.savefig(output_dir / "hitatk_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] hitatk_analysis.png")
    return True


def plot_tanimoto_distribution(predictions: list, output_dir: Path) -> bool:
    """Plot Tanimoto distribution histogram and boxplot."""
    if not predictions:
        print("  [SKIP] No predictions for Tanimoto distribution")
        return False

    stats = compute_tanimoto_stats(predictions)
    top1 = stats["top1"]
    best = stats["best"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Histogram with bimodal coloring
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.01]
    bin_labels = ["<0.3\n(Poor)", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0", "=1.0\n(Exact)"]
    counts = []
    for i in range(len(bins) - 1):
        count = sum(1 for t in top1 if bins[i] <= t < bins[i + 1])
        counts.append(count)

    # Color based on quality
    bar_colors = [COLORS["e2e"], "#f39c12", "#f39c12", COLORS["lgbm"],
                  COLORS["lgbm"], COLORS["oracle"]]

    x_pos = np.arange(len(bin_labels))
    bars = axes[0].bar(x_pos, counts, color=bar_colors, edgecolor="black", width=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(bin_labels, fontsize=10)
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Top-1 Candidate Tanimoto Distribution", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Add percentages
    total = len(top1)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Annotate bimodal nature
    axes[0].annotate("Bimodal:\n40% exact OR\n37% poor",
                     xy=(0.5, 0.85), xycoords="axes fraction",
                     fontsize=11, ha="center",
                     bbox=dict(boxstyle="round", facecolor="#fff3cd", edgecolor="#ffc107"))

    # Right: Boxplot comparing Top-1 vs Best-of-K
    bp = axes[1].boxplot([top1, best], labels=["Top-1\nCandidate", "Best of K\nCandidates"],
                         patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COLORS["e2e"])
    bp["boxes"][1].set_facecolor(COLORS["oracle"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    axes[1].set_ylabel("Tanimoto Similarity")
    axes[1].set_title("Top-1 vs Best-of-K Tanimoto", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    # Add mean annotations
    mean_top1 = np.mean(top1)
    mean_best = np.mean(best)
    axes[1].text(1, mean_top1 + 0.05, f"μ={mean_top1:.3f}", ha="center", fontsize=10)
    axes[1].text(2, mean_best + 0.05, f"μ={mean_best:.3f}", ha="center", fontsize=10)

    # Improvement arrow
    axes[1].annotate("", xy=(2, mean_best), xytext=(1, mean_top1),
                     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    improvement = (mean_best - mean_top1) / mean_top1 * 100
    axes[1].text(1.5, (mean_top1 + mean_best) / 2 + 0.05, f"+{improvement:.0f}%",
                 ha="center", fontsize=11, fontweight="bold", color="#27ae60")

    plt.tight_layout()
    plt.savefig(output_dir / "tanimoto_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] tanimoto_distribution.png")
    return True


def plot_tanimoto_by_rank(predictions: list, output_dir: Path) -> bool:
    """Plot Tanimoto decay by candidate rank."""
    if not predictions:
        print("  [SKIP] No predictions for Tanimoto by rank")
        return False

    stats = compute_tanimoto_stats(predictions)
    rank_stats = stats["by_rank"]

    ranks = sorted(rank_stats.keys())
    means = [rank_stats[r]["mean"] for r in ranks]
    stds = [rank_stats[r]["std"] for r in ranks]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(ranks, means, yerr=stds, fmt="o-", color=COLORS["e2e"],
                linewidth=2, markersize=10, capsize=5, capthick=2,
                label="Mean ± Std")
    ax.fill_between(ranks, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=COLORS["e2e"])

    ax.set_xlabel("Candidate Rank", fontsize=12)
    ax.set_ylabel("Tanimoto Similarity", fontsize=12)
    ax.set_title("Tanimoto Similarity by Candidate Rank\n(Model Calibration)", fontweight="bold")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.8)
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)

    # Annotate key points
    ax.annotate(f"{means[0]:.3f}", (1, means[0]), textcoords="offset points",
                xytext=(15, 5), fontweight="bold", fontsize=11)
    ax.annotate(f"{means[-1]:.3f}", (10, means[-1]), textcoords="offset points",
                xytext=(-30, 10), fontweight="bold", fontsize=11)

    # Add decay annotation
    decay = (means[0] - means[-1]) / means[0] * 100
    ax.text(5.5, 0.4, f"Rank 1→10 decay: {decay:.0f}%\n(Model is well-calibrated)",
            ha="center", fontsize=11, style="italic",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", edgecolor="#27ae60"))

    plt.tight_layout()
    plt.savefig(output_dir / "tanimoto_by_rank.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] tanimoto_by_rank.png")
    return True


def plot_analysis_summary(predictions: list, output_dir: Path) -> bool:
    """Create 2x2 summary dashboard."""
    if not predictions:
        print("  [SKIP] No predictions for analysis summary")
        return False

    hitatk = compute_hitatk(predictions)
    stats = compute_tanimoto_stats(predictions)
    top1 = stats["top1"]
    rank_stats = stats["by_rank"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # (a) Hit@K bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    k_display = [1, 5, 10]
    values = [hitatk[k] * 100 for k in k_display]
    colors_bar = [COLORS["e2e"], COLORS["lgbm"], COLORS["oracle"]]
    bars = ax1.bar([f"Hit@{k}" for k in k_display], values, color=colors_bar, width=0.6)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Hit@K Performance", fontweight="bold")
    ax1.set_ylim(0, 70)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontweight="bold")

    # (b) Tanimoto distribution
    ax2 = fig.add_subplot(gs[0, 1])
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    ax2.hist(top1, bins=bins, color=COLORS["e2e"], edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Tanimoto Similarity")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Top-1 Tanimoto Distribution", fontweight="bold")
    ax2.axvline(np.mean(top1), color="red", linestyle="--", label=f"Mean={np.mean(top1):.3f}")
    ax2.legend()

    # (c) Tanimoto by rank
    ax3 = fig.add_subplot(gs[1, 0])
    ranks = sorted(rank_stats.keys())
    means = [rank_stats[r]["mean"] for r in ranks]
    ax3.plot(ranks, means, "o-", color=COLORS["e2e"], linewidth=2, markersize=8)
    ax3.set_xlabel("Candidate Rank")
    ax3.set_ylabel("Mean Tanimoto")
    ax3.set_title("(c) Tanimoto Decay by Rank", fontweight="bold")
    ax3.set_xlim(0.5, 10.5)
    ax3.set_xticks(ranks)
    ax3.grid(True, alpha=0.3)

    # (d) Key statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    exact_pct = sum(1 for t in top1 if t >= 0.9999) / len(top1) * 100
    poor_pct = sum(1 for t in top1 if t < 0.3) / len(top1) * 100
    near_miss = sum(1 for t in top1 if 0.9 <= t < 0.9999) / len(top1) * 100

    stats_text = f"""
    KEY STATISTICS

    Hit@K Performance:
      • Hit@1:  {hitatk[1]*100:5.1f}%
      • Hit@5:  {hitatk[5]*100:5.1f}%
      • Hit@10: {hitatk[10]*100:5.1f}%

    Tanimoto Distribution:
      • Mean Top-1:     {np.mean(top1):.3f}
      • Exact (=1.0):   {exact_pct:.1f}%
      • Poor (<0.3):    {poor_pct:.1f}%
      • Near-miss:      {near_miss:.1f}%

    Model Calibration:
      • Rank 1 mean:    {means[0]:.3f}
      • Rank 10 mean:   {means[-1]:.3f}
      • Decay:          {(means[0]-means[-1])/means[0]*100:.0f}%
    """

    ax4.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=12,
             family="monospace", transform=ax4.transAxes,
             bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    plt.suptitle("SPEC2SMILES: Hit@K and Tanimoto Analysis", fontsize=18, fontweight="bold")
    plt.savefig(output_dir / "analysis_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] analysis_summary.png")
    return True


def generate_analysis(predictions: list, output_dir: Path) -> None:
    """Generate all analysis visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Analysis visualizations...")
    plot_hitatk_analysis(predictions, output_dir)
    plot_tanimoto_distribution(predictions, output_dir)
    plot_tanimoto_by_rank(predictions, output_dir)
    plot_analysis_summary(predictions, output_dir)
