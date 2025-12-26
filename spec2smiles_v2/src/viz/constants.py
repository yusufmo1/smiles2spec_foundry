"""Visualization constants and style configuration."""

import matplotlib.pyplot as plt
import seaborn as sns

# MassSpecGym baseline data (arXiv:2410.23326v3, Table 2)
MASSSPECGYM_BASELINES = {
    "Random": {"top10_accuracy": 0.00, "top10_tanimoto": 0.10},
    "SMILES Transformer": {"top10_accuracy": 0.00, "top10_tanimoto": 0.17},
    "SELFIES Transformer": {"top10_accuracy": 0.00, "top10_tanimoto": 0.15},
}

# Color palette for consistent styling
COLORS = {
    "oracle": "#27ae60",
    "e2e": "#e74c3c",
    "lgbm": "#3498db",
    "hybrid": "#9b59b6",
    "baseline": "#9E9E9E",
}


def setup_style() -> None:
    """Configure matplotlib for academic figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
    })
