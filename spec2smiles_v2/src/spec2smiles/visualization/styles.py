"""Colorblind-friendly styles and palettes for SPEC2SMILES visualizations."""

import matplotlib.pyplot as plt


# Colorblind-friendly palette (Wong 2011)
PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "yellow": "#F0E442",
    "green": "#009E73",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}

# Performance threshold colors
PERFORMANCE_COLORS = {
    "high": "#009E73",      # green - R² > 0.7
    "moderate": "#E69F00",  # orange - 0.5 < R² <= 0.7
    "low": "#D55E00",       # vermillion - R² <= 0.5
}


def set_style() -> None:
    """Set consistent matplotlib style for all plots."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def get_performance_color(r2_score: float) -> str:
    """Get color based on R² performance threshold."""
    if r2_score > 0.7:
        return PERFORMANCE_COLORS["high"]
    elif r2_score > 0.5:
        return PERFORMANCE_COLORS["moderate"]
    else:
        return PERFORMANCE_COLORS["low"]
