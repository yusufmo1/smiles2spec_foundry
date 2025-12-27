"""Visualization utilities for spectrum prediction."""

from smiles2spec.visualization.constants import COLORS, STYLES
from smiles2spec.visualization.diagnostics import (
    plot_predicted_vs_actual,
    plot_residual_distribution,
    plot_sample_spectra,
    plot_2x2_diagnostic,
)

__all__ = [
    "COLORS",
    "STYLES",
    "plot_predicted_vs_actual",
    "plot_residual_distribution",
    "plot_sample_spectra",
    "plot_2x2_diagnostic",
]
