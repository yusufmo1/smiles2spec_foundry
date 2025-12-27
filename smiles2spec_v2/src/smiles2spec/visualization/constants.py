"""Visualization constants and styles."""

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "neutral": "#6B7280",
    "target": "#1F2937",
    "predicted": "#3B82F6",
    "residual": "#EF4444",
}

# Model-specific colors
MODEL_COLORS = {
    "random_forest": "#10B981",
    "modular_net": "#3B82F6",
    "hierarchical_net": "#8B5CF6",
    "sparse_gated_net": "#F59E0B",
    "regional_expert_net": "#EC4899",
    "ensemble": "#1F2937",
}

# Plot styles
STYLES = {
    "figure_size": (10, 8),
    "figure_size_small": (6, 4),
    "figure_size_wide": (14, 6),
    "dpi": 150,
    "font_size": 12,
    "title_size": 14,
    "label_size": 12,
    "line_width": 1.5,
    "marker_size": 5,
    "alpha": 0.7,
    "grid_alpha": 0.3,
}

# m/z range colors
MZ_RANGE_COLORS = {
    "0-100": "#10B981",
    "100-200": "#3B82F6",
    "200-300": "#F59E0B",
    "300-400": "#EF4444",
    "400-500": "#8B5CF6",
}

# Matplotlib style settings
MPL_STYLE = {
    "figure.figsize": STYLES["figure_size"],
    "figure.dpi": STYLES["dpi"],
    "font.size": STYLES["font_size"],
    "axes.titlesize": STYLES["title_size"],
    "axes.labelsize": STYLES["label_size"],
    "axes.grid": True,
    "grid.alpha": STYLES["grid_alpha"],
    "lines.linewidth": STYLES["line_width"],
    "lines.markersize": STYLES["marker_size"],
}


def apply_style():
    """Apply custom matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(MPL_STYLE)
