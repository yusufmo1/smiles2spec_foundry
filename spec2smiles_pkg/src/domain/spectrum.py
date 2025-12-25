"""Pure spectrum processing functions.

These functions have no external dependencies beyond numpy,
making them easy to test and reuse.
"""

from typing import List, Literal, Tuple

import numpy as np


def bin_spectrum(
    peaks: List[Tuple[float, float]],
    n_bins: int = 500,
    bin_width: float = 1.0,
    max_mz: float = 500.0,
) -> np.ndarray:
    """Bin peak list into fixed-size spectrum array.

    Aggregates peak intensities into m/z bins.

    Args:
        peaks: List of (m/z, intensity) tuples
        n_bins: Number of bins in output spectrum
        bin_width: Width of each bin in m/z units
        max_mz: Maximum m/z value to consider

    Returns:
        Binned spectrum array of shape (n_bins,)

    Example:
        >>> peaks = [(50.0, 100.0), (50.5, 50.0), (100.0, 200.0)]
        >>> spectrum = bin_spectrum(peaks, n_bins=500)
        >>> spectrum[50]  # Contains 150.0 (aggregated)
        150.0
    """
    binned = np.zeros(n_bins, dtype=np.float32)

    for mz, intensity in peaks:
        if mz >= max_mz or mz < 0:
            continue
        bin_idx = int(mz / bin_width)
        if 0 <= bin_idx < n_bins:
            binned[bin_idx] += intensity

    return binned


def transform_spectrum(
    spectrum: np.ndarray,
    method: Literal["sqrt", "log", "none"] = "sqrt",
) -> np.ndarray:
    """Apply intensity transformation to spectrum.

    Transformations help stabilize variance and improve model performance.

    Args:
        spectrum: Input spectrum array
        method: Transformation method
            - "sqrt": Square root transformation
            - "log": Log1p transformation (log(1 + x))
            - "none": No transformation

    Returns:
        Transformed spectrum array

    Example:
        >>> spectrum = np.array([100.0, 400.0, 900.0])
        >>> transform_spectrum(spectrum, "sqrt")
        array([10., 20., 30.])
    """
    if method == "sqrt":
        return np.sqrt(spectrum)
    elif method == "log":
        return np.log1p(spectrum)
    elif method == "none":
        return spectrum.copy()
    else:
        raise ValueError(f"Unknown transform method: {method}")


def normalize_spectrum(
    spectrum: np.ndarray,
    method: Literal["max", "sum", "l2"] = "max",
) -> np.ndarray:
    """Normalize spectrum intensities.

    Args:
        spectrum: Input spectrum array
        method: Normalization method
            - "max": Divide by maximum value (range [0, 1])
            - "sum": Divide by sum (total intensity = 1)
            - "l2": L2 normalization (unit norm)

    Returns:
        Normalized spectrum array

    Example:
        >>> spectrum = np.array([0.0, 50.0, 100.0])
        >>> normalize_spectrum(spectrum, "max")
        array([0. , 0.5, 1. ])
    """
    if method == "max":
        max_val = spectrum.max()
        return spectrum / max_val if max_val > 0 else spectrum.copy()
    elif method == "sum":
        total = spectrum.sum()
        return spectrum / total if total > 0 else spectrum.copy()
    elif method == "l2":
        norm = np.linalg.norm(spectrum)
        return spectrum / norm if norm > 0 else spectrum.copy()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def process_spectrum(
    peaks: List[Tuple[float, float]],
    n_bins: int = 500,
    bin_width: float = 1.0,
    max_mz: float = 500.0,
    transform: Literal["sqrt", "log", "none"] = "sqrt",
    normalize: bool = True,
) -> np.ndarray:
    """Complete spectrum processing pipeline.

    Combines binning, transformation, and normalization.

    Args:
        peaks: List of (m/z, intensity) tuples
        n_bins: Number of bins in output spectrum
        bin_width: Width of each bin in m/z units
        max_mz: Maximum m/z value to consider
        transform: Transformation method ("sqrt", "log", "none")
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Processed spectrum array of shape (n_bins,)
    """
    spectrum = bin_spectrum(peaks, n_bins, bin_width, max_mz)
    spectrum = transform_spectrum(spectrum, transform)
    if normalize:
        spectrum = normalize_spectrum(spectrum, "max")
    return spectrum
