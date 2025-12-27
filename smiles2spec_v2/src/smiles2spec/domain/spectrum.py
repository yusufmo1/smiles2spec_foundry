"""Spectrum binning and transformation utilities.

Converts raw peak lists to fixed-length binned vectors.
"""

from typing import List, Literal, Tuple

import numpy as np

from smiles2spec.core.exceptions import DataError


def apply_transform(
    spectrum: np.ndarray, method: Literal["sqrt", "log", "none"] = "sqrt"
) -> np.ndarray:
    """Apply intensity transformation to spectrum.

    Args:
        spectrum: Binned spectrum array
        method: Transformation method ('sqrt', 'log', 'none')

    Returns:
        Transformed spectrum
    """
    if method == "sqrt":
        return np.sqrt(np.maximum(spectrum, 0))
    elif method == "log":
        return np.log1p(np.maximum(spectrum, 0))
    elif method == "none":
        return spectrum
    else:
        raise DataError(f"Unknown transform method: {method}")


def inverse_transform(
    spectrum: np.ndarray, method: Literal["sqrt", "log", "none"] = "sqrt"
) -> np.ndarray:
    """Apply inverse intensity transformation.

    Args:
        spectrum: Transformed spectrum array
        method: Original transformation method

    Returns:
        Original-scale spectrum
    """
    if method == "sqrt":
        return spectrum ** 2
    elif method == "log":
        return np.expm1(np.maximum(spectrum, 0))
    elif method == "none":
        return spectrum
    else:
        raise DataError(f"Unknown transform method: {method}")


class SpectrumBinner:
    """Convert peak lists to fixed-length binned vectors.

    Attributes:
        n_bins: Number of m/z bins
        bin_width: Width of each bin in m/z units
        max_mz: Maximum m/z value
        transform: Intensity transformation method
        normalize: Whether to normalize intensities
    """

    def __init__(
        self,
        n_bins: int = 500,
        bin_width: float = 1.0,
        max_mz: float = 499.0,
        transform: Literal["sqrt", "log", "none"] = "sqrt",
        normalize: bool = True,
    ):
        self.n_bins = n_bins
        self.bin_width = bin_width
        self.max_mz = max_mz
        self.transform = transform
        self.normalize = normalize

    def bin_spectrum(
        self, peaks: List[Tuple[float, float]], return_metadata: bool = False
    ) -> np.ndarray:
        """Convert peak list to binned spectrum.

        Args:
            peaks: List of (m/z, intensity) tuples
            return_metadata: If True, return dict with additional info

        Returns:
            Binned spectrum array of shape (n_bins,)
            If return_metadata=True, returns (spectrum, metadata_dict)
        """
        binned = np.zeros(self.n_bins, dtype=np.float32)
        peaks_beyond_max = 0

        for mz, intensity in peaks:
            if mz <= self.max_mz:
                bin_idx = min(int(mz / self.bin_width), self.n_bins - 1)
                # Take max intensity for overlapping peaks
                binned[bin_idx] = max(binned[bin_idx], intensity)
            else:
                peaks_beyond_max += 1

        # Normalize to [0, 1]
        if self.normalize:
            max_intensity = np.max(binned)
            if max_intensity > 0:
                binned = binned / max_intensity

        # Apply transform
        binned = apply_transform(binned, self.transform)

        if return_metadata:
            metadata = {
                "original_peak_count": len(peaks),
                "peaks_beyond_max": peaks_beyond_max,
                "max_intensity": float(np.max(binned)),
            }
            return binned, metadata

        return binned

    def batch_bin(self, peaks_list: List[List[Tuple[float, float]]]) -> np.ndarray:
        """Bin multiple spectra.

        Args:
            peaks_list: List of peak lists

        Returns:
            Array of shape (n_samples, n_bins)
        """
        return np.array([self.bin_spectrum(peaks) for peaks in peaks_list])

    def to_peaks(
        self, binned: np.ndarray, threshold: float = 0.01
    ) -> List[Tuple[float, float]]:
        """Convert binned spectrum back to peak list.

        Args:
            binned: Binned spectrum array
            threshold: Minimum intensity to include

        Returns:
            List of (m/z, intensity) tuples
        """
        # Reverse transform
        spectrum = inverse_transform(binned, self.transform)

        peaks = []
        for i, intensity in enumerate(spectrum):
            if intensity > threshold:
                mz = (i + 0.5) * self.bin_width  # Center of bin
                peaks.append((mz, float(intensity)))

        return peaks
