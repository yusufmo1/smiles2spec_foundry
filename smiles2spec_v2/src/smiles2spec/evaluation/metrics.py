"""Spectral similarity metrics for evaluation.

Implements cosine similarity, weighted dot product, and spectral contrast angle.
"""

from typing import Dict, Optional

import numpy as np
from scipy import stats


def cosine_similarity(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Compute cosine similarity between predicted and target spectra.

    Args:
        pred: Predictions (n_samples, n_bins) or (n_bins,)
        target: Targets (n_samples, n_bins) or (n_bins,)
        eps: Small value for numerical stability

    Returns:
        Cosine similarities (n_samples,) or scalar
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        target = target.reshape(1, -1)

    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True) + eps
    target_norm = np.linalg.norm(target, axis=1, keepdims=True) + eps

    pred_unit = pred / pred_norm
    target_unit = target / target_norm

    similarity = np.sum(pred_unit * target_unit, axis=1)
    return similarity.squeeze()


def weighted_dot_product(
    pred: np.ndarray,
    target: np.ndarray,
    mz_power: float = 0.5,
    intensity_power: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute weighted dot product similarity.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)
        mz_power: Power for m/z weighting
        intensity_power: Power for intensity weighting
        eps: Small value for numerical stability

    Returns:
        Weighted similarities (n_samples,)
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        target = target.reshape(1, -1)

    n_bins = pred.shape[1]
    mz_weights = np.arange(1, n_bins + 1) ** mz_power

    # Apply intensity power
    pred_weighted = (pred ** intensity_power) * mz_weights
    target_weighted = (target ** intensity_power) * mz_weights

    # Normalize
    pred_norm = np.sqrt(np.sum(pred_weighted ** 2, axis=1, keepdims=True)) + eps
    target_norm = np.sqrt(np.sum(target_weighted ** 2, axis=1, keepdims=True)) + eps

    pred_unit = pred_weighted / pred_norm
    target_unit = target_weighted / target_norm

    similarity = np.sum(pred_unit * target_unit, axis=1)
    return similarity.squeeze()


def spectral_contrast_angle(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Compute spectral contrast angle.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)
        eps: Small value for numerical stability

    Returns:
        Contrast angles in degrees (n_samples,)
    """
    cos_sim = cosine_similarity(pred, target, eps)
    # Clamp to valid range for arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle = np.arccos(cos_sim) * 180 / np.pi
    return angle


def mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(pred - target)))


def root_mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    compute_bootstrap: bool = False,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict:
    """Compute all spectral similarity metrics.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)
        compute_bootstrap: Whether to compute bootstrap CIs
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for intervals

    Returns:
        Dict of metric values
    """
    cos_sim = cosine_similarity(pred, target)
    wdp = weighted_dot_product(pred, target)
    sca = spectral_contrast_angle(pred, target)

    results = {
        "cosine_similarity": {
            "mean": float(np.mean(cos_sim)),
            "std": float(np.std(cos_sim)),
            "median": float(np.median(cos_sim)),
            "min": float(np.min(cos_sim)),
            "max": float(np.max(cos_sim)),
        },
        "weighted_dot_product": {
            "mean": float(np.mean(wdp)),
            "std": float(np.std(wdp)),
        },
        "spectral_contrast_angle": {
            "mean": float(np.mean(sca)),
            "std": float(np.std(sca)),
        },
        "mae": mean_absolute_error(pred, target),
        "rmse": root_mean_squared_error(pred, target),
        "n_samples": len(pred),
    }

    if compute_bootstrap:
        ci_low, ci_high = _bootstrap_ci(cos_sim, n_bootstrap, confidence)
        results["cosine_similarity"]["ci_low"] = ci_low
        results["cosine_similarity"]["ci_high"] = ci_high

    return results


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple:
    """Compute bootstrap confidence interval.

    Args:
        values: Array of values
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level

    Returns:
        Tuple of (lower, upper) bounds
    """
    n = len(values)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    ci_low = np.percentile(means, alpha * 100)
    ci_high = np.percentile(means, (1 - alpha) * 100)

    return float(ci_low), float(ci_high)


def compute_per_bin_metrics(pred: np.ndarray, target: np.ndarray) -> Dict:
    """Compute metrics per m/z bin.

    Args:
        pred: Predictions (n_samples, n_bins)
        target: Targets (n_samples, n_bins)

    Returns:
        Dict with per-bin statistics
    """
    n_bins = pred.shape[1]

    mae_per_bin = np.mean(np.abs(pred - target), axis=0)
    r2_per_bin = []

    for i in range(n_bins):
        if np.std(target[:, i]) > 1e-8:
            r2 = 1 - np.sum((pred[:, i] - target[:, i]) ** 2) / np.sum(
                (target[:, i] - target[:, i].mean()) ** 2
            )
        else:
            r2 = 0.0
        r2_per_bin.append(r2)

    return {
        "mae_per_bin": mae_per_bin,
        "r2_per_bin": np.array(r2_per_bin),
        "best_bins": np.argsort(mae_per_bin)[:10].tolist(),
        "worst_bins": np.argsort(mae_per_bin)[-10:].tolist(),
    }
