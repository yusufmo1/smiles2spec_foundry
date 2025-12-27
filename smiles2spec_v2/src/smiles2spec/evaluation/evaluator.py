"""Model evaluation utilities.

Handles comprehensive model evaluation and comparison.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from smiles2spec.evaluation.metrics import (
    compute_all_metrics,
    compute_per_bin_metrics,
    cosine_similarity,
)
from smiles2spec.models.base import BaseModel


class ModelEvaluator:
    """Comprehensive model evaluation utility."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        """Initialize evaluator.

        Args:
            output_dir: Directory for saving results
            verbose: Print evaluation progress
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: BaseModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        compute_bootstrap: bool = True,
    ) -> Dict:
        """Evaluate a single model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for reporting
            compute_bootstrap: Compute bootstrap CIs

        Returns:
            Evaluation results
        """
        if self.verbose:
            print(f"Evaluating {model_name}...")

        # Generate predictions
        predictions = model.predict(X_test)

        # Compute metrics
        metrics = compute_all_metrics(
            predictions, y_test, compute_bootstrap=compute_bootstrap
        )
        metrics["model_name"] = model_name

        # Per-bin analysis
        per_bin = compute_per_bin_metrics(predictions, y_test)
        metrics["per_bin"] = per_bin

        if self.verbose:
            self._print_summary(metrics)

        # Save results
        if self.output_dir:
            self._save_results(metrics, model_name)

        return metrics

    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Compare multiple models.

        Args:
            models: Dict of name -> model
            X_test: Test features
            y_test: Test targets

        Returns:
            Comparison results
        """
        results = {}

        for name, model in models.items():
            results[name] = self.evaluate(
                model, X_test, y_test, model_name=name, compute_bootstrap=False
            )

        # Rank models
        rankings = self._rank_models(results)
        comparison = {
            "models": results,
            "rankings": rankings,
            "best_model": rankings[0]["name"],
        }

        if self.output_dir:
            with open(self.output_dir / "comparison.json", "w") as f:
                json.dump(comparison, f, indent=2, default=str)

        return comparison

    def evaluate_by_mw_range(
        self,
        model: BaseModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        molecular_weights: np.ndarray,
        ranges: Optional[List[tuple]] = None,
    ) -> Dict:
        """Evaluate performance by molecular weight range.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            molecular_weights: MW for each sample
            ranges: List of (min, max) MW ranges

        Returns:
            Performance by MW range
        """
        if ranges is None:
            ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

        predictions = model.predict(X_test)
        results = {}

        for mw_min, mw_max in ranges:
            mask = (molecular_weights >= mw_min) & (molecular_weights < mw_max)
            n_samples = mask.sum()

            if n_samples > 0:
                cos_sim = cosine_similarity(predictions[mask], y_test[mask])
                results[f"{mw_min}-{mw_max}"] = {
                    "n_samples": int(n_samples),
                    "cosine_mean": float(np.mean(cos_sim)),
                    "cosine_std": float(np.std(cos_sim)),
                }
            else:
                results[f"{mw_min}-{mw_max}"] = {"n_samples": 0}

        return results

    def _print_summary(self, metrics: Dict) -> None:
        """Print evaluation summary."""
        cos = metrics["cosine_similarity"]
        print(f"\n{'=' * 50}")
        print(f"Model: {metrics['model_name']}")
        print(f"{'=' * 50}")
        print(f"Cosine Similarity: {cos['mean']:.4f} ± {cos['std']:.4f}")
        print(f"  Median: {cos['median']:.4f}")
        print(f"  Range: [{cos['min']:.4f}, {cos['max']:.4f}]")
        if "ci_low" in cos:
            print(f"  95% CI: [{cos['ci_low']:.4f}, {cos['ci_high']:.4f}]")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"N samples: {metrics['n_samples']}")

    def _save_results(self, metrics: Dict, model_name: str) -> None:
        """Save evaluation results."""
        if self.output_dir is None:
            return

        # Convert numpy arrays for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_path = self.output_dir / f"{model_name}_evaluation.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=convert)

    def _rank_models(self, results: Dict) -> List[Dict]:
        """Rank models by cosine similarity."""
        rankings = []
        for name, metrics in results.items():
            rankings.append({
                "name": name,
                "cosine_mean": metrics["cosine_similarity"]["mean"],
                "cosine_std": metrics["cosine_similarity"]["std"],
            })

        return sorted(rankings, key=lambda x: x["cosine_mean"], reverse=True)
