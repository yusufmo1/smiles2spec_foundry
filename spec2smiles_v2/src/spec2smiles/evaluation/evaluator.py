"""Pipeline evaluation utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
from tqdm import tqdm

from spec2smiles.core.config import PipelineConfig
from spec2smiles.data.loaders import DataLoader
from spec2smiles.models.pipeline import IntegratedPipeline
from spec2smiles.models.part_a.lgbm_ensemble import LGBMEnsemble
from spec2smiles.evaluation.metrics import (
    compute_r2,
    compute_rmse,
    compute_mae,
    compute_tanimoto,
    compute_hit_at_k,
    compute_formula_match_rate,
    compute_validity_rate,
    compute_uniqueness,
    compute_diversity,
)


class PipelineEvaluator:
    """Comprehensive evaluation for SPEC2SMILES pipeline.

    Evaluates:
    - Part A: Descriptor prediction accuracy (R2, RMSE, MAE)
    - Part B: Molecule generation quality
    - Integrated: End-to-end structure recovery
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize evaluator.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.results: Dict[str, Any] = {}

    def evaluate_part_a(
        self,
        model: LGBMEnsemble,
        X_test: np.ndarray,
        y_test: np.ndarray,
        descriptor_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate Part A (Spectrum -> Descriptors) model.

        Args:
            model: Trained LGBMEnsemble model
            X_test: Test spectra of shape (n_samples, 500)
            y_test: True descriptors of shape (n_samples, n_descriptors)
            descriptor_names: Optional descriptor names

        Returns:
            Dictionary with per-descriptor and aggregate metrics
        """
        y_pred = model.predict(X_test)

        if descriptor_names is None:
            descriptor_names = model.descriptor_names

        per_descriptor = {}
        r2_values = []
        rmse_values = []
        mae_values = []

        for i, name in enumerate(descriptor_names):
            y_t = y_test[:, i]
            y_p = y_pred[:, i]

            r2 = compute_r2(y_t, y_p)
            rmse = compute_rmse(y_t, y_p)
            mae = compute_mae(y_t, y_p)

            per_descriptor[name] = {
                "R2": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae),
            }

            r2_values.append(r2)
            rmse_values.append(rmse)
            mae_values.append(mae)

        results = {
            "per_descriptor": per_descriptor,
            "summary": {
                "mean_r2": float(np.mean(r2_values)),
                "median_r2": float(np.median(r2_values)),
                "std_r2": float(np.std(r2_values)),
                "min_r2": float(np.min(r2_values)),
                "max_r2": float(np.max(r2_values)),
                "mean_rmse": float(np.mean(rmse_values)),
                "mean_mae": float(np.mean(mae_values)),
                "best_descriptor": descriptor_names[np.argmax(r2_values)],
                "worst_descriptor": descriptor_names[np.argmin(r2_values)],
            },
            "n_samples": len(X_test),
        }

        self.results["part_a"] = results
        return results

    def evaluate_pipeline(
        self,
        pipeline: IntegratedPipeline,
        test_data: List[Dict],
        n_candidates: int = 50,
        temperature: float = 0.7,
        k_values: List[int] = [1, 5, 10, 50],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate integrated pipeline end-to-end.

        Args:
            pipeline: Loaded IntegratedPipeline
            test_data: Test data with spectra and SMILES
            n_candidates: Number of candidates to generate
            temperature: Sampling temperature
            k_values: K values for Hit@K metrics
            verbose: Whether to show progress

        Returns:
            Comprehensive evaluation metrics
        """
        from spec2smiles.data.loaders import extract_features_and_targets

        X_test, _, smiles_list = extract_features_and_targets(test_data)

        all_candidates = []
        all_similarities = []
        exact_matches = 0
        formula_matches = 0

        iterator = (
            tqdm(range(len(X_test)), desc="Evaluating pipeline")
            if verbose
            else range(len(X_test))
        )

        for i in iterator:
            result = pipeline.predict_and_rank(
                X_test[i],
                true_smiles=smiles_list[i],
                n_candidates=n_candidates,
                temperature=temperature,
            )

            all_candidates.append(result.get("ranked_candidates", result["candidates"]))
            all_similarities.append(result.get("best_similarity", 0.0))

            if result.get("exact_match", False):
                exact_matches += 1

        # Compute Hit@K for various K
        hit_at_k = {}
        for k in k_values:
            hit_rate = compute_hit_at_k(all_candidates, smiles_list, k=k)
            hit_at_k[f"hit@{k}"] = float(hit_rate)

        # Formula match rate
        formula_rate = compute_formula_match_rate(all_candidates, smiles_list)

        # Validity and uniqueness
        validity = compute_validity_rate(all_candidates)
        uniqueness = compute_uniqueness(all_candidates)

        # Diversity (sample for speed)
        sample_size = min(100, len(all_candidates))
        diversities = []
        for candidates in all_candidates[:sample_size]:
            if len(candidates) > 1:
                div = compute_diversity(candidates[:10])
                diversities.append(div)

        results = {
            "exact_match_rate": float(exact_matches / len(X_test)),
            "formula_match_rate": float(formula_rate),
            "hit_at_k": hit_at_k,
            "mean_tanimoto": float(np.mean(all_similarities)),
            "median_tanimoto": float(np.median(all_similarities)),
            "std_tanimoto": float(np.std(all_similarities)),
            "validity_rate": float(validity),
            "uniqueness": float(uniqueness),
            "mean_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "n_samples": len(X_test),
            "n_candidates": n_candidates,
            "temperature": temperature,
        }

        self.results["pipeline"] = results
        return results

    def evaluate_from_directories(
        self,
        data_dir: Path,
        part_a_dir: Path,
        part_b_dir: Path,
        n_candidates: int = 50,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run full evaluation from model directories.

        Args:
            data_dir: Directory with test data
            part_a_dir: Part A model directory
            part_b_dir: Part B model directory
            n_candidates: Number of candidates
            temperature: Sampling temperature
            verbose: Whether to show progress

        Returns:
            Combined evaluation results
        """
        from spec2smiles.data.loaders import extract_features_and_targets

        # Load test data
        if verbose:
            print("Loading test data...")

        train_data, val_data, test_data, metadata = DataLoader.load_processed_splits(
            Path(data_dir)
        )

        X_test, y_test, smiles_list = extract_features_and_targets(test_data)

        if verbose:
            print(f"Test samples: {len(test_data)}")

        # Evaluate Part A
        if verbose:
            print("\nEvaluating Part A (Spectrum -> Descriptors)...")

        part_a_model = LGBMEnsemble.load(Path(part_a_dir) / "lgbm_ensemble.pkl")
        part_a_results = self.evaluate_part_a(
            part_a_model, X_test, y_test, metadata.get("descriptor_names")
        )

        if verbose:
            print(f"  Mean R2: {part_a_results['summary']['mean_r2']:.3f}")
            print(f"  Median R2: {part_a_results['summary']['median_r2']:.3f}")

        # Evaluate integrated pipeline
        if verbose:
            print("\nEvaluating integrated pipeline...")

        pipeline = IntegratedPipeline.from_directories(
            part_a_dir, part_b_dir, verbose=False
        )

        pipeline_results = self.evaluate_pipeline(
            pipeline,
            test_data,
            n_candidates=n_candidates,
            temperature=temperature,
            verbose=verbose,
        )

        if verbose:
            print(f"\n  Exact match rate: {pipeline_results['exact_match_rate']:.1%}")
            print(f"  Hit@1: {pipeline_results['hit_at_k']['hit@1']:.1%}")
            print(f"  Hit@10: {pipeline_results['hit_at_k']['hit@10']:.1%}")
            print(f"  Mean Tanimoto: {pipeline_results['mean_tanimoto']:.3f}")

        return {
            "part_a": part_a_results,
            "pipeline": pipeline_results,
        }

    def save_results(self, output_path: Path) -> None:
        """Save evaluation results to JSON.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        if "part_a" in self.results:
            print("\nPart A (Spectrum -> Descriptors):")
            summary = self.results["part_a"]["summary"]
            print(f"  Mean R2: {summary['mean_r2']:.3f}")
            print(f"  Median R2: {summary['median_r2']:.3f}")
            print(f"  Best: {summary['best_descriptor']} (R2={summary['max_r2']:.3f})")
            print(f"  Worst: {summary['worst_descriptor']} (R2={summary['min_r2']:.3f})")

        if "pipeline" in self.results:
            print("\nIntegrated Pipeline (Spectrum -> SMILES):")
            pipeline = self.results["pipeline"]
            print(f"  Exact match rate: {pipeline['exact_match_rate']:.1%}")
            print(f"  Formula match rate: {pipeline['formula_match_rate']:.1%}")
            for k, v in pipeline["hit_at_k"].items():
                print(f"  {k.upper()}: {v:.1%}")
            print(f"  Mean Tanimoto: {pipeline['mean_tanimoto']:.3f}")
            print(f"  Validity rate: {pipeline['validity_rate']:.1%}")

        print("=" * 60)
