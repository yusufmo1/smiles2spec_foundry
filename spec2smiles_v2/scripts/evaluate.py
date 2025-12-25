#!/usr/bin/env python
"""Evaluate trained SPEC2SMILES pipeline.

Usage:
    python scripts/evaluate.py [--config config.yml]

Or via Makefile:
    make evaluate
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.config import settings, reload_config
from src.services.data_loader import DataLoaderService
from src.services.pipeline import PipelineService
from src.utils.metrics import (
    compute_batch_tanimoto,
    compute_hit_at_k,
    compute_validity_rate,
    compute_uniqueness,
)
from src.utils.paths import validate_input_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPEC2SMILES pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of candidates (default from config)"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Validate input directory exists
    validate_input_dir(settings.input_path, "Dataset")

    n_candidates = args.n_candidates or settings.n_candidates

    print("=" * 60)
    print("SPEC2SMILES Pipeline Evaluation")
    print("=" * 60)
    print(f"Dataset: {settings.dataset}")
    print()

    # Load pipeline
    print("Loading pipeline...")
    pipeline = PipelineService.from_directories(
        part_a_dir=settings.models_path / "part_a",
        part_b_dir=settings.models_path / "part_b",
        verbose=True,
    )
    print()

    # Load test data
    print("Loading test data...")
    data_loader = DataLoaderService(data_dir=Path(settings.data_input_dir) / settings.dataset)

    test_path = Path(settings.data_input_dir) / settings.dataset / "test_data.jsonl"
    if test_path.exists():
        test_data = data_loader.load_jsonl(test_path)
        X_test, y_test, smiles_test = data_loader.extract_features_and_targets(test_data)
    else:
        # Use raw data with split
        raw_data, _ = data_loader.load_raw_data()
        from sklearn.model_selection import train_test_split

        # Get test split
        _, test_data = train_test_split(
            raw_data,
            test_size=settings.test_ratio,
            random_state=settings.random_seed
        )

        from src.domain.spectrum import process_spectrum
        from src.domain.descriptors import calculate_descriptors

        X_test = []
        y_test = []
        smiles_test = []

        for sample in test_data:
            spectrum = process_spectrum(
                sample["peaks"],
                n_bins=settings.n_bins,
                transform=settings.transform,
                normalize=settings.normalize,
            )
            desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)
            if desc is not None:
                X_test.append(spectrum)
                y_test.append(desc)
                smiles_test.append(sample["smiles"])

        X_test = np.array(X_test)
        y_test = np.array(y_test)

    print(f"Test samples: {len(X_test)}")
    print()

    # Evaluate Part A
    print("Evaluating Part A (Spectrum -> Descriptors)...")
    print("-" * 40)

    summary = pipeline.part_a.get_summary_metrics(X_test, y_test)
    print(f"Mean R²:   {summary['mean_r2']:.4f}")
    print(f"Median R²: {summary['median_r2']:.4f}")
    print(f"Best:      {summary['best_descriptor']} (R² = {summary['best_r2']:.4f})")
    print(f"Worst:     {summary['worst_descriptor']} (R² = {summary['worst_r2']:.4f})")
    print()

    # Evaluate full pipeline
    print("Evaluating Full Pipeline (Spectrum -> SMILES)...")
    print("-" * 40)

    all_candidates = []
    for i in tqdm(range(len(X_test)), desc="Generating candidates"):
        result = pipeline.predict(
            X_test[i],
            n_candidates=n_candidates,
            temperature=settings.temperature,
        )
        all_candidates.append(result["candidates"])

    # Compute metrics
    best_sims, mean_sims = compute_batch_tanimoto(all_candidates, smiles_test)

    hit_1 = compute_hit_at_k(all_candidates, smiles_test, k=1)
    hit_5 = compute_hit_at_k(all_candidates, smiles_test, k=5)
    hit_10 = compute_hit_at_k(all_candidates, smiles_test, k=10)

    validity = compute_validity_rate(all_candidates)
    uniqueness = compute_uniqueness(all_candidates)

    print(f"\nResults:")
    print(f"  Hit@1:              {hit_1:.4f} ({hit_1 * 100:.1f}%)")
    print(f"  Hit@5:              {hit_5:.4f} ({hit_5 * 100:.1f}%)")
    print(f"  Hit@10:             {hit_10:.4f} ({hit_10 * 100:.1f}%)")
    print(f"  Mean Best Tanimoto: {np.mean(best_sims):.4f}")
    print(f"  Validity Rate:      {validity:.4f} ({validity * 100:.1f}%)")
    print(f"  Uniqueness:         {uniqueness:.4f} ({uniqueness * 100:.1f}%)")

    # Save results
    results = {
        "part_a": {
            "mean_r2": summary["mean_r2"],
            "median_r2": summary["median_r2"],
            "best_descriptor": summary["best_descriptor"],
            "best_r2": summary["best_r2"],
            "worst_descriptor": summary["worst_descriptor"],
            "worst_r2": summary["worst_r2"],
        },
        "pipeline": {
            "hit_at_1": hit_1,
            "hit_at_5": hit_5,
            "hit_at_10": hit_10,
            "mean_best_tanimoto": float(np.mean(best_sims)),
            "mean_mean_tanimoto": float(np.mean(mean_sims)),
            "validity_rate": validity,
            "uniqueness": uniqueness,
        },
        "config": {
            "n_candidates": n_candidates,
            "temperature": settings.temperature,
            "n_test_samples": len(X_test),
        }
    }

    output_path = settings.metrics_path / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
