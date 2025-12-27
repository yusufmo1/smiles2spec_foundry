#!/usr/bin/env python
"""Evaluate DirectDecoder with nucleus sampling (faster than beam search).

Usage:
    python scripts/eval_nucleus.py [--config config.yml] [--n-samples 100] [--temperature 0.8]

Or via Makefile:
    make eval-nucleus
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.models.direct_decoder import DirectDecoder
from src.models.selfies_encoder import SELFIESEncoder
from src.services.part_b import PartBService


def evaluate_nucleus(
    model: DirectDecoder,
    encoder: SELFIESEncoder,
    test_loader: DataLoader,
    test_smiles: list,
    device: torch.device,
    n_samples: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> tuple:
    """Evaluate model with nucleus sampling.

    Returns:
        Tuple of (metrics_dict, predictions_list)
    """
    model.eval()

    exact_matches = 0
    formula_matches = 0
    tanimoto_scores = []
    total = 0
    smiles_idx = 0
    all_predictions = []

    start_time = time.time()

    for batch_desc, in tqdm(test_loader, desc=f"Nucleus (n={n_samples}, t={temperature})"):
        batch_desc = batch_desc.to(device)
        batch_size = batch_desc.size(0)

        # Generate with nucleus sampling
        candidates = model.generate(
            batch_desc,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
        )

        for i in range(batch_size):
            if smiles_idx >= len(test_smiles):
                break

            true_smiles = test_smiles[smiles_idx]
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                smiles_idx += 1
                continue

            true_canonical = Chem.MolToSmiles(true_mol, canonical=True)
            true_formula = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)

            best_tanimoto = 0.0
            found_exact = False
            found_formula = False
            best_candidate = None
            candidate_list = []
            candidate_tanimotos = []

            for cand_tokens in candidates:
                cand_smiles = encoder.decode(cand_tokens[i].cpu().numpy().tolist())
                if cand_smiles is None:
                    continue

                cand_mol = Chem.MolFromSmiles(cand_smiles)
                if cand_mol is None:
                    continue

                cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)

                # Skip duplicates
                if cand_canonical in candidate_list:
                    continue

                if cand_canonical == true_canonical:
                    found_exact = True

                cand_formula = Chem.rdMolDescriptors.CalcMolFormula(cand_mol)
                if cand_formula == true_formula:
                    found_formula = True

                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                tanimoto = DataStructs.TanimotoSimilarity(true_fp, cand_fp)

                candidate_list.append(cand_canonical)
                candidate_tanimotos.append(tanimoto)

                if tanimoto > best_tanimoto:
                    best_tanimoto = tanimoto
                    best_candidate = cand_canonical

            if found_exact:
                exact_matches += 1
            if found_formula:
                formula_matches += 1
            tanimoto_scores.append(best_tanimoto)
            total += 1

            # Store prediction
            all_predictions.append({
                "index": smiles_idx,
                "true_smiles": true_smiles,
                "true_canonical": true_canonical,
                "best_candidate": best_candidate,
                "exact_match": found_exact,
                "best_tanimoto": best_tanimoto,
                "n_candidates": len(candidate_list),
                "all_candidates": candidate_list,
                "all_tanimotos": candidate_tanimotos,
            })

            smiles_idx += 1

    elapsed = time.time() - start_time

    # Calculate Hit@K
    n_preds = len(all_predictions) if all_predictions else 1
    hit_at_1 = sum(1 for p in all_predictions if p["all_candidates"] and p["all_candidates"][0] == p["true_canonical"]) / n_preds
    hit_at_5 = sum(1 for p in all_predictions if p["true_canonical"] in p["all_candidates"][:5]) / n_preds
    hit_at_10 = sum(1 for p in all_predictions if p["true_canonical"] in p["all_candidates"][:10]) / n_preds
    hit_at_50 = sum(1 for p in all_predictions if p["true_canonical"] in p["all_candidates"][:50]) / n_preds

    metrics = {
        "method": "nucleus",
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "exact_match_rate": exact_matches / total if total > 0 else 0,
        "formula_match_rate": formula_matches / total if total > 0 else 0,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "hit_at_50": hit_at_50,
        "mean_tanimoto": float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        "median_tanimoto": float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
        "total": total,
        "time_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }

    return metrics, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate DirectDecoder with nucleus sampling")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yml")
    parser.add_argument("--n-samples", type=int, default=100, help="Candidates per molecule")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    # Reload config if provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    device = settings.torch_device
    print(f"Device: {device}")

    # Load test data
    data_dir = Path(settings.data_input_dir) / settings.dataset
    test_path = data_dir / "test_data.jsonl"

    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Run preprocessing first to create train/val/test splits.")
        sys.exit(1)

    print("Loading test data...")
    with open(test_path) as f:
        test_data = [json.loads(line) for line in f]

    test_smiles = [d["smiles"] for d in test_data]
    test_descriptors = np.array([d["descriptors"] for d in test_data], dtype=np.float32)

    # Scale descriptors
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    test_descriptors_scaled = scaler.fit_transform(test_descriptors)

    print(f"Test set: {len(test_smiles)} molecules")

    # Load model
    model_dir = settings.models_path / "part_b"
    integration_path = model_dir / "integration_package.pkl"

    if not integration_path.exists():
        print(f"Error: Model not found at {model_dir}")
        print("Train the DirectDecoder first with: make train-part-b-direct")
        sys.exit(1)

    print("Loading model...")
    service = PartBService()
    service.load(model_dir)

    if not isinstance(service.model, DirectDecoder):
        print("Error: Loaded model is not a DirectDecoder. This script is for DirectDecoder only.")
        sys.exit(1)

    model = service.model
    encoder = service.encoder
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create test loader
    test_dataset = TensorDataset(torch.FloatTensor(test_descriptors_scaled))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if args.sweep:
        # Parameter sweep
        configs = [
            (100, 0.7, 0.9),   # Conservative
            (100, 0.8, 0.9),   # Balanced
            (100, 0.9, 0.9),   # More diverse
            (100, 1.0, 0.95),  # High diversity
            (200, 0.8, 0.9),   # More samples
        ]

        print("\n" + "=" * 70)
        print("NUCLEUS SAMPLING PARAMETER SWEEP")
        print("=" * 70)

        results = []
        for n_samples, temp, top_p in configs:
            metrics, _ = evaluate_nucleus(
                model, encoder, test_loader, test_smiles,
                device=device, n_samples=n_samples, temperature=temp, top_p=top_p,
            )
            results.append(metrics)

            print(f"\nn={n_samples}, temp={temp}, top_p={top_p}:")
            print(f"  Hit@1: {metrics['hit_at_1']:.1%}")
            print(f"  Hit@10: {metrics['hit_at_10']:.1%}")
            print(f"  Exact match: {metrics['exact_match_rate']:.1%}")
            print(f"  Mean Tanimoto: {metrics['mean_tanimoto']:.3f}")
            print(f"  Time: {metrics['time_seconds']:.1f}s ({metrics['samples_per_second']:.1f} mol/s)")

        # Find best config
        best = max(results, key=lambda x: x['hit_at_10'])
        print("\n" + "=" * 70)
        print("BEST CONFIG (by Hit@10):")
        print(f"  n={best['n_samples']}, temp={best['temperature']}, top_p={best['top_p']}")
        print(f"  Hit@1: {best['hit_at_1']:.1%}")
        print(f"  Hit@10: {best['hit_at_10']:.1%}")
        print(f"  Exact match: {best['exact_match_rate']:.1%}")
        print("=" * 70)

    else:
        # Single evaluation
        print(f"\nEvaluating with n={args.n_samples}, temp={args.temperature}, top_p={args.top_p}")
        metrics, predictions = evaluate_nucleus(
            model, encoder, test_loader, test_smiles,
            device=device,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results = [metrics]

        print("\n" + "=" * 70)
        print("NUCLEUS SAMPLING RESULTS")
        print("=" * 70)
        print(f"  Candidates per molecule: {metrics['n_samples']}")
        print(f"  Temperature: {metrics['temperature']}")
        print(f"  Top-p: {metrics['top_p']}")
        print(f"  Hit@1:  {metrics['hit_at_1']:.1%}")
        print(f"  Hit@5:  {metrics['hit_at_5']:.1%}")
        print(f"  Hit@10: {metrics['hit_at_10']:.1%}")
        print(f"  Hit@50: {metrics['hit_at_50']:.1%}")
        print(f"  Exact match rate: {metrics['exact_match_rate']:.1%}")
        print(f"  Formula match rate: {metrics['formula_match_rate']:.1%}")
        print(f"  Mean Tanimoto: {metrics['mean_tanimoto']:.3f}")
        print(f"  Median Tanimoto: {metrics['median_tanimoto']:.3f}")
        print(f"  Total molecules: {metrics['total']}")
        print(f"  Time: {metrics['time_seconds']:.1f}s ({metrics['samples_per_second']:.1f} mol/s)")
        print("=" * 70)

        # Save predictions
        preds_path = settings.metrics_path / "nucleus_predictions.jsonl"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"\nPredictions saved to {preds_path}")

    # Save metrics
    output_path = settings.metrics_path / "nucleus_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
