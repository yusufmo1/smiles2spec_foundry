#!/usr/bin/env python
"""Evaluate Mamba-Diffusion model on test set.

Usage:
    python scripts/eval_mamba_diffusion.py --config config_mamba_diffusion.yml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
import yaml

from src.models.mamba_diffusion import MambaDiffusion, MambaDiffusionPipeline
from src.models.selfies_encoder import SELFIESEncoder
from src.domain.spectrum import process_spectrum

RDLogger.DisableLog("rdApp.*")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_test_data(config: dict) -> list:
    """Load test data."""
    data_dir = Path(config["data_input_dir"]) / config["dataset"]
    test_path = data_dir / "test_data.jsonl"

    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    return test_data


def compute_exact_match(candidates: list, true_smiles: str) -> bool:
    """Check if any candidate exactly matches the true SMILES."""
    mol = Chem.MolFromSmiles(true_smiles)
    if mol is None:
        return False
    true_canonical = Chem.MolToSmiles(mol, canonical=True)

    for cand in candidates:
        cand_mol = Chem.MolFromSmiles(cand)
        if cand_mol is not None:
            cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
            if cand_canonical == true_canonical:
                return True
    return False


def compute_best_tanimoto(candidates: list, true_smiles: str) -> float:
    """Compute best Tanimoto similarity among candidates."""
    mol = Chem.MolFromSmiles(true_smiles)
    if mol is None:
        return 0.0
    true_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

    best_sim = 0.0
    for cand in candidates:
        cand_mol = Chem.MolFromSmiles(cand)
        if cand_mol is not None:
            cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
            sim = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
            best_sim = max(best_sim, sim)

    return best_sim


def compute_hit_at_k(candidates: list, true_smiles: str, k: int) -> bool:
    """Check if true SMILES is in top-k candidates."""
    mol = Chem.MolFromSmiles(true_smiles)
    if mol is None:
        return False
    true_canonical = Chem.MolToSmiles(mol, canonical=True)

    for cand in candidates[:k]:
        cand_mol = Chem.MolFromSmiles(cand)
        if cand_mol is not None:
            cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
            if cand_canonical == true_canonical:
                return True
    return False


def compute_validity(candidates: list) -> float:
    """Compute fraction of valid SMILES among candidates."""
    if not candidates:
        return 0.0

    valid = sum(1 for c in candidates if Chem.MolFromSmiles(c) is not None)
    return valid / len(candidates)


def compute_diversity(candidates: list) -> float:
    """Compute diversity (unique molecules) among valid candidates."""
    unique = set()
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            unique.add(canonical)

    return len(unique) / len(candidates) if candidates else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mamba-Diffusion model")
    parser.add_argument("--config", type=Path, default="config_mamba_diffusion.yml")
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=None, help="Limit test samples")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    if config["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["device"]
    print(f"Using device: {device}")

    # Load model
    model_dir = Path(config["models_path"])
    print(f"\nLoading model from {model_dir}")

    model = MambaDiffusion.load(model_dir, device)
    model.eval()

    # Load encoder
    with open(model_dir / "encoder.pkl", "rb") as f:
        encoder_state = pickle.load(f)
    encoder = SELFIESEncoder.from_state(encoder_state)

    # Print model info
    info = model.get_architecture_info()
    print(f"\nModel architecture:")
    print(f"  Encoder: {info['encoder_type']}")
    print(f"  Total params: {info['total_params']:,}")
    print(f"  Vocab size: {info['vocab_size']}")

    # Load test data
    test_data = load_test_data(config)
    print(f"\nTest samples: {len(test_data)}")

    if args.n_samples:
        test_data = test_data[:args.n_samples]
        print(f"Using first {args.n_samples} samples")

    # Process test data
    n_bins = config["spectrum"]["n_bins"]
    transform = config["spectrum"]["transform"]
    normalize = config["spectrum"].get("normalize", True)

    spectra = []
    true_smiles = []

    for sample in test_data:
        if "spectrum" in sample:
            spec = np.array(sample["spectrum"])
        else:
            spec = process_spectrum(
                sample["peaks"],
                n_bins=n_bins,
                transform=transform,
                normalize=normalize,
            )
        spectra.append(spec)
        true_smiles.append(sample["smiles"])

    spectra = torch.tensor(np.array(spectra), dtype=torch.float32)

    # Generate predictions
    print(f"\nGenerating {args.n_candidates} candidates per sample...")
    print(f"  Steps: {args.n_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Batch size: {args.batch_size}")

    all_candidates = []
    n_batches = (len(spectra) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(spectra))
        batch_spectra = spectra[start:end].to(device)

        # Generate tokens
        tokens = model.generate(
            batch_spectra,
            n_candidates=args.n_candidates,
            n_steps=args.n_steps,
            temperature=args.temperature,
        )

        # Reshape to [batch_size, n_candidates, seq_len]
        batch_size = end - start
        tokens = tokens.view(batch_size, args.n_candidates, -1)

        # Decode to SMILES
        for b in range(batch_size):
            candidates = []
            for c in range(args.n_candidates):
                token_list = tokens[b, c].cpu().tolist()
                smiles = encoder.decode(token_list)
                if smiles is not None:
                    candidates.append(smiles)
            all_candidates.append(candidates)

    # Compute metrics
    print("\nComputing metrics...")

    exact_matches = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    hit_at_50 = 0
    tanimoto_scores = []
    validity_scores = []
    diversity_scores = []

    detailed_results = []

    for i, (candidates, true_smi) in enumerate(zip(all_candidates, true_smiles)):
        # Metrics
        exact = compute_exact_match(candidates, true_smi)
        tanimoto = compute_best_tanimoto(candidates, true_smi)
        validity = compute_validity(candidates)
        diversity = compute_diversity(candidates)

        exact_matches += int(exact)
        hit_at_1 += int(compute_hit_at_k(candidates, true_smi, 1))
        hit_at_5 += int(compute_hit_at_k(candidates, true_smi, 5))
        hit_at_10 += int(compute_hit_at_k(candidates, true_smi, 10))
        hit_at_50 += int(compute_hit_at_k(candidates, true_smi, 50))
        tanimoto_scores.append(tanimoto)
        validity_scores.append(validity)
        diversity_scores.append(diversity)

        # Detailed result
        detailed_results.append({
            "index": i,
            "true_smiles": true_smi,
            "exact_match": exact,
            "best_tanimoto": tanimoto,
            "n_valid": int(validity * len(candidates)),
            "n_unique": int(diversity * len(candidates)),
            "top_candidates": candidates[:5],
        })

    n_samples = len(true_smiles)

    # Results
    results = {
        "metrics": {
            "exact_match": exact_matches / n_samples,
            "hit_at_1": hit_at_1 / n_samples,
            "hit_at_5": hit_at_5 / n_samples,
            "hit_at_10": hit_at_10 / n_samples,
            "hit_at_50": hit_at_50 / n_samples,
            "mean_best_tanimoto": np.mean(tanimoto_scores),
            "mean_validity": np.mean(validity_scores),
            "mean_diversity": np.mean(diversity_scores),
        },
        "config": {
            "n_candidates": args.n_candidates,
            "n_steps": args.n_steps,
            "temperature": args.temperature,
            "n_samples": n_samples,
        },
        "model": info,
    }

    # Print results
    print("\n" + "=" * 60)
    print("MAMBA-DIFFUSION RESULTS")
    print("=" * 60)
    print(f"  Hit@1:              {results['metrics']['hit_at_1']:.4f} ({results['metrics']['hit_at_1']*100:.1f}%)")
    print(f"  Hit@5:              {results['metrics']['hit_at_5']:.4f} ({results['metrics']['hit_at_5']*100:.1f}%)")
    print(f"  Hit@10:             {results['metrics']['hit_at_10']:.4f} ({results['metrics']['hit_at_10']*100:.1f}%)")
    print(f"  Hit@50:             {results['metrics']['hit_at_50']:.4f} ({results['metrics']['hit_at_50']*100:.1f}%)")
    print(f"  Exact Match:        {results['metrics']['exact_match']:.4f} ({results['metrics']['exact_match']*100:.1f}%)")
    print(f"  Mean Best Tanimoto: {results['metrics']['mean_best_tanimoto']:.4f}")
    print(f"  Mean Validity:      {results['metrics']['mean_validity']:.4f} ({results['metrics']['mean_validity']*100:.1f}%)")
    print(f"  Mean Diversity:     {results['metrics']['mean_diversity']:.4f} ({results['metrics']['mean_diversity']*100:.1f}%)")
    print()
    print(f"  Candidates per sample: {args.n_candidates}")
    print(f"  Denoising steps: {args.n_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Test samples: {n_samples}")

    # Save results
    metrics_dir = Path(config["metrics_path"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "mamba_diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {metrics_dir / 'mamba_diffusion_results.json'}")

    # Save detailed predictions
    with open(metrics_dir / "mamba_diffusion_predictions.jsonl", "w") as f:
        for item in detailed_results:
            f.write(json.dumps(item) + "\n")
    print(f"Predictions saved to {metrics_dir / 'mamba_diffusion_predictions.jsonl'}")

    # Print some examples
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Exact matches
    matches = [r for r in detailed_results if r["exact_match"]][:3]
    print(f"\nExact Matches ({exact_matches} total):")
    for r in matches:
        print(f"  [{r['index']}] {r['true_smiles'][:50]}...")
        print(f"       → {r['top_candidates'][0][:50] if r['top_candidates'] else 'N/A'}...")

    # Best non-exact
    non_exact = sorted(
        [r for r in detailed_results if not r["exact_match"]],
        key=lambda x: x["best_tanimoto"],
        reverse=True,
    )[:3]
    print(f"\nBest Non-Exact Matches:")
    for r in non_exact:
        print(f"  [{r['index']}] True: {r['true_smiles'][:40]}...")
        if r["top_candidates"]:
            print(f"       Pred: {r['top_candidates'][0][:40]}... (Tanimoto: {r['best_tanimoto']:.3f})")


if __name__ == "__main__":
    main()
