#!/usr/bin/env python
"""Evaluate model with nucleus sampling (faster than beam search)."""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.data.datasets import MolecularDataset
from spec2smiles.models.part_b.encoder import SELFIESEncoder
from spec2smiles.models.part_b.direct_decoder import DirectDecoder


def evaluate_nucleus(
    model: DirectDecoder,
    encoder: SELFIESEncoder,
    test_loader: DataLoader,
    test_selfies: list,
    device: torch.device,
    n_samples: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict:
    """Evaluate model with nucleus sampling."""
    model.eval()

    exact_matches = 0
    formula_matches = 0
    tanimoto_scores = []
    total = 0
    selfies_idx = 0

    start_time = time.time()

    for batch in tqdm(test_loader, desc=f"Nucleus (n={n_samples}, t={temperature})"):
        descriptors = batch["descriptors"].to(device)
        batch_size = descriptors.size(0)

        # Generate with nucleus sampling - this is parallelized!
        candidates = model.generate(
            descriptors,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
        )

        for i in range(batch_size):
            if selfies_idx >= len(test_selfies):
                break

            true_selfies = test_selfies[selfies_idx]
            true_smiles = encoder.selfies_to_smiles(true_selfies)

            if true_smiles is None:
                selfies_idx += 1
                continue

            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                selfies_idx += 1
                continue

            true_canonical = Chem.MolToSmiles(true_mol, canonical=True)
            true_formula = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)

            best_tanimoto = 0.0
            found_exact = False
            found_formula = False

            for cand_tokens in candidates:
                cand_smiles = encoder.decode(cand_tokens[i].cpu().numpy().tolist())
                if cand_smiles is None:
                    continue

                cand_mol = Chem.MolFromSmiles(cand_smiles)
                if cand_mol is None:
                    continue

                cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
                if cand_canonical == true_canonical:
                    found_exact = True

                cand_formula = Chem.rdMolDescriptors.CalcMolFormula(cand_mol)
                if cand_formula == true_formula:
                    found_formula = True

                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                tanimoto = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                best_tanimoto = max(best_tanimoto, tanimoto)

            if found_exact:
                exact_matches += 1
            if found_formula:
                formula_matches += 1
            tanimoto_scores.append(best_tanimoto)
            total += 1
            selfies_idx += 1

    elapsed = time.time() - start_time

    return {
        "method": "nucleus",
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "exact_match_rate": exact_matches / total if total > 0 else 0,
        "formula_match_rate": formula_matches / total if total > 0 else 0,
        "mean_tanimoto": float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        "median_tanimoto": float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
        "total": total,
        "time_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }


def main():
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Paths - use the 10x augmented model
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data_30desc"
    model_dir = script_dir / "models_30desc" / "direct_decoder_aug10x"

    if not model_dir.exists():
        model_dir = script_dir / "models_30desc" / "direct_decoder_aug"

    print(f"Model: {model_dir.name}")

    # Load test data
    print("Loading test data...")
    with open(data_dir / "test_data.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    # Load encoder from saved state (must match training vocab!)
    encoder_path = model_dir / "encoder.pkl"
    if encoder_path.exists():
        print(f"Loading encoder from {encoder_path}")
        with open(encoder_path, "rb") as f:
            encoder_state = pickle.load(f)
        encoder = SELFIESEncoder(max_len=100)
        encoder.set_state(encoder_state)
    else:
        print("WARNING: No saved encoder found, building from test data (may cause issues)")
        encoder = SELFIESEncoder(max_len=100)
        test_smiles = [d["smiles"] for d in test_data]
        encoder.build_vocab_from_smiles(test_smiles, verbose=True)

    test_smiles = [d["smiles"] for d in test_data]

    # Get test SELFIES
    test_selfies = []
    test_valid = []
    for i, smi in enumerate(test_smiles):
        sf = encoder.smiles_to_selfies(smi)
        if sf is not None:
            test_selfies.append(sf)
            test_valid.append(i)
    test_data_valid = [test_data[i] for i in test_valid]
    test_smiles_valid = [test_smiles[i] for i in test_valid]

    # Extract descriptors
    X_test = np.array([d["descriptors_scaled"] for d in test_data_valid])

    # Create dataset
    test_dataset = MolecularDataset(
        X_test, test_smiles_valid,
        encoder=encoder, max_len=100
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    print("Loading model...")
    model = DirectDecoder.load(model_dir / "model.pt", device=device)
    model.eval()

    print(f"\nTest set: {len(test_selfies)} molecules")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test different nucleus sampling configs
    configs = [
        # (n_samples, temperature, top_p)
        (100, 0.7, 0.9),   # Conservative
        (100, 0.8, 0.9),   # Balanced
        (100, 0.9, 0.9),   # More diverse
        (100, 1.0, 0.95),  # High diversity
        (200, 0.8, 0.9),   # More samples
    ]

    print("\n" + "=" * 70)
    print("NUCLEUS SAMPLING EVALUATION")
    print("=" * 70)

    results = []

    for n_samples, temp, top_p in configs:
        result = evaluate_nucleus(
            model, encoder, test_loader, test_selfies,
            device=device,
            n_samples=n_samples,
            temperature=temp,
            top_p=top_p,
        )
        results.append(result)

        print(f"\nn={n_samples}, temp={temp}, top_p={top_p}:")
        print(f"  Exact match: {result['exact_match_rate']:.1%}")
        print(f"  Formula match: {result['formula_match_rate']:.1%}")
        print(f"  Mean Tanimoto: {result['mean_tanimoto']:.3f}")
        print(f"  Time: {result['time_seconds']:.1f}s ({result['samples_per_second']:.1f} samples/s)")

    # Find best config
    best = max(results, key=lambda x: x['exact_match_rate'])

    print("\n" + "=" * 70)
    print("BEST NUCLEUS CONFIG:")
    print(f"  n={best['n_samples']}, temp={best['temperature']}, top_p={best['top_p']}")
    print(f"  Exact match: {best['exact_match_rate']:.1%}")
    print(f"  Time: {best['time_seconds']:.1f}s")
    print("=" * 70)

    # Save results
    output_path = model_dir / "nucleus_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
