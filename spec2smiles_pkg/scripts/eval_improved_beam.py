#!/usr/bin/env python
"""Evaluate 10x augmented model with improved beam search."""

import json
import pickle
import sys
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


def evaluate_beam_config(
    model: DirectDecoder,
    encoder: SELFIESEncoder,
    test_loader: DataLoader,
    test_selfies: list,
    beam_width: int,
    n_return: int,
    length_penalty: float,
    diversity_penalty: float,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Evaluate model with specific beam search configuration."""
    model.eval()

    exact_matches = 0
    formula_matches = 0
    tanimoto_scores = []
    total = 0
    selfies_idx = 0

    desc = f"beam={beam_width}, lp={length_penalty}, div={diversity_penalty}"
    iterator = tqdm(test_loader, desc=desc) if verbose else test_loader

    for batch in iterator:
        descriptors = batch["descriptors"].to(device)
        batch_size = descriptors.size(0)

        candidates = model.generate_beam(
            descriptors,
            beam_width=beam_width,
            n_return=n_return,
            length_penalty=length_penalty,
            diversity_penalty=diversity_penalty,
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

            true_smiles_canonical = Chem.MolToSmiles(true_mol, canonical=True)
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
                if cand_canonical == true_smiles_canonical:
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

    return {
        "beam_width": beam_width,
        "n_return": n_return,
        "length_penalty": length_penalty,
        "diversity_penalty": diversity_penalty,
        "exact_match_rate": exact_matches / total if total > 0 else 0,
        "formula_match_rate": formula_matches / total if total > 0 else 0,
        "mean_tanimoto": float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        "median_tanimoto": float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
        "total": total,
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

    # Paths
    data_dir = Path("/tmp/data_30desc")
    model_dir = Path("/tmp/augmented_10x_model")

    # Load test data
    print("Loading test data...")
    with open(data_dir / "test_data.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    # Load encoder
    with open(model_dir / "encoder.pkl", "rb") as f:
        encoder_state = pickle.load(f)
    encoder = SELFIESEncoder(max_len=100)
    encoder.set_state(encoder_state)

    # Get test SELFIES
    test_smiles = [d["smiles"] for d in test_data]
    test_selfies = []
    test_valid = []
    for i, smi in enumerate(test_smiles):
        sf = encoder.smiles_to_selfies(smi)
        if sf is not None:
            test_selfies.append(sf)
            test_valid.append(i)
    test_data = [test_data[i] for i in test_valid]

    # Extract descriptors
    X_test = np.array([d["descriptors_scaled"] for d in test_data])

    # Create dataset
    test_dataset = MolecularDataset(
        X_test, [test_smiles[i] for i in test_valid],
        encoder=encoder, max_len=100
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    print("Loading model...")
    model = DirectDecoder.load(model_dir / "model.pt", device=device)
    model.eval()

    print(f"\nTest set: {len(test_selfies)} molecules")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test configurations
    configs = [
        # (beam_width, n_return, length_penalty, diversity_penalty)
        (80, 100, 0.0, 0.0),   # Baseline - larger beam, no normalization
        (80, 100, 0.6, 0.0),   # Standard length normalization
        (80, 100, 0.7, 0.0),   # Stronger length normalization
        (80, 100, 0.6, 0.5),   # Length norm + diversity
        (100, 150, 0.6, 0.0),  # Even larger beam
        (100, 150, 0.6, 0.3),  # Larger beam + mild diversity
    ]

    print("\n" + "="*70)
    print("IMPROVED BEAM SEARCH EVALUATION")
    print("="*70)

    results = []
    best_exact = 0
    best_config = None

    for beam_width, n_return, lp, div in configs:
        result = evaluate_beam_config(
            model, encoder, test_loader, test_selfies,
            beam_width=beam_width,
            n_return=n_return,
            length_penalty=lp,
            diversity_penalty=div,
            device=device,
        )
        results.append(result)

        print(f"\nBeam={beam_width}, Return={n_return}, LP={lp}, Div={div}:")
        print(f"  Exact match: {result['exact_match_rate']:.1%}")
        print(f"  Formula match: {result['formula_match_rate']:.1%}")
        print(f"  Mean Tanimoto: {result['mean_tanimoto']:.3f}")

        if result['exact_match_rate'] > best_exact:
            best_exact = result['exact_match_rate']
            best_config = (beam_width, n_return, lp, div)

    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print(f"  Beam={best_config[0]}, Return={best_config[1]}, LP={best_config[2]}, Div={best_config[3]}")
    print(f"  Exact match: {best_exact:.1%}")
    print("="*70)

    # Save results
    with open(model_dir / "beam_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_exact, best_config


if __name__ == "__main__":
    main()
