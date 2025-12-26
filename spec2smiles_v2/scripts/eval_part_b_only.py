#!/usr/bin/env python
"""Evaluate Part B only (Descriptors -> SMILES).

Usage:
    python scripts/eval_part_b_only.py --config config_gnps_unique28.yml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from rdkit import Chem

from src.config import settings, reload_config
from src.services.part_b import PartBService
from src.services.data_loader import DataLoaderService
from src.domain.descriptors import calculate_descriptors


def compute_exact_match(candidates_list, true_smiles_list):
    """Compute exact match rate."""
    matches = 0
    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        # Canonicalize true SMILES
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_canonical = Chem.MolToSmiles(mol, canonical=True)

        # Check if any candidate matches
        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
                if cand_canonical == true_canonical:
                    matches += 1
                    break

    return matches / len(true_smiles_list)


def compute_tanimoto(candidates_list, true_smiles_list):
    """Compute mean best Tanimoto similarity."""
    from rdkit.Chem import AllChem, DataStructs

    best_sims = []
    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

        sims = []
        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                sim = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                sims.append(sim)

        if sims:
            best_sims.append(max(sims))

    return np.mean(best_sims) if best_sims else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part B only")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=None, help="Limit test samples")
    args = parser.parse_args()

    # Load config
    global settings
    settings = reload_config(args.config)

    print("=" * 60)
    print("Part B Evaluation (Descriptors -> SMILES)")
    print("=" * 60)
    print(f"Dataset: {settings.dataset}")
    print(f"Model: {settings.part_b_model}")
    print()

    # Load Part B model
    print("Loading Part B model...")
    service = PartBService()
    service.load(settings.models_path / "part_b")
    print(f"  Vocab size: {service.encoder.vocab_size}")
    print(f"  Device: {service.device}")
    print()

    # Load test data
    print("Loading test data...")
    data_loader = DataLoaderService(
        data_dir=Path(settings.data_input_dir) / settings.dataset
    )
    raw_data, _ = data_loader.load_raw_data()

    # Split to get test set
    from sklearn.model_selection import train_test_split
    train_val, test_data = train_test_split(
        raw_data,
        test_size=settings.test_ratio,
        random_state=settings.random_seed
    )

    # Calculate descriptors for test data
    smiles_list = []
    descriptors_list = []

    for sample in test_data:
        desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)
        if desc is not None:
            smiles_list.append(sample["smiles"])
            descriptors_list.append(desc)

    print(f"Test samples: {len(smiles_list)}")

    # Limit samples if specified
    if args.n_samples:
        smiles_list = smiles_list[:args.n_samples]
        descriptors_list = descriptors_list[:args.n_samples]
        print(f"Using first {args.n_samples} samples")

    # Scale descriptors using Part B's scaler
    descriptors = np.array(descriptors_list)
    descriptors_scaled = service.scaler.transform(descriptors)

    print()
    print("Generating candidates...")

    # Generate candidates
    all_candidates = []
    for i in tqdm(range(len(descriptors_scaled))):
        desc = descriptors_scaled[i:i+1]
        candidates = service.generate(
            desc,
            n_candidates=args.n_candidates,
            temperature=settings.temperature,
        )
        all_candidates.append(candidates[0] if candidates else [])

    # Compute metrics
    print()
    print("Computing metrics...")

    exact_match = compute_exact_match(all_candidates, smiles_list)
    mean_tanimoto = compute_tanimoto(all_candidates, smiles_list)

    # Validity and uniqueness
    valid_count = sum(
        1 for cands in all_candidates
        for c in cands
        if Chem.MolFromSmiles(c) is not None
    )
    total_count = sum(len(cands) for cands in all_candidates)
    validity = valid_count / total_count if total_count > 0 else 0.0

    unique_smiles = set()
    for cands in all_candidates:
        for c in cands:
            mol = Chem.MolFromSmiles(c)
            if mol is not None:
                unique_smiles.add(Chem.MolToSmiles(mol, canonical=True))
    uniqueness = len(unique_smiles) / total_count if total_count > 0 else 0.0

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Exact Match:        {exact_match:.4f} ({exact_match * 100:.1f}%)")
    print(f"  Mean Best Tanimoto: {mean_tanimoto:.4f}")
    print(f"  Validity:           {validity:.4f} ({validity * 100:.1f}%)")
    print(f"  Uniqueness:         {uniqueness:.4f} ({uniqueness * 100:.1f}%)")
    print()
    print(f"  Candidates per sample: {args.n_candidates}")
    print(f"  Test samples: {len(smiles_list)}")

    # Save results
    results = {
        "exact_match": exact_match,
        "mean_best_tanimoto": mean_tanimoto,
        "validity": validity,
        "uniqueness": uniqueness,
        "n_candidates": args.n_candidates,
        "n_samples": len(smiles_list),
        "model": settings.part_b_model,
    }

    output_path = settings.metrics_path / "part_b_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
