#!/usr/bin/env python
"""Fast GNPS evaluation with beam=5 and massive batch sizes for H200."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.models.part_b.encoder import SELFIESEncoder
from spec2smiles.models.part_b.direct_decoder import DirectDecoder

DESCRIPTOR_NAMES = [
    'MolWt', 'HeavyAtomCount', 'NumHeteroatoms', 'NumAromaticRings',
    'RingCount', 'NOCount', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'MolLogP', 'NumRotatableBonds', 'FractionCSP3',
    'NumAliphaticRings', 'NumSaturatedRings', 'NumAromaticHeterocycles',
    'NumAliphaticHeterocycles', 'NumSaturatedHeterocycles',
    'NHOHCount', 'NumAliphaticCarbocycles', 'NumSaturatedCarbocycles',
    'NumAromaticCarbocycles', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'HallKierAlpha',
    'Kappa1', 'Kappa2', 'Kappa3'
]


def calculate_descriptors(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = []
    for name in DESCRIPTOR_NAMES:
        try:
            func = getattr(Descriptors, name, None)
            if func:
                descriptors.append(float(func(mol)))
            else:
                descriptors.append(0.0)
        except:
            descriptors.append(0.0)
    return descriptors


def load_gnps_data(path: Path) -> list:
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            smiles = item.get('smiles')
            if not smiles:
                continue
            desc = calculate_descriptors(smiles)
            if desc is None:
                continue
            data.append({'smiles': smiles, 'descriptors': desc})
    return data


def evaluate_model(
    model_path: Path,
    test_data: list,
    encoder: SELFIESEncoder,
    scaler: StandardScaler,
    device: str,
    model_size: str,
    batch_size: int = 2400,  # H200 141GB - entire test set in one batch
    beam_width: int = 5,    # Fast beam
    n_return: int = 10,     # 10 candidates enough
):
    """Evaluate a trained model."""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_size} model: {model_path}")
    print(f"Batch size: {batch_size}, Beam: {beam_width}, Candidates: {n_return}")
    print(f"{'='*70}")

    # Prepare test data
    test_selfies = []
    test_descriptors = []

    for item in test_data:
        selfies = encoder.smiles_to_selfies(item['smiles'])
        if selfies:
            test_selfies.append(selfies)
            test_descriptors.append(item['descriptors'])

    print(f"Test samples: {len(test_selfies)}")

    # Scale descriptors
    test_desc_scaled = scaler.transform(test_descriptors)
    test_tensor = torch.tensor(test_desc_scaled, dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model configs from training scripts
    if model_size == "large":
        config = {"hidden_dim": 768, "n_layers": 6, "n_heads": 12, "dropout": 0.1}
    elif model_size == "small":
        config = {"hidden_dim": 512, "n_layers": 4, "n_heads": 8, "dropout": 0.1}
    else:  # tiny
        config = {"hidden_dim": 256, "n_layers": 3, "n_heads": 4, "dropout": 0.1}

    # Load model
    model = DirectDecoder(
        vocab_size=encoder.vocab_size,
        descriptor_dim=30,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_len=100,
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    # Evaluate
    exact_matches = 0
    formula_matches = 0
    tanimoto_scores = []
    total = 0
    selfies_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {model_size}"):
            descriptors = batch[0].to(device)
            batch_size_actual = descriptors.size(0)

            # Generate with beam search
            candidates = model.generate_beam(
                descriptors,
                beam_width=beam_width,
                n_return=n_return,
                length_penalty=0.6,
                diversity_penalty=0.0
            )

            for i in range(batch_size_actual):
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

    results = {
        'model_size': model_size,
        'exact_match_rate': exact_matches / total if total > 0 else 0,
        'formula_match_rate': formula_matches / total if total > 0 else 0,
        'mean_tanimoto': float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        'median_tanimoto': float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
        'total': total,
        'beam_width': beam_width,
        'n_return': n_return,
    }

    print(f"\n{model_size.upper()} Results:")
    print(f"  Exact Match: {results['exact_match_rate']*100:.1f}%")
    print(f"  Formula Match: {results['formula_match_rate']*100:.1f}%")
    print(f"  Mean Tanimoto: {results['mean_tanimoto']:.3f}")
    print(f"  Median Tanimoto: {results['median_tanimoto']:.3f}")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Paths
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"

    # Load data
    print("Loading GNPS data...")
    data = load_gnps_data(gnps_path)
    print(f"Total molecules: {len(data)}")

    # Split (same as training)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Test set: {len(test_data)} samples")

    # Build encoder from training data
    print("Building encoder...")
    train_smiles = [item['smiles'] for item in train_data]

    encoder = SELFIESEncoder(max_len=128)
    train_selfies, _ = encoder.build_vocab_from_smiles(train_smiles, verbose=True)
    print(f"Vocab size: {encoder.vocab_size}")

    # Fit scaler on training data
    train_descriptors = [item['descriptors'] for item in train_data]
    scaler = StandardScaler()
    scaler.fit(train_descriptors)

    # Model paths
    models = {
        "large": Path("/tmp/gnps_model/best_model.pt"),
        "small": Path("/tmp/gnps_model_small/best_model.pt"),
        "tiny": Path("/tmp/gnps_model_tiny/best_model.pt"),
    }

    # Evaluate all models
    all_results = {}
    for size, path in models.items():
        if path.exists():
            results = evaluate_model(
                model_path=path,
                test_data=test_data,
                encoder=encoder,
                scaler=scaler,
                device=device,
                model_size=size,
                batch_size=2400,  # H200 141GB - entire test set
                beam_width=5,    # Fast
                n_return=10,     # 10 candidates
            )
            all_results[size] = results
        else:
            print(f"Model not found: {path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - GNPS Model Comparison (beam=5, 10 candidates)")
    print("="*70)
    print(f"{'Model':<10} {'Exact%':<10} {'Formula%':<10} {'Tanimoto':<10}")
    print("-"*40)
    for size in ["large", "small", "tiny"]:
        if size in all_results:
            r = all_results[size]
            print(f"{size:<10} {r['exact_match_rate']*100:<10.1f} {r['formula_match_rate']*100:<10.1f} {r['mean_tanimoto']:<10.3f}")

    # Save results
    with open("/tmp/gnps_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /tmp/gnps_eval_results.json")


if __name__ == "__main__":
    main()
