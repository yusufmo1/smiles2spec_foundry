#!/usr/bin/env python3
"""Quick evaluation of trained Part B v2 model."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.models.part_b.encoder import SELFIESEncoder
from spec2smiles.models.part_b.transformer_decoder import TransformerSMILESDecoder
from spec2smiles.models.part_b.transformer_trainer import compute_morgan_fingerprint


def evaluate(model_dir: Path, data_dir: Path, n_candidates: int = 10, n_samples: int = 272):
    """Evaluate trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # Load data to rebuild encoder
    with open(data_dir / "train_data.jsonl") as f:
        train_data = [json.loads(line) for line in f]
    with open(data_dir / "val_data.jsonl") as f:
        val_data = [json.loads(line) for line in f]

    # Rebuild encoder
    encoder = SELFIESEncoder()
    all_smiles = [d["smiles"] for d in train_data + val_data]
    encoder.build_vocab_from_smiles(all_smiles)
    print(f"Built encoder with vocab size: {encoder.vocab_size}")

    # Load checkpoint and rebuild model
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)

    # Model config (from training command)
    model = TransformerSMILESDecoder(
        vocab_size=encoder.vocab_size,
        fp_dim=2048,
        d_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=6,
        max_len=150,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from checkpoint")

    # Select samples
    indices = np.random.choice(len(val_data), min(n_samples, len(val_data)), replace=False)
    samples = [val_data[i] for i in indices]

    print(f"Evaluating on {len(samples)} samples with {n_candidates} candidates each...")

    # Evaluate
    exact_matches = 0
    tanimoto_scores = []

    for sample in tqdm(samples, desc="Evaluating"):
        true_smiles = sample["smiles"]

        # Compute fingerprint
        fp = compute_morgan_fingerprint(true_smiles)
        fp_tensor = torch.FloatTensor(fp).unsqueeze(0).to(device)

        # Generate candidates
        with torch.no_grad():
            generated = model.generate(fp_tensor, temperature=0.8, n_samples=n_candidates)

        # Check candidates
        best_tanimoto = 0.0
        for tokens in generated:
            decoded = encoder.decode(tokens[0].cpu().numpy().tolist())
            if decoded is not None:
                try:
                    gen_mol = Chem.MolFromSmiles(decoded)
                    true_mol = Chem.MolFromSmiles(true_smiles)

                    if gen_mol and true_mol:
                        gen_can = Chem.MolToSmiles(gen_mol, canonical=True)
                        true_can = Chem.MolToSmiles(true_mol, canonical=True)

                        if gen_can == true_can:
                            exact_matches += 1
                            best_tanimoto = 1.0
                            break

                        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2)
                        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)
                        tanimoto = DataStructs.TanimotoSimilarity(gen_fp, true_fp)
                        best_tanimoto = max(best_tanimoto, tanimoto)
                except:
                    pass

        tanimoto_scores.append(best_tanimoto)

    # Results
    exact_match_rate = exact_matches / len(samples)
    mean_tanimoto = np.mean(tanimoto_scores)

    print("\n" + "="*60)
    print("PART B v2 ORACLE EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(samples)}")
    print(f"Candidates per sample: {n_candidates}")
    print(f"Exact Match Rate: {exact_match_rate:.2%} ({exact_matches}/{len(samples)})")
    print(f"Mean Tanimoto: {mean_tanimoto:.4f}")
    print("="*60)

    return {
        "exact_match": exact_match_rate,
        "mean_tanimoto": mean_tanimoto,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("./models_gpu/part_b_v2"))
    parser.add_argument("--data-dir", type=Path, default=Path("../spec2smiles/data/processed/hpj"))
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=272)
    args = parser.parse_args()

    evaluate(args.model_dir, args.data_dir, args.n_candidates, args.n_samples)
