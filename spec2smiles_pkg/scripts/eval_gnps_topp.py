#!/usr/bin/env python
"""Evaluate GNPS models using top-p (nucleus) sampling - much faster than beam search."""

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
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


def generate_topp(
    model: DirectDecoder,
    descriptors: torch.Tensor,
    encoder: SELFIESEncoder,
    top_p: float = 0.9,
    temperature: float = 1.0,
    n_samples: int = 10,
    max_len: int = 100,
) -> List[List[int]]:
    """Generate molecules using nucleus (top-p) sampling.

    Args:
        model: DirectDecoder model
        descriptors: (batch, descriptor_dim) tensor
        encoder: SELFIESEncoder for vocab
        top_p: Cumulative probability threshold
        temperature: Sampling temperature
        n_samples: Number of samples per input
        max_len: Maximum sequence length

    Returns:
        List of n_samples token sequences per batch item
    """
    device = descriptors.device
    batch_size = descriptors.shape[0]

    # Project descriptors once
    desc_proj = model.descriptor_proj(descriptors)  # (batch, hidden)
    memory = desc_proj.unsqueeze(1)  # (batch, 1, hidden)

    all_samples = []

    for _ in range(n_samples):
        # Start with SOS token
        tokens = torch.full((batch_size, 1), SELFIESEncoder.SOS_IDX,
                           dtype=torch.long, device=device)

        for step in range(max_len - 1):
            seq_len = tokens.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            # Embeddings
            tok_emb = model.token_embedding(tokens)
            pos_emb = model.pos_embedding(positions)
            x = model.dropout(tok_emb + pos_emb)

            # Causal mask
            causal_mask = model._generate_causal_mask(seq_len, device)

            # Decode
            decoded = model.transformer_decoder(x, memory, tgt_mask=causal_mask)
            logits = model.output_proj(decoded[:, -1, :])  # (batch, vocab)

            # Apply temperature
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find cutoff
            cutoff_mask = cumsum_probs > top_p
            cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
            cutoff_mask[:, 0] = False

            # Zero out low prob tokens
            sorted_probs[cutoff_mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sample
            next_token_sorted = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(-1, next_token_sorted)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Check if all sequences have EOS
            if (tokens == SELFIESEncoder.EOS_IDX).any(dim=1).all():
                break

        all_samples.append(tokens)

    return all_samples


def evaluate_model(
    model_path: Path,
    model_config: dict,
    model_name: str,
    test_data: list,
    encoder: SELFIESEncoder,
    scaler: StandardScaler,
    device: str,
    top_p_values: List[float] = [0.7, 0.8, 0.9, 0.95],
    n_samples: int = 10,
):
    """Evaluate a model with multiple top-p values."""

    if not model_path.exists():
        print(f"[{model_name}] Model not found: {model_path}")
        return None

    model = DirectDecoder(
        vocab_size=encoder.vocab_size,
        descriptor_dim=30,
        hidden_dim=model_config["hidden_dim"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        max_len=100,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    # Prepare test data
    test_selfies = []
    test_descriptors = []
    for item in test_data:
        selfies = encoder.smiles_to_selfies(item['smiles'])
        if selfies:
            test_selfies.append(selfies)
            test_descriptors.append(item['descriptors'])

    test_desc_scaled = scaler.transform(test_descriptors)

    results = {}

    for top_p in top_p_values:
        print(f"[{model_name}] Evaluating top_p={top_p}...")

        exact_matches = 0
        formula_matches = 0
        tanimoto_scores = []
        valid_count = 0
        total = 0

        batch_size = 64
        test_tensor = torch.tensor(test_desc_scaled, dtype=torch.float32)

        with torch.no_grad():
            for start_idx in tqdm(range(0, len(test_tensor), batch_size),
                                  desc=f"[{model_name}] p={top_p}"):
                end_idx = min(start_idx + batch_size, len(test_tensor))
                batch_desc = test_tensor[start_idx:end_idx].to(device)
                batch_size_actual = batch_desc.shape[0]

                # Generate samples
                samples = generate_topp(
                    model, batch_desc, encoder,
                    top_p=top_p, temperature=1.0, n_samples=n_samples
                )

                for i in range(batch_size_actual):
                    idx = start_idx + i
                    if idx >= len(test_selfies):
                        break

                    true_selfies = test_selfies[idx]
                    true_smiles = encoder.selfies_to_smiles(true_selfies)

                    if true_smiles is None:
                        continue

                    true_mol = Chem.MolFromSmiles(true_smiles)
                    if true_mol is None:
                        continue

                    true_canonical = Chem.MolToSmiles(true_mol, canonical=True)
                    true_formula = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
                    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)

                    best_tanimoto = 0.0
                    found_exact = False
                    found_formula = False
                    sample_valid = False

                    for sample_tokens in samples:
                        tokens = sample_tokens[i].cpu().numpy().tolist()
                        cand_smiles = encoder.decode(tokens)
                        if cand_smiles is None:
                            continue

                        cand_mol = Chem.MolFromSmiles(cand_smiles)
                        if cand_mol is None:
                            continue

                        sample_valid = True

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
                    if sample_valid:
                        valid_count += 1
                    tanimoto_scores.append(best_tanimoto)
                    total += 1

        results[top_p] = {
            'exact_match_rate': exact_matches / total if total > 0 else 0,
            'formula_match_rate': formula_matches / total if total > 0 else 0,
            'mean_tanimoto': float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
            'median_tanimoto': float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
            'valid_rate': valid_count / total if total > 0 else 0,
            'total': total,
        }

        print(f"[{model_name}] p={top_p}: Exact={results[top_p]['exact_match_rate']*100:.1f}%, "
              f"Tani={results[top_p]['mean_tanimoto']:.3f}")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"

    # Model configs
    models = {
        'large': {
            'path': Path("/tmp/gnps_model/best_model.pt"),
            'config': {"hidden_dim": 768, "n_layers": 6, "n_heads": 12},
        },
        'small': {
            'path': Path("/tmp/gnps_model_small/best_model.pt"),
            'config': {"hidden_dim": 512, "n_layers": 4, "n_heads": 8},
        },
        'tiny': {
            'path': Path("/tmp/gnps_model_tiny/best_model.pt"),
            'config': {"hidden_dim": 256, "n_layers": 3, "n_heads": 4},
        },
    }

    # Load data
    print("Loading GNPS data...")
    data = load_gnps_data(gnps_path)
    print(f"Total molecules: {len(data)}")

    # Split
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Test set: {len(test_data)} samples")

    # Build encoder
    print("Building encoder...")
    train_smiles = [item['smiles'] for item in train_data]
    encoder = SELFIESEncoder(max_len=100)
    encoder.build_vocab_from_smiles(train_smiles, verbose=False)
    print(f"Vocab size: {encoder.vocab_size}")

    # Fit scaler
    train_descriptors = [item['descriptors'] for item in train_data]
    scaler = StandardScaler()
    scaler.fit(train_descriptors)

    # Evaluate each model
    all_results = {}

    for model_name, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} model")
        print(f"{'='*60}")

        results = evaluate_model(
            model_path=model_info['path'],
            model_config=model_info['config'],
            model_name=model_name,
            test_data=test_data,
            encoder=encoder,
            scaler=scaler,
            device=device,
            top_p_values=[0.7, 0.8, 0.9, 0.95],
            n_samples=10,
        )

        if results:
            all_results[model_name] = results

    # Save results
    output_path = Path("/tmp/gnps_topp_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        for top_p, metrics in results.items():
            print(f"  p={top_p}: Exact={metrics['exact_match_rate']*100:.1f}%, "
                  f"Formula={metrics['formula_match_rate']*100:.1f}%, "
                  f"Tani={metrics['mean_tanimoto']:.3f}")


if __name__ == "__main__":
    main()
