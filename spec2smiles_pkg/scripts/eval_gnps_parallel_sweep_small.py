#!/usr/bin/env python
"""
MASSIVE PARALLEL top-p/temperature sweep for GNPS models.

Generates ALL candidates in parallel - fully utilizes H200's 141GB VRAM.
Sweeps: top_p × temperature × n_samples × all_test_samples AT ONCE.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
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


@torch.inference_mode()
def generate_parallel_topp(
    model: DirectDecoder,
    descriptors: torch.Tensor,
    top_p: float,
    temperature: float,
    n_samples: int,
    max_len: int = 100,
) -> torch.Tensor:
    """Generate n_samples for each input IN PARALLEL.

    Args:
        descriptors: (batch, descriptor_dim)

    Returns:
        tokens: (batch, n_samples, max_len) - all samples generated in parallel
    """
    device = descriptors.device
    batch_size = descriptors.shape[0]

    # Expand descriptors for parallel sampling: (batch * n_samples, descriptor_dim)
    desc_expanded = descriptors.unsqueeze(1).expand(-1, n_samples, -1)
    desc_expanded = desc_expanded.reshape(batch_size * n_samples, -1)

    # Project descriptors
    desc_proj = model.descriptor_proj(desc_expanded)
    memory = desc_proj.unsqueeze(1)  # (batch*n_samples, 1, hidden)

    total_samples = batch_size * n_samples

    # Start with SOS token
    tokens = torch.full((total_samples, 1), SELFIESEncoder.START_IDX,
                       dtype=torch.long, device=device)

    finished = torch.zeros(total_samples, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        if finished.all():
            break

        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(total_samples, -1)

        # Embeddings
        tok_emb = model.token_embedding(tokens)
        pos_emb = model.pos_embedding(positions)
        x = tok_emb + pos_emb

        # Causal mask
        causal_mask = model._generate_causal_mask(seq_len, device)

        # Decode
        decoded = model.transformer_decoder(x, memory, tgt_mask=causal_mask)
        logits = model.output_proj(decoded[:, -1, :])  # (total_samples, vocab)

        # Apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p sampling (vectorized)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
        cutoff_mask[:, 0] = False

        # Zero out low prob tokens
        sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Sample
        next_token_sorted = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices.gather(-1, next_token_sorted)

        # For finished sequences, pad with PAD token
        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token, SELFIESEncoder.PAD_IDX),
            next_token
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        # Update finished
        finished = finished | (next_token.squeeze(-1) == SELFIESEncoder.END_IDX)

    # Reshape: (batch, n_samples, seq_len)
    tokens = tokens.reshape(batch_size, n_samples, -1)

    return tokens


def evaluate_samples(
    all_tokens: torch.Tensor,  # (n_test, n_samples, seq_len)
    test_selfies: List[str],
    encoder: SELFIESEncoder,
) -> Dict:
    """Evaluate generated samples against ground truth."""

    n_test = all_tokens.shape[0]
    n_samples = all_tokens.shape[1]

    exact_matches = 0
    formula_matches = 0
    tanimoto_scores = []
    valid_count = 0
    total = 0

    for i in tqdm(range(n_test), desc="Evaluating"):
        true_selfies = test_selfies[i]
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
        any_valid = False

        for j in range(n_samples):
            tokens = all_tokens[i, j].cpu().numpy().tolist()
            cand_smiles = encoder.decode(tokens)

            if cand_smiles is None:
                continue

            cand_mol = Chem.MolFromSmiles(cand_smiles)
            if cand_mol is None:
                continue

            any_valid = True

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
        if any_valid:
            valid_count += 1
        tanimoto_scores.append(best_tanimoto)
        total += 1

    return {
        'exact_match_rate': exact_matches / total if total > 0 else 0,
        'formula_match_rate': formula_matches / total if total > 0 else 0,
        'mean_tanimoto': float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        'median_tanimoto': float(np.median(tanimoto_scores)) if tanimoto_scores else 0,
        'valid_rate': valid_count / total if total > 0 else 0,
        'total': total,
        'exact_matches': exact_matches,
        'formula_matches': formula_matches,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Sweep parameters
    TOP_P_VALUES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    TEMPERATURE_VALUES = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    N_SAMPLES = 20  # Per config
    BATCH_SIZE = 256  # H200 + small model

    print(f"\nSweep configuration:")
    print(f"  top_p values: {TOP_P_VALUES}")
    print(f"  temperature values: {TEMPERATURE_VALUES}")
    print(f"  samples per config: {N_SAMPLES}")
    print(f"  total configs: {len(TOP_P_VALUES) * len(TEMPERATURE_VALUES)}")

    # Paths
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"

    # Model configs - SMALL (medium) model
    models = {
        'small': {
            'path': Path("/tmp/gnps_model_small/best_model.pt"),
            'config': {"hidden_dim": 512, "n_layers": 4, "n_heads": 8},
        },
    }

    # Load data
    print("\nLoading GNPS data...")
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

    # Prepare test data
    test_selfies = []
    test_descriptors = []
    for item in test_data:
        selfies = encoder.smiles_to_selfies(item['smiles'])
        if selfies:
            test_selfies.append(selfies)
            test_descriptors.append(item['descriptors'])

    test_desc_scaled = scaler.transform(test_descriptors)
    test_tensor = torch.tensor(test_desc_scaled, dtype=torch.float32, device=device)

    print(f"Valid test samples: {len(test_selfies)}")

    all_results = {}

    for model_name, model_info in models.items():
        if not model_info['path'].exists():
            print(f"\n[{model_name}] Model not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*60}")

        # Load model
        model = DirectDecoder(
            vocab_size=encoder.vocab_size,
            descriptor_dim=30,
            hidden_dim=model_info['config']["hidden_dim"],
            n_layers=model_info['config']["n_layers"],
            n_heads=model_info['config']["n_heads"],
            max_len=100,
            dropout=0.0,  # No dropout for inference
        ).to(device)

        model.load_state_dict(torch.load(model_info['path'], weights_only=True, map_location=device))
        model.eval()

        model_results = {}

        for top_p in TOP_P_VALUES:
            for temp in TEMPERATURE_VALUES:
                config_key = f"p{top_p}_t{temp}"
                print(f"\n[{model_name}] top_p={top_p}, temp={temp}")

                start_time = time.time()

                # Generate all samples in batches
                all_tokens = []

                for start_idx in tqdm(range(0, len(test_tensor), BATCH_SIZE),
                                     desc=f"Generating"):
                    end_idx = min(start_idx + BATCH_SIZE, len(test_tensor))
                    batch_desc = test_tensor[start_idx:end_idx]

                    tokens = generate_parallel_topp(
                        model, batch_desc,
                        top_p=top_p,
                        temperature=temp,
                        n_samples=N_SAMPLES,
                    )
                    all_tokens.append(tokens.cpu())

                all_tokens = torch.cat(all_tokens, dim=0)

                gen_time = time.time() - start_time
                print(f"  Generation time: {gen_time:.1f}s")

                # Evaluate
                results = evaluate_samples(all_tokens, test_selfies, encoder)
                results['generation_time'] = gen_time

                model_results[config_key] = results

                print(f"  Exact: {results['exact_match_rate']*100:.1f}% "
                      f"({results['exact_matches']}/{results['total']})")
                print(f"  Formula: {results['formula_match_rate']*100:.1f}%")
                print(f"  Tanimoto: {results['mean_tanimoto']:.3f}")

        all_results[model_name] = model_results

        # Find best config for this model
        best_config = max(model_results.items(),
                         key=lambda x: x[1]['exact_match_rate'])
        print(f"\n[{model_name}] BEST CONFIG: {best_config[0]}")
        print(f"  Exact: {best_config[1]['exact_match_rate']*100:.1f}%")
        print(f"  Tanimoto: {best_config[1]['mean_tanimoto']:.3f}")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = Path("/tmp/gnps_parallel_sweep_small_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY - BEST CONFIGS PER MODEL")
    print("="*80)

    for model_name, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]['exact_match_rate'])
        print(f"\n{model_name.upper()}:")
        print(f"  Best config: {best[0]}")
        print(f"  Exact Match: {best[1]['exact_match_rate']*100:.2f}%")
        print(f"  Formula Match: {best[1]['formula_match_rate']*100:.2f}%")
        print(f"  Mean Tanimoto: {best[1]['mean_tanimoto']:.4f}")


if __name__ == "__main__":
    main()
