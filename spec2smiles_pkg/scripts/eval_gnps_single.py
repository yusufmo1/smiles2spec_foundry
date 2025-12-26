#!/usr/bin/env python
"""Quick single-config evaluation for GNPS models."""

import json
import sys
import time
from pathlib import Path

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

def calculate_descriptors(smiles):
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

def load_gnps_data(path):
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
def generate_parallel_topp(model, descriptors, top_p, temperature, n_samples, max_len=100):
    device = descriptors.device
    batch_size = descriptors.shape[0]
    
    desc_expanded = descriptors.unsqueeze(1).expand(-1, n_samples, -1)
    desc_expanded = desc_expanded.reshape(batch_size * n_samples, -1)
    
    desc_proj = model.descriptor_proj(desc_expanded)
    memory = desc_proj.unsqueeze(1)
    
    total_samples = batch_size * n_samples
    tokens = torch.full((total_samples, 1), SELFIESEncoder.START_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(total_samples, dtype=torch.bool, device=device)
    
    for step in range(max_len - 1):
        if finished.all():
            break
        
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(total_samples, -1)
        
        tok_emb = model.token_embedding(tokens)
        pos_emb = model.pos_embedding(positions)
        x = tok_emb + pos_emb
        
        causal_mask = model._generate_causal_mask(seq_len, device)
        decoded = model.transformer_decoder(x, memory, tgt_mask=causal_mask)
        logits = model.output_proj(decoded[:, -1, :])
        
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
        cutoff_mask[:, 0] = False
        
        sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        
        next_token_sorted = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices.gather(-1, next_token_sorted)
        
        next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, SELFIESEncoder.PAD_IDX), next_token)
        tokens = torch.cat([tokens, next_token], dim=1)
        finished = finished | (next_token.squeeze(-1) == SELFIESEncoder.END_IDX)
    
    return tokens.reshape(batch_size, n_samples, -1)

def evaluate_samples(all_tokens, test_selfies, encoder):
    n_test = all_tokens.shape[0]
    n_samples = all_tokens.shape[1]
    
    exact_matches = 0
    tanimoto_scores = []
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
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)
        
        best_tanimoto = 0.0
        found_exact = False
        
        for j in range(n_samples):
            tokens = all_tokens[i, j].cpu().numpy().tolist()
            cand_smiles = encoder.decode(tokens)
            if cand_smiles is None:
                continue
            
            cand_mol = Chem.MolFromSmiles(cand_smiles)
            if cand_mol is None:
                continue
            
            cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
            if cand_canonical == true_canonical:
                found_exact = True
            
            cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
            tanimoto = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
            best_tanimoto = max(best_tanimoto, tanimoto)
        
        if found_exact:
            exact_matches += 1
        tanimoto_scores.append(best_tanimoto)
        total += 1
    
    return {
        'exact_match_rate': exact_matches / total if total > 0 else 0,
        'mean_tanimoto': float(np.mean(tanimoto_scores)) if tanimoto_scores else 0,
        'total': total,
        'exact_matches': exact_matches,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='small', choices=['tiny', 'small', 'large'])
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=1.2)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    configs = {
        'tiny': {"hidden_dim": 256, "n_layers": 3, "n_heads": 4},
        'small': {"hidden_dim": 512, "n_layers": 4, "n_heads": 8},
        'large': {"hidden_dim": 768, "n_layers": 6, "n_heads": 12},
    }
    
    model_path = Path(f"/tmp/gnps_model_{args.model}/best_model.pt")
    config = configs[args.model]
    
    print(f"\nModel: {args.model} | top_p={args.top_p} | temp={args.temp} | samples={args.samples}")
    
    # Load data
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"
    
    print("Loading data...")
    data = load_gnps_data(gnps_path)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    _, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Encoder
    train_smiles = [item['smiles'] for item in train_data]
    encoder = SELFIESEncoder(max_len=100)
    encoder.build_vocab_from_smiles(train_smiles, verbose=False)
    
    # Scaler
    train_descriptors = [item['descriptors'] for item in train_data]
    scaler = StandardScaler()
    scaler.fit(train_descriptors)
    
    # Test data
    test_selfies = []
    test_descriptors = []
    for item in test_data:
        selfies = encoder.smiles_to_selfies(item['smiles'])
        if selfies:
            test_selfies.append(selfies)
            test_descriptors.append(item['descriptors'])
    
    test_desc_scaled = scaler.transform(test_descriptors)
    test_tensor = torch.tensor(test_desc_scaled, dtype=torch.float32, device=device)
    
    print(f"Test samples: {len(test_selfies)}")
    
    # Load model
    model = DirectDecoder(
        vocab_size=encoder.vocab_size,
        descriptor_dim=30,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_len=100,
        dropout=0.0,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    
    # Generate
    print("Generating...")
    start = time.time()
    all_tokens = []
    
    for start_idx in tqdm(range(0, len(test_tensor), args.batch)):
        end_idx = min(start_idx + args.batch, len(test_tensor))
        batch_desc = test_tensor[start_idx:end_idx]
        tokens = generate_parallel_topp(model, batch_desc, args.top_p, args.temp, args.samples)
        all_tokens.append(tokens.cpu())
    
    all_tokens = torch.cat(all_tokens, dim=0)
    gen_time = time.time() - start
    print(f"Generation: {gen_time:.1f}s")
    
    # Evaluate
    results = evaluate_samples(all_tokens, test_selfies, encoder)
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.model.upper()} | p={args.top_p} | t={args.temp}")
    print(f"{'='*50}")
    print(f"Exact Match: {results['exact_match_rate']*100:.1f}% ({results['exact_matches']}/{results['total']})")
    print(f"Tanimoto: {results['mean_tanimoto']:.3f}")

if __name__ == "__main__":
    main()
