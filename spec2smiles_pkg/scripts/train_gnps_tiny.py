#!/usr/bin/env python
"""Train TINY DirectDecoder on GNPS - testing if smaller is better."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.data.datasets import MolecularDataset
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
            func = getattr(Descriptors, name)
            val = func(mol)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            descriptors.append(float(val))
        except:
            descriptors.append(0.0)
    return descriptors


def load_gnps_data(data_path: Path):
    print(f"Loading GNPS data...")
    data = []
    with open(data_path) as f:
        for line in tqdm(f, desc="Loading"):
            data.append(json.loads(line))

    print("Calculating descriptors...")
    valid_data = []
    for record in tqdm(data, desc="Descriptors"):
        mol = Chem.MolFromSmiles(record['smiles'])
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        descriptors = calculate_descriptors(canonical)
        if descriptors:
            valid_data.append({'smiles': canonical, 'descriptors': descriptors})
    print(f"Valid: {len(valid_data)}")
    return valid_data


def train_model(train_data, val_data, test_data, output_dir, n_epochs=100,
                batch_size=256, hidden_dim=256, n_layers=3, n_heads=4, device=None):
    """Train TINY model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("TINY MODEL CONFIG:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  batch_size: {batch_size}")
    print(f"{'='*60}")

    # Scale descriptors
    X_train = np.array([d['descriptors'] for d in train_data])
    X_val = np.array([d['descriptors'] for d in val_data])
    X_test = np.array([d['descriptors'] for d in test_data])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_smiles = [d['smiles'] for d in train_data]
    val_smiles = [d['smiles'] for d in val_data]
    test_smiles = [d['smiles'] for d in test_data]

    # Build vocab
    encoder = SELFIESEncoder(max_len=100)
    train_selfies, train_valid = encoder.build_vocab_from_smiles(train_smiles, verbose=True)
    train_smiles_valid = [train_smiles[i] for i in train_valid]
    X_train_valid = X_train_scaled[train_valid]

    val_selfies, val_valid = [], []
    for i, smi in enumerate(val_smiles):
        sf = encoder.smiles_to_selfies(smi)
        if sf:
            val_selfies.append(sf)
            val_valid.append(i)
    X_val_valid = X_val_scaled[val_valid]
    val_smiles_valid = [val_smiles[i] for i in val_valid]

    test_selfies, test_valid = [], []
    for i, smi in enumerate(test_smiles):
        sf = encoder.smiles_to_selfies(smi)
        if sf:
            test_selfies.append(sf)
            test_valid.append(i)
    X_test_valid = X_test_scaled[test_valid]
    test_smiles_valid = [test_smiles[i] for i in test_valid]

    # Datasets
    train_dataset = MolecularDataset(X_train_valid, train_smiles_valid, encoder=encoder, max_len=100)
    val_dataset = MolecularDataset(X_val_valid, val_smiles_valid, encoder=encoder, max_len=100)
    test_dataset = MolecularDataset(X_test_valid, test_smiles_valid, encoder=encoder, max_len=100)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # TINY model
    model = DirectDecoder(
        vocab_size=encoder.vocab_size,
        descriptor_dim=len(DESCRIPTOR_NAMES),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1,
        max_len=100,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=5e-4, epochs=n_epochs, steps_per_epoch=len(train_loader), pct_start=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=SELFIESEncoder.PAD_IDX, label_smoothing=0.1)

    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            descriptors = batch['descriptors'].to(device)
            optimizer.zero_grad()
            logits, targets = model(tokens, descriptors)
            loss = criterion(logits.view(-1, encoder.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                descriptors = batch['descriptors'].to(device)
                logits, targets = model(tokens, descriptors)
                loss = criterion(logits.view(-1, encoder.vocab_size), targets.reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    print(f"Best epoch: {best_epoch+1} (val_loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    # Evaluate
    print("\nEvaluating (beam=40)...")
    model.eval()
    exact, formula, tanimotos, total = 0, 0, [], 0
    selfies_idx = 0

    for batch in tqdm(test_loader, desc="Eval"):
        descriptors = batch['descriptors'].to(device)
        candidates = model.generate_beam(descriptors, beam_width=40, n_return=100, length_penalty=0.6)

        for i in range(descriptors.size(0)):
            if selfies_idx >= len(test_selfies):
                break
            true_sf = test_selfies[selfies_idx]
            true_smi = encoder.selfies_to_smiles(true_sf)
            if not true_smi:
                selfies_idx += 1
                continue
            true_mol = Chem.MolFromSmiles(true_smi)
            if not true_mol:
                selfies_idx += 1
                continue

            true_can = Chem.MolToSmiles(true_mol, canonical=True)
            true_form = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)

            best_tan, found_ex, found_form = 0, False, False
            for ct in candidates:
                csmi = encoder.decode(ct[i].cpu().numpy().tolist())
                if not csmi:
                    continue
                cmol = Chem.MolFromSmiles(csmi)
                if not cmol:
                    continue
                ccan = Chem.MolToSmiles(cmol, canonical=True)
                if ccan == true_can:
                    found_ex = True
                if Chem.rdMolDescriptors.CalcMolFormula(cmol) == true_form:
                    found_form = True
                cfp = AllChem.GetMorganFingerprintAsBitVect(cmol, 2)
                best_tan = max(best_tan, DataStructs.TanimotoSimilarity(true_fp, cfp))

            if found_ex:
                exact += 1
            if found_form:
                formula += 1
            tanimotos.append(best_tan)
            total += 1
            selfies_idx += 1

    results = {
        'model_size': 'tiny',
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'total_params': total_params,
        'exact_match_rate': exact / total if total > 0 else 0,
        'formula_match_rate': formula / total if total > 0 else 0,
        'mean_tanimoto': float(np.mean(tanimotos)),
        'median_tanimoto': float(np.median(tanimotos)),
        'total': total,
    }

    print(f"\n{'='*60}")
    print("GNPS TINY Model Results:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Exact match: {results['exact_match_rate']:.1%}")
    print(f"  Formula match: {results['formula_match_rate']:.1%}")
    print(f"  Mean Tanimoto: {results['mean_tanimoto']:.3f}")
    print(f"{'='*60}")

    model.save(output_dir / "model.pt")
    with open(output_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder.get_state(), f)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"
    output_dir = Path("/tmp/gnps_model_tiny")

    data = load_gnps_data(gnps_path)
    train_data, temp = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_model(
        train_data, val_data, test_data,
        output_dir=output_dir,
        n_epochs=100,
        hidden_dim=256,   # TINY
        n_layers=3,       # TINY
        n_heads=4,        # TINY
        batch_size=256,   # LARGE batch
    )


if __name__ == "__main__":
    main()
