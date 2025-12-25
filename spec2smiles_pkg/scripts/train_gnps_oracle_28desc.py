#!/usr/bin/env python
"""
Train GNPS Oracle models using 28 high-R² descriptors from Part A analysis.
These are the descriptors that Part A can predict best (R² > 0.53).

Trains: Tiny, Small, Large models using TRUE descriptors (oracle baseline).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.models.part_b.encoder import SELFIESEncoder
from spec2smiles.models.part_b.direct_decoder import DirectDecoder

# 28 High-R² descriptors from Part A analysis
DESCRIPTOR_NAMES_28 = [
    'fr_phos_ester',           # R2=0.7230
    'Chi3n',                   # R2=0.7183
    'Chi2n',                   # R2=0.7141
    'NumAtomStereoCenters',    # R2=0.6964
    'SMR_VSA1',                # R2=0.6724
    'SMR_VSA4',                # R2=0.6676
    'NumSaturatedCarbocycles', # R2=0.6565
    'SMR_VSA5',                # R2=0.6564
    'EState_VSA10',            # R2=0.6558
    'PEOE_VSA7',               # R2=0.6299
    'NumSaturatedRings',       # R2=0.6237
    'fr_Al_OH',                # R2=0.6202
    'VSA_EState3',             # R2=0.6178
    'PEOE_VSA10',              # R2=0.6152
    'SPS',                     # R2=0.6148
    'fr_ether',                # R2=0.6066
    'SMR_VSA3',                # R2=0.5978
    'qed',                     # R2=0.5949
    'SlogP_VSA5',              # R2=0.5931
    'AvgIpc',                  # R2=0.5890
    'FractionCSP3',            # R2=0.5876
    'SlogP_VSA3',              # R2=0.5873
    'Kappa2',                  # R2=0.5599
    'RingCount',               # R2=0.5541
    'SlogP_VSA11',             # R2=0.5513
    'VSA_EState2',             # R2=0.5506
    'HallKierAlpha',           # R2=0.5499
    'NumAromaticRings',        # R2=0.5330
]

# Import additional descriptor functions
from rdkit.Chem import Fragments
from rdkit.Chem.QED import qed as calc_qed


def calculate_28_descriptors(smiles: str) -> list:
    """Calculate the 28 high-R² descriptors using Descriptors module."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = []
    for name in DESCRIPTOR_NAMES_28:
        try:
            # Most descriptors are in Descriptors module
            func = getattr(Descriptors, name, None)
            if func:
                val = float(func(mol))
            # Some fragment descriptors
            elif hasattr(Fragments, name):
                func = getattr(Fragments, name)
                val = float(func(mol))
            # QED special case
            elif name == 'qed':
                val = float(calc_qed(mol))
            else:
                # Fallback - try Descriptors with exact name
                val = 0.0

            # Handle NaN/Inf
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            descriptors.append(val)
        except Exception as e:
            descriptors.append(0.0)

    return descriptors


def load_gnps_data(path: Path) -> list:
    """Load GNPS data and calculate 28 descriptors."""
    data = []
    with open(path) as f:
        for line in tqdm(f, desc="Loading GNPS"):
            item = json.loads(line)
            smiles = item.get('smiles')
            if not smiles:
                continue
            desc = calculate_28_descriptors(smiles)
            if desc is None:
                continue
            data.append({'smiles': smiles, 'descriptors': desc})
    return data


def train_model(
    model_name: str,
    model_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: SELFIESEncoder,
    device: str,
    output_dir: Path,
    epochs: int = 50,
    lr: float = 1e-4,
):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} Oracle (28 descriptors)")
    print(f"Config: {model_config}")
    print(f"{'='*60}")

    model = DirectDecoder(
        vocab_size=encoder.vocab_size,
        descriptor_dim=28,  # 28 descriptors now!
        hidden_dim=model_config["hidden_dim"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        max_len=100,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=SELFIESEncoder.PAD_IDX)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_desc, batch_tokens in pbar:
            batch_desc = batch_desc.to(device)
            batch_tokens = batch_tokens.to(device)

            # Teacher forcing
            input_tokens = batch_tokens[:, :-1]
            target_tokens = batch_tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(batch_desc, input_tokens)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch_desc, batch_tokens in val_loader:
                batch_desc = batch_desc.to(device)
                batch_tokens = batch_tokens.to(device)

                input_tokens = batch_tokens[:, :-1]
                target_tokens = batch_tokens[:, 1:]

                logits = model(batch_desc, input_tokens)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save final config
    config = {
        'model_name': model_name,
        'model_config': model_config,
        'descriptor_dim': 28,
        'descriptor_names': DESCRIPTOR_NAMES_28,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return best_val_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model configs
    models = {
        'tiny': {"hidden_dim": 256, "n_layers": 3, "n_heads": 4},
        'small': {"hidden_dim": 512, "n_layers": 4, "n_heads": 8},
        'large': {"hidden_dim": 768, "n_layers": 6, "n_heads": 12},
    }

    # Paths
    script_dir = Path(__file__).parent.parent
    foundry_dir = script_dir.parent
    gnps_path = foundry_dir / "smiles2spec/data/input/GNPS/spectral_data.jsonl"

    # Load data
    print(f"\nLoading GNPS data with 28 descriptors...")
    data = load_gnps_data(gnps_path)
    print(f"Total molecules: {len(data)}")

    # Split
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Build encoder
    print("Building SELFIES encoder...")
    train_smiles = [item['smiles'] for item in train_data]
    encoder = SELFIESEncoder(max_len=100)
    encoder.build_vocab_from_smiles(train_smiles, verbose=False)
    print(f"Vocab size: {encoder.vocab_size}")

    # Fit scaler on training descriptors
    train_descriptors = np.array([item['descriptors'] for item in train_data])
    scaler = StandardScaler()
    scaler.fit(train_descriptors)

    # Prepare datasets
    def prepare_dataset(data_list):
        descriptors = []
        tokens = []
        for item in data_list:
            selfies = encoder.smiles_to_selfies(item['smiles'])
            if selfies is None:
                continue
            encoded = encoder.encode(selfies)
            if encoded is None:
                continue
            descriptors.append(item['descriptors'])
            tokens.append(encoded)

        desc_scaled = scaler.transform(descriptors)
        desc_tensor = torch.tensor(desc_scaled, dtype=torch.float32)

        # Pad tokens
        max_len = max(len(t) for t in tokens)
        padded = []
        for t in tokens:
            padded.append(t + [SELFIESEncoder.PAD_IDX] * (max_len - len(t)))
        tokens_tensor = torch.tensor(padded, dtype=torch.long)

        return TensorDataset(desc_tensor, tokens_tensor)

    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Train all models
    results = {}
    for model_name, model_config in models.items():
        output_dir = Path(f"/tmp/gnps_oracle_28desc_{model_name}")

        best_loss = train_model(
            model_name=model_name,
            model_config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            encoder=encoder,
            device=device,
            output_dir=output_dir,
            epochs=50,
            lr=1e-4,
        )

        results[model_name] = {
            'best_val_loss': best_loss,
            'output_dir': str(output_dir),
        }

        # Clear GPU memory between models
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - 28 Descriptor Oracle Models")
    print("="*60)
    for name, result in results.items():
        print(f"{name.upper()}: val_loss={result['best_val_loss']:.4f} -> {result['output_dir']}")

    # Save overall results
    with open("/tmp/gnps_oracle_28desc_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
