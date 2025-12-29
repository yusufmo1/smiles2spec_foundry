#!/usr/bin/env python
"""Train Mamba-Diffusion model for end-to-end spectrum to SMILES.

This script trains a novel architecture combining:
- Mamba-2 encoder for efficient O(n) spectrum processing
- Masked diffusion decoder for parallel SELFIES generation

Usage:
    python scripts/train_mamba_diffusion.py --config config_mamba_diffusion.yml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import yaml

from src.models.mamba_diffusion import MambaDiffusion
from src.models.selfies_encoder import SELFIESEncoder
from src.domain.spectrum import process_spectrum


class SpectrumSmilesDataset(Dataset):
    """Dataset for spectrum-to-SMILES training."""

    def __init__(
        self,
        spectra: np.ndarray,
        tokens: np.ndarray,
        smiles: list,
    ):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.smiles = smiles

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return {
            "spectrum": self.spectra[idx],
            "tokens": self.tokens[idx],
            "smiles": self.smiles[idx],
        }


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> tuple:
    """Load and preprocess training data."""
    data_dir = Path(config["data_input_dir"]) / config["dataset"]

    # Load train and validation data
    train_path = data_dir / "train_data.jsonl"
    val_path = data_dir / "val_data.jsonl"

    train_data = []
    with open(train_path) as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    with open(val_path) as f:
        for line in f:
            val_data.append(json.loads(line))

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")

    return train_data, val_data


def prepare_datasets(
    train_data: list,
    val_data: list,
    config: dict,
) -> tuple:
    """Prepare datasets with SELFIES encoding."""
    n_bins = config["spectrum"]["n_bins"]
    transform = config["spectrum"]["transform"]
    normalize = config["spectrum"].get("normalize", True)
    max_len = config["tokenizer"]["max_length"]

    # Build SELFIES vocabulary from training data
    print("\nBuilding SELFIES vocabulary...")
    encoder = SELFIESEncoder(max_len=max_len)

    train_smiles = [d["smiles"] for d in train_data]
    train_selfies, valid_indices = encoder.build_vocab_from_smiles(train_smiles)

    print(f"Vocabulary size: {encoder.vocab_size}")
    print(f"Valid SMILES: {len(valid_indices)} / {len(train_smiles)}")

    # Filter training data to valid entries
    train_data = [train_data[i] for i in valid_indices]

    # Process spectra and encode SMILES
    def process_data(data_list, desc="Processing"):
        spectra = []
        tokens = []
        smiles_list = []

        for sample in tqdm(data_list, desc=desc):
            # Process spectrum
            if "spectrum" in sample:
                spec = np.array(sample["spectrum"])
            else:
                spec = process_spectrum(
                    sample["peaks"],
                    n_bins=n_bins,
                    transform=transform,
                    normalize=normalize,
                )

            # Encode SMILES to SELFIES
            selfies = encoder.smiles_to_selfies(sample["smiles"])
            if selfies is None:
                continue

            tok = encoder.encode(selfies)

            spectra.append(spec)
            tokens.append(tok)
            smiles_list.append(sample["smiles"])

        return np.array(spectra), np.array(tokens), smiles_list

    train_spectra, train_tokens, train_smiles = process_data(train_data, "Processing train")
    val_spectra, val_tokens, val_smiles = process_data(val_data, "Processing val")

    print(f"\nFinal training samples: {len(train_smiles)}")
    print(f"Final validation samples: {len(val_smiles)}")

    # Create datasets
    train_dataset = SpectrumSmilesDataset(train_spectra, train_tokens, train_smiles)
    val_dataset = SpectrumSmilesDataset(val_spectra, val_tokens, val_smiles)

    return train_dataset, val_dataset, encoder


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    config: dict,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_acc = 0
    n_batches = 0

    # Progressive masking schedule
    warmup_epochs = config["training"]["mask_schedule"]["warmup_epochs"]
    start_ratio = config["training"]["mask_schedule"]["start_ratio"]
    end_ratio = config["training"]["mask_schedule"]["end_ratio"]

    if epoch < warmup_epochs:
        mask_ratio = start_ratio + (end_ratio - start_ratio) * (epoch / warmup_epochs)
    else:
        mask_ratio = end_ratio

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]")
    for batch in pbar:
        spectrum = batch["spectrum"].to(device)
        tokens = batch["tokens"].to(device)

        optimizer.zero_grad()

        # Forward pass with loss
        loss, metrics = model.compute_loss(spectrum, tokens, mask_ratio=mask_ratio)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["training"]["gradient_clip"],
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{metrics['accuracy']:.3f}",
            "mask": f"{mask_ratio:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "mask_ratio": mask_ratio,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0
    total_acc = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        spectrum = batch["spectrum"].to(device)
        tokens = batch["tokens"].to(device)

        loss, metrics = model.compute_loss(spectrum, tokens, mask_ratio=0.5)

        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-Diffusion model")
    parser.add_argument("--config", type=Path, default="config_mamba_diffusion.yml")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    if config["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["device"]
    print(f"Using device: {device}")

    # Load data
    train_data, val_data = load_data(config)

    # Prepare datasets
    train_dataset, val_dataset, encoder = prepare_datasets(train_data, val_data, config)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    enc_config = config["encoder"]
    dec_config = config["decoder"]

    model = MambaDiffusion(
        vocab_size=encoder.vocab_size,
        max_seq_len=config["tokenizer"]["max_length"],
        spectrum_dim=config["spectrum"]["n_bins"],
        d_model=enc_config["d_model"],
        encoder_type=enc_config["type"],
        encoder_n_layers=enc_config["n_layers"],
        encoder_d_state=enc_config.get("d_state", 128),
        decoder_n_layers=dec_config["n_layers"],
        decoder_n_heads=dec_config["n_heads"],
        decoder_d_ff=dec_config["d_ff"],
        dropout=dec_config["dropout"],
        mask_token_id=SELFIESEncoder.MASK_IDX,
        pad_token_id=SELFIESEncoder.PAD_IDX,
    )
    model = model.to(device)

    # Print model info
    info = model.get_architecture_info()
    print(f"\nModel architecture:")
    print(f"  Encoder: {info['encoder_type']}")
    print(f"  Encoder params: {info['encoder_params']:,}")
    print(f"  Decoder params: {info['decoder_params']:,}")
    print(f"  Total params: {info['total_params']:,}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),  # Restart every epoch
        T_mult=2,
        eta_min=1e-6,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    # Create output directory
    output_dir = Path(config["models_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save encoder
    encoder_state = encoder.get_state()
    with open(output_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder_state, f)

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}")

    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    training_log = []

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, config
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "mask_ratio": train_metrics["mask_ratio"],
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Mask Ratio: {train_metrics['mask_ratio']:.2f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            model.save(output_dir)
            print(f"  [NEW BEST] Saved to {output_dir}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

        # Save training log
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
