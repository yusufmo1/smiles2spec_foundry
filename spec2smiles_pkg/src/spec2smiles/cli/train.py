"""Training commands for SPEC2SMILES."""

from pathlib import Path
from typing import Optional

import click

from spec2smiles.core.config import PipelineConfig


@click.group()
def train():
    """Train SPEC2SMILES models."""
    pass


@train.command("part-a")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed train/val/test JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for model and results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML configuration file",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show training progress",
)
def train_part_a(
    data_dir: Path,
    output_dir: Path,
    config: Optional[Path],
    verbose: bool,
):
    """Train Part A model (Spectrum -> Descriptors).

    Trains a LightGBM ensemble with one model per molecular descriptor.
    Uses early stopping on validation set.

    Example:
        spec2smiles train part-a -d ./data/processed/hpj -o ./models/part_a
    """
    from spec2smiles.models.part_a.trainer import PartATrainer

    # Load config
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    # Train
    trainer = PartATrainer(config=pipeline_config)
    model = trainer.train(data_dir, output_dir, verbose=verbose)

    if verbose:
        click.echo(f"\nPart A training complete!")
        click.echo(f"Model saved to: {output_dir}")


@train.command("part-b")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed train/val/test JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for model and results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML configuration file",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device for training (cuda, cpu, mps)",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show training progress",
)
def train_part_b(
    data_dir: Path,
    output_dir: Path,
    config: Optional[Path],
    device: Optional[str],
    verbose: bool,
):
    """Train Part B model (Descriptors -> SMILES).

    Trains a Conditional VAE with SELFIES encoding for molecular generation.
    Uses cyclical KL annealing and early stopping.

    Example:
        spec2smiles train part-b -d ./data/processed/hpj -o ./models/part_b
    """
    import torch
    from spec2smiles.models.part_b.trainer import PartBTrainer

    # Load config
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    # Set device
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = None

    # Train
    trainer = PartBTrainer(config=pipeline_config, device=torch_device)
    model = trainer.train(data_dir, output_dir, verbose=verbose)

    if verbose:
        click.echo(f"\nPart B training complete!")
        click.echo(f"Model saved to: {output_dir}")


@train.command("full")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed train/val/test JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for models and results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML configuration file",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device for training (cuda, cpu, mps)",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show training progress",
)
def train_full(
    data_dir: Path,
    output_dir: Path,
    config: Optional[Path],
    device: Optional[str],
    verbose: bool,
):
    """Train full pipeline (Part A + Part B).

    Trains both components sequentially:
    1. Part A: LightGBM ensemble for descriptor prediction
    2. Part B: Conditional VAE for SMILES generation

    Example:
        spec2smiles train full -d ./data/processed/hpj -o ./models
    """
    import torch
    from spec2smiles.models.part_a.trainer import PartATrainer
    from spec2smiles.models.part_b.trainer import PartBTrainer

    # Load config
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    # Set device
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = None

    output_dir = Path(output_dir)
    part_a_dir = output_dir / "part_a"
    part_b_dir = output_dir / "part_b"

    # Train Part A
    if verbose:
        click.echo("=" * 60)
        click.echo("TRAINING PART A: Spectrum -> Descriptors")
        click.echo("=" * 60)

    trainer_a = PartATrainer(config=pipeline_config)
    trainer_a.train(data_dir, part_a_dir, verbose=verbose)

    # Train Part B
    if verbose:
        click.echo("\n" + "=" * 60)
        click.echo("TRAINING PART B: Descriptors -> SMILES")
        click.echo("=" * 60)

    trainer_b = PartBTrainer(config=pipeline_config, device=torch_device)
    trainer_b.train(data_dir, part_b_dir, verbose=verbose)

    # Save config
    pipeline_config.to_yaml(output_dir / "config.yaml")

    if verbose:
        click.echo("\n" + "=" * 60)
        click.echo("FULL PIPELINE TRAINING COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Part A model: {part_a_dir}")
        click.echo(f"Part B model: {part_b_dir}")
        click.echo(f"Config saved: {output_dir / 'config.yaml'}")


@train.command("transformer")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed train/val/test JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for model and results",
)
@click.option(
    "--d-model",
    type=int,
    default=256,
    help="Model dimension (default: 256)",
)
@click.option(
    "--n-heads",
    type=int,
    default=8,
    help="Number of attention heads (default: 8)",
)
@click.option(
    "--n-layers",
    type=int,
    default=6,
    help="Number of transformer layers (default: 6)",
)
@click.option(
    "--d-ff",
    type=int,
    default=1024,
    help="Feed-forward dimension (default: 1024)",
)
@click.option(
    "--patch-size",
    type=int,
    default=10,
    help="Spectrum bins per patch (default: 10)",
)
@click.option(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout rate (default: 0.1)",
)
@click.option(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate (default: 1e-4)",
)
@click.option(
    "--epochs",
    type=int,
    default=200,
    help="Maximum epochs (default: 200)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size (default: 32)",
)
@click.option(
    "--patience",
    type=int,
    default=30,
    help="Early stopping patience (default: 30)",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device for training (cuda, cpu, mps)",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show training progress",
)
def train_transformer_cmd(
    data_dir: Path,
    output_dir: Path,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    patch_size: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    verbose: bool,
):
    """Train Spectrum Transformer (Part A alternative).

    Trains a Transformer model that predicts all molecular descriptors
    jointly using multi-task learning and self-attention.

    This is an alternative to the LightGBM ensemble that may achieve
    higher accuracy by learning cross-descriptor correlations.

    Example:
        spec2smiles train transformer -d ./data/processed/hpj -o ./models/transformer
        spec2smiles train transformer -d ./data/processed/hpj -o ./models/transformer --d-model 512 --n-layers 8
    """
    import json
    import numpy as np
    from spec2smiles.models.part_a.transformer import SpectrumTransformerConfig
    from spec2smiles.models.part_a.transformer_trainer import train_transformer
    from spec2smiles.core.config import DescriptorConfig

    if verbose:
        click.echo("=" * 60)
        click.echo("TRAINING SPECTRUM TRANSFORMER: Spectrum -> Descriptors")
        click.echo("=" * 60)

    # Load data
    if verbose:
        click.echo(f"Loading data from {data_dir}...")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train/val data
    train_data = []
    with open(data_dir / "train_data.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    with open(data_dir / "val_data.jsonl") as f:
        for line in f:
            val_data.append(json.loads(line))

    # Get descriptor names from config
    desc_config = DescriptorConfig()
    descriptor_names = list(desc_config.names)

    # Extract spectra and descriptors (use pre-scaled descriptors)
    X_train = np.array([d["spectrum"] for d in train_data], dtype=np.float32)
    y_train = np.array([d["descriptors_scaled"] for d in train_data], dtype=np.float32)

    X_val = np.array([d["spectrum"] for d in val_data], dtype=np.float32)
    y_val = np.array([d["descriptors_scaled"] for d in val_data], dtype=np.float32)

    if verbose:
        click.echo(f"  Train: {len(X_train)} samples")
        click.echo(f"  Val: {len(X_val)} samples")
        click.echo(f"  Spectrum bins: {X_train.shape[1]}")
        click.echo(f"  Descriptors: {len(descriptor_names)}")

    # Create config
    config = SpectrumTransformerConfig(
        n_bins=X_train.shape[1],
        n_descriptors=len(descriptor_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        patch_size=patch_size,
        dropout=dropout,
        learning_rate=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    if verbose:
        click.echo(f"\nTransformer configuration:")
        click.echo(f"  d_model: {d_model}")
        click.echo(f"  n_heads: {n_heads}")
        click.echo(f"  n_layers: {n_layers}")
        click.echo(f"  d_ff: {d_ff}")
        click.echo(f"  patch_size: {patch_size}")
        click.echo(f"  dropout: {dropout}")
        click.echo(f"  learning_rate: {lr}")
        click.echo(f"  batch_size: {batch_size}")
        click.echo(f"  max_epochs: {epochs}")
        click.echo(f"  patience: {patience}")
        click.echo(f"  device: {device}")

    # Train
    results = train_transformer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        descriptor_names=descriptor_names,
        output_dir=output_dir,
        config=config,
        device=device,
    )

    if verbose:
        click.echo(f"\nTransformer training complete!")
        click.echo(f"Best validation R²: {results['val_r2']:.4f}")
        click.echo(f"Model saved to: {output_dir}")


@train.command("hybrid")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed train/val/test JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for model and results",
)
@click.option(
    "--cnn-hidden",
    type=int,
    default=256,
    help="CNN hidden dimension (default: 256)",
)
@click.option(
    "--transformer-dim",
    type=int,
    default=256,
    help="Transformer dimension (default: 256)",
)
@click.option(
    "--n-heads",
    type=int,
    default=8,
    help="Number of attention heads (default: 8)",
)
@click.option(
    "--n-layers",
    type=int,
    default=4,
    help="Number of transformer layers (default: 4)",
)
@click.option(
    "--d-ff",
    type=int,
    default=1024,
    help="Feed-forward dimension (default: 1024)",
)
@click.option(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout rate (default: 0.1)",
)
@click.option(
    "--lr",
    type=float,
    default=3e-4,
    help="Learning rate (default: 3e-4)",
)
@click.option(
    "--epochs",
    type=int,
    default=300,
    help="Maximum epochs (default: 300)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size (default: 32)",
)
@click.option(
    "--patience",
    type=int,
    default=40,
    help="Early stopping patience (default: 40)",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device for training (cuda, cpu, mps)",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show training progress",
)
def train_hybrid_cmd(
    data_dir: Path,
    output_dir: Path,
    cnn_hidden: int,
    transformer_dim: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    verbose: bool,
):
    """Train Hybrid CNN-Transformer (Part A alternative).

    Combines CNN for local pattern extraction with Transformer for global attention.
    The CNN captures isotope patterns and fragmentation signatures while the
    Transformer models long-range dependencies across the spectrum.

    Example:
        spec2smiles train hybrid -d ./data/processed/hpj -o ./models/hybrid
        spec2smiles train hybrid -d ./data/processed/hpj -o ./models/hybrid --cnn-hidden 512 --transformer-dim 512
    """
    import json
    import numpy as np
    from spec2smiles.models.part_a.hybrid_cnn_transformer import HybridConfig
    from spec2smiles.models.part_a.hybrid_trainer import train_hybrid
    from spec2smiles.core.config import DescriptorConfig

    if verbose:
        click.echo("=" * 60)
        click.echo("TRAINING HYBRID CNN-TRANSFORMER: Spectrum -> Descriptors")
        click.echo("=" * 60)

    # Load data
    if verbose:
        click.echo(f"Loading data from {data_dir}...")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = []
    with open(data_dir / "train_data.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    with open(data_dir / "val_data.jsonl") as f:
        for line in f:
            val_data.append(json.loads(line))

    desc_config = DescriptorConfig()
    descriptor_names = list(desc_config.names)

    # Extract spectra and descriptors (use pre-scaled descriptors)
    X_train = np.array([d["spectrum"] for d in train_data], dtype=np.float32)
    y_train = np.array([d["descriptors_scaled"] for d in train_data], dtype=np.float32)

    X_val = np.array([d["spectrum"] for d in val_data], dtype=np.float32)
    y_val = np.array([d["descriptors_scaled"] for d in val_data], dtype=np.float32)

    if verbose:
        click.echo(f"  Train: {len(X_train)} samples")
        click.echo(f"  Val: {len(X_val)} samples")
        click.echo(f"  Spectrum bins: {X_train.shape[1]}")
        click.echo(f"  Descriptors: {len(descriptor_names)}")

    # Create config
    config = HybridConfig(
        n_bins=X_train.shape[1],
        n_descriptors=len(descriptor_names),
        cnn_hidden=cnn_hidden,
        transformer_dim=transformer_dim,
        n_heads=n_heads,
        n_transformer_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        learning_rate=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    if verbose:
        click.echo(f"\nHybrid CNN-Transformer configuration:")
        click.echo(f"  cnn_hidden: {cnn_hidden}")
        click.echo(f"  transformer_dim: {transformer_dim}")
        click.echo(f"  n_heads: {n_heads}")
        click.echo(f"  n_layers: {n_layers}")
        click.echo(f"  d_ff: {d_ff}")
        click.echo(f"  dropout: {dropout}")
        click.echo(f"  learning_rate: {lr}")
        click.echo(f"  batch_size: {batch_size}")
        click.echo(f"  max_epochs: {epochs}")
        click.echo(f"  patience: {patience}")
        click.echo(f"  device: {device}")

    # Train
    results = train_hybrid(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        descriptor_names=descriptor_names,
        output_dir=output_dir,
        config=config,
        device=device,
    )

    if verbose:
        click.echo(f"\nHybrid CNN-Transformer training complete!")
        click.echo(f"Best validation R²: {results['val_r2']:.4f}")
        click.echo(f"Model saved to: {output_dir}")
