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
