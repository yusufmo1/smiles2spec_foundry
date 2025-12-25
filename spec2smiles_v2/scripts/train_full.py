#!/usr/bin/env python
"""Train complete SPEC2SMILES pipeline (Part A + Part B).

Usage:
    python scripts/train_full.py [--config config.yml] [--model lgbm|transformer]

Or via Makefile:
    make train-full-lgbm-vae
    make train-full-transformer-vae
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config


def main():
    parser = argparse.ArgumentParser(description="Train complete pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "transformer"],
        default=None,
        help="Part A model type (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show training progress"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    model_type = args.model or settings.part_a_model

    scripts_dir = Path(__file__).parent

    print("=" * 60)
    print("SPEC2SMILES Full Pipeline Training")
    print("=" * 60)
    print(f"Part A Model: {model_type}")
    print(f"Dataset:      {settings.dataset}")
    print()

    # Train Part A
    print("Stage 1: Training Part A (Spectrum -> Descriptors)")
    print("-" * 60)

    cmd = [
        sys.executable,
        str(scripts_dir / "train_part_a.py"),
        "--model", model_type,
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Part A training failed!")
        sys.exit(1)

    print()
    print("=" * 60)

    # Train Part B
    print("Stage 2: Training Part B (Descriptors -> SMILES)")
    print("-" * 60)

    cmd = [
        sys.executable,
        str(scripts_dir / "train_part_b.py"),
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Part B training failed!")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Full pipeline training complete!")
    print()
    print("Models saved to:")
    print(f"  Part A: {settings.models_path / 'part_a'}")
    print(f"  Part B: {settings.models_path / 'part_b'}")
    print()
    print("Next steps:")
    print("  make predict     - Run predictions")
    print("  make evaluate    - Evaluate model performance")
    print("  make visualize   - Generate figures")


if __name__ == "__main__":
    main()
