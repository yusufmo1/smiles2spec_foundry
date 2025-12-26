#!/usr/bin/env python
"""Create physical train/val/test splits from raw spectral data.

This ensures no data leakage between training and evaluation.
Creates JSONL files with processed spectra and descriptors.

Usage:
    python scripts/preprocess_splits.py --config config_gnps_unique28.yml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors_batch
from src.utils.paths import validate_input_dir


def main():
    parser = argparse.ArgumentParser(description="Create physical train/val/test splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Validate input directory exists
    validate_input_dir(settings.input_path, "Dataset")

    output_dir = Path(settings.data_input_dir) / settings.dataset

    # Check if splits already exist
    train_path = output_dir / "train_data.jsonl"
    if train_path.exists() and not args.force:
        print(f"Split files already exist in {output_dir}")
        print("Use --force to overwrite")
        return

    print("=" * 60)
    print("Creating Physical Data Splits")
    print("=" * 60)
    print(f"Dataset:     {settings.dataset}")
    print(f"Split ratio: {settings.train_ratio}/{settings.val_ratio}/{settings.test_ratio}")
    print(f"Seed:        {settings.random_seed}")
    print(f"Descriptors: {len(settings.descriptor_names)}")
    print()

    # Load raw data
    raw_path = output_dir / "spectral_data.jsonl"
    print(f"Loading raw data from {raw_path}...")

    raw_data = []
    with open(raw_path) as f:
        for line in f:
            sample = json.loads(line)
            # Basic validation
            if "smiles" in sample and "peaks" in sample and len(sample["peaks"]) >= 3:
                raw_data.append(sample)

    print(f"Loaded {len(raw_data)} valid samples")

    # Extract SMILES for descriptor calculation
    smiles_list = [sample["smiles"] for sample in raw_data]

    # Calculate descriptors in parallel
    print(f"\nCalculating {len(settings.descriptor_names)} descriptors...")
    all_descriptors, valid_mask = calculate_descriptors_batch(
        smiles_list,
        settings.descriptor_names,
        return_valid_mask=True
    )

    # Process spectra and filter to valid samples
    print("\nProcessing spectra...")
    valid_samples = []
    desc_idx = 0

    for i, sample in enumerate(tqdm(raw_data, desc="Processing")):
        if valid_mask[i]:
            spectrum = process_spectrum(
                sample["peaks"],
                n_bins=settings.n_bins,
                bin_width=settings.bin_width,
                max_mz=settings.max_mz,
                transform=settings.transform,
                normalize=settings.normalize,
            )
            valid_samples.append({
                "smiles": sample["smiles"],
                "spectrum": spectrum.tolist(),
                "descriptors": all_descriptors[desc_idx].tolist(),
            })
            desc_idx += 1

    print(f"\nValid samples after processing: {len(valid_samples)}")

    # Split data
    print("\nSplitting data...")
    train_data, temp_data = train_test_split(
        valid_samples,
        test_size=settings.val_ratio + settings.test_ratio,
        random_state=settings.random_seed
    )

    test_fraction = settings.test_ratio / (settings.val_ratio + settings.test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_fraction,
        random_state=settings.random_seed
    )

    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(valid_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/len(valid_samples)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/len(valid_samples)*100:.1f}%)")

    # Save split files
    print(f"\nSaving to {output_dir}...")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    save_jsonl(train_data, output_dir / "train_data.jsonl")
    save_jsonl(val_data, output_dir / "val_data.jsonl")
    save_jsonl(test_data, output_dir / "test_data.jsonl")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "dataset": settings.dataset,
        "split_seed": settings.random_seed,
        "split_ratio": {
            "train": settings.train_ratio,
            "val": settings.val_ratio,
            "test": settings.test_ratio,
        },
        "counts": {
            "total": len(valid_samples),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "n_descriptors": len(settings.descriptor_names),
        "descriptor_names": list(settings.descriptor_names),
        "spectrum_config": {
            "n_bins": settings.n_bins,
            "bin_width": settings.bin_width,
            "max_mz": settings.max_mz,
            "transform": settings.transform,
            "normalize": settings.normalize,
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFiles created:")
    print(f"  - train_data.jsonl ({len(train_data)} samples)")
    print(f"  - val_data.jsonl ({len(val_data)} samples)")
    print(f"  - test_data.jsonl ({len(test_data)} samples)")
    print(f"  - metadata.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
