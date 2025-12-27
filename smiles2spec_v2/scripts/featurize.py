#!/usr/bin/env python
"""Extract molecular features from SMILES and cache them.

Usage:
    poetry run python scripts/featurize.py
    poetry run python scripts/featurize.py --config config.yml
    poetry run python scripts/featurize.py --type 3d
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smiles2spec.core.loader import load_settings
from smiles2spec.data.loader import DataLoader
from smiles2spec.domain.spectrum import SpectrumBinner
from smiles2spec.services.featurizer import FeaturizationService


def main():
    parser = argparse.ArgumentParser(description="Extract molecular features")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Config file path"
    )
    parser.add_argument(
        "--type", type=str, choices=["2d", "3d", "combined"], help="Feature type"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching"
    )
    args = parser.parse_args()

    # Load settings
    config_path = Path(__file__).parent.parent / args.config
    settings = load_settings(config_path)

    print("=" * 60)
    print("SMILES2SPEC: Feature Extraction")
    print("=" * 60)
    print(f"Dataset: {settings.dataset}")
    print(f"Feature type: {args.type or settings.features.type}")

    # Load data (input_path.parent.parent goes from data/input/hpj -> data/input -> data)
    loader = DataLoader(
        data_dir=settings.input_path.parent.parent,
        dataset=settings.dataset,
    )
    smiles_list, peaks_list = loader.extract_smiles_and_peaks()
    print(f"Loaded {len(smiles_list)} molecules")

    # Extract features
    featurizer = FeaturizationService(
        config=settings.features,
        cache_dir=settings.cache_path,
        use_cache=not args.no_cache,
    )

    feature_type = args.type or settings.features.type
    features, feature_names, failed = featurizer.extract(smiles_list, feature_type)

    print(f"\nFeature extraction complete:")
    print(f"  Successful: {len(features)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Feature dimension: {features.shape[1]}")

    # Bin spectra
    print("\nBinning spectra...")
    binner = SpectrumBinner(
        n_bins=settings.spectrum.n_bins,
        bin_width=settings.spectrum.bin_width,
        max_mz=settings.spectrum.max_mz,
        transform=settings.spectrum.transform,
    )

    valid_peaks = [p for i, p in enumerate(peaks_list) if i not in failed]
    spectra = binner.batch_bin(valid_peaks)
    print(f"  Spectrum shape: {spectra.shape}")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
