#!/usr/bin/env python
"""Run predictions with trained SPEC2SMILES pipeline.

Usage:
    python scripts/predict.py --spectrum peaks.json [--config config.yml]
    python scripts/predict.py --input spectra.jsonl --output predictions.jsonl

Or via Makefile:
    make predict
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.services.pipeline import PipelineService


def main():
    parser = argparse.ArgumentParser(description="Run SPEC2SMILES predictions")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yml file"
    )
    parser.add_argument(
        "--spectrum",
        type=str,
        help="Path to JSON file with single spectrum (peaks list)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSONL file with multiple spectra"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSONL file (for batch mode)"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of candidates to generate (default from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default from config)"
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    n_candidates = args.n_candidates or settings.n_candidates
    temperature = args.temperature or settings.temperature

    # Load pipeline
    print("Loading SPEC2SMILES pipeline...")
    pipeline = PipelineService.from_directories(
        part_a_dir=settings.models_path / "part_a",
        part_b_dir=settings.models_path / "part_b",
        verbose=True,
    )
    print()

    if args.spectrum:
        # Single spectrum prediction
        with open(args.spectrum) as f:
            data = json.load(f)

        peaks = data.get("peaks", data)  # Handle both {"peaks": [...]} and [...]

        print(f"Predicting from {len(peaks)} peaks...")
        result = pipeline.predict_from_peaks(
            peaks,
            n_candidates=n_candidates,
            temperature=temperature,
            return_descriptors=True,
        )

        print(f"\nGenerated {result['n_unique']} unique candidates")
        print(f"\nTop 10 candidates:")
        for i, cand in enumerate(result["candidates"][:10], 1):
            print(f"  {i}. {cand}")

        print(f"\nPredicted descriptors:")
        for name, value in zip(result["descriptor_names"], result["descriptors"]):
            print(f"  {name}: {value:.2f}")

    elif args.input:
        # Batch prediction
        print(f"Loading spectra from {args.input}...")

        with open(args.input) as f:
            data = [json.loads(line) for line in f]

        print(f"Processing {len(data)} spectra...")

        results = []
        for i, sample in enumerate(data):
            peaks = sample.get("peaks", sample)
            result = pipeline.predict_from_peaks(
                peaks,
                n_candidates=n_candidates,
                temperature=temperature,
            )

            # Add original data if available
            if "smiles" in sample:
                result["true_smiles"] = sample["smiles"]

            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(data)}")

        # Save results
        output_path = args.output or "predictions.jsonl"
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"\nPredictions saved to {output_path}")

    else:
        # Interactive mode
        print("Interactive mode - enter peaks as: m/z1,intensity1 m/z2,intensity2 ...")
        print("Example: 50,100 77,50 105,80")
        print("Type 'quit' to exit\n")

        while True:
            try:
                line = input("Peaks> ").strip()
                if line.lower() in ["quit", "exit", "q"]:
                    break

                if not line:
                    continue

                # Parse peaks
                peaks = []
                for pair in line.split():
                    mz, intensity = pair.split(",")
                    peaks.append((float(mz), float(intensity)))

                result = pipeline.predict_from_peaks(
                    peaks,
                    n_candidates=n_candidates,
                    temperature=temperature,
                )

                print(f"\nTop 5 candidates:")
                for i, cand in enumerate(result["candidates"][:5], 1):
                    print(f"  {i}. {cand}")
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                print()

    print("\nDone!")


if __name__ == "__main__":
    main()
