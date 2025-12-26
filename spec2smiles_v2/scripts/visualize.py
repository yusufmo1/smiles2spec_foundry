#!/usr/bin/env python
"""Generate all SPEC2SMILES visualizations.

Usage:
    python scripts/visualize.py              # All figures
    python scripts/visualize.py --part-a     # Part A only
    python scripts/visualize.py --part-b     # Part B only
    python scripts/visualize.py --pipeline   # Pipeline comparisons only
    python scripts/visualize.py --benchmark  # MassSpecGym comparison only
    python scripts/visualize.py --analysis   # Hit@K and Tanimoto analysis

Or via Makefile:
    make viz              # All figures
    make viz-part-a       # Part A only
    make viz-benchmark    # Benchmark comparison only
    make viz-analysis     # Hit@K and Tanimoto analysis
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, reload_config
from src.viz import (
    setup_style,
    load_all_metrics,
    load_e2e_predictions,
    generate_part_a,
    generate_part_b,
    generate_pipeline,
    generate_benchmark,
    generate_analysis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPEC2SMILES visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualize.py              # All figures
    python scripts/visualize.py --part-a     # Part A only
    python scripts/visualize.py --benchmark  # MassSpecGym comparison
        """,
    )
    parser.add_argument("--config", type=Path, help="Path to config.yml file")
    parser.add_argument("--part-a", action="store_true", help="Generate Part A figures only")
    parser.add_argument("--part-b", action="store_true", help="Generate Part B figures only")
    parser.add_argument("--pipeline", action="store_true", help="Generate pipeline comparisons only")
    parser.add_argument("--benchmark", action="store_true", help="Generate MassSpecGym comparison only")
    parser.add_argument("--analysis", action="store_true", help="Generate Hit@K and Tanimoto analysis")
    args = parser.parse_args()

    # Reload config if provided
    global settings
    if args.config:
        settings = reload_config(args.config)

    # Setup paths
    metrics_dir = settings.metrics_path
    output_dir = settings.figures_path
    log_dir = settings.logs_path

    print("=" * 60)
    print("SPEC2SMILES Visualization Generator")
    print("=" * 60)
    print(f"Metrics:  {metrics_dir}")
    print(f"Output:   {output_dir}")
    print()

    # Setup style and load metrics
    setup_style()
    metrics = load_all_metrics(metrics_dir)
    print(f"Loaded metrics: {list(metrics.keys())}")
    print()

    # Load predictions for analysis
    predictions = load_e2e_predictions(metrics_dir)
    if predictions:
        print(f"Loaded {len(predictions)} E2E predictions")
    print()

    # Generate requested visualizations
    if args.part_a:
        generate_part_a(metrics, output_dir, log_dir, metrics_dir)
    elif args.part_b:
        generate_part_b(metrics, output_dir, log_dir)
    elif args.pipeline:
        generate_pipeline(metrics, output_dir)
    elif args.benchmark:
        generate_benchmark(metrics, output_dir)
    elif args.analysis:
        generate_analysis(predictions, output_dir)
    else:
        # Generate all
        generate_part_a(metrics, output_dir, log_dir, metrics_dir)
        generate_part_b(metrics, output_dir, log_dir)
        generate_pipeline(metrics, output_dir)
        generate_benchmark(metrics, output_dir)
        generate_analysis(predictions, output_dir)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
