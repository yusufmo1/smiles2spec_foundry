"""Evaluation commands for SPEC2SMILES."""

from pathlib import Path
from typing import Optional

import click


@click.group()
def evaluate():
    """Evaluate trained models."""
    pass


@evaluate.command("part-a")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing Part A model",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing test data",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file for results",
)
def evaluate_part_a(
    model_dir: Path,
    data_dir: Path,
    output: Optional[Path],
):
    """Evaluate Part A model on test set.

    Reports per-descriptor R2, RMSE, MAE and aggregate statistics.

    Example:
        spec2smiles evaluate part-a -m ./models/part_a -d ./data/processed/hpj
    """
    import numpy as np
    from spec2smiles.data.loaders import DataLoader, extract_features_and_targets
    from spec2smiles.services.part_a import PartAService
    from spec2smiles.evaluation.evaluator import PipelineEvaluator

    # Load data
    click.echo("Loading test data...")
    train_data, val_data, test_data, metadata = DataLoader.load_processed_splits(
        Path(data_dir)
    )
    X_test, y_test, _ = extract_features_and_targets(test_data)
    click.echo(f"Test samples: {len(test_data)}")

    # Load model (auto-detects model type)
    click.echo("Loading model...")
    service = PartAService()
    service.load(Path(model_dir))
    model = service.model

    # Evaluate
    evaluator = PipelineEvaluator()
    results = evaluator.evaluate_part_a(model, X_test, y_test)

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("PART A EVALUATION RESULTS")
    click.echo("=" * 50)

    summary = results["summary"]
    click.echo(f"\nAggregate Metrics:")
    click.echo(f"  Mean R2:   {summary['mean_r2']:.3f}")
    click.echo(f"  Median R2: {summary['median_r2']:.3f}")
    click.echo(f"  Std R2:    {summary['std_r2']:.3f}")
    click.echo(f"  Mean RMSE: {summary['mean_rmse']:.3f}")
    click.echo(f"  Mean MAE:  {summary['mean_mae']:.3f}")

    click.echo(f"\nBest:  {summary['best_descriptor']} (R2={summary['max_r2']:.3f})")
    click.echo(f"Worst: {summary['worst_descriptor']} (R2={summary['min_r2']:.3f})")

    click.echo("\nPer-Descriptor:")
    for name, metrics in results["per_descriptor"].items():
        click.echo(f"  {name:20s}: R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@evaluate.command("pipeline")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing models (with part_a/ and part_b/ subdirs)",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing test data",
)
@click.option(
    "--n-candidates",
    "-n",
    type=int,
    default=50,
    help="Number of candidates to generate",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file for results",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device for inference",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Show progress",
)
def evaluate_pipeline(
    model_dir: Path,
    data_dir: Path,
    n_candidates: int,
    temperature: float,
    output: Optional[Path],
    device: Optional[str],
    verbose: bool,
):
    """Evaluate integrated pipeline end-to-end.

    Reports Hit@K, Tanimoto similarity, exact match rate, and more.

    Example:
        spec2smiles evaluate pipeline -m ./models -d ./data/processed/hpj -n 50
    """
    from spec2smiles.evaluation.evaluator import PipelineEvaluator

    model_dir = Path(model_dir)
    part_a_dir = model_dir / "part_a"
    part_b_dir = model_dir / "part_b"

    if not part_a_dir.exists():
        part_a_dir = model_dir
        part_b_dir = model_dir

    evaluator = PipelineEvaluator()
    results = evaluator.evaluate_from_directories(
        data_dir=Path(data_dir),
        part_a_dir=part_a_dir,
        part_b_dir=part_b_dir,
        n_candidates=n_candidates,
        temperature=temperature,
        verbose=verbose,
    )

    # Print summary
    evaluator.print_summary()

    if output:
        evaluator.save_results(output)
        click.echo(f"\nResults saved to: {output}")
