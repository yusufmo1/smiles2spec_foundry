"""Visualization commands for SPEC2SMILES."""

from pathlib import Path
from typing import Optional
import json

import click


@click.group()
def visualize():
    """Generate visualizations from training results."""
    pass


@visualize.command("part-a")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing Part A model and visualization_data.json",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for figures (defaults to model-dir/figures)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format for figures",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
def visualize_part_a(
    model_dir: Path,
    output_dir: Optional[Path],
    format: str,
    dpi: int,
):
    """Generate Part A visualizations (regression analysis, performance summary).

    Creates:
    - Parity plots showing predicted vs actual for key descriptors
    - R² performance summary bar chart
    - Feature importance analysis plots

    Example:
        spec2smiles visualize part-a -m ./models/part_a
    """
    import numpy as np
    from spec2smiles.visualization import (
        set_style,
        plot_part_a_regression_analysis,
        plot_part_a_performance_summary,
        plot_part_a_feature_importance,
        save_figure,
    )

    model_dir = Path(model_dir)
    output_dir = output_dir or model_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load visualization data
    viz_path = model_dir / "visualization_data.json"
    if not viz_path.exists():
        raise click.ClickException(
            f"Visualization data not found: {viz_path}\n"
            "Re-run training to generate visualization data."
        )

    with open(viz_path) as f:
        viz_data = json.load(f)

    # Load metrics
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise click.ClickException(f"Metrics file not found: {metrics_path}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    set_style()

    y_true = np.array(viz_data["y_true"])
    y_pred = np.array(viz_data["y_pred"])
    descriptor_names = viz_data["descriptor_names"]
    feature_importances = {
        k: np.array(v) for k, v in viz_data["feature_importances"].items()
    }

    click.echo("Generating Part A visualizations...")

    # Regression analysis plots
    fig = plot_part_a_regression_analysis(
        y_true=y_true,
        y_pred=y_pred,
        descriptor_names=descriptor_names,
        metrics=metrics["test"],
    )
    save_figure(fig, output_dir / f"regression_analysis.{format}", dpi=dpi)
    click.echo(f"  - Saved regression_analysis.{format}")

    # Performance summary
    fig = plot_part_a_performance_summary(
        metrics=metrics["test"],
        descriptor_names=descriptor_names,
    )
    save_figure(fig, output_dir / f"performance_summary.{format}", dpi=dpi)
    click.echo(f"  - Saved performance_summary.{format}")

    # Feature importance
    fig = plot_part_a_feature_importance(
        feature_importances=feature_importances,
        descriptor_names=descriptor_names,
    )
    save_figure(fig, output_dir / f"feature_importance.{format}", dpi=dpi)
    click.echo(f"  - Saved feature_importance.{format}")

    click.echo(f"\nPart A figures saved to: {output_dir}")


@visualize.command("part-b")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing Part B model and visualization_data.json",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for figures (defaults to model-dir/figures)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format for figures",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
def visualize_part_b(
    model_dir: Path,
    output_dir: Optional[Path],
    format: str,
    dpi: int,
):
    """Generate Part B visualizations (training dynamics).

    Creates:
    - Training dynamics dashboard showing loss curves, KL divergence,
      beta schedule, learning rate, and Tanimoto distribution

    Example:
        spec2smiles visualize part-b -m ./models/part_b
    """
    from spec2smiles.visualization import (
        set_style,
        plot_part_b_training_dynamics,
        save_figure,
    )

    model_dir = Path(model_dir)
    output_dir = output_dir or model_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load visualization data
    viz_path = model_dir / "visualization_data.json"
    if not viz_path.exists():
        raise click.ClickException(
            f"Visualization data not found: {viz_path}\n"
            "Re-run training to generate visualization data."
        )

    with open(viz_path) as f:
        viz_data = json.load(f)

    set_style()

    click.echo("Generating Part B visualizations...")

    # Training dynamics
    fig = plot_part_b_training_dynamics(
        history=viz_data["history"],
        tanimoto_scores=viz_data.get("tanimoto_scores", []),
    )
    save_figure(fig, output_dir / f"training_dynamics.{format}", dpi=dpi)
    click.echo(f"  - Saved training_dynamics.{format}")

    click.echo(f"\nPart B figures saved to: {output_dir}")


@visualize.command("pipeline")
@click.option(
    "--results-file",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pipeline evaluation results JSON file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for figures",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format for figures",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
def visualize_pipeline(
    results_file: Path,
    output_dir: Optional[Path],
    format: str,
    dpi: int,
):
    """Generate integrated pipeline visualizations.

    Creates:
    - Pipeline performance dashboard with Hit@K curves, Tanimoto
      distribution, and component comparison

    Example:
        spec2smiles visualize pipeline -r ./results/pipeline_results.json
    """
    from spec2smiles.visualization import (
        set_style,
        plot_pipeline_performance,
        save_figure,
    )

    results_file = Path(results_file)
    output_dir = output_dir or results_file.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file) as f:
        results = json.load(f)

    set_style()

    click.echo("Generating pipeline visualizations...")

    fig = plot_pipeline_performance(results)
    save_figure(fig, output_dir / f"pipeline_performance.{format}", dpi=dpi)
    click.echo(f"  - Saved pipeline_performance.{format}")

    click.echo(f"\nPipeline figures saved to: {output_dir}")


@visualize.command("all")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing models (with part_a and part_b subdirectories)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for all figures",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format for figures",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
def visualize_all(
    model_dir: Path,
    output_dir: Optional[Path],
    format: str,
    dpi: int,
):
    """Generate all visualizations from a full training run.

    Expects model_dir to contain part_a/ and part_b/ subdirectories.

    Example:
        spec2smiles visualize all -m ./models
    """
    import numpy as np
    from spec2smiles.visualization import (
        set_style,
        plot_part_a_regression_analysis,
        plot_part_a_performance_summary,
        plot_part_a_feature_importance,
        plot_part_b_training_dynamics,
        save_figure,
    )

    model_dir = Path(model_dir)
    output_dir = output_dir or model_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()

    # Part A visualizations
    part_a_dir = model_dir / "part_a"
    if part_a_dir.exists():
        viz_path = part_a_dir / "visualization_data.json"
        metrics_path = part_a_dir / "metrics.json"

        if viz_path.exists() and metrics_path.exists():
            click.echo("Generating Part A visualizations...")

            with open(viz_path) as f:
                viz_data = json.load(f)
            with open(metrics_path) as f:
                metrics = json.load(f)

            y_true = np.array(viz_data["y_true"])
            y_pred = np.array(viz_data["y_pred"])
            descriptor_names = viz_data["descriptor_names"]
            feature_importances = {
                k: np.array(v) for k, v in viz_data["feature_importances"].items()
            }

            fig = plot_part_a_regression_analysis(
                y_true=y_true,
                y_pred=y_pred,
                descriptor_names=descriptor_names,
                metrics=metrics["test"],
            )
            save_figure(fig, output_dir / f"part_a_regression_analysis.{format}", dpi=dpi)
            click.echo(f"  - Saved part_a_regression_analysis.{format}")

            fig = plot_part_a_performance_summary(
                metrics=metrics["test"],
                descriptor_names=descriptor_names,
            )
            save_figure(fig, output_dir / f"part_a_performance_summary.{format}", dpi=dpi)
            click.echo(f"  - Saved part_a_performance_summary.{format}")

            fig = plot_part_a_feature_importance(
                feature_importances=feature_importances,
                descriptors_to_compare=descriptor_names,
            )
            save_figure(fig, output_dir / f"part_a_feature_importance.{format}", dpi=dpi)
            click.echo(f"  - Saved part_a_feature_importance.{format}")
        else:
            click.echo("Part A visualization data not found, skipping...")

    # Part B visualizations
    part_b_dir = model_dir / "part_b"
    if part_b_dir.exists():
        viz_path = part_b_dir / "visualization_data.json"

        if viz_path.exists():
            click.echo("\nGenerating Part B visualizations...")

            with open(viz_path) as f:
                viz_data = json.load(f)

            fig = plot_part_b_training_dynamics(
                history=viz_data["history"],
                tanimoto_scores=viz_data.get("tanimoto_scores", []),
            )
            save_figure(fig, output_dir / f"part_b_training_dynamics.{format}", dpi=dpi)
            click.echo(f"  - Saved part_b_training_dynamics.{format}")
        else:
            click.echo("Part B visualization data not found, skipping...")

    click.echo(f"\nAll figures saved to: {output_dir}")
