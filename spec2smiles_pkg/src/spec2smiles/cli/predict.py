"""Prediction commands for SPEC2SMILES."""

from pathlib import Path
from typing import Optional
import json

import click


@click.group()
def predict():
    """Run predictions with trained models."""
    pass


@predict.command("single")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing trained models (with part_a/ and part_b/ subdirs)",
)
@click.option(
    "--spectrum",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to spectrum JSON file with 'peaks' field",
)
@click.option(
    "--n-candidates",
    "-n",
    type=int,
    default=50,
    help="Number of candidate structures to generate",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature (lower = more conservative)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file (default: print to stdout)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device for inference (cuda, cpu, mps)",
)
def predict_single(
    model_dir: Path,
    spectrum: Path,
    n_candidates: int,
    temperature: float,
    output: Optional[Path],
    device: Optional[str],
):
    """Predict molecular structure from a single spectrum.

    The spectrum file should be a JSON file with a 'peaks' field containing
    a list of [m/z, intensity] pairs.

    Example:
        spec2smiles predict single -m ./models -s spectrum.json -n 50
    """
    import torch
    from spec2smiles.models.pipeline import IntegratedPipeline

    # Load spectrum
    with open(spectrum) as f:
        data = json.load(f)

    if "peaks" not in data:
        raise click.ClickException("Spectrum file must contain 'peaks' field")

    peaks = [(p[0], p[1]) for p in data["peaks"]]

    # Set device
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = None

    # Load pipeline
    model_dir = Path(model_dir)
    part_a_dir = model_dir / "part_a"
    part_b_dir = model_dir / "part_b"

    if not part_a_dir.exists() or not part_b_dir.exists():
        # Try model_dir directly
        part_a_dir = model_dir
        part_b_dir = model_dir

    click.echo("Loading models...")
    pipeline = IntegratedPipeline(device=torch_device)
    pipeline.load(part_a_dir, part_b_dir, verbose=False)

    # Predict
    click.echo("Generating candidates...")
    result = pipeline.predict_from_peaks(
        peaks,
        n_candidates=n_candidates,
        temperature=temperature,
        return_descriptors=True,
    )

    # Format output
    output_data = {
        "candidates": result["candidates"][:20],  # Top 20 for display
        "n_candidates": len(result["candidates"]),
        "n_valid": result["n_valid"],
        "predicted_descriptors": dict(zip(
            result.get("descriptor_names", []),
            result.get("descriptors", []),
        )),
    }

    if output:
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo("\nTop predicted structures:")
        for i, smiles in enumerate(result["candidates"][:10], 1):
            click.echo(f"  {i}. {smiles}")

        click.echo(f"\nTotal candidates: {len(result['candidates'])}")
        click.echo(f"Valid molecules: {result['n_valid']}")


@predict.command("batch")
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing trained models",
)
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with spectra",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for predictions",
)
@click.option(
    "--n-candidates",
    "-n",
    type=int,
    default=50,
    help="Number of candidate structures per spectrum",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature",
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
    help="Show progress bar",
)
def predict_batch(
    model_dir: Path,
    input: Path,
    output: Path,
    n_candidates: int,
    temperature: float,
    device: Optional[str],
    verbose: bool,
):
    """Batch prediction for multiple spectra.

    Input file should be JSONL format with 'spectrum' or 'peaks' field.
    Output is JSONL with predictions for each input.

    Example:
        spec2smiles predict batch -m ./models -i spectra.jsonl -o predictions.jsonl
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from spec2smiles.models.pipeline import IntegratedPipeline

    # Set device
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = None

    # Load pipeline
    model_dir = Path(model_dir)
    part_a_dir = model_dir / "part_a"
    part_b_dir = model_dir / "part_b"

    if not part_a_dir.exists():
        part_a_dir = model_dir
        part_b_dir = model_dir

    click.echo("Loading models...")
    pipeline = IntegratedPipeline(device=torch_device)
    pipeline.load(part_a_dir, part_b_dir, verbose=False)

    # Load input data
    input_data = []
    with open(input) as f:
        for line in f:
            input_data.append(json.loads(line))

    click.echo(f"Processing {len(input_data)} spectra...")

    # Process
    results = []
    iterator = tqdm(input_data, desc="Predicting") if verbose else input_data

    for item in iterator:
        # Get spectrum
        if "spectrum" in item:
            spectrum = np.array(item["spectrum"])
        elif "peaks" in item:
            peaks = [(p[0], p[1]) for p in item["peaks"]]
            spectrum = pipeline.spectrum_processor.process(peaks)
        else:
            results.append({"error": "No spectrum or peaks field"})
            continue

        # Predict
        result = pipeline.predict(
            spectrum,
            n_candidates=n_candidates,
            temperature=temperature,
        )

        results.append({
            "candidates": result["candidates"],
            "n_valid": result["n_valid"],
        })

    # Save results
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    click.echo(f"Results saved to: {output}")
