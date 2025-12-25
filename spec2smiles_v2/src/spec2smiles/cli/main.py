"""Main CLI entry point for SPEC2SMILES."""

import click

from spec2smiles.cli.train import train
from spec2smiles.cli.predict import predict
from spec2smiles.cli.evaluate import evaluate
from spec2smiles.cli.visualize import visualize


@click.group()
@click.version_option(version="0.1.0", prog_name="spec2smiles")
def cli():
    """SPEC2SMILES: Mass Spectrum to Molecular Structure Prediction.

    A two-stage pipeline for molecular structure identification from
    electron ionization mass spectra:

    \b
    1. Part A: Spectrum -> Molecular Descriptors (Hybrid CNN-Transformer)
    2. Part B: Descriptors -> SMILES candidates (Conditional VAE)

    Use 'spec2smiles COMMAND --help' for more information on each command.
    """
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(visualize)


if __name__ == "__main__":
    cli()
