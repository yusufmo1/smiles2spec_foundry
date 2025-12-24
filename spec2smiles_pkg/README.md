# SPEC2SMILES

Mass Spectrum to Molecular Structure Prediction Pipeline.

## Overview

SPEC2SMILES is a two-stage machine learning pipeline for molecular structure identification from electron ionization mass spectra:

1. **Part A**: Spectrum → Molecular Descriptors (LightGBM ensemble)
2. **Part B**: Descriptors → SMILES candidates (Conditional VAE with SELFIES)

## Installation

```bash
# Using Poetry (recommended)
cd spec2smiles_pkg
poetry install

# Or using pip
pip install -e .
```

## Quick Start

### Training

```bash
# Train full pipeline
spec2smiles train full --data-dir ./data/processed/hpj --output-dir ./models

# Or train components separately
spec2smiles train part-a --data-dir ./data/processed/hpj --output-dir ./models/part_a
spec2smiles train part-b --data-dir ./data/processed/hpj --output-dir ./models/part_b
```

### Prediction

```bash
# Single spectrum prediction
spec2smiles predict single --model-dir ./models --spectrum peaks.json

# Batch prediction
spec2smiles predict batch --model-dir ./models --input spectra.jsonl --output predictions.jsonl
```

### Evaluation

```bash
# Evaluate Part A
spec2smiles evaluate part-a --model-dir ./models/part_a --data-dir ./data/processed/hpj

# Evaluate full pipeline
spec2smiles evaluate pipeline --model-dir ./models --data-dir ./data/processed/hpj
```

## Python API

```python
from spec2smiles.models.pipeline import IntegratedPipeline

# Load trained pipeline
pipeline = IntegratedPipeline.from_directories(
    part_a_dir="./models/part_a",
    part_b_dir="./models/part_b"
)

# Predict from peaks
result = pipeline.predict_from_peaks(
    peaks=[(50.0, 100.0), (77.0, 50.0), (105.0, 80.0)],
    n_candidates=50
)

print(result["candidates"][:5])
```

## Data Format

Input JSONL files should contain:
```json
{"smiles": "CCO", "peaks": [[31, 100], [45, 80], [46, 10]], "spectrum": [0, 0, ...]}
```

## Architecture

- **Part A**: 12 independent LightGBM regressors (one per descriptor)
- **Part B**: Conditional VAE with BiLSTM encoder and LSTM decoder
- **Representation**: SELFIES for guaranteed chemical validity

## Performance

- Part A Mean R²: ~0.65
- Integrated Exact Match: ~2.9%
- Mean Tanimoto Similarity: ~0.42
