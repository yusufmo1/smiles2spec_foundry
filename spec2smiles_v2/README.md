# SPEC2SMILES Pipeline

Mass Spectrum to Molecular Structure Prediction using a two-stage machine learning approach.

## Architecture

```
Spectrum (peaks) → [Part A] → Descriptors → [Part B] → SMILES candidates
```

### Part A: Spectrum → Descriptors
- **Hybrid CNN-Transformer**: Combines local (CNN) and global (Transformer) pattern recognition

### Part B: Descriptors → SMILES
- **VAE**: Conditional Variational Autoencoder with SELFIES encoding

## Project Structure

```
spec2smiles_pkg/
├── pyproject.toml          # Poetry configuration
├── Makefile                 # Build commands
├── .env.example             # Environment variable template
├── data/
│   ├── input/hpj/           # Input spectral data
│   └── output/              # Models, metrics, figures
├── src/
│   ├── config/              # Pydantic settings
│   ├── domain/              # Pure business logic
│   ├── services/            # Orchestration layer
│   ├── models/              # ML model definitions
│   └── utils/               # Utilities and metrics
└── scripts/                 # Executable pipeline stages
```

## Quick Start

```bash
# Install dependencies
make install

# Train Part A (Hybrid CNN-Transformer)
make train-part-a

# Train Part B
make train-part-b-vae

# Or train full pipeline
make train-full

# Run predictions
make predict

# Evaluate performance
make evaluate

# Generate visualizations
make visualize
```

## Configuration

All settings are in `config.yml`:

```yaml
# Dataset and device
dataset: hpj
device: auto  # cuda, mps, cpu, or auto

# Part A settings (Hybrid CNN-Transformer)
part_a:
  cnn_hidden: 256
  transformer_dim: 256
  n_heads: 8
  n_transformer_layers: 4

# Inference settings
inference:
  n_candidates: 50
  temperature: 0.7
```

Use a custom config:
```bash
make train-part-a CONFIG=my_config.yml
# Or directly:
python scripts/train_part_a.py --config my_config.yml
```

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies with Poetry |
| `make train-part-a` | Train Part A (Spectrum → Descriptors) |
| `make train-part-b` | Train Part B (Descriptors → SMILES) |
| `make train-full` | Train complete pipeline |
| `make predict` | Run predictions |
| `make evaluate` | Compute metrics on test set |
| `make visualize` | Generate figures |
| `make clean` | Remove generated files |

## Python Usage

```python
from src.services.pipeline import PipelineService
from src.config import settings

# Load trained pipeline
pipeline = PipelineService.from_directories(
    part_a_dir=settings.models_path / "part_a",
    part_b_dir=settings.models_path / "part_b",
)

# Predict from peaks
result = pipeline.predict_from_peaks(
    peaks=[(50.0, 100.0), (77.0, 50.0), (105.0, 80.0)],
    n_candidates=50,
)

print(result["candidates"][:5])
```

## Data Format

Input JSONL files should contain:
```json
{"smiles": "CCO", "peaks": [[31, 100], [45, 80], [46, 10]]}
```

## Performance

- Part A Mean R²: ~0.65
- Integrated Hit@10: ~78.5%
- Mean Tanimoto Similarity: ~0.42

## Requirements

- Python 3.10-3.11 (RDKit compatibility)
- Poetry for dependency management
- ~8GB RAM for training

## License

MIT
