# SPEC2SMILES Pipeline

Mass Spectrum to Molecular Structure Prediction using a two-stage machine learning approach.

## Architecture

```
Spectrum (peaks) → [Part A] → Descriptors → [Part B] → SMILES candidates
```

### Part A: Spectrum → Descriptors
- **LightGBM**: Gradient boosting ensemble (R²=0.80)
- **Hybrid CNN-Transformer**: Alternative neural network approach

### Part B: Descriptors → SMILES
- **DirectDecoder**: Transformer-based autoregressive decoder with SELFIES encoding

## Project Structure

```
spec2smiles_v2/
├── pyproject.toml           # Poetry configuration
├── Makefile                 # Build commands
├── config.yml               # Main configuration
├── data/
│   ├── input/gnps/          # Input spectral data
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

# Train Part A (LightGBM - recommended)
make train-part-a-lgbm

# Train Part A (Hybrid CNN-Transformer - alternative)
make train-part-a

# Train Part B (DirectDecoder)
make train-part-b

# Or train full pipeline
make train-full

# Run end-to-end evaluation
make evaluate

# Generate visualizations
make viz
```

## Configuration

All settings are in `config.yml`:

```yaml
# Dataset and device
dataset: gnps
device: auto  # cuda, mps, cpu, or auto

# 28 optimized descriptors
descriptors:
  - fr_phos_ester
  - Chi3n
  - Chi2n
  # ... (28 total)

# Part B settings (DirectDecoder)
part_b:
  model: direct
  direct:
    hidden_dim: 768
    n_layers: 6
    n_heads: 12

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
| `make train-part-a` | Train Part A Hybrid CNN-Transformer |
| `make train-part-a-lgbm` | Train Part A LightGBM (recommended) |
| `make train-part-b` | Train Part B DirectDecoder |
| `make train-full` | Train complete pipeline |
| `make evaluate` | Compute metrics on test set |
| `make viz` | Generate figures |
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

## Performance (GNPS Dataset)

| Metric | Value |
|--------|-------|
| Part A Mean R² (LightGBM) | 0.80 |
| Hit@1 | 37.0% |
| Hit@10 | 53.2% |
| Mean Best Tanimoto | 0.712 |
| SMILES Validity | 100% |

## Requirements

- Python 3.10-3.11 (RDKit compatibility)
- Poetry for dependency management
- ~8GB RAM for training

## License

MIT
