# Stage 1: Data Collection

## Overview
This stage implements a robust conversion pipeline from Mass Spectrometry Pattern (MSP) format to JavaScript Object Notation Lines (JSONL) format, preparing spectral data for machine learning pipeline processing.

**Conversion Process:**
- SMILES → Molecular Structure representation
- Peaks → [(m/z₁, I₁), (m/z₂, I₂), ..., (m/zₙ, Iₙ)]

Where m/z = mass-to-charge ratio and I = intensity value.

**Key Features:**
- Robust SMILES extraction from various MSP formats
- Peak data validation and normalization
- Error tracking for corrupted records
- Compatibility with multiple MSP sources (GNPS, MoNA, HMDB, etc.)
- Auto-detection of MSP files in dataset directories

## Methodology

### Input Format
- **Source**: MSP files containing electron ionization mass spectra
- **Structure**: Text-based format with metadata fields and peak lists
- **Datasets**: GNPS, HMDB, and proprietary HPJ collections

### Processing Pipeline
1. **MSP File Detection**: Auto-detect or specify MSP files in dataset directory
2. **MSP Parsing**: Extract entries using double-newline delimiters
3. **SMILES Extraction**: Multiple regex patterns for various formats:
   - GNPS format: `"SMILES=..."`
   - MoNA format: `"computed SMILES=..."`
   - Variations in capitalization and spacing
4. **Validation**: 
   - Chemical structure verification using extended SMILES character set
   - Peak data integrity checks (m/z and intensity ranges)
   - Minimum SMILES length validation
5. **Conversion**: Transform to standardized JSONL format with error tracking

### Output Format
```json
{"smiles": "CC(=O)O", "peaks": [[43.0, 100.0], [45.0, 85.2]]}
```

## Key Results (GNPS Dataset Example)
- **Total Spectra**: 23,630 successfully converted
- **Success Rate**: 99.3% (171 corrupted records excluded)
- **Peak Statistics**: Mean 878.7 peaks per spectrum (range: 1-361,421)
- **m/z Range**: 20.4-4,971.1 Da
- **SMILES Length**: Mean 71.8 characters (σ=43.2)
- **Intensity Statistics**: 
  - Range: 0.00e+00 - 1.04e+09
  - Mean: 7.61e+02
  - Median: 1.00e-01

## Configuration

The conversion process is controlled by a comprehensive configuration dictionary:

```python
CONVERSION_CONFIG = {
    'dataset': {
        'name': 'GNPS',  # Dataset name in data/raw/
        'msp_filename': None,  # Specific file or None for auto-detect
    },
    'validation': {
        'min_peaks': 1,
        'max_mz': 5000,
        'max_intensity': 1e10,
        'min_smiles_length': 2,
    },
    'processing': {
        'verbose': True,
        'save_corrupted': True,
    }
}
```

## Quality Control
- **Validation Parameters**:
  - Minimum peaks threshold: 1
  - Maximum m/z: 5,000 Da
  - Maximum intensity: 1e10
  - Minimum SMILES length: 2 characters
- **SMILES Validation**: Extended character set including `CNOSFPB[]()=#+\\/-@123456789.%`
- **Error Tracking**: Corrupted records saved with error descriptions and context

## Dependencies
- Python 3.10+
- NumPy for statistical analysis
- TQDM for progress monitoring

## Error Analysis

**Common Error Types:**
1. **Invalid or missing SMILES**: Comments field doesn't contain valid SMILES notation
2. **Invalid peaks**: Peak data doesn't meet validation criteria
3. **Conversion errors**: Unexpected errors during processing

Error records include:
- Entry index and name
- Error description
- Relevant context (e.g., Comments excerpt for SMILES errors)

## Usage

```python
# Basic usage - configure dataset name in notebook
CONVERSION_CONFIG['dataset']['name'] = 'GNPS'

# Run conversion pipeline
jupyter nbconvert --execute --to notebook --inplace 01_data_conversion.ipynb
```

## Output Files
- `data/input/{dataset}/spectral_data.jsonl`: Successfully converted spectra
- `data/input/{dataset}/corrupted_records.jsonl`: Failed conversions with error details

## Directory Structure
```
data/
├── raw/
│   └── {dataset_name}/
│       └── *.msp
└── input/
    └── {dataset_name}/
        ├── spectral_data.jsonl
        └── corrupted_records.jsonl
```

## Reference
Notebook: `01_data_conversion.ipynb`