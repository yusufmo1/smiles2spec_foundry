# SPEC2SMILES: Inverse Mass Spectrometry Pipeline for Molecular Structure Elucidation

## Overview
SPEC2SMILES is a machine learning pipeline for reconstructing molecular structures from electron ionization mass spectra. This inverse prediction system employs a two-stage architecture combining gradient boosting and variational autoencoders to generate molecular candidates from spectral data. The pipeline demonstrates the feasibility of structure elucidation through physicochemical property prediction, achieving 10.7% exact match rate at the descriptor level and exploring the challenges of inverse mass spectrometry prediction.

## System Architecture

The pipeline implements a sequential two-stage workflow:

```
Mass Spectrum → Molecular Descriptors → SELFIES Encoding → SMILES Generation → Structure Ranking
     ↓                    ↓                    ↓                   ↓                  ↓
  Binning (500)     12 LGBM Models      Conditional VAE      ~24 Candidates   Tanimoto Ranking
```

This architecture decouples the complex inverse problem into interpretable stages: first predicting molecular properties from spectra, then generating structures constrained by these properties.

## Pipeline Components

### Data Preparation Stage
Implements spectrum preprocessing and molecular descriptor computation for 2,720 experimental spectra. Performs m/z binning (500 bins, 0-499 Da), square-root transformation for variance stabilization, and extracts 12 RDKit molecular descriptors. Achieves 100% data processing success rate with comprehensive validation.

### Part A: Spectrum-to-Descriptors (LGBM Ensemble)
Deploys 12 LightGBM models, one per molecular descriptor, achieving mean R² of 0.657 across all properties. Key predicted descriptors include NumAromaticRings (R²=0.876), MolWeight (R²=0.839), and HeavyAtomCount (R²=0.842). Implements parallel training architecture with feature importance analysis and early stopping (~306 iterations average). Total training time: 64.3 seconds for complete ensemble.

### Part B: Descriptors-to-SMILES (Conditional VAE)
Conditional Variational Autoencoder with SELFIES molecular representation ensuring 100% chemical validity. Architecture: 128-dimensional latent space, 3-layer encoder/decoder with dropout regularization. Implements teacher forcing during training. Generates candidate structures per spectrum with vocabulary of 39 tokens and maximum sequence length of 71.

### Integrated Pipeline
End-to-end system combining both stages with candidate ranking. Preprocesses input spectra, predicts 12 molecular descriptors via LGBM ensemble, conditions VAE generation on predicted properties, produces SELFIES sequences converted to SMILES, and ranks candidates by Tanimoto similarity.

## Performance Achievements

### Primary Metrics (Integrated Pipeline)
- **Exact Match**: 2.9% - Exact structure recovery
- **Formula Match**: 27.6% - Correct molecular formula
- **Hit@1**: 2.9% - Correct structure in top prediction
- **Hit@5**: 2.9% - Correct structure in top 5 candidates
- **Hit@10**: 2.9% - Correct structure in top 10 candidates
- **Hit@50**: 2.9% - Correct structure in all candidates
- **Mean Tanimoto Similarity**: 0.420 ± 0.232
- **Median Tanimoto Similarity**: 0.360

### Component Performance
#### Part A (Descriptor Prediction)
- **Mean R² Score**: 0.657 across 12 descriptors
- **Median R² Score**: 0.744
- **Best Predictions**: NumAromaticRings (0.876), HeavyAtomCount (0.842), MolWeight (0.839)
- **Challenging Properties**: NumHDonors (0.407), NumHAcceptors (0.428), TPSA (0.448)
- **High Performance**: 7/12 descriptors with R² > 0.7 (58%)

#### Part B (Structure Generation from True Descriptors)
- **Exact Match Rate**: 10.7% from true descriptors
- **Formula Match Rate**: 56.6%
- **Mean Tanimoto Similarity**: 0.533 (Median: 0.500)
- **Chemical Validity**: 100% valid SELFIES
- **Training Loss**: 0.1607 (final, converged at epoch 50/100)

### Performance Distribution
- **High R² (>0.7)**: 7 descriptors (58%)
- **Moderate R² (0.5-0.7)**: 1 descriptor (8%)
- **Low R² (<0.5)**: 4 descriptors (33%)

## Technical Specifications

### Data
- **Training**: 2,176 spectra with validated SMILES
- **Validation**: 272 spectra for hyperparameter tuning
- **Test**: 272 spectra for final evaluation
- **Input**: 500-dimensional binned spectra (0-499 m/z)
- **Intermediate**: 12 molecular descriptors
- **Output**: ~24 SMILES candidates per spectrum (average)

### Model Architecture Details
- **LGBM Models**: 12 independent gradient boosting machines
  - n_estimators: 1000 with early stopping
  - Learning rate: 0.05
  - num_leaves: 31
  - Regularization: L1=0.1, L2=0.1
  - Early stopping rounds: 50
  
- **Conditional VAE**: 
  - Parameters: 4,137,511 total
  - Encoder: Input→256→128 (latent)
  - Latent: 128-dimensional Gaussian
  - Decoder: (128+12)→256→512→Vocabulary(39)
  - Optimizer: Adam with cosine annealing
  - Vocabulary: 39 SELFIES tokens

### Computational Requirements
- **Memory**: 4-6GB RAM for full pipeline
- **Storage**: 
  - Part A models: 17.3 MB
  - Integration package: 205.3 MB
- **Training Time**: 
  - Part A: 64.3 seconds (12 LGBM models)
  - Part B: 100 epochs (~2 minutes on GPU, converges at epoch 50)
- **Inference Speed**: 1.675 seconds per molecule
- **Throughput**: 0.59 molecules/second
- **Dependencies**: Python 3.10-3.11, RDKit, LightGBM, PyTorch, SELFIES

## Repository Structure

```
spec2smiles/
├── notebooks/
│   ├── 00_data_preperation.ipynb      # Spectrum preprocessing & descriptors
│   ├── 01_spectra_to_descriptors.ipynb # Part A: LGBM training
│   ├── 02_descriptors_to_smiles.ipynb  # Part B: VAE training
│   ├── 03_spectra_to_smiles.ipynb      # Integrated pipeline evaluation
│   └── 04_lgbm_HPO.ipynb               # Bayesian hyperparameter optimization
├── data/
│   ├── input/                          # Raw spectral data (JSONL)
│   ├── processed/                      # Preprocessed train/val/test splits
│   ├── models/                         # Trained model checkpoints
│   │   ├── part_a/                     # LGBM ensemble models
│   │   └── part_b/                     # VAE model and tokenizer
│   └── results/                        # Evaluation metrics and figures
└── README.md                           # This file
```

## Key Findings

1. **Error Cascading**: Descriptor prediction errors cause 73% performance degradation (10.7% → 2.9% exact match) in the integrated pipeline
2. **Descriptor Quality**: 58% of descriptors achieve R² > 0.7, with structural features (rings, atoms) outperforming chemical properties (H-bonding, TPSA)
3. **SELFIES Validity**: 100% valid structures generated, demonstrating robustness of SELFIES representation
4. **Limited Recovery**: Hit@K rates plateau at 2.9% from Hit@1 through Hit@50, indicating no improvement with additional candidates
5. **Generation Bottleneck**: VAE generates average 24.1 candidates despite requesting 50, suggesting early convergence limitations
6. **Structural Similarity**: Mean Tanimoto of 0.420 ± 0.232 shows moderate structural resemblance despite low exact matches

## Notable Implementation Details

### Spectrum Preprocessing
- **Binning Strategy**: 1 Da bins from 0-499 m/z for consistent dimensionality
- **Transformation**: Square-root scaling stabilizes variance across intensity range
- **Normalization**: Max-norm scaling preserves relative peak relationships

### Descriptor Selection
- **Focused Set**: 12 RDKit descriptors covering key molecular properties
- **Properties**: MolWt, HeavyAtomCount, NumHeteroatoms, NumAromaticRings, RingCount, NOCount, NumHDonors, NumHAcceptors, TPSA, MolLogP, NumRotatableBonds, FractionCsp3
- **Standardization**: Zero-mean, unit-variance scaling on training set

### SELFIES Encoding
- **Vocabulary**: 39 unique tokens from training set
- **Max Length**: 71 tokens (maximum observed)
- **Special Tokens**: ['<PAD>', '<START>', '<END>', '<UNK>'] for sequence modeling

### Candidate Ranking
- **Primary Metric**: Tanimoto similarity of Morgan fingerprints (radius=2)
- **Tie Breaking**: Descriptor similarity for equivalent Tanimoto scores
- **Diversity Preservation**: Maintains structural variety in top-K results

## Hyperparameter Optimization

### Bayesian Optimization Results
Applied Ax Platform with BoTorch backend for systematic hyperparameter tuning:
- **Trials**: 50 per descriptor (600 total evaluations)
- **Search Space**: n_estimators [100-1000], learning_rate [0.001-0.3], num_leaves [10-100]
- **Outcome**: Mean R² of 0.637 (slight degradation from baseline 0.657)
- **Best Optimized**: MolWt (R²=0.889), NumAromaticRings (R²=0.889)
- **Key Finding**: Baseline hyperparameters already near-optimal, marginal improvements observed
- **Time Investment**: ~60 minutes for complete optimization

## Academic Contribution

SPEC2SMILES explores the challenges of inverse mass spectrometry through:
- **Two-Stage Architecture**: Decomposition into descriptor prediction and structure generation stages
- **Interpretable Intermediates**: 12 molecular descriptors as bridging representation
- **Valid Generation**: 100% chemically valid structures through SELFIES encoding
- **Challenge Documentation**: Quantifies the difficulty of spectrum-to-structure inversion
- **Efficient Implementation**: Sub-minute training for LGBM ensemble
- **Systematic Optimization**: Bayesian hyperparameter tuning with 600 trials

The pipeline reveals significant challenges in inverse mass spectrometry, with descriptor prediction errors causing 73% performance degradation (10.7% → 2.9%) through the generation stage. While achieving moderate structural similarity (0.420 Tanimoto), exact structure recovery remains elusive (2.9%), highlighting the complexity of this inverse problem.

## Statistical Validation
- **Test Set Performance**: 2.9% exact match across 272 test molecules
- **Descriptor R² Distribution**: Mean 0.657, Median 0.744
- **Best Descriptor**: NumAromaticRings (R² = 0.876)
- **Worst Descriptor**: NumHDonors (R² = 0.407)
- **Training Convergence**: VAE loss plateaued at epoch 50/100
- **Generation Statistics**: 24.1 average candidates (std: varies per molecule)
- **Performance Gap**: 73% degradation from descriptor errors (10.7% → 2.9%)

