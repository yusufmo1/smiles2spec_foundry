# SPEC2SMILES Notebooks: Inverse Mass Spectrometry Pipeline

## Overview
This directory contains the complete implementation of the SPEC2SMILES inverse prediction pipeline, which reconstructs molecular structures from electron ionization mass spectra through a two-stage machine learning approach. The notebooks implement sequential data preprocessing, gradient boosting for descriptor prediction, variational autoencoder for structure generation, and integrated pipeline evaluation.

## Methodology

### Two-Stage Architecture
The pipeline decomposes the complex inverse problem into two interpretable stages:
1. **Spectrum → Descriptors**: Predict molecular properties from spectral features
2. **Descriptors → SMILES**: Generate structures constrained by predicted properties

### Mathematical Framework
- **Stage A**: $f: \mathbb{R}^{500} → \mathbb{R}^{12}$ mapping spectra to descriptors
- **Stage B**: $g: \mathbb{R}^{12} → \mathcal{S}^{n}$ generating SMILES candidates
- **Integration**: $h = g \circ f$ with ranking function $r: \mathcal{S}^{n} → \mathcal{S}$

### Data Flow
```
Raw Spectra → Binning → LGBM Ensemble → Descriptor Vectors → Conditional VAE → SELFIES → SMILES → Ranking
```

## Notebook Descriptions

### 00_data_preperation.ipynb
**Purpose**: Spectrum preprocessing and molecular descriptor computation

**Key Components**:
- **Data Loading**: JSONL parsing with validation (2,720 spectra)
- **Spectrum Processing**: 
  - Binning to 500 dimensions (0-499 m/z, 1 Da bins)
  - Square-root transformation for variance stabilization
  - Max-norm scaling for intensity normalization
- **Descriptor Calculation**: 12 RDKit molecular descriptors
  - MolWt, HeavyAtomCount, NumHeteroatoms, NumAromaticRings
  - RingCount, NOCount, NumHDonors, NumHAcceptors
  - TPSA, MolLogP, NumRotatableBonds, FractionCsp3
- **Data Splitting**: 80/10/10 train/validation/test (2176/272/272 samples)
- **Quality Control**: Validation of chemical structures and spectrum integrity

**Key Results**:
- Processing success rate: 100% (all 2,720 entries valid)
- Descriptor matrix: (2720, 12) shape
- Mean non-zero bins: 128.1 ± 65.1 per spectrum
- Mean molecular weight: 284.8 ± 89.6 Da
- SELFIES conversion: 100% success rate
- Processing time: 4.90s for descriptor calculation
- Data splits: Train=2176, Val=272, Test=272

### 01_spectra_to_descriptors.ipynb
**Purpose**: Part A implementation - Training LGBM models for descriptor prediction

**Key Components**:
- **Model Architecture**: 12 independent LightGBM regressors
- **Hyperparameters**:
  - n_estimators: 1000 with early stopping
  - learning_rate: 0.05
  - num_leaves: 31
  - L1 regularization: 0.1
  - L2 regularization: 0.1
- **Training Strategy**:
  - Parallel training with joblib
  - Early stopping with 50-round patience
  - Validation-based convergence monitoring
- **Feature Analysis**: Feature importance for top m/z bins
- **Ensemble Packaging**: Serialization with preprocessing pipelines

**Key Results**:
- Test Set Mean R²: 0.657 (Median: 0.744)  
- Best Predictions: NumAromaticRings (0.876), HeavyAtomCount (0.842), MolWeight (0.839)
- Challenging Properties: NumHDonors (0.407), NumHAcceptors (0.428), TPSA (0.448)
- High Performance (R² > 0.7): 7/12 descriptors (58%)
- Training Time: 64.3 seconds for complete ensemble
- Model Size: 17.3 MB (integration package)
- Average Early Stopping: ~306 iterations per model
- Feature Importance: m/z bins 98, 51, 183, 77, 43 most informative

### 02_descriptors_to_smiles.ipynb
**Purpose**: Part B implementation - Training Conditional VAE for structure generation

**Key Components**:
- **SELFIES Encoding**:
  - Vocabulary construction: 39 unique tokens
  - Maximum sequence length: 71 tokens
  - Special tokens: ['<PAD>', '<START>', '<END>', '<UNK>']
- **VAE Architecture**:
  - Parameters: 4,137,511 total
  - Encoder: Input → 256 → 128 (latent)
  - Decoder: (128 + 12) → 256 → 512 → Vocabulary(39)
  - Activation: ReLU with Dropout(0.2)
  - Output: LogSoftmax for token probabilities
- **Conditional Mechanism**: Descriptor concatenation in decoder
- **Loss Function**: 
  - Reconstruction: Cross-entropy loss
  - Regularization: KL divergence with annealing
  - Total: ELBO maximization
- **Training Configuration**:
  - Optimizer: Adam with cosine annealing
  - Batch size: 64
  - Epochs: 100 (best at epoch 50)
  - Learning rate schedule: 1e-3 → 3.9e-6

**Key Results**:
- Final Loss: 0.1607 (reconstruction: 0.1607, KL: ~0.0000)
- Valid SELFIES: 100% chemical validity
- Exact Match Rate: 10.7% from true descriptors
- Formula Match Rate: 56.6%
- Mean Tanimoto Similarity: 0.533 (Median: 0.500)
- Training Time: ~2 minutes on GPU (100 epochs)
- Best Epoch: 50 (early convergence)
- VAE Parameters: 4,137,511 total
- Vocabulary Size: 39 SELFIES tokens
- Max Sequence Length: 71 tokens
- Candidates per molecule: 50

### 03_spectra_to_smiles.ipynb
**Purpose**: Integrated pipeline evaluation and end-to-end testing

**Key Components**:
- **Pipeline Integration**:
  - Model loading and compatibility verification
  - Preprocessing pipeline restoration
  - VAE weight initialization
- **Inference Pipeline**:
  1. Spectrum preprocessing (binning, transformation)
  2. Descriptor prediction (12 LGBM models)
  3. Structure generation (candidates via VAE)
  4. SELFIES to SMILES conversion
  5. Candidate ranking (Tanimoto similarity)
- **Evaluation Metrics**:
  - Hit@K for K ∈ {1, 5, 10, 20, 50}
  - Exact match rate
  - Formula match rate
  - Average Tanimoto similarity
- **Performance Analysis**:
  - Descriptor prediction error impact
  - Generation quality assessment
  - Example predictions with error analysis
- **Visualization**: Pipeline performance plots and example predictions

**Key Results**:
- Exact Match: 2.9% structure recovery
- Formula Match: 27.6% correct molecular formula
- Hit@1: 2.9% (top prediction)
- Hit@5: 2.9% (no improvement)
- Hit@10: 2.9% (no improvement)
- Hit@50: 2.9% (all candidates)
- Mean Tanimoto: 0.420 ± 0.232
- Median Tanimoto: 0.360
- Average candidates generated: 24.1 (despite requesting 50)
- Inference Speed: 1.675 seconds per molecule
- Throughput: 0.59 molecules/second (457.3s for 272 molecules)
- Performance Gap: 73% degradation (10.7% → 2.9%) from descriptor prediction errors

## Dependencies
- **Core Libraries**: NumPy, Pandas, Scikit-learn
- **Chemistry**: RDKit (≥2022.03), SELFIES (≥2.1.0)
- **Machine Learning**: LightGBM (≥3.3.0), PyTorch (≥1.12.0)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Utilities**: TQDM, Joblib

## Output Files

### Processed Data
- `data/processed/{dataset}/train_data.jsonl`: Training set with descriptors
- `data/processed/{dataset}/val_data.jsonl`: Validation set
- `data/processed/{dataset}/test_data.jsonl`: Test set
- `data/processed/{dataset}/descriptor_scaler.pkl`: Standardization parameters

### Trained Models
- `data/models/part_a/{dataset}/spectra_to_descriptors_ensemble.pkl`: 12 LGBM models
- `data/models/part_a/{dataset}/integration_package.pkl`: Part A package (17.3 MB)
- `data/models/part_b/{dataset}/best_model.pt`: VAE checkpoint
- `data/models/part_b/{dataset}/integration_package.pkl`: Complete pipeline (205.3 MB)

### Results
- `data/results/part_a/{dataset}/`: Descriptor prediction analysis
- `data/results/part_b/{dataset}/`: VAE training curves
- `data/results/integration/{dataset}/`: Pipeline evaluation metrics

### 04_lgbm_HPO.ipynb (Optional)
**Purpose**: Bayesian optimization of LGBM hyperparameters for descriptor prediction

**Key Components**:
- **Optimization Framework**: Ax Platform with Bayesian optimization
- **Search Space**:
  - n_estimators: [100, 1000]
  - learning_rate: [0.001, 0.3] (log scale)
  - num_leaves: [10, 100]
  - min_child_samples: [5, 50]
  - regularization: L1 [0, 1], L2 [0, 1]
- **Trials**: 50 per descriptor (600 total)
- **Objective**: Maximize R² on validation subset (50% for speed)
- **Backend**: BoTorch with Sobol+GP strategy

**Key Results**:
- Mean R² after optimization: 0.637 (slight degradation from baseline 0.657)
- Best optimized descriptors: MolWt (R²=0.889), NumAromaticRings (R²=0.889), HeavyAtomCount (R²=0.887)
- Worst optimized: NumHDonors (R²=0.394), TPSA (R²=0.598), NumHAcceptors (R²=0.657)
- Total optimization time: ~3600 seconds (60 minutes)
- Observation: Marginal improvements, baseline hyperparameters already near-optimal

## Execution Order
Notebooks must be executed sequentially as each depends on outputs from previous stages:
1. `00_data_preperation.ipynb` → Preprocessed data
2. `01_spectra_to_descriptors.ipynb` → LGBM models
3. `02_descriptors_to_smiles.ipynb` → VAE model
4. `03_spectra_to_smiles.ipynb` → Pipeline evaluation
5. `04_lgbm_HPO.ipynb` (Optional) → Optimized LGBM models

## Performance Summary
| Stage | Metric | Value |
|-------|--------|-------|
| Data Preparation | Success Rate | 100% |
| Data Preparation | Valid SELFIES | 100% |
| Part A (LGBM) | Mean R² | 0.657 |
| Part A (LGBM) | Median R² | 0.744 |
| Part A (LGBM) | Best Descriptor R² | 0.876 (NumAromaticRings) |
| Part A (LGBM) | Training Time | 64.3 seconds |
| Part B (VAE) | Valid SELFIES | 100% |
| Part B (VAE) | Exact Match (true desc) | 10.7% |
| Part B (VAE) | Formula Match (true desc) | 56.6% |
| Part B (VAE) | Mean Tanimoto (true desc) | 0.533 |
| Integration | Exact Match | 2.9% |
| Integration | Formula Match | 27.6% |
| Integration | Mean Tanimoto | 0.420 |
| Integration | Inference Speed | 1.675 sec/molecule |

## Key Insights
1. **Descriptor Bridge Challenge**: Error propagation from descriptor prediction (R²=0.657) causes 73% performance degradation (10.7% → 2.9%)
2. **SELFIES Validity**: 100% chemical validity achieved through SELFIES encoding
3. **Descriptor Quality**: 7/12 descriptors achieve R² > 0.7, with structural features (rings, atoms) outperforming chemical properties (H-bonding, TPSA)
4. **Generation Bottleneck**: VAE generates only ~24 candidates despite requesting 50, with Hit@K metrics showing no improvement beyond Hit@1
5. **Early Convergence**: VAE converges at epoch 50/100, suggesting potential for faster training
6. **Inverse Problem Difficulty**: Low exact match (2.9%) despite moderate structural similarity (0.420 Tanimoto) highlights the fundamental challenge of spectrum-to-structure inversion

