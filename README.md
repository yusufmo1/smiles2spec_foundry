# SMILES2SPEC Foundry: Bidirectional Mass Spectrometry Prediction Pipeline

**MSc AI in Biosciences Dissertation Project**  
**Queen Mary University of London**  
**Author:** Yusuf Mohammed  
**Supervisor:** Dr Mohammed Elbadawi  

<div align="center">

[![MSc Dissertation](https://img.shields.io/badge/MSc%20Dissertation-Queen%20Mary%20University%20of%20London-003E74?style=for-the-badge&logo=graduation-cap)](https://www.qmul.ac.uk/)
[![Academic Year](https://img.shields.io/badge/Academic%20Year-2024--2025-blue?style=for-the-badge)](https://github.com/yusufmo1)

### Core Technologies

[![Python](https://img.shields.io/badge/Python-3.10--3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2025.3.2-07598D?style=flat-square&logo=molecule&logoColor=white)](https://www.rdkit.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-02569B?style=flat-square&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833?style=flat-square&logo=anaconda&logoColor=white)](https://conda.io/)

### Machine Learning & Performance

[![Models Tested](https://img.shields.io/badge/Models%20Tested-100+-purple?style=flat-square)](https://github.com/yusufmo1)
[![Best Performance](https://img.shields.io/badge/Cosine%20Similarity-0.8164-brightgreen?style=flat-square)](https://github.com/yusufmo1)
[![Training Samples](https://img.shields.io/badge/Training%20Samples-2,720-blue?style=flat-square)](https://github.com/yusufmo1)
[![Features](https://img.shields.io/badge/Molecular%20Features-7,137-orange?style=flat-square)](https://github.com/yusufmo1)
[![Improvement](https://img.shields.io/badge/vs%20NEIMS-33.1%25%20Better-success?style=flat-square)](https://github.com/yusufmo1)
[![Throughput](https://img.shields.io/badge/Throughput-3,152%20samples%2Fs-yellow?style=flat-square)](https://github.com/yusufmo1)
[![Bayesian Trials](https://img.shields.io/badge/Bayesian%20Trials-300-red?style=flat-square)](https://github.com/yusufmo1)
[![Hit@10](https://img.shields.io/badge/Hit%4010%20(Inverse)-2.9%25-teal?style=flat-square)](https://github.com/yusufmo1)

</div>

---

## Quick Navigation

[Executive Summary](#executive-summary) • 
[Research Architecture](#research-architecture) • 
[Pipeline Components](#pipeline-components) • 
[Performance Achievements](#performance-achievements) • 
[Installation](#installation-and-setup) • 
[Running Pipelines](#running-the-pipelines) • 
[Technical Specifications](#technical-specifications) • 
[Citation](#citation)

---

## Executive Summary

The SMILES2SPEC Foundry is a comprehensive machine learning research pipeline developed as part of an MSc dissertation exploring artificial intelligence integration into electronic laboratory notebooks. This research infrastructure implements bidirectional molecular structure-spectrum prediction through systematic evaluation of 100+ model architectures, achieving state-of-the-art performance among rule-free methods with a 33.1% improvement over NEIMS.

The foundry comprises two complementary pipelines: **SMILES2SPEC** for forward prediction (molecular structure to mass spectrum) achieving 0.8164 cosine similarity through bin-by-bin ensemble optimization, and **SPEC2SMILES** for inverse prediction (mass spectrum to molecular structure) achieving 2.9% exact match rate with 100% valid SMILES generation. These research pipelines form the scientific foundation for understanding the challenges and opportunities in computational mass spectrometry.

## Research Architecture

### Bidirectional Pipeline Structure

```
                    SMILES2SPEC (Forward)
    SMILES → Features → Models → Spectrum Prediction
       ↓         ↓          ↓            ↓
    RDKit    7,137 dims  100+ tested  0.8164 cosine

                    SPEC2SMILES (Inverse)
    Spectrum → Descriptors → VAE → SMILES Generation
       ↓           ↓          ↓          ↓
    500 bins   12 LGBM     SELFIES   78.5% Hit@10
```

### Research Methodology

The foundry implements a rigorous scientific methodology through:

1. **Systematic Model Evaluation**: 100+ models tested across traditional ML, deep learning, and ensemble methods
2. **Comprehensive Feature Engineering**: 7,137 molecular descriptors with multiple reduction strategies
3. **Bayesian Hyperparameter Optimisation**: 300 trials across 6 model architectures
4. **Statistical Validation**: Bootstrap confidence intervals (1,000 iterations) for all metrics
5. **Literature Benchmarking**: Comparison with CFM-ID, NEIMS, and other state-of-the-art methods

## Pipeline Components

### SMILES2SPEC Pipeline (Forward Prediction)

#### Stage 1: Data Collection
Converts Mass Spectrometry Pattern (MSP) files to standardised JSONL format, processing 2,720 experimental spectra with 99.3% success rate.

#### Stage 2: Molecular Featurisation
Extracts 7,137 molecular descriptors and fingerprints using RDKit:
- Morgan fingerprints (radii 1-3, size 1024)
- MACCS keys (166 structural keys)
- Topological and pattern fingerprints
- Electronic properties and 3D conformer features

#### Stage 3: Data Exploration
Systematic evaluation of 31 model architectures:
- 18 baseline ML algorithms (Linear, Ridge, KNN, Trees, Random Forest, Extra Trees)
- 10 neural architectures (ModularNet, HierarchicalPredictionNet, etc.)
- 3 transformer models (SpectraFormer, ViT-1D, LinearAttention)

#### Stage 4: Feature Engineering
- Statistical selection (Mutual Information, F-test, Variance)
- Model-based selection (LASSO, Random Forest importance)
- Dimensionality reduction (PCA, UMAP, Autoencoders)
- Achieves 86% feature reduction with <2% accuracy loss

#### Stage 5: Hyperparameter Optimisation
Multi-objective Bayesian optimisation using Ax Platform:
- 300 total trials across 6 model types
- NEHVI acquisition function with Sobol initialisation
- Parallel asynchronous evaluation

#### Stage 6: Model Training
Production model development with ensemble strategies:
- Simple weighted ensemble: 0.8037 cosine similarity
- Bin-by-bin ensemble: 0.8164 cosine similarity (best overall)
- 2D features outperform 3D by 12.2%

#### Stage 7: Model Evaluation
Comprehensive evaluation framework:
- Bootstrap confidence intervals (1,000 iterations)
- SHAP value analysis for interpretability
- Throughput analysis: up to 3,152 samples/second
- Literature benchmarking: 33.1% improvement over NEIMS

### SPEC2SMILES Pipeline (Inverse Prediction)

#### Part A: Spectrum-to-Descriptors
- 12 LightGBM models (one per molecular descriptor)
- Mean R² of 0.653 across all properties
- Best predictions: NumAromaticRings (R²=0.874), MolWeight (R²=0.849)
- Training time: 56.7 seconds for complete ensemble

#### Part B: Descriptors-to-SMILES
- Conditional Variational Autoencoder (VAE)
- SELFIES encoding for 100% chemical validity
- 128-dimensional latent space with 4.1M parameters
- Generates ~24 candidate structures per spectrum

#### Integrated Pipeline Performance
- Exact Match: 2.9% (challenging inverse problem)
- Formula Match: 27.6% (correct molecular formula)
- Hit@10: 2.9% (limited by descriptor prediction errors)
- Mean Tanimoto Similarity: 0.420 ± 0.232

## Performance Achievements

### Forward Prediction (SMILES2SPEC)

| Model Category | Best Model | Cosine Similarity | Training Time |
|----------------|------------|-------------------|---------------|
| **Ensemble** | Bin-by-bin | **0.8164** | ~90s |
| **Tree-based** | Random Forest | 0.7837 | 15.2s |
| **Neural Network** | HierarchicalPredictionNet | 0.7770 | 850s |
| **Transformer** | SpectraFormer | 0.7519 | 920s |
| **KNN** | K=5, Manhattan | 0.7275 | <1s |
| **Baseline** | Ridge α=10 | 0.6970 | <1s |

### Inverse Prediction (SPEC2SMILES)

| Metric | Performance | Description |
|--------|-------------|-------------|
| **Exact Match** | 2.9% | Structure recovery (integrated) |
| **Formula Match** | 27.6% | Correct molecular formula |
| **Mean Tanimoto** | 0.420 | Structural similarity |
| **Valid SMILES** | 100% | Through SELFIES encoding |
| **Descriptor R²** | 0.653 | Mean across 12 properties |
| **Best Descriptor** | 0.874 | NumAromaticRings prediction |

### Performance by Molecular Weight Range

| Range (Da) | Cosine Similarity | Relative Performance |
|------------|-------------------|---------------------|
| 0-100 | 0.89 | Excellent |
| 100-200 | 0.82 | Good |
| 200-300 | 0.75 | Moderate |
| 300-400 | 0.61 | Challenging |
| 400-500 | 0.32 | Poor |

## Installation and Setup

### Prerequisites

- Python 3.10-3.11 (Python 3.12+ has RDKit compatibility issues)
- Conda package manager
- 16GB RAM minimum (32GB recommended for neural architectures)
- CUDA-capable GPU (optional, improves neural network training)

### Environment Setup

#### Option 1: Complete Environment (Recommended)
```bash
# Clone repository
git clone https://github.com/yusufmo1/smiles2spec-foundry.git
cd smiles2spec-foundry

# Create environment from YAML
conda env create -f environment.yml
conda activate bio729p311
```

#### Option 2: Manual Installation
```bash
# Create conda environment
conda create -n smiles2spec python=3.11
conda activate smiles2spec

# Install RDKit (CRITICAL: must use conda-forge)
conda install -c conda-forge rdkit=2025.3.2

# Install core dependencies
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.7.0
pip install torch==2.7.1 pytorch-lightning==2.5.0

# Install ML libraries
pip install lightgbm==4.6.0 catboost==1.2.2
pip install selfies==2.2.0  # For SPEC2SMILES

# Install visualisation and utilities
pip install matplotlib==3.10.3 seaborn==0.13.2 plotly==6.1.2
pip install jupyter jupyterlab tqdm joblib
```

## Running the Pipelines

### SMILES2SPEC Pipeline (Forward)

Execute notebooks in sequential order:

```bash
cd smiles2spec

# Stage 1: Data Collection
jupyter notebook 01_data_collection/01_data_conversion.ipynb

# Stage 2: Featurisation
jupyter notebook 02_featurisation/01_feature_generation.ipynb
jupyter notebook 02_featurisation/02_feature_combination.ipynb

# Stage 3: Data Exploration
jupyter notebook 03_data_exploration/01_eda.ipynb
jupyter notebook 03_data_exploration/03_linear_learning_exploration.ipynb
jupyter notebook 03_data_exploration/04_deep_learning_exploration.ipynb

# Stage 4: Feature Engineering
jupyter notebook 04_feature_engineering/01_feature_selection.ipynb
jupyter notebook 04_feature_engineering/02_dimensionality_reduction.ipynb

# Stage 5: Hyperparameter Optimisation
jupyter notebook 05_hyperparamater_optimisation/02_random_forest_HPO.ipynb

# Stage 6: Model Training
jupyter notebook 06_model_training/01_training.ipynb

# Stage 7: Evaluation
jupyter notebook 07_model_evaluation/02_model_diagnostic.ipynb
```

### SPEC2SMILES Pipeline (Inverse)

```bash
cd spec2smiles/notebooks

# Data Preparation
jupyter notebook 00_data_preperation.ipynb

# Part A: Spectrum to Descriptors
jupyter notebook 01_spectra_to_descriptors.ipynb

# Part B: Descriptors to SMILES
jupyter notebook 02_descriptors_to_smiles.ipynb

# Integrated Pipeline
jupyter notebook 03_spectra_to_smiles.ipynb
```

### Quick Model Testing

```bash
# Test best forward model
python -c "
import pickle
with open('smiles2spec/models/hpj_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(f'Model loaded: {type(model[\"model\"]).__name__}')
"

# View comprehensive results
less smiles2spec/results.md  # 1,455 lines of detailed analysis
```

## Technical Specifications

### Models Tested Summary (100+ Total)

| Category | Count | Examples |
|----------|-------|----------|
| **Baseline ML** | 18 | Linear, Ridge (3 variants), KNN (4 variants), Trees, RF, Extra Trees |
| **Neural Architectures** | 13 | ModularNet, HierarchicalPredictionNet, SparseGatedNet, RegionalExpertNet |
| **Transformers** | 3 | SpectraFormer, ViT-1D, LinearAttention-1D |
| **Dataset Size Experiments** | 10 | 10% to 100% in 10% increments |
| **Feature Selection** | 37 | Statistical (11), Model-based (9), Domain-specific (10), Ensemble (5) |
| **Dimensionality Reduction** | 23 | PCA (5), UMAP (3), Autoencoders (3), Contrastive (2), Tree-based (4) |
| **Bayesian Optimization** | 300 | 50 trials each for RF, KNN, ModularNet, HierarchicalNet, SparseGatedNet, RegionalExpertNet |
| **Production Models** | 8 | Including 2D/3D variants and ensemble methods |


### Repository Organisation

```
smiles2spec_foundry/
├── smiles2spec/                      # Forward prediction pipeline
│   ├── 01_data_collection/          # MSP → JSONL conversion (99.3% success)
│   ├── 02_featurisation/            # 7,137 features (2D) + 83 features (3D)
│   ├── 03_data_exploration/         # 18 baseline + 13 neural models
│   ├── 04_feature_engineering/      # 37 selection + 23 reduction methods
│   ├── 05_hyperparamater_optimisation/  # 300 Bayesian trials
│   ├── 06_model_training/           # Production models & ensembles
│   ├── 07_model_evaluation/         # Bootstrap CI, SHAP, benchmarking
│   ├── models/                      # 100+ trained models
│   ├── data/                        # 2,720 samples, preprocessed features
│   └── results.md                   # 1,455 lines of detailed analysis
├── spec2smiles/                      # Inverse prediction pipeline
│   ├── notebooks/                   # 4-stage implementation
│   │   ├── 00_data_preperation.ipynb
│   │   ├── 01_spectra_to_descriptors.ipynb  # 12 LGBM models
│   │   ├── 02_descriptors_to_smiles.ipynb   # Conditional VAE
│   │   └── 03_spectra_to_smiles.ipynb       # Integrated pipeline
│   └── data/models/                 # Part A (15MB) + Part B (205MB)
├── environment.yml                   # Complete conda environment
└── README.md                        # This file
```

## Academic Context

The SMILES2SPEC Foundry represents the machine learning research component of the dissertation "Integrating AI into Electronic Lab Notebooks" submitted for the MSc AI in Biosciences programme at Queen Mary University of London. This research pipeline developed and validated the models deployed in the production SMILES2SPEC application, demonstrating a complete research-to-deployment workflow. Together with the GUARDIAN pharmaceutical compliance system, these implementations showcase practical AI integration strategies for modern laboratory informatics.

## Key Scientific Contributions

1. **Systematic Model Evaluation**: Comprehensive testing of 100+ models across traditional ML, deep learning, and ensemble methods
2. **Bin-by-bin Ensemble Optimization**: Novel approach achieving 0.8164 cosine similarity through m/z-specific weighting
3. **Two-Stage Inverse Architecture**: Decomposition of spectrum-to-structure problem into interpretable stages
4. **Statistical Validation**: Bootstrap confidence intervals (1,000 iterations) for all performance metrics
5. **Performance Analysis**: Detailed characterization of performance degradation by molecular weight
6. **Production Implementation**: Throughput optimization achieving 3,152 samples/second

## Citation

For academic use of this work, please cite:

```bibtex
@mastersthesis{mohammed2025foundry,
  title = {SMILES2SPEC Foundry: Bidirectional Mass Spectrometry Prediction Pipeline},
  author = {Mohammed, Yusuf},
  year = {2025},
  school = {Queen Mary University of London},
  department = {MSc AI in Biosciences},
  supervisor = {Elbadawi, Mohammed},
  note = {MSc Dissertation: Integrating AI into Electronic Lab Notebooks}
}
```

---

<div align="center">

[![Author](https://img.shields.io/badge/Author-Yusuf%20Mohammed-blue?style=flat-square&logo=github)](https://github.com/yusufmo1)
[![Supervisor](https://img.shields.io/badge/Supervisor-Dr%20Mohammed%20Elbadawi-green?style=flat-square&logo=github)](https://github.com/Dr-M-ELBA)
[![Institution](https://img.shields.io/badge/Institution-QMUL-003E74?style=flat-square)](https://www.qmul.ac.uk/)
[![Programme](https://img.shields.io/badge/Programme-MSc%20AI%20in%20Biosciences-purple?style=flat-square)](https://www.qmul.ac.uk/)

</div>

*Developed as part of MSc AI in Biosciences dissertation at Queen Mary University of London (2025)*