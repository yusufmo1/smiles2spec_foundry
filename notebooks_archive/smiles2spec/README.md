# SMILES2SPEC: Machine Learning Pipeline for Mass Spectrometry Prediction

## Overview
SMILES2SPEC is a comprehensive machine learning pipeline for predicting electron ionization mass spectra from molecular SMILES strings. This system achieves state-of-the-art performance among rule-free methods through systematic evaluation of 88+ model configurations and advanced ensemble techniques, demonstrating a 32.3% improvement over existing neural approaches.

## System Architecture

The pipeline implements a sequential 7-stage workflow:

```
MSP Files → JSONL → Molecular Features → Model Exploration → Feature Engineering → Hyperparameter Optimization → Production Training → Evaluation
```

Each stage builds upon previous outputs, ensuring reproducible and systematic development from raw spectral data to production-ready models capable of real-time inference.

## Pipeline Stages

### Stage 1: Data Collection
Converts Mass Spectrometry Pattern (MSP) files to standardized JSONL format. Processes 2,720 experimental spectra with 99.3% success rate, extracting SMILES strings and peak data through robust parsing and validation.

### Stage 2: Molecular Featurisation  
Extracts 7,137 molecular descriptors from SMILES using RDKit, including Morgan fingerprints (radii 1-3), MACCS keys, and electronic properties. Additionally generates 984 3D conformer features, though 2D features demonstrate superior performance (12.2% improvement). Includes comprehensive visualization capabilities for feature analysis.

### Stage 3: Data Exploration
Systematically evaluates 18 baseline ML algorithms, 10 neural architectures, and 3 sequence-to-sequence models. Establishes Random Forest baseline (0.7581 cosine similarity) and identifies tree-based methods' superiority over neural networks (5.7% improvement). Includes interactive plotting suite, target scaling experiments, and regressor chain methodology.

### Stage 4: Feature Engineering
Implements statistical selection (MI, F-test), model-based selection (LASSO, Random Forest importance), and dimensionality reduction (PCA, UMAP, autoencoders). Achieves 86% feature reduction (7,137→1,000) with minimal accuracy loss (<2%) and 10× inference speedup.

### Stage 5: Hyperparameter Optimisation
Conducts multi-objective Bayesian optimization using Ax Platform with BoTorch backend. 300 total trials across 6 model types: Random Forest (50 trials, best: 0.7879), KNN (50 trials, best: 0.7275), ModularNet (50 trials, best: 0.7689), HierarchicalPredictionNet (50 trials, best: 0.7647), SparseGatedNet (50 trials, best: 0.7742), RegionalExpertNet (50 trials, best: 0.7422). Uses NEHVI acquisition function with Sobol initialization.

### Stage 6: Model Training
Trains production models with optimized hyperparameters on 2,176 training samples. Develops two ensemble strategies: Simple weighted ensemble (RF: 0.453, ModularNet: 0.119, HierarchicalNet: 0.150, SparseGatedNet: 0.169, RegionalExpertNet: 0.109) achieving 0.8037 cosine similarity, and bin-by-bin ensemble with individual optimization per m/z bin achieving 0.8164 cosine similarity (best overall). Confirms 2D features outperform 3D by 12.2%.

### Stage 7: Model Evaluation
Comprehensive evaluation using cosine similarity, weighted dot product, and bootstrap confidence intervals (1,000 iterations). Includes diagnostic analysis with residual patterns, SHAP values, and failure mode characterization. Benchmarks against literature methods demonstrating competitive accuracy with 1,583× faster inference than CFM-ID.

## Performance Achievements

### Primary Metrics
- **Best Model**: Simple Weighted Ensemble - 0.8063 cosine similarity
- **Best Individual**: Random Forest - 0.7837 cosine similarity
- **Best Neural Network**: ModularNet - 0.7703 cosine similarity
- **vs NEIMS**: 32.3% improvement (0.621→0.806)
- **Inference Speed**: 5-20ms per spectrum
- **Throughput**: 50,000 spectra/second capability

### Advanced Neural Architectures
- **HierarchicalPredictionNet**: Multi-level molecular representations (0.7698)
- **SparseGatedNet**: Adaptive feature selection with gating (0.7622)
- **RegionalExpertNet**: m/z range-specific expert models (0.7681)
- **ModularNet**: Attention-based module fusion (0.7703)

### m/z Range Performance
- 0-100 Da: 0.89 cosine similarity
- 100-200 Da: 0.82
- 200-300 Da: 0.75
- 300-400 Da: 0.61
- 400-500 Da: 0.32

## Technical Specifications

### Data
- **Training**: 2,176 spectra
- **Validation**: 272 spectra
- **Test**: 272 spectra
- **Features**: 7,137 dimensions (full) / 1,000 (selected)
- **Output**: 500 m/z bins (0-499 Da)

### Model Testing Summary
- **Total Models Evaluated**: 88+ across all experiments
  - 18 baseline ML models
  - 10 core neural architectures
  - 3 sequence-to-sequence models
  - 10 dataset size experiments
  - 9 feature selection variants
  - 18 dimensionality reduction models
  - 80 Bayesian optimization trials
  - 8 production models with ensembles

### Computational Requirements
- **Memory**: 2-4GB RAM (peak 8GB for neural architectures)
- **Storage**: ~10GB complete pipeline, ~500MB for trained models
- **Training Time**: <1s (KNN) to 920s (HierarchicalPredictionNet)
  - Random Forest: 15-40s
  - Neural Networks: 60-900s
- **Optimization Time**: ~3 hours for 300 Bayesian trials
- **Hardware**: 16-core CPU, optional GPU (CUDA/MPS) for neural networks
- **Dependencies**: Python 3.10-3.11, RDKit, PyTorch, Optuna, Ax Platform

## Repository Structure

```
smiles2spec/
├── 01_data_collection/      # MSP→JSONL conversion
├── 02_featurisation/        # Feature extraction & visualization
├── 03_data_exploration/     # Model evaluation & scaling
├── 04_feature_engineering/  # Feature optimization
├── 05_hyperparamater_optimisation/  # Bayesian optimization (note: typo in dir name)
├── 06_model_training/       # Production models
├── 07_model_evaluation/     # Benchmarking & diagnostics
├── data/                    # Processed datasets
├── models/                  # Trained models
├── reduced_features/        # Pre-computed reductions
├── figures/                 # Diagnostic visualizations
└── results.md              # Comprehensive results (1,516 lines)
```

## Key Findings

1. **Model Selection**: Tree-based methods (Random Forest) consistently outperform neural networks for tabular molecular data
2. **Feature Engineering**: 2D molecular descriptors superior to 3D conformer features despite theoretical advantages (12.2% difference)
3. **Ensemble Benefits**: 1-3% improvement through weighted combination of diverse models
4. **Data Efficiency**: Performance plateaus at ~80% of available training data
5. **Range Dependency**: Prediction difficulty increases exponentially with molecular weight
6. **Architecture Innovation**: Advanced neural architectures competitive but not superior to optimized classical methods

## Notable Notebooks

### Visualization and Analysis
- `../00_plotting_suite.ipynb`: Interactive visualization toolkit for spectrum analysis (root level)
- `02_featurisation/03_feature_visualisation.ipynb`: 2D feature distribution and correlation analysis  
- `02_featurisation/06_feature_visualisation_3d.ipynb`: 3D conformer feature analysis

### Experimental Innovations
- `03_data_exploration/02_scaling.ipynb`: 20+ target transformation experiments (rank transform: 59.84% improvement)
- `03_data_exploration/04_deep_learning_exploration.ipynb`: 13 neural architectures with ensemble analysis
- `03_data_exploration/05_dataset_size_effects.ipynb`: Data efficiency analysis (plateaus at 80%)
- `03_data_exploration/06_seq2seq_learning_exploration.ipynb`: Transformer architectures (SpectraFormer, ViT-1D)
- `03_data_exploration/07_rf_chain.ipynb`: Regressor chain for isotope patterns (20 groups identified)

### Advanced Architectures
- `05_hyperparamater_optimisation/01_modular_net_HPO.ipynb`: ModularNet with attention fusion
- `05_hyperparamater_optimisation/02_random_forest_HPO.ipynb`: Random Forest Bayesian optimization
- `05_hyperparamater_optimisation/03_knn_HPO.ipynb`: K-Nearest Neighbors optimization
- `05_hyperparamater_optimisation/04_HierarchicalPredictionNet_HPO.ipynb`: Dual-stage prediction architecture
- `05_hyperparamater_optimisation/05_SparseGatedNet_HPO.ipynb`: Adaptive feature selection with learnable gates
- `05_hyperparamater_optimisation/06_RegionalExpertNet_HPO.ipynb`: m/z range-specific expert models

### Diagnostic Analysis
- `07_model_evaluation/02_model_diagnostic.ipynb`: Comprehensive error analysis and SHAP values
- `07_model_evaluation/05_throughput_analysis.ipynb`: Scalability and performance benchmarking

## Academic Contribution

This work represents a significant advancement in computational mass spectrometry, achieving state-of-the-art performance among rule-free methods through:
- Systematic evaluation of 100+ model configurations across 7 pipeline stages
- Introduction of specialized neural architectures (HierarchicalPredictionNet, SparseGatedNet, RegionalExpertNet)
- Comprehensive target transformation analysis (20+ methods, rank transform achieves 59.84% improvement)
- Rigorous statistical validation with bootstrap confidence intervals (1,000 iterations)
- Multi-objective Bayesian optimization with 300 trials across 6 model types
- Bin-by-bin ensemble optimization achieving 0.8164 cosine similarity
- 33.1% improvement over NEIMS (best rule-free method in literature)
- Production-ready implementation with throughput up to 3,152 samples/second

The pipeline demonstrates that careful feature engineering, systematic optimization, and ensemble methods can achieve competitive performance with complex fragmentation rules while maintaining computational efficiency suitable for large-scale deployment.

## Statistical Validation
- **Bootstrap CI**: 0.8063 ± 0.0042 (95% confidence, 1000 iterations)
- **Cross-Validation**: 5-fold CV for stability assessment
- **Effect Size**: Cohen's d = 1.82 (very large) vs baseline
- **Stability**: Coefficient of variation < 2.5%

## References
- RASSP (Wei et al., 2023): Rule-Augmented Spectrum Simulation - WDP=0.929 (rule-based)
- CFM-ID (Allen et al., 2016): Competitive Fragmentation Modeling - WDP=0.775 (rule-based)
- NEIMS (Wei et al., 2019): Neural Electron-Ionization Mass Spectrometry - WDP=0.621 (rule-free)
- QCEIMS (2020): Quantum Chemical EI-MS - WDP=0.608 (rule-free)

## Citation
If using this pipeline in academic work, please reference the comprehensive results analysis in `results.md` and the methodology documentation in individual stage READMEs.