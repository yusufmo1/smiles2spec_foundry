# Stage 5: Hyperparameter Optimisation

## Overview
This stage implements multi-objective Bayesian optimization to systematically identify optimal hyperparameters for both classical machine learning models and advanced neural architectures, conducting 300 optimization trials across 6 different model types using state-of-the-art acquisition functions.

## Methodology

### Optimization Framework
- **Method**: Multi-Objective Bayesian Optimization
- **Library**: Ax Platform with BoTorch backend
- **Objectives**: 
  - Maximize cosine similarity (primary metric)
  - Minimize RMSE (secondary metric)
- **Acquisition Function**: Noisy Expected Hypervolume Improvement (NEHVI)
- **Initialization**: Sobol sequence sampling (14-36 trials)
- **Validation Strategy**: Subsampled validation for speed, full validation for final evaluation

### Classical Models Optimized

#### Random Forest
- **Search Space**: 7-dimensional
- **Parameters**: n_estimators (50-500), max_depth (10-50), min_samples_split (2-20)
- **Feature Selection**: max_features (0.1-1.0)
- **Bootstrap**: True/False with oob_score
- **Trials**: 50 (14 Sobol + 36 Bayesian)
- **Optimization Time**: 35.4 minutes
- **Best Configuration**: 290 estimators, depth 25, min_split 2, max_features 0.3
- **Best Validation**: 0.7999 cosine, 0.1062 RMSE
- **Test Performance**: 0.7879 cosine, 0.0658 RMSE

#### XGBoost (Optimized in Training Stage)
- **Note**: XGBoost hyperparameter optimization is performed during the training stage rather than as a separate optimization notebook
- **Search Space**: 15-dimensional
- **Parameters**: n_estimators (50-500), learning_rate (0.01-0.3)
- **Tree Parameters**: max_depth (3-10), subsample (0.5-1.0)
- **Regularization**: reg_alpha (0-1), reg_lambda (0-1)
- **Boosting**: DART vs GBTREE methods
- **Best Configuration**: 350 estimators, lr 0.18, depth 7

#### K-Nearest Neighbors (KNN)
- **Search Space**: 6-dimensional
- **Parameters**: n_neighbors (3-50), weights (uniform/distance)
- **Distance Metrics**: manhattan, euclidean, minkowski, cosine
- **Algorithm**: auto, ball_tree, kd_tree, brute
- **Trials**: 50 (12 Sobol + 38 Bayesian)
- **Optimization Time**: 5.9 minutes
- **Best Configuration**: k=6, distance weighting, manhattan metric, brute algorithm
- **Best Validation**: 0.7231 cosine, 0.0839 RMSE
- **Test Performance**: 0.7275 cosine, 0.0721 RMSE

### Advanced Neural Architectures

#### ModularNet
- **Search Space**: 13-dimensional
- **Parameters**: Number of modules (1-6), layers (1-3), hidden units (32-512)
- **Regularization**: Dropout (0.0-0.5), weight decay (0.0-0.01)
- **Training**: Learning rate (1e-5 to 1e-2), batch size (8-128)
- **Trials**: 50 (14 Sobol + 36 Bayesian)
- **Optimization Time**: 71.5 minutes
- **Best Configuration**: 5 modules, 2 layers, 393 units, ReLU activation, AdamW optimizer
- **Best Validation**: 0.7894 cosine, 0.1098 RMSE
- **Test Performance**: 0.7689 cosine, 0.0704 RMSE
- **Model Size**: 18.7M parameters

#### HierarchicalPredictionNet
- **Architecture**: Dual-stage prediction (presence + intensity)
- **Search Space**: 15-dimensional with hierarchical components
- **Specialized Parameters**: 
  - Presence network: 1-4 layers, 32-512 hidden units
  - Intensity network: 1-5 layers, 64-1024 hidden units
  - Conditional dropout, calibration layers
- **Key Innovation**: Separates peak presence from intensity prediction
- **Trials**: 50 (30 Sobol + 20 Bayesian)
- **Optimization Time**: 24.9 minutes
- **Best Configuration**: 4 presence layers (32 units), 1 intensity layer (119 units)
- **Best Validation**: 0.7121 cosine, 0.1133 RMSE
- **Test Performance**: 0.7647 cosine, 0.0702 RMSE, 86.97% presence accuracy
- **Model Size**: 1.5M parameters

#### SparseGatedNet
- **Architecture**: Adaptive feature selection with learnable gates
- **Search Space**: 18-dimensional with gating controls
- **Specialized Parameters**: 
  - Gate temperature (0.1-10.0)
  - Sparsity threshold (0.0-0.1)
  - Gate hidden units (32-256)
  - Zero/non-zero weights for balanced prediction
- **Key Innovation**: Dynamic feature importance learning
- **Trials**: 50 (36 Sobol + 14 Bayesian)
- **Optimization Time**: 15.9 minutes
- **Best Configuration**: 1 layer, 441 hidden, gate_hidden 102, temperature 4.04
- **Best Validation**: 0.7528 cosine, 0.1139 RMSE
- **Test Performance**: 0.7742 cosine, 0.0718 RMSE
- **Model Size**: 4.3M parameters

#### RegionalExpertNet
- **Architecture**: m/z range-specific expert models with router
- **Search Space**: 21-dimensional with regional specialization
- **Specialized Parameters**: 
  - Number of regions (2-10)
  - Region size (50-500 bins)
  - Expert layers (1-4), hidden units (32-512)
  - Router temperature and overlap bins
- **Key Innovation**: Learns specialized models for different m/z ranges
- **Trials**: 50 (42 Sobol + 8 Bayesian)
- **Optimization Time**: 18.5 minutes
- **Best Configuration**: 2 regions, 252 bin size, 3 expert layers, 227 hidden units
- **Best Validation**: 0.7638 cosine, 0.1095 RMSE
- **Test Performance**: 0.7422 cosine, 0.0732 RMSE
- **Model Size**: 4.3M parameters


## Key Results

### Optimal Configurations Summary (Test Set Performance)
| Model | Cosine Similarity | RMSE | Optimization Time | Training Time | Parameters |
|-------|------------------|------|-------------------|---------------|------------|
| **Random Forest** | 0.7879 | 0.0658 | 35.4 min | 40.4s | 290 trees |
| **SparseGatedNet** | 0.7742 | 0.0718 | 15.9 min | 75.8s | 4.3M |
| **ModularNet** | 0.7689 | 0.0704 | 71.5 min | 721.3s | 18.7M |
| **HierarchicalPredictionNet** | 0.7647 | 0.0702 | 24.9 min | 396.3s | 1.5M |
| **RegionalExpertNet** | 0.7422 | 0.0732 | 18.5 min | 233.0s | 4.3M |
| **KNN** | 0.7275 | 0.0721 | 5.9 min | <1s | N/A |


### Pareto Frontier Analysis
Each optimization identified Pareto-optimal solutions balancing cosine similarity and RMSE:
- **ModularNet**: 3 Pareto solutions found
- **Random Forest**: 2 Pareto solutions found  
- **KNN**: 5 Pareto solutions found
- **HierarchicalPredictionNet**: 4 Pareto solutions found
- **SparseGatedNet**: 4 Pareto solutions found
- **RegionalExpertNet**: 4 Pareto solutions found

### Optimization Insights
- **Multi-Objective Benefits**: NEHVI acquisition effectively balances accuracy metrics
- **Initialization Impact**: Sobol initialization (14-42 trials) crucial for exploration
- **Parallel Evaluation**: 8-12 parallel workers significantly reduce wall time
- **Architecture Matters**: Simpler architectures often outperform complex ones
- **Classical vs Neural**: Random Forest achieves best neural network performance with 100× faster inference

## Computational Requirements
- **Total Trials**: 300 trials (50 per model × 6 models)
- **Total Optimization Time**: ~3 hours cumulative
  - Classical models: ~41 minutes (RF + KNN)
  - Neural networks: ~150 minutes (4 architectures)
- **Hardware**: 16-core CPU (Apple M-series or x86-64)
- **Memory Usage**: Peak 8GB for neural architectures
- **GPU Support**: Optional (MPS on Apple Silicon, CUDA on NVIDIA)

## Output Files
- `../models/bayesian_optimized/`: Optimized model checkpoints (.pth, .pkl)
- `../models/hierarchical_bayesian_optimized/`: HierarchicalPredictionNet models
- `../models/bayesian_sparse_gated/`: SparseGatedNet models
- `../models/knn_bayesian_optimized/`: KNN models
- `*_optimization_results.pkl`: Complete trial histories with Pareto frontiers
- `*_optimization_results.json`: Human-readable optimization summaries
- `*_optimization_report.txt`: Detailed text reports
- `*_optimization_analysis.png`: Convergence and Pareto visualizations

## Reference
Notebooks:
- `01_modular_net_HPO.ipynb` - ModularNet optimization
- `02_random_forest_HPO.ipynb` - Random Forest optimization  
- `03_knn_HPO.ipynb` - K-Nearest Neighbors optimization
- `04_HierarchicalPredictionNet_HPO.ipynb` - Hierarchical architecture
- `05_SparseGatedNet_HPO.ipynb` - Sparse gating mechanisms
- `06_RegionalExpertNet_HPO.ipynb` - Regional expert models

