# Stage 4: Feature Engineering

## Overview
This stage implements comprehensive feature selection and dimensionality reduction techniques to optimize the 7,137-dimensional feature space for improved model performance and computational efficiency. The implementation includes parallel processing optimizations and extensive evaluation with standard visualization suites.

## Methodology

### 1. Feature Selection Methods (`01_feature_selection.ipynb`)

#### Statistical Methods
- **Mutual Information**: Non-linear dependency measurement with parallel computation (n_neighbors=3)
- **F-test Regression**: Linear dependency scoring for target correlation
- **Variance Threshold**: Remove low-variance features (threshold: 0.01)
- **Correlation Filtering**: Remove redundant features (r > 0.95)

#### Model-Based Selection
- **Random Forest Importance**: Mean decrease in impurity with 100 estimators
- **LASSO (L1) Regularization**: Sparse linear model with 5-fold CV optimization
- **Recursive Feature Elimination (RFE)**: Iterative removal with 10-20% step size
  - Optimized with subset sampling for computational efficiency

#### Domain-Specific Methods
- **Fingerprint Type Analysis**: 
  - Morgan (3072 features), MACCS (166 features), RDKit (2048 features)
  - Descriptor features (1849 features)
  - Combined fingerprint evaluations
- **Spectral Range Correlation**: Features predictive of specific m/z ranges
  - Low m/z (0-200): Small fragments
  - Mid m/z (200-350): Medium fragments  
  - High m/z (350-500): Large fragments/molecular ion

#### Ensemble Selection
- **Voting-Based Selection**: Features selected by multiple methods (2+, 3+, 5+ votes)
- **Stability Analysis**: Bootstrap consistency with Jaccard index (10 iterations)
- **Stable Feature Identification**: Features appearing in 80% of bootstrap samples

### 2. Dimensionality Reduction Techniques (`02_dimensionality_reduction.ipynb`)

#### Classical Methods
- **PCA**: Principal Component Analysis
  - Components tested: [50, 100, 200, 500, 1000]
  - Explained variance: 54.7% (50 comp) to 98.5% (1000 comp)
- **UMAP**: Uniform Manifold Approximation and Projection
  - Components: [50, 100, 200]
  - Parameters: n_neighbors=15, min_dist=0.1

#### Tree-based Feature Learning
- **Random Forest Leaf Encoding**: 
  - Extract leaf indices from 100 trees (max_depth=15)
  - One-hot encoding → 45,100 sparse features
  - PCA post-processing: [100, 200, 500] components

#### Deep Learning Approaches

##### Standard Autoencoder
- **Architectures**: 
  - ae_50: [4096, 2048, 512, 50] latent dimensions
  - ae_100: [4096, 2048, 1024, 512, 100] latent dimensions
  - ae_200: [4096, 2048, 1024, 512, 200] latent dimensions
- **Training**: 50 epochs, batch_size=128, Adam optimizer (lr=1e-3)

##### Contrastive Learning
- **Dual Encoder Architecture**: Feature and target encoders with projection heads
- **NT-Xent Loss**: Temperature=0.1 for contrastive alignment
- **Latent dimensions**: [128, 256] with 512D embedding output
- **Training**: 30 contrastive epochs + 50 regression epochs

##### Supervised Dual Autoencoder
- **Architecture**: Joint encoding of features and targets in shared latent space
- **Multi-objective Loss**:
  - Feature reconstruction (α=2.0)
  - Target reconstruction from features (α=3.0)
  - Latent space alignment (α=1.0)
  - Cosine similarity optimization (α=2.0)
- **Shared dimensions**: [64, 128, 256]
- **Training**: 60 epochs with early stopping (patience=15)

#### Hybrid Approaches
- **RF Leaves + UMAP**: Tree features with manifold learning (50D)
- **Contrastive + PCA**: Learned representations with classical reduction (100D)
- **PCA + Autoencoder**: Concatenated classical and deep features (250D)
- **Dual AE + PCA**: Supervised embeddings with PCA features (356D)

## Evaluation Framework

### Metrics
- **Primary**: Cosine similarity between predicted and actual spectra
- **Secondary**: MSE, R², MAE
- **Peak Detection**: Precision, Recall, F1-score (threshold=0.01)
- **Range-Specific**: Performance by m/z ranges with uncertainty quantification

### Fixed Random Forest Configuration
All methods evaluated with consistent RF hyperparameters:
- n_estimators: 100
- max_depth: 25
- min_samples_split: 3
- min_samples_leaf: 1

### Standard Visualizations
- **2x2 Diagnostic Plots**: 
  1. Cosine similarity distribution
  2. Performance by m/z range with uncertainties
  3. Intensity correlation with violin density
  4. Peak count comparison
- **Random Spectrum Examples**: 2x2 grid with molecular structures

## Key Results

### Feature Selection Performance
- **Best Method**: Voting_2+ (6042 features)
  - Cosine Similarity: 0.7805
  - 0.14% improvement over baseline
  - Near-zero compression (0.1% reduction)
- **Most Efficient**: MI_1000 (1000 features)
  - Cosine Similarity: 0.7672
  - 86% feature reduction
  - Training time: 31.0s

### Dimensionality Reduction Performance
- **Best Classical**: PCA-100
  - Cosine Similarity: 0.7364
  - 98.6% compression (71.3× smaller)
  - Explained variance: 65.7%
- **Best Tree-based**: RF-Leaves-PCA500
  - Cosine Similarity: 0.7574
  - 93% compression (14.3× smaller)
- **Best Deep Learning**: DualAE-256
  - Cosine Similarity: 0.7554
  - 96.4% compression (27.9× smaller)
- **Best Hybrid**: Contrastive+PCA100
  - Cosine Similarity: 0.7534
  - 98.6% compression (71.3× smaller)

### Performance by m/z Range
- **0-100 Da**: Excellent (>0.85 cosine similarity)
- **100-200 Da**: Good (0.75-0.85)
- **200-300 Da**: Moderate (0.65-0.75)
- **300+ Da**: Poor (<0.65)

## Computational Impact

### Feature Selection
- **Parallel Processing**: Threading backend for macOS compatibility
- **Training Time Reduction**: 33.6s → 14.8s (56% reduction with F-test_50)
- **Memory Efficiency**: Correlation matrix computed in chunks

### Dimensionality Reduction
- **Compression Ratios**: Up to 142.7× data size reduction (PCA-50, UMAP-50)
- **Training Speed**: PCA-50 (4.9s) vs Baseline (29.7s) - 83% faster
- **GPU Acceleration**: Available for deep learning methods (CUDA/MPS)

## Output Files

### Feature Selection
- `../data/results/selected_features/`:
  - `{method}_selection_results.pkl`: Complete results with metrics
  - `feature_selection_summary.pkl`: Comparative analysis
- `../data/results/{dataset}/selected_{method}/`:
  - `{train,val,test}_data.jsonl`: Selected feature datasets
  - `feature_selection_info.json`: Selection metadata

### Dimensionality Reduction  
- `../reduced_features/`:
  - `{method}_{dim}.pkl`: Reduced features with models
  - `{method}_{dim}_{train,val,test}.jsonl`: Split datasets
  - `results_summary.pkl`: Performance comparison

## Technical Configuration

### Master Configuration (Feature Selection)
```python
{
    'selection': {
        'target_features': [50, 100, 200, 500, 1000],
        'variance_threshold': 0.01,
        'correlation_threshold': 0.95,
        'stability_bootstraps': 10
    },
    'rf_eval': {
        'n_estimators': 100,
        'max_depth': 25,
        'min_samples_split': 3,
        'min_samples_leaf': 1
    }
}
```

### Master Configuration (Dimensionality Reduction)
```python
{
    'pca': {'n_components': [50, 100, 200, 500, 1000]},
    'umap': {'n_components': [50, 100, 200], 'n_neighbors': 15},
    'autoencoder': {'epochs': 50, 'batch_size': 128, 'lr': 1e-3},
    'contrastive': {'temperature': 0.1, 'epochs': 30},
    'dual_ae': {'epochs': 60, 'patience': 15}
}
```

## Reference
- **Notebooks**: 
  - `01_feature_selection.ipynb`: Comprehensive feature selection with parallel evaluation
  - `02_dimensionality_reduction.ipynb`: Classical and deep learning reduction methods
- **Dependencies**: scikit-learn, XGBoost, PyTorch, UMAP, joblib
- **Random Seed**: 42 (consistent throughout)