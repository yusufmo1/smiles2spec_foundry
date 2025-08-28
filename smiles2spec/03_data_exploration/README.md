# Stage 3: Data Exploration

## Overview
This stage implements systematic evaluation of baseline methods and neural architectures, establishing performance benchmarks for mass spectrometry prediction. Comprehensive analysis across 18+ baseline models, 10 neural architectures, and advanced target transformation techniques.

## Dataset Statistics
- **Total Samples**: 2,720 molecules with experimental mass spectra
- **Data Split**: Train (2,176, 80%), Validation (272, 10%), Test (272, 10%)
- **Feature Dimensions**: 7,137 molecular descriptors and fingerprints
- **Target Dimensions**: 500 m/z points (binned spectra)

## Methodology

### Exploratory Data Analysis (01_eda.ipynb)
- **Target Sparsity**: 74.4% zeros (typical for mass spectra)
- **Dynamic Range**: 1.0e+03 (3 orders of magnitude)
- **Mean Peaks per Spectrum**: 6.4 ± 10.0
- **Mean Spectral Entropy**: 4.65 ± 1.16
- **Feature Quality**: 901 highly correlated pairs, 0 zero-variance features
- **Data Integrity**: Zero molecular overlap between splits confirmed (no leakage)

### Molecular Properties
- **Molecular Weight**: 284-289 Da average across splits
- **LogP**: 3.21-3.31 (moderately lipophilic)
- **TPSA**: 40-43 Ų (good oral bioavailability)
- **QED Score**: 0.654-0.662 (drug-like)
- **Heavy Atoms**: 20.3 ± 6.5 (expect 41-61 possible fragments)
- **Aromatic Rings**: 1.6 ± 1.1 (stable aromatic fragments likely)
- **Veber Compliance**: 97-98% (good oral bioavailability)

### Target Transformation Analysis (02_scaling.ipynb)
Systematic evaluation of 20+ transformation methods to optimize cosine similarity:

#### Top Transformations (Ranked by Performance)
1. **Rank Transform**: 0.9753 cosine (59.84% improvement) - Best overall
2. **Power Optimal (α=0.100)**: 0.9178 cosine - Best power transform
3. **Fourth Root (α=0.250)**: 0.8588 cosine
4. **Yeo-Johnson**: 0.8578 cosine - Best statistical
5. **Cubic Root (α=0.333)**: 0.8377 cosine

#### Categories Tested
- **Standard**: Power, logarithmic, asinh transformations
- **Statistical**: Box-Cox, Yeo-Johnson, quantile, rank
- **Custom**: Adaptive intensity-based, m/z-dependent

#### Key Findings
- Baseline (no transform): 0.610 cosine similarity
- Best improvement: 59.84% with rank transformation
- Power transforms with α < 0.25 show best balance
- Statistical methods outperform standard transformations

### Visualization Suite
- **Interactive Plotting Tools**: Comprehensive visualization library (00_plotting_suite.ipynb)
- **Spectrum Grid Visualizations**: Multi-panel comparisons of predictions
- **Error Distribution Plots**: Regional and molecular property-based analysis
- **Transformation Effects**: Visual comparison of different scaling methods

### Baseline Methods (03_linear_learning_exploration.ipynb)

#### All 18 Models Ranked by Performance

| Rank | Model | Cosine Similarity | MSE | Peak F1 | Training Time |
|------|-------|------------------|-----|---------|---------------|
| 1 | **Random Forest (n=50)** | **0.758** | 0.0047 | 0.656 | 15.9s |
| 2 | Extra Trees (n=50) | 0.726 | 0.0055 | 0.710 | 19.5s |
| 3 | KNN Regression (k=5) | 0.714 | 0.0054 | 0.756 | 0.003s |
| 4 | KNN Regression (k=10) | 0.713 | 0.0054 | 0.735 | 0.003s |
| 5 | KNN (k=10, uniform) | 0.703 | 0.0055 | 0.732 | 0.003s |
| 6 | KNN Regression (k=20) | 0.698 | 0.0055 | 0.715 | 0.003s |
| 7 | Ridge (α=10.0) | 0.697 | 0.0091 | 0.596 | 0.53s |
| 8 | Decision Tree (depth=10) | 0.674 | 0.0063 | 0.669 | 3.4s |
| 9 | Decision Tree (depth=20) | 0.670 | 0.0071 | 0.745 | 4.6s |
| 10 | Ridge (α=1.0) | 0.637 | 0.0151 | 0.564 | 0.46s |
| 11 | Ridge (α=0.1) | 0.572 | 0.0267 | 0.543 | 0.37s |
| 12 | Radius Neighbors | 0.561 | 0.0083 | 0.532 | 0.003s |
| 13 | Mean Spectrum | 0.549 | 0.0085 | 0.525 | 0.0002s |
| 14 | Frequency-Weighted | 0.535 | 0.0090 | 0.628 | 0.001s |
| 15 | Median Spectrum | 0.495 | 0.0096 | 0.629 | 0.006s |
| 16 | Random Noise | 0.299 | 0.306 | 0.417 | 0.0002s |
| 17 | Linear Regression (OLS) | 0.268 | 3.31 | 0.454 | 25.0s |
| 18 | Zero Predictor | 0.000 | 0.012 | 0.000 | 0.0s |

#### Performance by Category
- **Tree-based**: 0.707 mean cosine (best category)
- **Neighbors**: 0.678 mean cosine
- **Linear**: 0.544 mean cosine
- **Statistical**: 0.376 mean cosine

### Neural Architectures (04_deep_learning_exploration.ipynb)

#### 13 Models Evaluated (GPU-accelerated with CUDA)

##### Traditional Architectures (9 models)
1. **SimpleMLP**: Classical deep network - 0.7575 cosine, 17.5M params
2. **ResidualNet**: Skip connections - 0.6970 cosine, 16.2M params
3. **AttentionNet**: Self-attention mechanisms - 0.7332 cosine, 7.1M params
4. **DenseNet**: Dense connectivity - 0.7370 cosine, 1.2M params
5. **DeepSetNet**: Set-based architecture - 0.7411 cosine, 5.0M params
6. **HybridNet**: Mixed architecture - 0.7411 cosine, 7.1M params
7. **GatedNet**: Highway gating - 0.7587 cosine, 10.1M params
8. **EnsembleNet**: Multiple sub-networks - 0.7627 cosine, 11.7M params
9. **ModularNet**: Specialized modules - 0.7751 cosine, 9.5M params (best traditional)

##### Specialized Architectures (4 models)
1. **MultiHeadAttentionNet**: Multi-head attention - 0.6992 cosine, 13.4M params
2. **SparseGatedNet**: Sparse gating mechanisms - 0.7719 cosine, 11.4M params
3. **RegionalExpertNet**: m/z region experts - 0.7662 cosine, 22.7M params
4. **HierarchicalPredictionNet**: Hierarchical prediction - 0.7768 cosine, 8.3M params (best individual)

### Transformer Architectures (06_seq2seq_learning_exploration.ipynb)

#### Models Evaluated (3 architectures)
| Architecture | Type | Val Cosine | Test Cosine | Test MSE | Peak F1 | Parameters |
|--------------|------|------------|-------------|----------|---------|------------|
| **SpectraFormer** | Transformer | 0.7653 | 0.7519 | 0.0061 | 0.6830 | 13.9M |
| **ViT-1D** | Transformer | 0.7207 | 0.7217 | 0.0076 | 0.7443 | 28.9M |
| **LinearAttention-1D** | Transformer | 0.7195 | 0.7181 | 0.0087 | 0.6696 | 14.8M |

#### Ablation Studies

**Depth Ablation (SpectraFormer)**:
- Depth 4: 0.7031 val cosine, 0.7043 test cosine
- Depth 6: 0.7038 val cosine, 0.6986 test cosine  
- Depth 8: 0.7077 val cosine, 0.7054 test cosine (best)

**Patch Size Ablation (ViT-1D)**:
- Patch size 1: 0.5552 val cosine, 0.5648 test cosine
- Patch size 5: 0.5420 val cosine, 0.5510 test cosine
- Patch size 10: 0.5996 val cosine, 0.6069 test cosine (best)

#### Key Findings
- **Best Transformer**: SpectraFormer (0.7519 test cosine)
- **Gap to Baseline**: 0.0544 behind target (0.8063)
- **High m/z Performance**: Poor (0.425 cosine for SpectraFormer)
- **Training**: 100 epochs with early stopping, Adam optimizer

### Additional Experiments
- **Power transformations**: λ = 0.1, 0.2, 0.5 for variance stabilization
- **Regional specialization**: m/z range-specific model components
- **Reversible scaling**: Maintaining physical interpretability

### Target Transformation Experiments
- **Power Transformations**: λ = 0.1, 0.2, 0.5 for variance stabilization
- **Reversible Scaling**: Maintaining physical interpretability
- **Regional Analysis**: m/z range-specific transformations
- **Best Configuration**: Power transform λ=0.1 improves neural network convergence

### Regressor Chain Analysis (07_rf_chain.ipynb)

#### Isotope Pattern Discovery
- **20 isotope groups identified** through correlation analysis
- **Strongest correlations**: Bins 494-495 (r=1.00), 481-482 (r=1.00), 466-467 (r=1.00)
- **Largest group**: Bins 100-424 (325 connected bins)
- **Mean absolute correlation**: 0.0757 across all m/z bins

#### Model Comparison
| Model | Cosine Similarity | MSE | R² | Peak Precision | Peak Recall | Peak F1 |
|-------|------------------|-----|-----|----------------|-------------|----------|
| **MultiOutputRegressor** | **0.7662** | 0.0046 | 0.2192 | 0.4934 | 0.9907 | 0.6588 |
| **RegressorChain (isotope)** | 0.7552 | 0.0049 | 0.1695 | 0.5027 | 0.9887 | 0.6665 |
| **Hybrid Model** | 0.7557 | 0.0049 | 0.1698 | 0.5027 | 0.9887 | 0.6665 |

#### Regional Performance
- **Best regions**: m/z 44-99 (R² > 0.4 for most groups)
- **Correlation preservation**: >0.90 for isotope pairs in 35-99 m/z range
- **Challenging regions**: High m/z (>400) with negative R² values

#### Key Findings
- **MultiOutput baseline superior**: Simpler model outperforms chain methods
- **Isotope patterns preserved**: Both methods maintain correlations well
- **No benefit from chaining**: Sequential dependencies don't improve performance

## Key Results

### Baseline Methods Performance
- **Best Overall**: Random Forest (0.758 cosine similarity, n=50)
- **Best Neighbors**: KNN k=5 (0.714 cosine similarity)
- **Best Linear**: Ridge α=10.0 (0.697 cosine similarity)
- **Best Statistical**: Mean Spectrum (0.549 cosine similarity)
- **Fastest Accurate**: KNN methods (<0.003s training, >0.70 cosine)

### Target Transformation Impact
- **Best Transform**: Rank transformation (0.975 cosine, 59.84% improvement)
- **Power Optimal**: α=0.100 (0.918 cosine, practical for deployment)
- **Neural Network Benefit**: 3-5% improvement with power transforms
- **Implementation Note**: Always apply inverse transform for physical units

### Neural Architecture Performance

#### Deep Learning Models (04_deep_learning_exploration.ipynb)
- **Best Individual Model**: HierarchicalPredictionNet (0.7768 test cosine)
- **Best Traditional Architecture**: ModularNet (0.7751 test cosine)
- **Best Specialized Architecture**: HierarchicalPredictionNet (0.7768 test cosine)
- **Category Performance**: Specialized (0.7716 mean) > Traditional (0.7403 mean)

#### Transformer Models (06_seq2seq_learning_exploration.ipynb)
- **Best Transformer**: SpectraFormer (0.7519 test cosine)
- **Transformer vs Traditional**: ~2.5% lower performance than best neural networks
- **Parameter Efficiency**: ViT-1D has 2x parameters but lower performance

### Ensemble Performance
- **Optimal Ensemble**: K=4 models (0.7938 test cosine, 0.8059 validation)
  - Models: HierarchicalPredictionNet, RegionalExpertNet, SparseGatedNet, ModularNet
  - Equal weights (0.25 each)
  - Throughput: 3,553.5 samples/s
- **All Models Ensemble**: 0.7944 test cosine (13 models)
- **Improvement over best individual**: +2.27%

### Regressor Chain Performance
- **Best Approach**: MultiOutputRegressor baseline (0.7662 cosine)
- **Chain Methods**: No improvement despite isotope correlations
- **Isotope Groups**: 20 groups identified with correlations up to 1.0
- **Correlation Preservation**: >0.90 for most isotope pairs

### Performance Patterns by m/z Range
- **0-100 Da**: Excellent predictions (>0.85 cosine)
- **100-200 Da**: Good performance (0.75-0.85)
- **200-300 Da**: Moderate accuracy (0.65-0.75)
- **300+ Da**: Challenging (<0.65 cosine)

### Data Efficiency Insights
- **Saturation Analysis**: Performance plateaus at 80% of training data (1,740 samples)
- **Efficiency Metric**: 10% of data provides 83.8% of final performance
- **Predicted Maximum**: 0.8266 cosine at 2x data (logarithmic extrapolation)
- **Feature Stability**: Importance correlations increase from 0.61 (10% vs 50%) to 0.86 (50% vs 100%)

## Computational Requirements

### Training Times (Full Dataset)
- **Statistical Baselines**: <0.01s
- **KNN Methods**: <0.005s
- **Ridge Regression**: 0.4-0.5s
- **Decision Trees**: 3-5s
- **Random Forest (n=50)**: 16s
- **Extra Trees (n=50)**: 19s
- **Linear Regression (OLS)**: 25s (memory intensive)

### Memory Requirements
- **Feature Matrix**: ~120MB (2176 × 7137 float32)
- **Target Matrix**: ~4MB (2176 × 500 float32)
- **Model Storage**: 1-150MB depending on method

## Dataset Size Analysis (05_dataset_size_effects.ipynb)

### Performance Progression
| Training % | N Samples | Cosine Sim | R² | RMSE | MAE | Efficiency |
|------------|-----------|------------|-----|------|-----|------------|
| 10 | 217 | 0.6525±0.1348 | 0.0221 | 0.0799 | 0.0323 | 8.381 |
| 20 | 435 | 0.6863±0.1378 | 0.0065 | 0.0761 | 0.0303 | 4.407 |
| 30 | 652 | 0.7051±0.1383 | -0.0566 | 0.0746 | 0.0292 | 3.019 |
| 40 | 870 | 0.7229±0.1362 | -0.1019 | 0.0731 | 0.0282 | 2.321 |
| 50 | 1,088 | 0.7309±0.1412 | -0.0509 | 0.0720 | 0.0277 | 1.878 |
| 60 | 1,305 | 0.7454±0.1437 | -0.0209 | 0.0704 | 0.0267 | 1.596 |
| 70 | 1,523 | 0.7577±0.1379 | -0.0187 | 0.0694 | 0.0262 | 1.390 |
| 80 | 1,740 | 0.7666±0.1385 | 0.0135 | 0.0683 | 0.0257 | 1.231 |
| 90 | 1,958 | 0.7734±0.1363 | 0.0739 | 0.0675 | 0.0254 | 1.104 |
| 100 | 2,176 | 0.7786±0.1408 | 0.1575 | 0.0669 | 0.0249 | 1.000 |

### Key Findings
- **Overall Improvement**: +0.1260 cosine (19.3% relative) from 10% to 100%
- **Saturation Point**: Performance plateaus at ~80% of data
- **Most Data-Efficient**: 10% of data (efficiency: 8.381)
- **Logarithmic Fit**: R² = 0.9978, Performance = 0.0724 * log(0.005036 * n + 1) + 0.6000
- **Statistical Significance**: All comparisons significant (p < 0.05, Mann-Whitney U test)
- **Feature Importance Stability**: Correlation increases with data size (0.61→0.86)

## Neural Network Training Details

### Training Configuration
- **Hardware**: CUDA GPU acceleration
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 64
- **Epochs**: 100 with early stopping
- **Loss Functions**: MSE + Cosine similarity (weighted)
- **Validation**: 10% holdout for model selection

### Performance by Architecture Type
| Category | Mean Cosine | Std Dev | Best Model | Parameters |
|----------|------------|---------|------------|------------|
| Ensemble | 0.7938 | N/A | K=4 Optimal | - |
| Specialized | 0.7716 | 0.0053 | HierarchicalPredictionNet | 8.3M |
| Traditional | 0.7403 | 0.0258 | ModularNet | 9.5M |

### Throughput Analysis (K-Model Ensembles)
| K | Val Cosine | Test Cosine | Samples/s | Speedup |
|---|-----------|-------------|-----------|---------||
| 2 | 0.7992 | 0.7855 | 3,087.9 | 1.00x |
| 3 | 0.8043 | 0.7916 | 2,583.9 | 0.84x |
| 4 | 0.8059 | 0.7938 | 3,553.5 | 1.15x |
| 5 | 0.8062 | 0.7951 | 11,374.5 | 3.68x |
| 8 | 0.8077 | 0.7965 | 7,234.9 | 2.34x |
| 13 | 0.8050 | 0.7944 | 5,041.9 | 1.63x |

## Reference
Notebooks:
- `../../00_plotting_suite.ipynb` - Interactive visualization toolkit (at root level)
- `01_eda.ipynb` - Exploratory data analysis (dataset statistics, molecular properties)
- `02_scaling.ipynb` - Target transformation experiments (20+ methods tested)
- `03_linear_learning_exploration.ipynb` - Baseline ML methods (18 models)
- `04_deep_learning_exploration.ipynb` - Neural architectures (13 models + ensembles)
- `05_dataset_size_effects.ipynb` - Data efficiency analysis (10 increments, saturation study)
- `06_seq2seq_learning_exploration.ipynb` - Transformer models (SpectraFormer, ViT-1D, LinearAttn)
- `07_rf_chain.ipynb` - Regressor chain methodology (isotope pattern exploitation)