# Stage 6: Model Training

## Overview
This stage implements production model training using optimized hyperparameters, developing both individual models and ensemble methods for mass spectrometry prediction.

## Methodology

### Production Models

#### Individual Models
- **Random Forest**: 500 estimators, max_depth=30
- **K-Nearest Neighbors**: Optimized with k=5, distance weighting
- **ModularNet**: 4-module architecture with attention fusion (9,530,836 parameters)
- **HierarchicalPredictionNet**: Regional experts for m/z ranges (8,307,692 parameters)
- **SparseGatedNet**: Gated architecture with sparsity constraints (11,378,652 parameters)
- **RegionalExpertNet**: Multiple specialized networks by m/z region (22,748,253 parameters)
- **Feature Variants**: Separate models for 2D (7,137 dims) and 3D (83 dims) features

#### Ensemble Strategies

**Simple Weighted Ensemble**
- Optimal weights: RF (0.453), ModularNet (0.119), HierarchicalNet (0.150), SparseGatedNet (0.169), RegionalExpertNet (0.109)
- Weight optimization via differential evolution
- Performance: 0.8037 cosine similarity

**Bin-by-Bin Ensemble**
- Individual optimization per m/z bin (500 bins total)
- Range-specific model weighting
- Average weights: RF (0.384), KNN (0.109), ModularNet (0.119), HierarchicalNet (0.131), SparseGatedNet (0.129), RegionalExpertNet (0.128)
- Performance: 0.8164 cosine similarity (best overall)

### Training Configuration
- **Data**: 2,176 training samples
- **Validation**: 272 samples for early stopping
- **Test**: 272 samples for final evaluation
- **Features**: 
  - 2D features: 7,137 dimensions (molecular descriptors + fingerprints)
  - 3D features: 83 dimensions (conformer-based)
  - Combined: 7,220 dimensions
- **Target**: 500 m/z bins (0-499 Da)
- **Regularization**: L2 penalty, dropout, early stopping
- **Device**: MPS (Apple Silicon) support for neural networks

### Model Persistence
- **Format**: Pickle serialization with protocol 4
- **Components**: Model, preprocessor, feature mapping
- **Size**: 89MB (XGBoost) to 250MB (ensemble)
- **Compatibility**: scikit-learn 1.3.0+, XGBoost 1.7.0+

## Key Results

### Performance Metrics (Main Models - Notebook 01)
| Model | Cosine Similarity | MSE | R² Score |
|-------|------------------|-----|----------|
| **Bin-by-bin Ensemble** | **0.8164** | - | - |
| Simple Weighted Ensemble | 0.8037 | - | - |
| Random Forest | 0.7837 | 0.0044 | 0.1486 |
| HierarchicalPredictionNet | 0.7770 | 0.0049 | 0.0440 |
| ModularNet | 0.7691 | 0.0049 | 0.2186 |
| SparseGatedNet | 0.7674 | 0.0050 | 0.2640 |
| RegionalExpertNet | 0.7622 | 0.0053 | 0.2382 |
| K-Nearest Neighbors | 0.7325 | 0.0052 | -0.1260 |

### Training Times
- Random Forest: ~15 seconds
- Neural Networks: 60-100 epochs, 1-2 minutes each
- Ensemble optimization: ~90 seconds

### Feature Comparison (Notebook 02)
| Model Configuration | Features | Cosine Similarity |
|--------------------|----------|------------------|
| RF 2D Only | 7,137 | 0.7797 |
| RF 3D Only | 83 | 0.6950 |
| RF 2D+3D Combined | 7,220 | 0.7728 |
| 2D/3D Ensemble | - | 0.7831 |

### Key Insights
- **2D vs 3D**: 2D features superior by 12.2% (0.7797 vs 0.6950)
- **Feature Combination**: Combined features underperform 2D alone (likely overfitting)
- **Ensemble Gain**: 4.16% improvement over best individual model
- **Model Ranking**: Bin-by-bin ensemble > Simple ensemble > Random Forest > Neural Networks

## Output Files

### Individual Models
- `models/hpj_rf_model.pkl`: Random Forest model
- `models/hpj_knn_model.pkl`: K-Nearest Neighbors model
- `models/ModularNet_best.pth`: ModularNet neural network
- `models/HierarchicalPredictionNet_best.pth`: Hierarchical prediction network
- `models/SparseGatedNet_best.pth`: Sparse gated network
- `models/RegionalExpertNet_best.pth`: Regional expert network

### Feature-Specific Models
- `models/hpj_rf_2d_regression_model.pkl`: RF with 2D features only
- `models/hpj_rf_3d_regression_model.pkl`: RF with 3D features only
- `models/hpj_rf_combined_regression_model.pkl`: RF with combined features

### Ensemble Models
- `models/ensemble_results.pkl`: Main ensemble configurations and weights
- `models/ensemble_2d_3d_results.pkl`: 2D/3D ensemble results
- `models/all_model_predictions.pkl`: All model predictions for analysis
- `models/best_model.pkl`: Best performing model (Bin-by-bin ensemble)
- `models/model_metadata.json`: Metadata for deployment

## Summary & Recommendations

### Key Achievements
1. **Best Performance**: Bin-by-bin ensemble achieves 0.8164 cosine similarity
2. **Ensemble Effectiveness**: 4.16% improvement over best individual model
3. **Feature Analysis**: 2D molecular features significantly outperform 3D conformer features
4. **Model Diversity**: Six different architectures successfully trained and evaluated

### Production Recommendations
- **For Maximum Accuracy**: Use bin-by-bin ensemble (0.8164 cosine similarity)
- **For Speed/Accuracy Balance**: Use Random Forest model (0.7837 cosine similarity, 15s training)
- **For Feature Selection**: Use 2D features only (7,137 dimensions)
- **For Deployment**: Load `models/best_model.pkl` with associated metadata

### Computational Requirements
- **Memory**: 2-4GB RAM for training
- **Storage**: ~500MB for all models
- **Inference**: 5-50ms per spectrum depending on model choice

## Reference
Notebooks:
- `01_training.ipynb` - Core model training and ensemble creation
- `02_2d_3d_combined_training.ipynb` - Feature combination experiments