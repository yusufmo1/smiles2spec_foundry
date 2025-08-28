# Stage 2: Molecular Featurisation

## Overview
This stage extracts comprehensive molecular features from SMILES strings using RDKit, generating high-dimensional representations for mass spectrometry prediction.

## Methodology

### 2D Molecular Features (7,137 dimensions)
- **RDKit Descriptors**: 188 molecular properties (MW, LogP, TPSA, etc.)
- **Morgan Fingerprints**: Radii 1-3, 1024 bits each
- **MACCS Keys**: 166 structural keys
- **Topological Fingerprints**: RDKit (2048), Avalon (1024), Pattern (2048), Layered (2048)
- **Electronic Properties**: Gasteiger charges, PEOE_VSA descriptors

### 3D Conformer Features (984 dimensions)
- **Conformer Generation**: ETKDGv3 algorithm, 5 conformers per molecule
- **3D Descriptors**: PMI, NPR, asphericity, radius of gyration (11 features)
- **Advanced Descriptors**: AUTOCORR3D (80), RDF (210), MORSE (224), WHIM (114), GETAWAY (273)
- **Shape Recognition**: USR (12), USRCAT (60)

### Data Preprocessing
- **Variance Filtering**: Remove features with variance < 1e-8
- **Scaling**: StandardScaler normalization
- **NaN Handling**: Drop features with missing values
- **Final Dimensions**: 2D: 7,137 retained; 3D: 83 after filtering (91% reduction)

## Data Splitting
- **Training**: 80% (2,176 molecules)
- **Validation**: 10% (272 molecules)
- **Test**: 10% (272 molecules)
- **Strategy**: No molecular overlap between splits (seed=42)

## Key Results
- **2D Feature Extraction**: 100% success rate
- **3D Feature Extraction**: 100% success rate
- **Performance**: 2D features outperform 3D by 12.2%
- **Computational Time**: 3D extraction 40× slower than 2D

## Output Files
- `data/results/{dataset}/full_featurised/`: Complete feature matrices
- `feature_preprocessor.pkl`: Scaling pipeline
- `feature_mapping.json`: Feature name mappings

## Visualization Capabilities
- **Feature Distribution Analysis**: Histograms and density plots for all feature types
- **Correlation Heatmaps**: Inter-feature relationships and redundancy detection
- **PCA Visualization**: 2D/3D projections of chemical space coverage
- **Feature Importance Plots**: Initial importance rankings for feature selection

## Reference
Notebooks: 
- `01_feature_generation.ipynb` - 2D molecular descriptor extraction
- `02_feature_combination.ipynb` - Feature preprocessing and data splitting
- `03_feature_visualisation.ipynb` - 2D feature analysis and visualization
- `04_feature_generation_3d.ipynb` - 3D conformer feature extraction
- `05_feature_combination_3d.ipynb` - 3D feature preprocessing
- `06_feature_visualisation_3d.ipynb` - 3D feature analysis and visualization