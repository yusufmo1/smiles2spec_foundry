"""Feature extraction service for molecular structures.

Orchestrates 2D and 3D feature extraction with caching.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from smiles2spec.core.config import FeatureConfig
from smiles2spec.core.exceptions import FeatureError
from smiles2spec.data.cache import FeatureCache
from smiles2spec.domain.features_2d import Feature2DExtractor
from smiles2spec.domain.features_3d import Feature3DExtractor
from smiles2spec.domain.molecule import MoleculeProcessor


class FeaturizationService:
    """Service for extracting molecular features from SMILES.

    Supports 2D descriptors, 3D conformer features, and caching.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """Initialize featurization service.

        Args:
            config: Feature configuration
            cache_dir: Directory for feature cache
            use_cache: Whether to use caching
            verbose: Print progress
        """
        self.config = config or FeatureConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        self.verbose = verbose

        # Initialize extractors
        self._2d_extractor = Feature2DExtractor(
            include_descriptors=self.config.descriptors_enabled,
            include_fingerprints=True,
            include_electronic=True,
            morgan_radii=self.config.fingerprints.morgan.radii,
            morgan_bits=self.config.fingerprints.morgan.n_bits,
        )

        self._3d_extractor = None
        if self.config.enable_3d:
            self._3d_extractor = Feature3DExtractor(
                n_conformers=self.config.conformer.n_conformers,
            )

        # Initialize cache
        self._cache = None
        if self.use_cache and self.cache_dir:
            self._cache = FeatureCache(self.cache_dir)

    def extract(
        self,
        smiles_list: List[str],
        feature_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """Extract features from SMILES list.

        Args:
            smiles_list: List of SMILES strings
            feature_type: Override feature type ('2d', '3d', 'combined')

        Returns:
            Tuple of (features, feature_names, failed_indices)
        """
        feature_type = feature_type or self.config.type

        # Check cache
        if self._cache and self._cache.is_valid(smiles_list):
            if self.verbose:
                print("Loading features from cache...")
            cached = self._cache.get(smiles_list)
            if cached is not None:
                return cached, self._get_feature_names(feature_type), []

        # Extract features
        features_list = []
        failed_indices = []

        iterator = tqdm(smiles_list, desc="Extracting features") if self.verbose else smiles_list

        for i, smiles in enumerate(iterator):
            try:
                features = self._extract_single(smiles, feature_type)
                if features is not None:
                    features_list.append(features)
                else:
                    failed_indices.append(i)
            except Exception:
                failed_indices.append(i)

        if not features_list:
            raise FeatureError("No valid features extracted")

        features_array = np.array(features_list, dtype=np.float32)
        feature_names = self._get_feature_names(feature_type)

        # Cache results
        if self._cache:
            valid_smiles = [s for i, s in enumerate(smiles_list) if i not in failed_indices]
            self._cache.put(valid_smiles, features_array, feature_names)

        return features_array, feature_names, failed_indices

    def _extract_single(self, smiles: str, feature_type: str) -> Optional[np.ndarray]:
        """Extract features from a single SMILES.

        Args:
            smiles: SMILES string
            feature_type: Feature type

        Returns:
            Feature array or None if failed
        """
        if feature_type == "2d":
            return self._2d_extractor.extract_from_smiles(smiles)
        elif feature_type == "3d" and self._3d_extractor:
            return self._3d_extractor.extract_from_smiles(smiles)
        elif feature_type == "combined" and self._3d_extractor:
            feat_2d = self._2d_extractor.extract_from_smiles(smiles)
            feat_3d = self._3d_extractor.extract_from_smiles(smiles)
            if feat_2d is not None and feat_3d is not None:
                return np.concatenate([feat_2d, feat_3d])
            return None
        else:
            return self._2d_extractor.extract_from_smiles(smiles)

    def _get_feature_names(self, feature_type: str) -> List[str]:
        """Get feature names for given type."""
        names = []
        if feature_type in ("2d", "combined"):
            names.extend(self._2d_extractor.feature_names)
        if feature_type in ("3d", "combined") and self._3d_extractor:
            names.extend(self._3d_extractor.feature_names)
        return names

    @property
    def n_features(self) -> int:
        """Get total number of features."""
        total = self._2d_extractor.n_features
        if self._3d_extractor and self.config.type in ("3d", "combined"):
            if self.config.type == "3d":
                return self._3d_extractor.n_features
            total += self._3d_extractor.n_features
        return total

    def get_summary(self) -> Dict:
        """Get summary of feature configuration."""
        return {
            "type": self.config.type,
            "n_2d_features": self._2d_extractor.n_features,
            "n_3d_features": self._3d_extractor.n_features if self._3d_extractor else 0,
            "total_features": self.n_features,
            "cache_enabled": self.use_cache,
        }
