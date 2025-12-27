"""2D Molecular feature extraction using RDKit.

Extracts molecular descriptors and fingerprints:
- 208 RDKit molecular descriptors
- Morgan fingerprints (multiple radii)
- MACCS keys (167 bits)
- RDKit fingerprints
- Avalon fingerprints
- Electronic properties

Total: ~7,137 features (default configuration)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Descriptors import MoleculeDescriptors

from smiles2spec.core.exceptions import FeatureError


class DescriptorExtractor:
    """Extract RDKit molecular descriptors."""

    def __init__(self):
        self.descriptor_names = [desc[0] for desc in Descriptors._descList]
        self._calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.descriptor_names
        )

    def extract(self, mol: Chem.Mol) -> np.ndarray:
        """Extract all RDKit descriptors.

        Args:
            mol: RDKit Mol object

        Returns:
            Array of descriptor values (208 features)
        """
        try:
            descriptors = self._calculator.CalcDescriptors(mol)
            return np.array(descriptors, dtype=np.float32)
        except Exception as e:
            raise FeatureError(f"Descriptor extraction failed: {e}")

    @property
    def n_features(self) -> int:
        return len(self.descriptor_names)


class FingerprintExtractor:
    """Extract molecular fingerprints."""

    def __init__(
        self,
        morgan_radii: List[int] = [1, 2, 3],
        morgan_bits: int = 1024,
        rdkit_bits: int = 2048,
        avalon_bits: int = 1024,
        include_maccs: bool = True,
    ):
        self.morgan_radii = morgan_radii
        self.morgan_bits = morgan_bits
        self.rdkit_bits = rdkit_bits
        self.avalon_bits = avalon_bits
        self.include_maccs = include_maccs

        # Pre-create generators
        self._morgan_generators = {
            r: rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=morgan_bits)
            for r in morgan_radii
        }

    def extract_morgan(self, mol: Chem.Mol) -> Dict[int, np.ndarray]:
        """Extract Morgan fingerprints at multiple radii."""
        result = {}
        for radius, gen in self._morgan_generators.items():
            fp = gen.GetFingerprint(mol)
            result[radius] = np.array([int(b) for b in fp.ToBitString()], dtype=np.int8)
        return result

    def extract_maccs(self, mol: Chem.Mol) -> np.ndarray:
        """Extract MACCS keys (167 bits)."""
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.int8)

    def extract_rdkit_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Extract RDKit fingerprint."""
        fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_bits)
        return np.array([int(b) for b in fp.ToBitString()], dtype=np.int8)

    def extract_avalon(self, mol: Chem.Mol) -> np.ndarray:
        """Extract Avalon fingerprint."""
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.avalon_bits)
        return np.array([int(b) for b in fp.ToBitString()], dtype=np.int8)

    def extract_all(self, mol: Chem.Mol) -> np.ndarray:
        """Extract all fingerprints concatenated.

        Returns:
            Concatenated fingerprint array
        """
        fps = []

        # Morgan fingerprints (multiple radii)
        for radius in self.morgan_radii:
            fp = self._morgan_generators[radius].GetFingerprint(mol)
            fps.append(np.array([int(b) for b in fp.ToBitString()], dtype=np.int8))

        # MACCS keys
        if self.include_maccs:
            fps.append(self.extract_maccs(mol))

        # RDKit fingerprint
        fps.append(self.extract_rdkit_fp(mol))

        # Avalon fingerprint
        fps.append(self.extract_avalon(mol))

        return np.concatenate(fps)

    @property
    def n_features(self) -> int:
        total = len(self.morgan_radii) * self.morgan_bits
        total += self.rdkit_bits + self.avalon_bits
        if self.include_maccs:
            total += 167
        return total


class ElectronicExtractor:
    """Extract electronic properties."""

    def extract(self, mol: Chem.Mol) -> np.ndarray:
        """Extract electronic features.

        Includes Gasteiger charges, PEOE_VSA, Crippen contributions.

        Returns:
            Array of electronic features (~13 features)
        """
        features = []

        # Gasteiger charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [
                atom.GetDoubleProp("_GasteigerCharge")
                if atom.HasProp("_GasteigerCharge")
                else 0.0
                for atom in mol.GetAtoms()
            ]
            if charges:
                features.extend([
                    min(charges),
                    max(charges),
                    np.mean(charges),
                    np.std(charges),
                ])
            else:
                features.extend([0.0] * 4)
        except Exception:
            features.extend([0.0] * 4)

        # PEOE_VSA
        try:
            peoe = rdMolDescriptors.PEOE_VSA_(mol)
            features.extend(peoe[:3])
        except Exception:
            features.extend([0.0] * 3)

        # Crippen contributions
        try:
            crippen = rdMolDescriptors.GetCrippenContribs(mol)
            logp_vals = [c[0] for c in crippen]
            mr_vals = [c[1] for c in crippen]
            features.extend([
                max(logp_vals) if logp_vals else 0.0,
                min(logp_vals) if logp_vals else 0.0,
                max(mr_vals) if mr_vals else 0.0,
                min(mr_vals) if mr_vals else 0.0,
            ])
        except Exception:
            features.extend([0.0] * 4)

        # Count features
        try:
            bond_counts = defaultdict(int)
            for bond in mol.GetBonds():
                bond_counts[str(bond.GetBondType())] += 1
            features.extend([
                bond_counts.get("SINGLE", 0),
                bond_counts.get("DOUBLE", 0),
            ])
        except Exception:
            features.extend([0.0] * 2)

        return np.array(features, dtype=np.float32)

    @property
    def n_features(self) -> int:
        return 13


class Feature2DExtractor:
    """Unified 2D feature extraction.

    Combines descriptors, fingerprints, and electronic properties.
    """

    def __init__(
        self,
        include_descriptors: bool = True,
        include_fingerprints: bool = True,
        include_electronic: bool = True,
        morgan_radii: List[int] = [1, 2, 3],
        morgan_bits: int = 1024,
    ):
        self.include_descriptors = include_descriptors
        self.include_fingerprints = include_fingerprints
        self.include_electronic = include_electronic

        self._descriptor_extractor = DescriptorExtractor() if include_descriptors else None
        self._fingerprint_extractor = (
            FingerprintExtractor(morgan_radii=morgan_radii, morgan_bits=morgan_bits)
            if include_fingerprints
            else None
        )
        self._electronic_extractor = ElectronicExtractor() if include_electronic else None

    def extract(self, mol: Chem.Mol) -> np.ndarray:
        """Extract all 2D features.

        Args:
            mol: RDKit Mol object

        Returns:
            Concatenated feature array
        """
        if mol is None:
            raise FeatureError("Cannot extract features from None molecule")

        features = []

        if self._descriptor_extractor:
            features.append(self._descriptor_extractor.extract(mol))

        if self._fingerprint_extractor:
            features.append(self._fingerprint_extractor.extract_all(mol))

        if self._electronic_extractor:
            features.append(self._electronic_extractor.extract(mol))

        return np.concatenate(features).astype(np.float32)

    def extract_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """Extract features from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Feature array, or None if parsing fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.extract(mol)

    @property
    def n_features(self) -> int:
        total = 0
        if self._descriptor_extractor:
            total += self._descriptor_extractor.n_features
        if self._fingerprint_extractor:
            total += self._fingerprint_extractor.n_features
        if self._electronic_extractor:
            total += self._electronic_extractor.n_features
        return total

    @property
    def feature_names(self) -> List[str]:
        """Get feature names in order."""
        names = []

        if self._descriptor_extractor:
            names.extend(self._descriptor_extractor.descriptor_names)

        if self._fingerprint_extractor:
            fp = self._fingerprint_extractor
            for r in fp.morgan_radii:
                names.extend([f"Morgan_r{r}_{i}" for i in range(fp.morgan_bits)])
            if fp.include_maccs:
                names.extend([f"MACCS_{i}" for i in range(167)])
            names.extend([f"RDKit_fp_{i}" for i in range(fp.rdkit_bits)])
            names.extend([f"Avalon_{i}" for i in range(fp.avalon_bits)])

        if self._electronic_extractor:
            names.extend([
                "min_charge", "max_charge", "mean_charge", "std_charge",
                "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3",
                "max_logp", "min_logp", "max_mr", "min_mr",
                "single_bonds", "double_bonds",
            ])

        return names


def postprocess_descriptors(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove NaN values and zero-variance features.

    Args:
        features: Feature matrix (n_samples, n_features)

    Returns:
        Tuple of (cleaned_features, valid_mask)
    """
    # Handle NaN
    nan_mask = ~np.isnan(features).any(axis=0)

    # Handle zero variance
    var_mask = np.std(features, axis=0) > 0

    valid_mask = nan_mask & var_mask
    cleaned = features[:, valid_mask]

    return cleaned, valid_mask
