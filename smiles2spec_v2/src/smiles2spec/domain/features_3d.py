"""3D Molecular feature extraction (optional).

Extracts 3D conformer-based features:
- Basic 3D descriptors (PMI, NPR, shape)
- AUTOCORR3D (80 features)
- RDF (210 features)
- MORSE (224 features)
- WHIM (114 features)
- GETAWAY (273 features)
- USR/USRCAT (72 features)

Total: 984 features
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

from smiles2spec.core.exceptions import FeatureError


class Conformer3DGenerator:
    """Generate 3D conformers for molecules."""

    def __init__(
        self,
        n_conformers: int = 5,
        max_attempts: int = 100,
        prune_rms_thresh: float = 0.5,
        random_seed: int = 42,
    ):
        self.n_conformers = n_conformers
        self.max_attempts = max_attempts
        self.prune_rms_thresh = prune_rms_thresh
        self.random_seed = random_seed

    def generate(self, mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, List[int]]]:
        """Generate 3D conformers.

        Args:
            mol: RDKit Mol object (2D)

        Returns:
            Tuple of (3D mol, list of conformer IDs), or None if failed
        """
        try:
            mol_3d = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0
            params.pruneRmsThresh = self.prune_rms_thresh

            conf_ids = AllChem.EmbedMultipleConfs(
                mol_3d, numConfs=self.n_conformers, params=params
            )

            if len(conf_ids) == 0:
                result = AllChem.EmbedMolecule(
                    mol_3d, maxAttempts=self.max_attempts, randomSeed=self.random_seed
                )
                if result < 0:
                    return None
                conf_ids = [0]

            AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=0)
            return mol_3d, list(conf_ids)

        except Exception:
            return None


class Basic3DExtractor:
    """Extract basic 3D shape descriptors."""

    FEATURE_NAMES = [
        "PMI1", "PMI2", "PMI3", "NPR1", "NPR2", "Asphericity",
        "Eccentricity", "InertialShapeFactor", "SpherocityIndex",
        "RadiusOfGyration", "PBF",
    ]

    def extract(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """Extract basic 3D descriptors.

        Returns:
            Array of 11 basic 3D features
        """
        features = []

        try:
            features.append(Descriptors3D.PMI1(mol, confId=conf_id))
            features.append(Descriptors3D.PMI2(mol, confId=conf_id))
            features.append(Descriptors3D.PMI3(mol, confId=conf_id))
            features.append(Descriptors3D.NPR1(mol, confId=conf_id))
            features.append(Descriptors3D.NPR2(mol, confId=conf_id))
            features.append(Descriptors3D.Asphericity(mol, confId=conf_id))
            features.append(Descriptors3D.Eccentricity(mol, confId=conf_id))
            features.append(Descriptors3D.InertialShapeFactor(mol, confId=conf_id))
            features.append(Descriptors3D.SpherocityIndex(mol, confId=conf_id))
            features.append(Descriptors3D.RadiusOfGyration(mol, confId=conf_id))
            features.append(Descriptors3D.PBF(mol, confId=conf_id))
        except Exception:
            features = [0.0] * 11

        return np.array(features, dtype=np.float32)

    @property
    def n_features(self) -> int:
        return 11


class Advanced3DExtractor:
    """Extract advanced 3D descriptors (AUTOCORR3D, RDF, MORSE, WHIM, GETAWAY)."""

    def extract_autocorr3d(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """3D autocorrelation (80 features)."""
        try:
            return np.array(rdMolDescriptors.CalcAUTOCORR3D(mol, confId=conf_id))
        except Exception:
            return np.zeros(80, dtype=np.float32)

    def extract_rdf(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """Radial Distribution Function (210 features)."""
        try:
            return np.array(rdMolDescriptors.CalcRDF(mol, confId=conf_id))
        except Exception:
            return np.zeros(210, dtype=np.float32)

    def extract_morse(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """3D-MoRSE (224 features)."""
        try:
            return np.array(rdMolDescriptors.CalcMORSE(mol, confId=conf_id))
        except Exception:
            return np.zeros(224, dtype=np.float32)

    def extract_whim(
        self, mol: Chem.Mol, conf_id: int = 0, thresh: float = 0.001
    ) -> np.ndarray:
        """WHIM descriptors (114 features)."""
        try:
            return np.array(rdMolDescriptors.CalcWHIM(mol, confId=conf_id, thresh=thresh))
        except Exception:
            return np.zeros(114, dtype=np.float32)

    def extract_getaway(
        self, mol: Chem.Mol, conf_id: int = 0, precision: float = 2.0
    ) -> np.ndarray:
        """GETAWAY descriptors (273 features)."""
        try:
            return np.array(
                rdMolDescriptors.CalcGETAWAY(mol, confId=conf_id, precision=precision)
            )
        except Exception:
            return np.zeros(273, dtype=np.float32)

    def extract_usr(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """USR descriptors (12 features)."""
        try:
            return np.array(rdMolDescriptors.GetUSR(mol, confId=conf_id))
        except Exception:
            return np.zeros(12, dtype=np.float32)

    def extract_usrcat(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """USRCAT descriptors (60 features)."""
        try:
            return np.array(rdMolDescriptors.GetUSRCAT(mol, confId=conf_id))
        except Exception:
            return np.zeros(60, dtype=np.float32)


class Feature3DExtractor:
    """Unified 3D feature extraction.

    Generates conformers and extracts all 3D features.
    """

    def __init__(
        self,
        n_conformers: int = 5,
        aggregation: str = "mean",
        include_basic: bool = True,
        include_autocorr3d: bool = True,
        include_rdf: bool = True,
        include_morse: bool = True,
        include_whim: bool = True,
        include_getaway: bool = True,
        include_usr: bool = True,
    ):
        self.aggregation = aggregation
        self.include_basic = include_basic
        self.include_autocorr3d = include_autocorr3d
        self.include_rdf = include_rdf
        self.include_morse = include_morse
        self.include_whim = include_whim
        self.include_getaway = include_getaway
        self.include_usr = include_usr

        self._conformer_gen = Conformer3DGenerator(n_conformers=n_conformers)
        self._basic_extractor = Basic3DExtractor()
        self._advanced_extractor = Advanced3DExtractor()

    def extract(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """Extract all 3D features.

        Args:
            mol: RDKit Mol object (2D)

        Returns:
            Feature array, or None if conformer generation fails
        """
        result = self._conformer_gen.generate(mol)
        if result is None:
            return None

        mol_3d, conf_ids = result
        all_conformer_features = []

        for conf_id in conf_ids:
            features = []

            if self.include_basic:
                features.append(self._basic_extractor.extract(mol_3d, conf_id))

            if self.include_autocorr3d:
                features.append(
                    self._advanced_extractor.extract_autocorr3d(mol_3d, conf_id)
                )

            if self.include_rdf:
                features.append(self._advanced_extractor.extract_rdf(mol_3d, conf_id))

            if self.include_morse:
                features.append(self._advanced_extractor.extract_morse(mol_3d, conf_id))

            if self.include_whim:
                features.append(self._advanced_extractor.extract_whim(mol_3d, conf_id))

            if self.include_getaway:
                features.append(self._advanced_extractor.extract_getaway(mol_3d, conf_id))

            if self.include_usr:
                features.append(self._advanced_extractor.extract_usr(mol_3d, conf_id))
                features.append(self._advanced_extractor.extract_usrcat(mol_3d, conf_id))

            all_conformer_features.append(np.concatenate(features))

        # Aggregate across conformers
        stacked = np.stack(all_conformer_features)
        if self.aggregation == "mean":
            return np.mean(stacked, axis=0).astype(np.float32)
        elif self.aggregation == "max":
            return np.max(stacked, axis=0).astype(np.float32)
        elif self.aggregation == "min":
            return np.min(stacked, axis=0).astype(np.float32)
        else:
            return np.mean(stacked, axis=0).astype(np.float32)

    def extract_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """Extract 3D features from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.extract(mol)

    @property
    def n_features(self) -> int:
        total = 0
        if self.include_basic:
            total += 11
        if self.include_autocorr3d:
            total += 80
        if self.include_rdf:
            total += 210
        if self.include_morse:
            total += 224
        if self.include_whim:
            total += 114
        if self.include_getaway:
            total += 273
        if self.include_usr:
            total += 72  # USR (12) + USRCAT (60)
        return total

    @property
    def feature_names(self) -> List[str]:
        """Get feature names in order."""
        names = []

        if self.include_basic:
            names.extend(Basic3DExtractor.FEATURE_NAMES)
        if self.include_autocorr3d:
            names.extend([f"AUTOCORR3D_{i}" for i in range(80)])
        if self.include_rdf:
            names.extend([f"RDF_{i}" for i in range(210)])
        if self.include_morse:
            names.extend([f"MORSE_{i}" for i in range(224)])
        if self.include_whim:
            names.extend([f"WHIM_{i}" for i in range(114)])
        if self.include_getaway:
            names.extend([f"GETAWAY_{i}" for i in range(273)])
        if self.include_usr:
            names.extend([f"USR_{i}" for i in range(12)])
            names.extend([f"USRCAT_{i}" for i in range(60)])

        return names
