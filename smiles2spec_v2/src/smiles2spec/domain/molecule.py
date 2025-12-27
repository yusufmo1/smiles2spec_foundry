"""Molecular structure validation and basic processing.

Provides SMILES validation and RDKit molecule handling.
"""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from smiles2spec.core.exceptions import FeatureError


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


class MoleculeProcessor:
    """Process molecular structures for feature extraction.

    Provides methods for SMILES validation and basic molecular properties.
    """

    @staticmethod
    def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
        """Parse SMILES string to RDKit molecule.

        Args:
            smiles: SMILES string

        Returns:
            RDKit Mol object, or None if invalid
        """
        if not is_valid_smiles(smiles):
            return None
        return Chem.MolFromSmiles(smiles)

    @staticmethod
    def get_basic_properties(mol: Chem.Mol) -> dict:
        """Get basic molecular properties.

        Args:
            mol: RDKit Mol object

        Returns:
            Dictionary with molecular weight, exact mass, etc.
        """
        if mol is None:
            raise FeatureError("Cannot get properties from None molecule")

        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "exact_mass": Descriptors.ExactMolWt(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        }

    @staticmethod
    def canonicalize(smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES, or None if invalid
        """
        mol = MoleculeProcessor.parse_smiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    @staticmethod
    def add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
        """Add explicit hydrogens to molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            Molecule with explicit hydrogens
        """
        return Chem.AddHs(mol)
