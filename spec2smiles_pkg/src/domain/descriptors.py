"""Molecular descriptor definitions and calculations.

This module defines the molecular descriptors used in the pipeline
and provides functions to calculate them from SMILES strings.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Default descriptor names used throughout the pipeline
DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "MolWt",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "NumAromaticRings",
    "RingCount",
    "NOCount",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "MolLogP",
    "NumRotatableBonds",
    "FractionCSP3",
)

# Lazily initialized descriptor functions (RDKit import is expensive)
_descriptor_funcs: Optional[Dict[str, Callable]] = None


def _get_descriptor_functions() -> Dict[str, Callable]:
    """Lazy initialization of RDKit descriptor functions.

    Returns:
        Dictionary mapping descriptor names to RDKit functions.
    """
    global _descriptor_funcs

    if _descriptor_funcs is not None:
        return _descriptor_funcs

    from rdkit.Chem import Descriptors

    _descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "RingCount": Descriptors.RingCount,
        "NOCount": Descriptors.NOCount,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": Descriptors.TPSA,
        "MolLogP": Descriptors.MolLogP,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "FractionCSP3": Descriptors.FractionCSP3,
    }

    return _descriptor_funcs


# Alias for backwards compatibility
DESCRIPTOR_FUNCTIONS = _get_descriptor_functions


def calculate_descriptors(
    smiles: str,
    descriptor_names: Tuple[str, ...] = DESCRIPTOR_NAMES,
) -> Optional[np.ndarray]:
    """Calculate molecular descriptors for a SMILES string.

    Args:
        smiles: SMILES string representing the molecule
        descriptor_names: Tuple of descriptor names to calculate

    Returns:
        Array of descriptor values, or None if molecule is invalid

    Example:
        >>> descriptors = calculate_descriptors("CCO")  # Ethanol
        >>> descriptors[0]  # MolWt
        46.07
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    funcs = _get_descriptor_functions()

    try:
        values = [funcs[name](mol) for name in descriptor_names]
        return np.array(values, dtype=np.float32)
    except Exception:
        return None


def calculate_descriptors_batch(
    smiles_list: List[str],
    descriptor_names: Tuple[str, ...] = DESCRIPTOR_NAMES,
    return_valid_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculate descriptors for multiple molecules.

    Args:
        smiles_list: List of SMILES strings
        descriptor_names: Tuple of descriptor names to calculate
        return_valid_mask: Whether to return mask of valid molecules

    Returns:
        Tuple of (descriptors array, valid mask if requested)
        Descriptors array has shape (n_valid, n_descriptors)
    """
    from tqdm import tqdm

    descriptors = []
    valid_mask = []

    for smiles in tqdm(smiles_list, desc="Calculating descriptors"):
        desc = calculate_descriptors(smiles, descriptor_names)
        if desc is not None:
            descriptors.append(desc)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    result = np.array(descriptors) if descriptors else np.empty((0, len(descriptor_names)))

    if return_valid_mask:
        return result, np.array(valid_mask)
    return result, None
