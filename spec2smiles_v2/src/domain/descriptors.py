"""Molecular descriptor definitions and calculations.

This module defines the molecular descriptors used in the pipeline
and provides functions to calculate them from SMILES strings.
Supports ALL RDKit descriptors dynamically.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Default descriptor names used throughout the pipeline
# 30 descriptors for 100% uniqueness on HPJ dataset
# (Original 12 descriptors had only 88.9% uniqueness)
DESCRIPTOR_NAMES: Tuple[str, ...] = (
    # Original 12 descriptors
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
    # Extended 18 descriptors for uniqueness
    "ExactMolWt",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
    "NumAromaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi1",
    "Chi2n",
    "Chi3n",
    "Chi4n",
    "HallKierAlpha",
)

# Lazily initialized descriptor functions (RDKit import is expensive)
_descriptor_funcs: Optional[Dict[str, Callable]] = None


def _get_descriptor_functions() -> Dict[str, Callable]:
    """Lazy initialization of ALL RDKit descriptor functions.

    Returns:
        Dictionary mapping descriptor names to RDKit functions.
    """
    global _descriptor_funcs

    if _descriptor_funcs is not None:
        return _descriptor_funcs

    from rdkit.Chem import Descriptors

    # Build dictionary from ALL RDKit descriptors
    _descriptor_funcs = {name: func for name, func in Descriptors.descList}

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
        arr = np.array(values, dtype=np.float64)
        # Replace inf/nan and clip to float32 range to avoid overflow
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -3.4e38, 3.4e38)  # float32 max is ~3.4e38
        return arr.astype(np.float32)
    except Exception:
        return None


def _calc_desc_worker(args):
    """Worker function for parallel descriptor calculation."""
    smiles, descriptor_names = args
    return calculate_descriptors(smiles, descriptor_names)


def calculate_descriptors_batch(
    smiles_list: List[str],
    descriptor_names: Tuple[str, ...] = DESCRIPTOR_NAMES,
    return_valid_mask: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculate descriptors for multiple molecules in parallel.

    Args:
        smiles_list: List of SMILES strings
        descriptor_names: Tuple of descriptor names to calculate
        return_valid_mask: Whether to return mask of valid molecules
        n_jobs: Number of parallel workers (-1 for all CPUs)

    Returns:
        Tuple of (descriptors array, valid mask if requested)
        Descriptors array has shape (n_valid, n_descriptors)
    """
    import os
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Prepare arguments
    args = [(smiles, descriptor_names) for smiles in smiles_list]

    # Parallel processing with progress bar
    descriptors = []
    valid_mask = []

    with Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_calc_desc_worker, args, chunksize=100),
            total=len(smiles_list),
            desc=f"Calculating descriptors ({n_jobs} workers)"
        ))

    for desc in results:
        if desc is not None:
            descriptors.append(desc)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    result = np.array(descriptors) if descriptors else np.empty((0, len(descriptor_names)))

    if return_valid_mask:
        return result, np.array(valid_mask)
    return result, None
