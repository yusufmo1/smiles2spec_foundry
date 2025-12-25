"""Evaluation metrics for SPEC2SMILES pipeline."""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_tanimoto(smiles1: str, smiles2: str, radius: int = 2) -> float:
    """Compute Tanimoto similarity between two molecules.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        radius: Morgan fingerprint radius (default 2 = ECFP4)

    Returns:
        Tanimoto similarity between 0 and 1
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def compute_batch_tanimoto(
    candidates_list: List[List[str]],
    true_smiles_list: List[str],
    radius: int = 2,
) -> Tuple[List[float], List[float]]:
    """Compute best Tanimoto similarities for batch of predictions.

    Returns:
        Tuple of (best_similarities, mean_similarities)
    """
    best_sims = []
    mean_sims = []

    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        if not candidates:
            best_sims.append(0.0)
            mean_sims.append(0.0)
            continue

        sims = [compute_tanimoto(cand, true_smiles, radius) for cand in candidates]
        best_sims.append(max(sims))
        mean_sims.append(np.mean(sims))

    return best_sims, mean_sims


def compute_hit_at_k(
    candidates_list: List[List[str]],
    true_smiles_list: List[str],
    k: int = 1,
) -> float:
    """Compute Hit@K metric (exact match rate in top K candidates)."""
    hits = 0
    total = 0

    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                continue

            true_canonical = Chem.MolToSmiles(true_mol, canonical=True)
            top_k = candidates[:k]

            for cand in top_k:
                cand_mol = Chem.MolFromSmiles(cand)
                if cand_mol is not None:
                    cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
                    if cand_canonical == true_canonical:
                        hits += 1
                        break

            total += 1
        except Exception:
            continue

    return hits / total if total > 0 else 0.0


def compute_validity_rate(candidates_list: List[List[str]]) -> float:
    """Compute chemical validity rate of generated SMILES."""
    valid = 0
    total = 0

    for candidates in candidates_list:
        for smiles in candidates:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid += 1
            total += 1

    return valid / total if total > 0 else 0.0


def compute_uniqueness(candidates_list: List[List[str]]) -> float:
    """Compute uniqueness of generated molecules."""
    all_smiles = set()
    total = 0

    for candidates in candidates_list:
        for smiles in candidates:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                all_smiles.add(canonical)
                total += 1

    return len(all_smiles) / total if total > 0 else 0.0


def compute_diversity(candidates: List[str], radius: int = 2) -> float:
    """Compute diversity (average pairwise Tanimoto distance) of candidates."""
    if len(candidates) < 2:
        return 0.0

    fps = []
    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1 - sim)

    return np.mean(distances)
