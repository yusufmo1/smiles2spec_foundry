"""
Reward computation for the RL molecule generation game.

Multi-component reward function that balances:
- Spectrum similarity (main objective)
- Descriptor matching (auxiliary signal)
- Validity (guaranteed with SELFIES, but included for completeness)
- Length efficiency (encourage concise molecules)
"""

import numpy as np
from typing import Dict, Optional, Callable
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import selfies as sf


class RewardComputer:
    """
    Computes rewards for molecule generation.

    Terminal rewards are given at the end of an episode when the molecule
    is complete. Intermediate rewards provide shaping signal during generation.
    """

    def __init__(
        self,
        descriptor_calculator: Optional[Callable] = None,
        weights: Optional[Dict[str, float]] = None,
        intermediate_reward: float = 0.01,
    ):
        """
        Args:
            descriptor_calculator: Function that computes descriptors from SMILES
            weights: Dictionary of reward component weights
            intermediate_reward: Small reward given at each step
        """
        self.descriptor_calculator = descriptor_calculator
        self.weights = weights or {
            'spectrum_similarity': 0.4,
            'descriptor_match': 0.3,
            'validity': 0.2,
            'length_efficiency': 0.1,
        }
        self.intermediate_reward_value = intermediate_reward

    def terminal_reward(
        self,
        smiles: str,
        target_descriptors: np.ndarray,
        target_smiles: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute reward at end of episode.

        Args:
            smiles: Generated SMILES string
            target_descriptors: Target molecular descriptors (30-dim)
            target_smiles: Ground truth SMILES (optional, for Tanimoto)

        Returns:
            Dictionary with individual reward components and total
        """
        rewards = {}

        # 1. Validity check
        mol = Chem.MolFromSmiles(smiles)
        rewards['validity'] = 1.0 if mol is not None else 0.0

        if mol is None:
            # Invalid molecule gets minimal reward
            rewards['total'] = 0.0
            return rewards

        # 2. Descriptor matching
        if self.descriptor_calculator is not None:
            try:
                pred_descriptors = self.descriptor_calculator(smiles)
                # Normalized MAE -> similarity score
                desc_error = np.abs(pred_descriptors - target_descriptors)
                # Scale by typical descriptor ranges
                desc_ranges = np.abs(target_descriptors) + 1e-6
                normalized_error = (desc_error / desc_ranges).mean()
                rewards['descriptor_match'] = np.exp(-normalized_error)
            except Exception:
                rewards['descriptor_match'] = 0.0
        else:
            rewards['descriptor_match'] = 0.5  # Default if no calculator

        # 3. Tanimoto similarity (if target SMILES provided)
        if target_smiles is not None:
            tanimoto = self._compute_tanimoto(smiles, target_smiles)
            rewards['tanimoto'] = tanimoto
        else:
            rewards['tanimoto'] = 0.0

        # 4. Length efficiency (prefer concise molecules)
        canonical = Chem.MolToSmiles(mol)
        max_length = 150  # Typical max SMILES length
        rewards['length_efficiency'] = max(0, 1.0 - len(canonical) / max_length)

        # 5. Compute weighted total
        total = 0.0
        for key, weight in self.weights.items():
            if key in rewards:
                total += weight * rewards[key]

        # Add Tanimoto bonus if available (not in base weights)
        if target_smiles is not None:
            total += 0.3 * rewards['tanimoto']  # Extra bonus for structural match

        rewards['total'] = total
        return rewards

    def intermediate_reward(self, step: int, max_steps: int) -> float:
        """
        Small shaping reward during molecule construction.

        Encourages progress without overwhelming the terminal reward.

        Args:
            step: Current step number
            max_steps: Maximum steps allowed

        Returns:
            Small positive reward
        """
        # Slight decay to encourage finishing early
        decay = 1.0 - (step / max_steps) * 0.5
        return self.intermediate_reward_value * decay

    def _compute_tanimoto(self, smiles1: str, smiles2: str) -> float:
        """Compute Tanimoto similarity between two molecules."""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.0


def calculate_basic_descriptors(smiles: str) -> np.ndarray:
    """
    Calculate a subset of molecular descriptors for reward computation.

    Uses fast descriptors that don't require 3D conformer generation.

    Args:
        smiles: SMILES string

    Returns:
        Array of descriptor values (30 dimensions to match training)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(30)

    # Calculate commonly used descriptors
    descriptors = [
        Descriptors.MolWt(mol),           # Molecular weight
        Descriptors.MolLogP(mol),         # LogP
        Descriptors.TPSA(mol),            # Topological polar surface area
        Descriptors.NumHDonors(mol),      # H-bond donors
        Descriptors.NumHAcceptors(mol),   # H-bond acceptors
        Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
        Descriptors.NumAromaticRings(mol),   # Aromatic rings
        Descriptors.NumHeteroatoms(mol),     # Heteroatoms
        Descriptors.FractionCSP3(mol),       # Fraction sp3 carbons
        Descriptors.NumAliphaticRings(mol),  # Aliphatic rings
        Descriptors.HeavyAtomCount(mol),     # Heavy atoms
        Descriptors.NumValenceElectrons(mol),  # Valence electrons
        Descriptors.NHOHCount(mol),          # NH and OH count
        Descriptors.NOCount(mol),            # NO count
        Descriptors.RingCount(mol),          # Total rings
        Descriptors.BertzCT(mol),            # Bertz CT complexity
        Descriptors.Chi0(mol),               # Chi0
        Descriptors.Chi1(mol),               # Chi1
        Descriptors.HallKierAlpha(mol),      # Hall-Kier alpha
        Descriptors.Kappa1(mol),             # Kappa1
        Descriptors.Kappa2(mol),             # Kappa2
        Descriptors.LabuteASA(mol),          # Labute ASA
        Descriptors.MolMR(mol),              # Molar refractivity
        Descriptors.NumSaturatedRings(mol),  # Saturated rings
        Descriptors.NumAromaticHeterocycles(mol),  # Aromatic heterocycles
        Descriptors.NumSaturatedHeterocycles(mol), # Saturated heterocycles
        Descriptors.NumAliphaticHeterocycles(mol), # Aliphatic heterocycles
        Descriptors.NumAromaticCarbocycles(mol),   # Aromatic carbocycles
        Descriptors.NumSaturatedCarbocycles(mol),  # Saturated carbocycles
        Descriptors.NumAliphaticCarbocycles(mol),  # Aliphatic carbocycles
    ]

    return np.array(descriptors, dtype=np.float32)


def exact_match_reward(generated: str, target: str) -> float:
    """
    Binary reward for exact canonical SMILES match.

    Args:
        generated: Generated SMILES
        target: Target SMILES

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    try:
        mol1 = Chem.MolFromSmiles(generated)
        mol2 = Chem.MolFromSmiles(target)

        if mol1 is None or mol2 is None:
            return 0.0

        canon1 = Chem.MolToSmiles(mol1, canonical=True)
        canon2 = Chem.MolToSmiles(mol2, canonical=True)

        return 1.0 if canon1 == canon2 else 0.0
    except Exception:
        return 0.0
