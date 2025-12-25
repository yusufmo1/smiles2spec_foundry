"""
RL Molecule Game - Reinforcement Learning for Spectrum-to-SMILES Generation

This package implements a game-like environment where an RL agent learns to
build molecules token-by-token to match a target mass spectrum.

Architecture:
    Target Spectrum -> RL Agent -> Build Molecule -> Reward (spectrum match)
                         ^                              |
                         +-------- Learn <--------------+
"""

from .environment import MoleculeGameEnv
from .agent import MoleculePolicy, PPOAgent
from .rewards import RewardComputer

__version__ = "0.1.0"
__all__ = ["MoleculeGameEnv", "MoleculePolicy", "PPOAgent", "RewardComputer"]
