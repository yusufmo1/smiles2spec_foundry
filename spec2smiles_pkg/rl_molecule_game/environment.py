"""
Gym-style environment for RL molecule generation.

The agent builds molecules token-by-token (SELFIES) to match a target spectrum.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import selfies as sf

from .rewards import RewardComputer, calculate_basic_descriptors


@dataclass
class EnvConfig:
    """Configuration for the molecule game environment."""
    max_length: int = 100          # Max SELFIES tokens
    spectrum_dim: int = 500        # Spectrum bins
    descriptor_dim: int = 30       # Number of descriptors
    intermediate_reward: float = 0.01


class MoleculeGameEnv:
    """
    Gym-style environment for molecule building.

    State Space:
        - current_tokens: Token indices generated so far (max_length,)
        - target_spectrum: Target mass spectrum (500,)
        - target_descriptors: Target molecular descriptors (30,)
        - step_count: Current step number

    Action Space:
        - Discrete: Select next SELFIES token (vocab_size)

    Reward:
        - Terminal: Multi-component reward (spectrum similarity, Tanimoto, etc.)
        - Intermediate: Small positive reward for progress

    Done:
        - When END token is selected
        - When max_length is reached
    """

    def __init__(
        self,
        encoder,  # SELFIESEncoder instance
        config: Optional[EnvConfig] = None,
        reward_computer: Optional[RewardComputer] = None,
    ):
        """
        Initialize environment.

        Args:
            encoder: SELFIESEncoder with vocabulary
            config: Environment configuration
            reward_computer: Reward computation module
        """
        self.encoder = encoder
        self.config = config or EnvConfig()
        self.reward_computer = reward_computer or RewardComputer(
            descriptor_calculator=calculate_basic_descriptors,
            intermediate_reward=self.config.intermediate_reward,
        )

        # Action space is vocabulary size
        self.vocab_size = encoder.vocab_size
        self.action_dim = self.vocab_size

        # Special token indices from encoder
        self.pad_idx = encoder.PAD_IDX
        self.start_idx = encoder.START_IDX
        self.end_idx = encoder.END_IDX

        # Episode state
        self.current_tokens = []
        self.target_spectrum = None
        self.target_descriptors = None
        self.target_smiles = None
        self.step_count = 0
        self.done = False

    def reset(
        self,
        target_spectrum: np.ndarray,
        target_descriptors: np.ndarray,
        target_smiles: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reset environment for new episode.

        Args:
            target_spectrum: Target mass spectrum to match
            target_descriptors: Target molecular descriptors
            target_smiles: Ground truth SMILES (optional, for evaluation)

        Returns:
            Initial observation dictionary
        """
        self.target_spectrum = target_spectrum
        self.target_descriptors = target_descriptors
        self.target_smiles = target_smiles

        # Start with START token
        self.current_tokens = [self.start_idx]
        self.step_count = 0
        self.done = False

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Token index to add to the sequence

        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Add token to sequence
        self.current_tokens.append(action)
        self.step_count += 1

        # Check termination conditions
        self.done = (
            action == self.end_idx or
            self.step_count >= self.config.max_length
        )

        # Compute reward
        if self.done:
            reward, info = self._compute_terminal_reward()
        else:
            reward = self.reward_computer.intermediate_reward(
                self.step_count, self.config.max_length
            )
            info = {"intermediate": True}

        return self._get_observation(), reward, self.done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.

        Returns:
            Dictionary with:
                - tokens: Padded token sequence (max_length,)
                - token_mask: Mask for valid tokens (max_length,)
                - spectrum: Target spectrum (500,)
                - descriptors: Target descriptors (30,)
        """
        # Pad tokens to max length (truncate if needed)
        tokens = np.zeros(self.config.max_length, dtype=np.int64)
        n_tokens = min(len(self.current_tokens), self.config.max_length)
        tokens[:n_tokens] = self.current_tokens[:n_tokens]

        # Create attention mask
        mask = np.zeros(self.config.max_length, dtype=np.float32)
        mask[:n_tokens] = 1.0

        return {
            "tokens": tokens,
            "token_mask": mask,
            "spectrum": self.target_spectrum.astype(np.float32),
            "descriptors": self.target_descriptors.astype(np.float32),
            "step": np.array([self.step_count], dtype=np.int64),
        }

    def _compute_terminal_reward(self) -> Tuple[float, Dict]:
        """
        Compute reward at end of episode.

        Returns:
            reward: Total reward
            info: Dictionary with reward breakdown
        """
        # Decode generated tokens to SMILES
        generated_smiles = self.encoder.decode(self.current_tokens)

        if generated_smiles is None:
            # Failed to decode - minimal reward
            return 0.0, {
                "valid": False,
                "smiles": None,
                "tanimoto": 0.0,
            }

        # Compute multi-component reward
        reward_dict = self.reward_computer.terminal_reward(
            smiles=generated_smiles,
            target_descriptors=self.target_descriptors,
            target_smiles=self.target_smiles,
        )

        info = {
            "valid": True,
            "smiles": generated_smiles,
            "tanimoto": reward_dict.get("tanimoto", 0.0),
            "descriptor_match": reward_dict.get("descriptor_match", 0.0),
            "length_efficiency": reward_dict.get("length_efficiency", 0.0),
            "reward_breakdown": reward_dict,
        }

        # Check for exact match
        if self.target_smiles is not None:
            from rdkit import Chem
            gen_mol = Chem.MolFromSmiles(generated_smiles)
            target_mol = Chem.MolFromSmiles(self.target_smiles)
            if gen_mol and target_mol:
                gen_canon = Chem.MolToSmiles(gen_mol, canonical=True)
                target_canon = Chem.MolToSmiles(target_mol, canonical=True)
                info["exact_match"] = (gen_canon == target_canon)
            else:
                info["exact_match"] = False

        return reward_dict["total"], info

    def get_generated_smiles(self) -> Optional[str]:
        """Get the SMILES string for the current token sequence."""
        return self.encoder.decode(self.current_tokens)

    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions at current state.

        For SELFIES, all tokens are always valid (guaranteed validity).
        But we can mask PAD token since it shouldn't be generated.

        Returns:
            Boolean mask of valid actions (vocab_size,)
        """
        valid = np.ones(self.vocab_size, dtype=bool)
        valid[self.pad_idx] = False  # Never generate PAD
        return valid

    def render(self) -> str:
        """Render current state as string."""
        smiles = self.get_generated_smiles() or "[incomplete]"
        return (
            f"Step: {self.step_count}/{self.config.max_length}\n"
            f"Tokens: {self.current_tokens}\n"
            f"SMILES: {smiles}\n"
            f"Done: {self.done}"
        )


class BatchMoleculeGameEnv:
    """
    Batched version of MoleculeGameEnv for efficient parallel rollouts.
    """

    def __init__(
        self,
        encoder,
        batch_size: int,
        config: Optional[EnvConfig] = None,
        reward_computer: Optional[RewardComputer] = None,
    ):
        """
        Initialize batched environment.

        Args:
            encoder: SELFIESEncoder
            batch_size: Number of parallel environments
            config: Environment configuration
            reward_computer: Reward computation module
        """
        self.encoder = encoder
        self.batch_size = batch_size
        self.config = config or EnvConfig()
        self.reward_computer = reward_computer or RewardComputer(
            descriptor_calculator=calculate_basic_descriptors,
        )

        self.vocab_size = encoder.vocab_size
        self.pad_idx = encoder.PAD_IDX
        self.start_idx = encoder.START_IDX
        self.end_idx = encoder.END_IDX

        # Batch state
        self.current_tokens = None  # (batch_size, max_length)
        self.token_lengths = None   # (batch_size,)
        self.target_spectra = None
        self.target_descriptors = None
        self.target_smiles = None
        self.done_mask = None       # (batch_size,)

    def reset(
        self,
        target_spectra: np.ndarray,
        target_descriptors: np.ndarray,
        target_smiles: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset all environments in batch."""
        batch_size = target_spectra.shape[0]

        self.target_spectra = target_spectra
        self.target_descriptors = target_descriptors
        self.target_smiles = target_smiles

        # Initialize with START tokens
        self.current_tokens = np.full(
            (batch_size, self.config.max_length),
            self.pad_idx,
            dtype=np.int64
        )
        self.current_tokens[:, 0] = self.start_idx
        self.token_lengths = np.ones(batch_size, dtype=np.int64)
        self.done_mask = np.zeros(batch_size, dtype=bool)

        return self._get_batch_observation()

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, list]:
        """Take batched step."""
        batch_size = actions.shape[0]

        # Add tokens only for non-done environments
        for i in range(batch_size):
            if not self.done_mask[i]:
                pos = self.token_lengths[i]
                if pos < self.config.max_length:
                    self.current_tokens[i, pos] = actions[i]
                    self.token_lengths[i] += 1

                # Check done
                if actions[i] == self.end_idx or self.token_lengths[i] >= self.config.max_length:
                    self.done_mask[i] = True

        # Compute rewards
        rewards = np.zeros(batch_size, dtype=np.float32)
        infos = []

        for i in range(batch_size):
            if self.done_mask[i]:
                smiles = self.encoder.decode(self.current_tokens[i, :self.token_lengths[i]].tolist())
                target_smi = self.target_smiles[i] if self.target_smiles else None

                if smiles is None:
                    rewards[i] = 0.0
                    infos.append({"valid": False})
                else:
                    reward_dict = self.reward_computer.terminal_reward(
                        smiles=smiles,
                        target_descriptors=self.target_descriptors[i],
                        target_smiles=target_smi,
                    )
                    rewards[i] = reward_dict["total"]
                    infos.append({"valid": True, "smiles": smiles, "reward": reward_dict})
            else:
                rewards[i] = self.reward_computer.intermediate_reward(
                    int(self.token_lengths[i]), self.config.max_length
                )
                infos.append({"intermediate": True})

        return self._get_batch_observation(), rewards, self.done_mask.copy(), infos

    def _get_batch_observation(self) -> Dict[str, np.ndarray]:
        """Get batched observations."""
        batch_size = self.current_tokens.shape[0]

        # Create masks
        masks = np.zeros((batch_size, self.config.max_length), dtype=np.float32)
        for i in range(batch_size):
            masks[i, :self.token_lengths[i]] = 1.0

        return {
            "tokens": self.current_tokens.copy(),
            "token_mask": masks,
            "spectrum": self.target_spectra.astype(np.float32),
            "descriptors": self.target_descriptors.astype(np.float32),
        }
