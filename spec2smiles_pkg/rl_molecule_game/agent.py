"""
PPO Agent for molecule generation.

Implements actor-critic architecture where:
- Actor: Outputs action probabilities (next token)
- Critic: Estimates state value for variance reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the PPO agent."""
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    # PPO hyperparameters
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class MoleculePolicy(nn.Module):
    """
    Policy network for molecule generation.

    Architecture:
    - Spectrum encoder: MLP to encode target spectrum
    - Descriptor encoder: MLP to encode target descriptors
    - Token encoder: LSTM to encode current sequence
    - Policy head: Outputs action logits
    - Value head: Outputs state value
    """

    def __init__(
        self,
        vocab_size: int,
        spectrum_dim: int = 500,
        descriptor_dim: int = 30,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Spectrum encoder
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Descriptor encoder
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Token embedding and LSTM encoder
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Combine all encodings
        combined_dim = hidden_dim * 3  # spectrum + descriptor + sequence

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        spectrum: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            tokens: Token indices (batch, seq_len)
            token_mask: Attention mask (batch, seq_len)
            spectrum: Target spectrum (batch, 500)
            descriptors: Target descriptors (batch, 30)

        Returns:
            logits: Action logits (batch, vocab_size)
            value: State value (batch, 1)
        """
        batch_size = tokens.shape[0]

        # Encode spectrum
        spec_enc = self.spectrum_encoder(spectrum)  # (batch, hidden)

        # Encode descriptors
        desc_enc = self.descriptor_encoder(descriptors)  # (batch, hidden)

        # Encode token sequence
        tok_emb = self.token_embedding(tokens)  # (batch, seq, hidden)

        # Pack and run through LSTM
        lengths = token_mask.sum(dim=1).long().cpu()
        lengths = lengths.clamp(min=1)  # Ensure at least 1

        packed = nn.utils.rnn.pack_padded_sequence(
            tok_emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.sequence_encoder(packed)

        # Take last layer's hidden state
        seq_enc = hidden[-1]  # (batch, hidden)

        # Combine all encodings
        combined = torch.cat([spec_enc, desc_enc, seq_enc], dim=-1)

        # Get policy logits and value
        logits = self.policy_head(combined)
        value = self.value_head(combined)

        return logits, value

    def get_action(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        spectrum: torch.Tensor,
        descriptors: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            tokens, token_mask, spectrum, descriptors: Observation tensors
            deterministic: If True, take argmax action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of distribution
            value: State value estimate
        """
        logits, value = self.forward(tokens, token_mask, spectrum, descriptors)

        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        spectrum: torch.Tensor,
        descriptors: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given actions.

        Used during PPO update to compute ratio and value loss.

        Args:
            tokens, token_mask, spectrum, descriptors: Observation tensors
            actions: Actions taken

        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of distribution
            value: State value estimate
        """
        logits, value = self.forward(tokens, token_mask, spectrum, descriptors)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO agent for molecule generation.

    Handles:
    - Action selection
    - Experience storage
    - PPO updates
    """

    def __init__(
        self,
        policy: MoleculePolicy,
        config: Optional[AgentConfig] = None,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.config = config or AgentConfig()
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
        )

        # Experience buffer
        self.buffer = {
            "tokens": [],
            "token_masks": [],
            "spectra": [],
            "descriptors": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.

        Args:
            observation: Dictionary with tokens, spectrum, descriptors
            deterministic: Whether to use greedy action selection

        Returns:
            action: Selected action
            log_prob: Log probability
            value: Value estimate
        """
        with torch.no_grad():
            tokens = torch.tensor(observation["tokens"], device=self.device).unsqueeze(0)
            mask = torch.tensor(observation["token_mask"], device=self.device).unsqueeze(0)
            spectrum = torch.tensor(observation["spectrum"], device=self.device).unsqueeze(0)
            descriptors = torch.tensor(observation["descriptors"], device=self.device).unsqueeze(0)

            action, log_prob, _, value = self.policy.get_action(
                tokens, mask, spectrum, descriptors, deterministic
            )

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.buffer["tokens"].append(observation["tokens"])
        self.buffer["token_masks"].append(observation["token_mask"])
        self.buffer["spectra"].append(observation["spectrum"])
        self.buffer["descriptors"].append(observation["descriptors"])
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["values"].append(value)
        self.buffer["dones"].append(done)

    def clear_buffer(self):
        """Clear experience buffer."""
        for key in self.buffer:
            self.buffer[key] = []

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.buffer["rewards"])
        values = np.array(self.buffer["values"])
        dones = np.array(self.buffer["dones"])

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        # Compute GAE
        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t + 1])

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(
        self,
        n_epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            n_epochs: Number of epochs to train on collected data
            batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            Dictionary with loss statistics
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns(gamma, gae_lambda)

        # Convert buffer to tensors
        tokens = torch.tensor(np.array(self.buffer["tokens"]), device=self.device)
        masks = torch.tensor(np.array(self.buffer["token_masks"]), device=self.device)
        spectra = torch.tensor(np.array(self.buffer["spectra"]), device=self.device)
        descriptors = torch.tensor(np.array(self.buffer["descriptors"]), device=self.device)
        actions = torch.tensor(np.array(self.buffer["actions"]), device=self.device)
        old_log_probs = torch.tensor(np.array(self.buffer["log_probs"]), device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n_samples = len(tokens)
        indices = np.arange(n_samples)

        # Training statistics
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
        }

        for _ in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                batch_tokens = tokens[batch_idx]
                batch_masks = masks[batch_idx]
                batch_spectra = spectra[batch_idx]
                batch_desc = descriptors[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_tokens, batch_masks, batch_spectra, batch_desc, batch_actions
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track stats
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean().item()
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(-entropy_loss.item())
                    stats["kl_div"].append(kl)

        # Clear buffer after update
        self.clear_buffer()

        return {k: np.mean(v) for k, v in stats.items()}

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "config" in checkpoint:
            self.config = checkpoint["config"]
