"""Part B service - Descriptors to SMILES using Conditional VAE."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import settings
from src.models.selfies_encoder import SELFIESEncoder
from src.models.vae import ConditionalVAE
from src.utils.exceptions import ModelError


class PartBService:
    """Service for training and inference of Descriptors -> SMILES model."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize Part B service.

        Args:
            device: PyTorch device (defaults to settings.torch_device)
        """
        self.device = device or settings.torch_device
        self.encoder: Optional[SELFIESEncoder] = None
        self.model: Optional[ConditionalVAE] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained

    def prepare_data(
        self,
        smiles_list: List[str],
        descriptors: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare training data by building vocabulary and encoding SMILES.

        Args:
            smiles_list: List of SMILES strings
            descriptors: Descriptor array of shape (n_samples, n_descriptors)
            verbose: Whether to show progress

        Returns:
            Tuple of (encoded_tokens, filtered_descriptors, valid_indices)
        """
        # Initialize encoder and build vocabulary
        self.encoder = SELFIESEncoder(max_len=settings.vae_max_seq_len)
        _, valid_indices = self.encoder.build_vocab_from_smiles(smiles_list, verbose=verbose)

        # Encode SMILES to tokens
        encoded, _, _ = self.encoder.batch_encode(
            [smiles_list[i] for i in valid_indices], verbose=verbose
        )

        # Filter descriptors to match valid molecules
        filtered_descriptors = descriptors[valid_indices]

        return encoded, filtered_descriptors, valid_indices

    def train(
        self,
        encoded_tokens: np.ndarray,
        descriptors: np.ndarray,
        val_tokens: Optional[np.ndarray] = None,
        val_descriptors: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train Conditional VAE on encoded data.

        Args:
            encoded_tokens: Encoded SELFIES tokens of shape (n_samples, max_len)
            descriptors: Descriptor array of shape (n_samples, n_descriptors)
            val_tokens: Optional validation tokens
            val_descriptors: Optional validation descriptors
            verbose: Whether to show progress

        Returns:
            Dictionary of training history (loss curves)
        """
        if self.encoder is None:
            raise ModelError("Encoder not initialized. Call prepare_data first.")

        # Create model
        self.model = ConditionalVAE(
            vocab_size=self.encoder.vocab_size,
            descriptor_dim=descriptors.shape[1],
            latent_dim=settings.vae_latent_dim,
            hidden_dim=settings.vae_hidden_dim,
            n_layers=settings.vae_n_layers,
            dropout=settings.vae_dropout,
            max_len=settings.vae_max_seq_len,
        )
        self.model.to(self.device)

        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=settings.vae_learning_rate
        )

        history = {"train_loss": [], "val_loss": [], "kl_loss": [], "recon_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        n_epochs = settings.vae_n_epochs
        batch_size = settings.vae_batch_size

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = []

            # Create batches
            indices = np.random.permutation(len(encoded_tokens))
            n_batches = len(indices) // batch_size

            iterator = range(n_batches)
            if verbose:
                iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{n_epochs}")

            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]

                tokens = torch.LongTensor(encoded_tokens[batch_indices]).to(self.device)
                desc = torch.FloatTensor(descriptors[batch_indices]).to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits, targets, q_mean, q_logvar, p_mean, p_logvar = self.model(
                    tokens, desc
                )

                # Compute loss
                recon_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SELFIESEncoder.PAD_IDX,
                )

                # KL divergence between posterior and prior
                kl_loss = 0.5 * torch.mean(
                    p_logvar - q_logvar
                    + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2)
                    / torch.exp(p_logvar)
                    - 1
                )

                # Cyclical KL annealing
                cycle_pos = epoch % settings.vae_kl_cycle_length
                beta = min(1.0, cycle_pos / (settings.vae_kl_cycle_length / 2))

                loss = recon_loss + beta * kl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), settings.vae_gradient_clip
                )
                optimizer.step()

                epoch_losses.append(loss.item())

            # Record training loss
            train_loss = np.mean(epoch_losses)
            history["train_loss"].append(train_loss)

            # Validation
            if val_tokens is not None and val_descriptors is not None:
                val_loss = self._compute_validation_loss(val_tokens, val_descriptors)
                history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch + 1}: train_loss={train_loss:.4f}"
                if val_tokens is not None:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}"
                print(msg)

        self._trained = True
        return history

    def _compute_validation_loss(
        self, tokens: np.ndarray, descriptors: np.ndarray
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        with torch.no_grad():
            tokens_t = torch.LongTensor(tokens).to(self.device)
            desc_t = torch.FloatTensor(descriptors).to(self.device)

            logits, targets, q_mean, q_logvar, p_mean, p_logvar = self.model(
                tokens_t, desc_t
            )

            recon_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=SELFIESEncoder.PAD_IDX,
            )

            kl_loss = 0.5 * torch.mean(
                p_logvar - q_logvar
                + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2)
                / torch.exp(p_logvar)
                - 1
            )

            return (recon_loss + kl_loss).item()

    def generate(
        self,
        descriptors: np.ndarray,
        n_candidates: int = 50,
        temperature: float = 0.7,
    ) -> List[List[str]]:
        """Generate SMILES candidates from descriptors.

        Args:
            descriptors: Descriptor array of shape (n_samples, n_descriptors)
            n_candidates: Number of candidates per sample
            temperature: Sampling temperature

        Returns:
            List of candidate SMILES lists (one per sample)
        """
        if not self._trained:
            raise ModelError("Model must be trained before generation")

        self.model.eval()
        all_candidates = []

        desc_tensor = torch.FloatTensor(descriptors).to(self.device)

        for i in range(len(descriptors)):
            sample_desc = desc_tensor[i : i + 1]
            candidates = self.model.generate(
                sample_desc, n_samples=n_candidates, temperature=temperature
            )

            # Decode candidates
            smiles_candidates = []
            for cand_tokens in candidates:
                smiles = self.encoder.decode(cand_tokens[0].cpu().numpy().tolist())
                if smiles is not None:
                    smiles_candidates.append(smiles)

            # Remove duplicates
            unique_candidates = list(dict.fromkeys(smiles_candidates))
            all_candidates.append(unique_candidates)

        return all_candidates

    # ===========================================
    # Persistence
    # ===========================================
    def save(self, output_dir: Path) -> None:
        """Save trained model and encoder.

        Args:
            output_dir: Output directory
        """
        if not self._trained:
            raise ModelError("Cannot save untrained model")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as integration package
        package = {
            "model_state_dict": self.model.state_dict(),
            "encoder_state": self.encoder.get_state(),
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "descriptor_dim": self.model.descriptor_dim,
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "n_layers": self.model.n_layers,
                "max_len": self.model.max_len,
            },
        }

        with open(output_dir / "integration_package.pkl", "wb") as f:
            pickle.dump(package, f)

        # Also save separate model file
        self.model.save(output_dir / "vae_model.pt")

    def load(self, model_dir: Path) -> "PartBService":
        """Load trained model and encoder.

        Args:
            model_dir: Directory containing model files

        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)

        # Try integration package first
        integration_path = model_dir / "integration_package.pkl"
        if integration_path.exists():
            with open(integration_path, "rb") as f:
                package = pickle.load(f)

            # Load encoder
            self.encoder = SELFIESEncoder.from_state(package["encoder_state"])

            # Load model
            config = package["model_config"]
            self.model = ConditionalVAE(
                vocab_size=config["vocab_size"],
                descriptor_dim=config["descriptor_dim"],
                latent_dim=config["latent_dim"],
                hidden_dim=config["hidden_dim"],
                n_layers=config["n_layers"],
                max_len=config["max_len"],
            )
            self.model.load_state_dict(package["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        else:
            # Load from separate files
            encoder_path = model_dir / "encoder.pkl"
            if not encoder_path.exists():
                raise ModelError(f"Encoder not found: {encoder_path}")

            with open(encoder_path, "rb") as f:
                encoder_state = pickle.load(f)
            self.encoder = SELFIESEncoder.from_state(encoder_state)

            vae_path = model_dir / "vae_model.pt"
            if not vae_path.exists():
                vae_path = model_dir / "best_model.pt"

            if not vae_path.exists():
                raise ModelError(f"VAE model not found in {model_dir}")

            self.model = ConditionalVAE.load(vae_path, self.device)
            self.model.eval()

        self._trained = True
        return self
