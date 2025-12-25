"""Part B service - Descriptors to SMILES using Conditional VAE or DirectDecoder."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import settings
from src.data.augment import augment_dataset
from src.models.direct_decoder import DirectDecoder
from src.models.selfies_encoder import SELFIESEncoder
from src.models.vae import ConditionalVAE
from src.services.scaler import ScalerService
from src.utils.exceptions import ModelError
from src.utils.logging import TrainingLogger


class PartBService:
    """Service for training and inference of Descriptors -> SMILES model.

    Supports both ConditionalVAE and DirectDecoder models.
    Uses ScalerService for descriptor scaling consistency with Part A.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        scaler: Optional[ScalerService] = None,
        model_type: Optional[str] = None,
    ):
        """Initialize Part B service.

        Args:
            device: PyTorch device (defaults to settings.torch_device)
            scaler: ScalerService instance (uses singleton if not provided)
            model_type: "vae" or "direct" (defaults to settings.part_b_model)
        """
        self.device = device or settings.torch_device
        self.scaler = scaler or ScalerService.get_instance()
        self.model_type = model_type or settings.part_b_model
        self.encoder: Optional[SELFIESEncoder] = None
        self.model: Optional[Union[ConditionalVAE, DirectDecoder]] = None
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
        augment: Optional[bool] = None,
        n_augment: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare training data by building vocabulary and encoding SMILES.

        Args:
            smiles_list: List of SMILES strings
            descriptors: Descriptor array of shape (n_samples, n_descriptors)
            verbose: Whether to show progress
            augment: Whether to augment data (defaults to settings.augment_enabled)
            n_augment: Number of augmentations per sample (defaults to settings.n_augment)

        Returns:
            Tuple of (encoded_tokens, filtered_descriptors, valid_indices)
        """
        augment = augment if augment is not None else settings.augment_enabled
        n_augment = n_augment if n_augment is not None else settings.n_augment

        # Apply SMILES augmentation if enabled
        if augment and n_augment > 0:
            if verbose:
                print(f"Applying {n_augment}x SMILES augmentation...")
            smiles_list, descriptors = augment_dataset(smiles_list, descriptors, n_augment)
            if verbose:
                print(f"  Augmented dataset size: {len(smiles_list)}")

        # Get max_seq_len from appropriate config
        max_len = settings.vae.max_seq_len if self.model_type == "vae" else settings.direct.max_seq_len

        # Initialize encoder and build vocabulary
        self.encoder = SELFIESEncoder(max_len=max_len)
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
        log_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Train model on encoded data (VAE or DirectDecoder based on model_type).

        Args:
            encoded_tokens: Encoded SELFIES tokens of shape (n_samples, max_len)
            descriptors: Descriptor array of shape (n_samples, n_descriptors)
            val_tokens: Optional validation tokens
            val_descriptors: Optional validation descriptors
            verbose: Whether to show progress
            log_dir: Optional directory for live epoch logging

        Returns:
            Dictionary of training history (loss curves)
        """
        if self.encoder is None:
            raise ModelError("Encoder not initialized. Call prepare_data first.")

        if self.model_type == "direct":
            return self._train_direct(encoded_tokens, descriptors, val_tokens, val_descriptors, verbose, log_dir)

        # VAE training (default)
        cfg = settings.vae
        self.model = ConditionalVAE(
            vocab_size=self.encoder.vocab_size,
            descriptor_dim=descriptors.shape[1],
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_seq_len,
        )
        self.model.to(self.device)

        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate
        )

        history = {"train_loss": [], "val_loss": [], "kl_loss": [], "recon_loss": [], "beta": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        n_epochs = cfg.n_epochs
        batch_size = cfg.batch_size

        # Setup live epoch logging
        epoch_logger = None
        if log_dir is not None:
            epoch_logger = TrainingLogger(
                log_dir=Path(log_dir),
                model_name="vae",
                metrics=["train_loss", "val_loss", "recon_loss", "kl_loss", "beta"],
            )
            if verbose:
                print(f"  - Epoch log: {epoch_logger.filepath}")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

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
                cycle_pos = epoch % cfg.kl_cycle_length
                beta = min(1.0, cycle_pos / (cfg.kl_cycle_length / 2))

                loss = recon_loss + beta * kl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.gradient_clip
                )
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_recon_losses.append(recon_loss.item())
                epoch_kl_losses.append(kl_loss.item())

            # Record training metrics
            train_loss = np.mean(epoch_losses)
            epoch_recon = np.mean(epoch_recon_losses)
            epoch_kl = np.mean(epoch_kl_losses)
            history["train_loss"].append(train_loss)
            history["recon_loss"].append(epoch_recon)
            history["kl_loss"].append(epoch_kl)
            history["beta"].append(beta)

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

            # Live epoch logging
            if epoch_logger is not None:
                epoch_logger.log_epoch(epoch, {
                    "train_loss": train_loss,
                    "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
                    "recon_loss": epoch_recon,
                    "kl_loss": epoch_kl,
                    "beta": beta,
                })

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch + 1}: train_loss={train_loss:.4f}"
                if val_tokens is not None:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}"
                print(msg)

        # Close epoch logger
        if epoch_logger is not None:
            epoch_logger.close()

        self._trained = True
        return history

    def _train_direct(
        self, tokens: np.ndarray, descriptors: np.ndarray,
        val_tokens: Optional[np.ndarray], val_descriptors: Optional[np.ndarray],
        verbose: bool, log_dir: Optional[Path],
    ) -> Dict[str, List[float]]:
        """Train DirectDecoder model."""
        cfg = settings.direct
        self.model = DirectDecoder(
            vocab_size=self.encoder.vocab_size,
            descriptor_dim=descriptors.shape[1],
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            max_len=cfg.max_seq_len,
        ).to(self.device)

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"DirectDecoder: {n_params:,} parameters")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=SELFIESEncoder.PAD_IDX, label_smoothing=cfg.label_smoothing)

        train_dataset = TensorDataset(torch.LongTensor(tokens), torch.FloatTensor(descriptors))
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss, patience_counter = float("inf"), 0

        for epoch in range(cfg.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_tokens, batch_desc in train_loader:
                batch_tokens, batch_desc = batch_tokens.to(self.device), batch_desc.to(self.device)
                optimizer.zero_grad()
                logits, targets = self.model(batch_tokens, batch_desc)
                loss = criterion(logits.reshape(-1, self.encoder.vocab_size), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(train_loss)

            if val_tokens is not None:
                val_loss = self._compute_direct_val_loss(val_tokens, val_descriptors, criterion)
                history["val_loss"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss, patience_counter = val_loss, 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
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

    def _compute_direct_val_loss(self, tokens: np.ndarray, descriptors: np.ndarray, criterion) -> float:
        """Compute validation loss for DirectDecoder."""
        self.model.eval()
        with torch.no_grad():
            tokens_t = torch.LongTensor(tokens).to(self.device)
            desc_t = torch.FloatTensor(descriptors).to(self.device)
            logits, targets = self.model(tokens_t, desc_t)
            return criterion(logits.reshape(-1, self.encoder.vocab_size), targets.reshape(-1)).item()

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
        """Save trained model, encoder, and scaler state.

        The integration package contains everything needed for inference:
        - Model weights
        - Encoder vocabulary
        - Scaler parameters (for descriptor normalization)
        - Model configuration and type

        Args:
            output_dir: Output directory
        """
        if not self._trained:
            raise ModelError("Cannot save untrained model")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build model config based on model type
        if self.model_type == "direct":
            model_config = {
                "vocab_size": self.model.vocab_size,
                "descriptor_dim": self.model.descriptor_dim,
                "hidden_dim": self.model.hidden_dim,
                "n_layers": self.model.n_layers,
                "n_heads": self.model.n_heads,
                "max_len": self.model.max_len,
            }
            model_filename = "direct_model.pt"
        else:
            model_config = {
                "vocab_size": self.model.vocab_size,
                "descriptor_dim": self.model.descriptor_dim,
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "n_layers": self.model.n_layers,
                "max_len": self.model.max_len,
            }
            model_filename = "vae_model.pt"

        # Save as integration package (includes scaler for reproducibility)
        package = {
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "encoder_state": self.encoder.get_state(),
            "scaler_state": self.scaler.get_state(),
            "model_config": model_config,
        }

        with open(output_dir / "integration_package.pkl", "wb") as f:
            pickle.dump(package, f)

        # Also save separate model file
        self.model.save(output_dir / model_filename)

    def load(self, model_dir: Path) -> "PartBService":
        """Load trained model, encoder, and scaler state.

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

            # Load scaler state (critical for consistent scaling)
            if "scaler_state" in package:
                self.scaler.from_state(package["scaler_state"])

            # Determine model type from package or use current setting
            self.model_type = package.get("model_type", self.model_type)
            config = package["model_config"]

            # Load model based on type
            if self.model_type == "direct":
                self.model = DirectDecoder(
                    vocab_size=config["vocab_size"],
                    descriptor_dim=config["descriptor_dim"],
                    hidden_dim=config["hidden_dim"],
                    n_layers=config["n_layers"],
                    n_heads=config.get("n_heads", 8),
                    max_len=config["max_len"],
                )
            else:
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
            # Load from separate files (legacy format or standalone model files)
            encoder_path = model_dir / "encoder.pkl"
            if not encoder_path.exists():
                raise ModelError(f"Encoder not found: {encoder_path}")

            with open(encoder_path, "rb") as f:
                encoder_state = pickle.load(f)
            self.encoder = SELFIESEncoder.from_state(encoder_state)

            # Try to load scaler
            scaler_path = model_dir / "descriptor_scaler.pkl"
            if scaler_path.exists():
                self.scaler.load(scaler_path)

            # Try DirectDecoder first, then VAE
            direct_path = model_dir / "direct_model.pt"
            vae_path = model_dir / "vae_model.pt"

            if direct_path.exists():
                self.model_type = "direct"
                self.model = DirectDecoder.load(direct_path, self.device)
            elif vae_path.exists():
                self.model_type = "vae"
                self.model = ConditionalVAE.load(vae_path, self.device)
            else:
                # Legacy fallback
                legacy_path = model_dir / "best_model.pt"
                if not legacy_path.exists():
                    raise ModelError(f"No model file found in {model_dir}")
                self.model_type = "vae"
                self.model = ConditionalVAE.load(legacy_path, self.device)

            self.model.eval()

        self._trained = True
        return self
