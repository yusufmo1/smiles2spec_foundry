"""Pipeline configuration using Pydantic Settings.

Environment variables override defaults with SPEC2SMILES_ prefix.
Example: SPEC2SMILES_N_BINS=500 overrides n_bins setting.
"""

from pathlib import Path
from typing import Literal, Tuple

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SPEC2SMILES pipeline configuration.

    All settings can be overridden via environment variables with SPEC2SMILES_ prefix.
    Example: SPEC2SMILES_DATASET=hmdb will set dataset to "hmdb".
    """

    model_config = SettingsConfigDict(
        env_prefix="SPEC2SMILES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===========================================
    # Paths
    # ===========================================
    data_input_dir: str = "data/input"
    data_output_dir: str = "data/output"
    dataset: str = "hpj"

    # ===========================================
    # Spectrum Processing
    # ===========================================
    n_bins: int = 500
    bin_width: float = 1.0
    max_mz: float = 500.0
    transform: Literal["sqrt", "log", "none"] = "sqrt"
    normalize: bool = True

    # ===========================================
    # Descriptors
    # ===========================================
    descriptor_names: Tuple[str, ...] = (
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

    # ===========================================
    # Part A: LightGBM Configuration
    # ===========================================
    lgbm_n_estimators: int = 1000
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_feature_fraction: float = 0.9
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 5
    lgbm_reg_alpha: float = 0.1
    lgbm_reg_lambda: float = 0.1
    lgbm_early_stopping_rounds: int = 50
    lgbm_min_child_samples: int = 20
    lgbm_n_jobs: int = 8

    # ===========================================
    # Part B: VAE Configuration
    # ===========================================
    vae_latent_dim: int = 128
    vae_hidden_dim: int = 256
    vae_n_layers: int = 2
    vae_dropout: float = 0.2
    vae_max_seq_len: int = 100
    vae_learning_rate: float = 0.001
    vae_n_epochs: int = 100
    vae_batch_size: int = 64
    vae_kl_cycle_length: int = 20
    vae_gradient_clip: float = 1.0

    # ===========================================
    # Inference
    # ===========================================
    n_candidates: int = 50
    temperature: float = 0.7
    top_p: float = 0.9

    # ===========================================
    # Training
    # ===========================================
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # ===========================================
    # Device
    # ===========================================
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"

    # ===========================================
    # Computed Properties
    # ===========================================
    @property
    def n_descriptors(self) -> int:
        """Number of molecular descriptors."""
        return len(self.descriptor_names)

    @property
    def input_path(self) -> Path:
        """Path to input data directory."""
        return Path(self.data_input_dir) / self.dataset

    @property
    def output_path(self) -> Path:
        """Path to output directory."""
        return Path(self.data_output_dir)

    @property
    def models_path(self) -> Path:
        """Path to trained models."""
        return self.output_path / "models"

    @property
    def metrics_path(self) -> Path:
        """Path to evaluation metrics."""
        return self.output_path / "metrics"

    @property
    def figures_path(self) -> Path:
        """Path to visualization figures."""
        return self.output_path / "figures"

    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device based on configuration."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    # ===========================================
    # LightGBM Helper
    # ===========================================
    def to_lgbm_params(self) -> dict:
        """Convert settings to LightGBM parameter dictionary."""
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": self.lgbm_n_estimators,
            "num_leaves": self.lgbm_num_leaves,
            "learning_rate": self.lgbm_learning_rate,
            "feature_fraction": self.lgbm_feature_fraction,
            "bagging_fraction": self.lgbm_bagging_fraction,
            "bagging_freq": self.lgbm_bagging_freq,
            "reg_alpha": self.lgbm_reg_alpha,
            "reg_lambda": self.lgbm_reg_lambda,
            "min_child_samples": self.lgbm_min_child_samples,
            "n_jobs": self.lgbm_n_jobs,
            "verbose": -1,
            "random_state": self.random_seed,
        }

    # ===========================================
    # Validators
    # ===========================================
    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """Ensure ratios are valid probabilities."""
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


# Global settings instance
settings = Settings()
