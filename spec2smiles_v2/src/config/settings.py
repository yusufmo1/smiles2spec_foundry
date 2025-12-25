"""Pipeline configuration from YAML file.

Load config from config.yml (or custom path via --config flag).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch

from src.config.loader import dict_to_dataclass, find_config_path, load_yaml


@dataclass
class SpectrumConfig:
    n_bins: int = 500
    bin_width: float = 1.0
    max_mz: float = 500.0
    transform: Literal["sqrt", "log", "none"] = "sqrt"
    normalize: bool = True


@dataclass
class LGBMConfig:
    n_estimators: int = 1000
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 50
    min_child_samples: int = 20
    n_jobs: int = 8

    def to_lgbm_params(self, seed: int = 42) -> dict:
        """Convert to LightGBM parameter dictionary."""
        return {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": self.n_estimators, "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate, "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction, "bagging_freq": self.bagging_freq,
            "reg_alpha": self.reg_alpha, "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples, "n_jobs": self.n_jobs,
            "verbose": -1, "random_state": seed,
        }


@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    patch_size: int = 10
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 200
    batch_size: int = 32
    gradient_clip: float = 1.0
    patience: int = 30


@dataclass
class HybridConfig:
    """Configuration for HybridCNNTransformer model."""
    cnn_hidden: int = 256
    transformer_dim: int = 256
    n_heads: int = 8
    n_transformer_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 300
    batch_size: int = 32
    gradient_clip: float = 1.0
    patience: int = 40


@dataclass
class PartAConfig:
    model: Literal["lgbm", "transformer", "hybrid"] = "lgbm"
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)


@dataclass
class VAEConfig:
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    max_seq_len: int = 100
    learning_rate: float = 0.001
    n_epochs: int = 100
    batch_size: int = 64
    kl_cycle_length: int = 20
    gradient_clip: float = 1.0


@dataclass
class PartBConfig:
    model: Literal["vae"] = "vae"
    vae: VAEConfig = field(default_factory=VAEConfig)


@dataclass
class InferenceConfig:
    n_candidates: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42


@dataclass
class Settings:
    """Main configuration container."""
    data_input_dir: str = "data/input"
    data_output_dir: str = "data/output"
    dataset: str = "hpj"
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"

    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    descriptors: List[str] = field(default_factory=lambda: [
        "MolWt", "HeavyAtomCount", "NumHeteroatoms", "NumAromaticRings",
        "RingCount", "NOCount", "NumHDonors", "NumHAcceptors",
        "TPSA", "MolLogP", "NumRotatableBonds", "FractionCSP3",
    ])
    part_a: PartAConfig = field(default_factory=PartAConfig)
    part_b: PartBConfig = field(default_factory=PartBConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    # Computed properties - spectrum shortcuts
    @property
    def n_bins(self) -> int: return self.spectrum.n_bins
    @property
    def bin_width(self) -> float: return self.spectrum.bin_width
    @property
    def max_mz(self) -> float: return self.spectrum.max_mz
    @property
    def transform(self) -> str: return self.spectrum.transform
    @property
    def normalize(self) -> bool: return self.spectrum.normalize

    # Descriptor properties
    @property
    def n_descriptors(self) -> int: return len(self.descriptors)
    @property
    def descriptor_names(self) -> tuple: return tuple(self.descriptors)

    # Path properties
    @property
    def input_path(self) -> Path: return Path(self.data_input_dir) / self.dataset
    @property
    def output_path(self) -> Path: return Path(self.data_output_dir)
    @property
    def models_path(self) -> Path: return self.output_path / "models"
    @property
    def metrics_path(self) -> Path: return self.output_path / "metrics"
    @property
    def figures_path(self) -> Path: return self.output_path / "figures"

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    # Model config accessors
    @property
    def part_a_model(self) -> str: return self.part_a.model
    @property
    def lgbm(self) -> LGBMConfig: return self.part_a.lgbm
    @property
    def transformer(self) -> TransformerConfig: return self.part_a.transformer
    @property
    def hybrid(self) -> HybridConfig: return self.part_a.hybrid
    @property
    def vae(self) -> VAEConfig: return self.part_b.vae

    # Split config accessors
    @property
    def train_ratio(self) -> float: return self.split.train
    @property
    def val_ratio(self) -> float: return self.split.val
    @property
    def test_ratio(self) -> float: return self.split.test
    @property
    def random_seed(self) -> int: return self.split.seed

    # Inference config accessors
    @property
    def n_candidates(self) -> int: return self.inference.n_candidates
    @property
    def temperature(self) -> float: return self.inference.temperature

    def to_lgbm_params(self) -> dict:
        """Get LightGBM parameters."""
        return self.lgbm.to_lgbm_params(self.split.seed)


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from YAML file."""
    path = find_config_path(config_path)
    if path is None:
        print("Warning: config.yml not found, using defaults")
        return Settings()

    data = load_yaml(path)
    kwargs = {}

    # Simple fields
    if "paths" in data:
        kwargs["data_input_dir"] = data["paths"].get("data_input", "data/input")
        kwargs["data_output_dir"] = data["paths"].get("data_output", "data/output")
    for key in ("dataset", "device", "descriptors"):
        if key in data:
            kwargs[key] = data[key]

    # Nested configs
    if "spectrum" in data:
        kwargs["spectrum"] = dict_to_dataclass(SpectrumConfig, data["spectrum"])

    if "part_a" in data:
        pa = data["part_a"]
        pa_kwargs = {"model": pa.get("model", "lgbm")}
        if "lgbm" in pa:
            pa_kwargs["lgbm"] = dict_to_dataclass(LGBMConfig, pa["lgbm"])
        if "transformer" in pa:
            pa_kwargs["transformer"] = dict_to_dataclass(TransformerConfig, pa["transformer"])
        if "hybrid" in pa:
            pa_kwargs["hybrid"] = dict_to_dataclass(HybridConfig, pa["hybrid"])
        kwargs["part_a"] = PartAConfig(**pa_kwargs)

    if "part_b" in data:
        pb = data["part_b"]
        pb_kwargs = {"model": pb.get("model", "vae")}
        if "vae" in pb:
            pb_kwargs["vae"] = dict_to_dataclass(VAEConfig, pb["vae"])
        kwargs["part_b"] = PartBConfig(**pb_kwargs)

    if "inference" in data:
        kwargs["inference"] = dict_to_dataclass(InferenceConfig, data["inference"])
    if "split" in data:
        kwargs["split"] = dict_to_dataclass(SplitConfig, data["split"])

    return Settings(**kwargs)


# Global settings instance
settings = load_config()


def reload_config(config_path: Optional[Path] = None) -> Settings:
    """Reload configuration from file."""
    global settings
    settings = load_config(config_path)
    return settings
