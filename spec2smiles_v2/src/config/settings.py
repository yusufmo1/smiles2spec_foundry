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
class PartAConfig:
    """Part A configuration (Hybrid CNN-Transformer)."""
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
class DirectDecoderConfig:
    """Configuration for DirectDecoder (transformer-based, no VAE).

    Default: 768 hidden, 6 layers, 12 heads = ~57.5M parameters.
    """
    hidden_dim: int = 768
    n_layers: int = 6
    n_heads: int = 12
    d_ff: int = 3072  # Feed-forward dimension (4x hidden)
    dropout: float = 0.1
    max_seq_len: int = 100
    learning_rate: float = 3e-4
    n_epochs: int = 200
    batch_size: int = 64  # Reduced due to larger model
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    patience: int = 30


@dataclass
class AugmentConfig:
    """Configuration for SMILES augmentation."""
    enabled: bool = False
    n_augment: int = 5  # 5 augmented versions = 6x total data


@dataclass
class DescriptorAugmentConfig:
    """Configuration for descriptor noise augmentation.

    Adds Gaussian noise matching LightGBM prediction errors during training.
    This helps Part B generalize to imperfect descriptor predictions.
    """
    enabled: bool = False
    noise_prob: float = 0.5  # Probability of adding noise per sample
    noise_scale: float = 1.0  # Multiplier for RMSE (1.0 = full error)
    rmse_path: Optional[str] = None  # Path to Part A metrics JSON


@dataclass
class PartBConfig:
    model: Literal["vae", "direct"] = "vae"
    vae: VAEConfig = field(default_factory=VAEConfig)
    direct: DirectDecoderConfig = field(default_factory=DirectDecoderConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    descriptor_augment: DescriptorAugmentConfig = field(default_factory=DescriptorAugmentConfig)


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
    logs_dir: str = "logs"
    dataset: str = "hpj"
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"

    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    # 30 descriptors for 100% uniqueness on HPJ dataset
    # (Original 12 descriptors had only 88.9% uniqueness)
    descriptors: List[str] = field(default_factory=lambda: [
        # Original 12 descriptors
        "MolWt", "HeavyAtomCount", "NumHeteroatoms", "NumAromaticRings",
        "RingCount", "NOCount", "NumHDonors", "NumHAcceptors",
        "TPSA", "MolLogP", "NumRotatableBonds", "FractionCSP3",
        # Extended 18 descriptors for uniqueness
        "ExactMolWt", "NumAliphaticRings", "NumSaturatedRings",
        "NumAromaticHeterocycles", "NumAromaticCarbocycles",
        "NumAliphaticHeterocycles", "NumAliphaticCarbocycles",
        "NumSaturatedHeterocycles", "NumSaturatedCarbocycles",
        "LabuteASA", "BalabanJ", "BertzCT",
        "Chi0", "Chi1", "Chi2n", "Chi3n", "Chi4n", "HallKierAlpha",
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
    def logs_path(self) -> Path: return Path(self.logs_dir)

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
    def vae(self) -> VAEConfig: return self.part_b.vae
    @property
    def direct(self) -> DirectDecoderConfig: return self.part_b.direct
    @property
    def part_b_model(self) -> str: return self.part_b.model
    @property
    def augment_enabled(self) -> bool: return self.part_b.augment.enabled
    @property
    def n_augment(self) -> int: return self.part_b.augment.n_augment

    # Descriptor augmentation accessors
    @property
    def desc_augment_enabled(self) -> bool: return self.part_b.descriptor_augment.enabled
    @property
    def desc_noise_prob(self) -> float: return self.part_b.descriptor_augment.noise_prob
    @property
    def desc_noise_scale(self) -> float: return self.part_b.descriptor_augment.noise_scale
    @property
    def desc_rmse_path(self) -> Optional[str]: return self.part_b.descriptor_augment.rmse_path

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
        kwargs["part_a"] = dict_to_dataclass(PartAConfig, data["part_a"])

    if "part_b" in data:
        pb = data["part_b"]
        pb_kwargs = {"model": pb.get("model", "vae")}
        if "vae" in pb:
            pb_kwargs["vae"] = dict_to_dataclass(VAEConfig, pb["vae"])
        if "direct" in pb:
            pb_kwargs["direct"] = dict_to_dataclass(DirectDecoderConfig, pb["direct"])
        if "augment" in pb:
            pb_kwargs["augment"] = dict_to_dataclass(AugmentConfig, pb["augment"])
        if "descriptor_augment" in pb:
            pb_kwargs["descriptor_augment"] = dict_to_dataclass(
                DescriptorAugmentConfig, pb["descriptor_augment"]
            )
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
