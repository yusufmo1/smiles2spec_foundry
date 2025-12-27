"""Pipeline configuration from YAML file.

Load config from config.yml (or custom path via --config flag).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch

from smiles2spec.core.loader import dict_to_dataclass, find_config_path, load_yaml


@dataclass
class SpectrumConfig:
    """Spectrum output configuration."""

    n_bins: int = 500
    bin_width: float = 1.0
    max_mz: float = 499.0
    transform: Literal["sqrt", "log", "none"] = "sqrt"
    normalize: bool = True


@dataclass
class MorganFPConfig:
    """Morgan fingerprint configuration."""

    enabled: bool = True
    radii: List[int] = field(default_factory=lambda: [1, 2, 3])
    n_bits: int = 1024


@dataclass
class FingerprintConfig:
    """Fingerprint feature configuration."""

    morgan: MorganFPConfig = field(default_factory=MorganFPConfig)
    maccs: bool = True
    rdkit_fp: bool = True
    rdkit_fp_bits: int = 2048


@dataclass
class ConformerConfig:
    """3D conformer configuration."""

    n_conformers: int = 5
    optimize: bool = True
    energy_window: float = 50.0


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""

    type: Literal["2d", "3d", "combined"] = "2d"
    descriptors_enabled: bool = True
    fingerprints: FingerprintConfig = field(default_factory=FingerprintConfig)
    enable_3d: bool = False
    conformer: ConformerConfig = field(default_factory=ConformerConfig)


@dataclass
class RandomForestConfig:
    """Random Forest configuration (best individual: 0.7837)."""

    n_estimators: int = 290
    max_depth: int = 25
    max_features: float = 0.3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class ModularNetConfig:
    """ModularNet configuration (0.7691 cosine sim)."""

    n_modules: int = 4
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class HierarchicalNetConfig:
    """HierarchicalPredictionNet configuration (0.7770)."""

    hidden_dim: int = 512
    n_layers: int = 3
    dropout: float = 0.2


@dataclass
class SparseGatedNetConfig:
    """SparseGatedNet configuration (0.7674)."""

    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1


@dataclass
class RegionalExpertConfig:
    """RegionalExpertNet configuration (0.7622)."""

    n_regions: int = 5
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class NeuralNetworkConfig:
    """Neural network training configuration (matched to original notebook)."""

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 64
    patience: int = 20
    gradient_clip: float = 1.0
    num_modules: int = 4  # Default for ModularNet
    dropout: float = 0.1  # Default dropout
    modular_net: ModularNetConfig = field(default_factory=ModularNetConfig)
    hierarchical_net: HierarchicalNetConfig = field(default_factory=HierarchicalNetConfig)
    sparse_gated_net: SparseGatedNetConfig = field(default_factory=SparseGatedNetConfig)
    regional_expert: RegionalExpertConfig = field(default_factory=RegionalExpertConfig)


@dataclass
class WeightedEnsembleConfig:
    """Weighted ensemble configuration."""

    optimize_weights: bool = True


@dataclass
class BinByBinConfig:
    """Bin-by-bin ensemble configuration (best: 0.8164)."""

    optimize_per_bin: bool = True


@dataclass
class EnsembleConfig:
    """Ensemble model configuration."""

    method: Literal["weighted", "bin_by_bin"] = "bin_by_bin"
    models: List[str] = field(default_factory=lambda: ["random_forest", "modular_net"])
    weighted: WeightedEnsembleConfig = field(default_factory=WeightedEnsembleConfig)
    bin_by_bin: BinByBinConfig = field(default_factory=BinByBinConfig)


@dataclass
class SplitConfig:
    """Data split configuration."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""

    early_stopping: bool = True
    save_best: bool = True
    log_interval: int = 10


@dataclass
class Settings:
    """Main configuration container."""

    data_input_dir: str = "data/input"
    data_output_dir: str = "data/output"
    cache_dir: str = "data/cache"
    logs_dir: str = "logs"
    dataset: str = "hpj"
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"

    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    neural_network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Spectrum shortcuts
    @property
    def n_bins(self) -> int:
        return self.spectrum.n_bins

    @property
    def bin_width(self) -> float:
        return self.spectrum.bin_width

    @property
    def max_mz(self) -> float:
        return self.spectrum.max_mz

    @property
    def transform(self) -> str:
        return self.spectrum.transform

    @property
    def normalize(self) -> bool:
        return self.spectrum.normalize

    # Path properties
    @property
    def input_path(self) -> Path:
        return Path(self.data_input_dir) / self.dataset

    @property
    def output_path(self) -> Path:
        return Path(self.data_output_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir) / self.dataset

    @property
    def models_path(self) -> Path:
        return self.output_path / "models"

    @property
    def metrics_path(self) -> Path:
        return self.output_path / "metrics"

    @property
    def figures_path(self) -> Path:
        return self.output_path / "figures"

    @property
    def logs_path(self) -> Path:
        return Path(self.logs_dir)

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    # Feature shortcuts
    @property
    def feature_type(self) -> str:
        return self.features.type

    @property
    def enable_3d(self) -> bool:
        return self.features.enable_3d

    # Split shortcuts
    @property
    def train_ratio(self) -> float:
        return self.split.train

    @property
    def val_ratio(self) -> float:
        return self.split.val

    @property
    def test_ratio(self) -> float:
        return self.split.test

    @property
    def random_seed(self) -> int:
        return self.split.seed


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
        kwargs["cache_dir"] = data["paths"].get("cache", "data/cache")

    for key in ("dataset", "device"):
        if key in data:
            kwargs[key] = data[key]

    # Nested configs
    if "spectrum" in data:
        kwargs["spectrum"] = dict_to_dataclass(SpectrumConfig, data["spectrum"])

    if "features" in data:
        feat_data = data["features"]
        feat_kwargs = {"type": feat_data.get("type", "2d")}
        if "descriptors" in feat_data:
            feat_kwargs["descriptors_enabled"] = feat_data["descriptors"].get("enabled", True)
        if "enable_3d" in feat_data:
            feat_kwargs["enable_3d"] = feat_data["enable_3d"]
        if "fingerprints" in feat_data:
            fp_data = feat_data["fingerprints"]
            fp_kwargs = {}
            if "morgan" in fp_data:
                fp_kwargs["morgan"] = dict_to_dataclass(MorganFPConfig, fp_data["morgan"])
            if "maccs" in fp_data:
                fp_kwargs["maccs"] = fp_data["maccs"].get("enabled", True)
            if "rdkit_fp" in fp_data:
                fp_kwargs["rdkit_fp"] = fp_data["rdkit_fp"].get("enabled", True)
                fp_kwargs["rdkit_fp_bits"] = fp_data["rdkit_fp"].get("n_bits", 2048)
            feat_kwargs["fingerprints"] = FingerprintConfig(**fp_kwargs)
        if "conformer" in feat_data:
            feat_kwargs["conformer"] = dict_to_dataclass(ConformerConfig, feat_data["conformer"])
        kwargs["features"] = FeatureConfig(**feat_kwargs)

    if "random_forest" in data:
        kwargs["random_forest"] = dict_to_dataclass(RandomForestConfig, data["random_forest"])

    if "neural_network" in data:
        nn_data = data["neural_network"]
        nn_kwargs = {}
        for key in ("learning_rate", "weight_decay", "max_epochs", "batch_size", "patience",
                    "gradient_clip"):
            if key in nn_data:
                nn_kwargs[key] = nn_data[key]
        if "modular_net" in nn_data:
            nn_kwargs["modular_net"] = dict_to_dataclass(ModularNetConfig, nn_data["modular_net"])
        if "hierarchical_net" in nn_data:
            nn_kwargs["hierarchical_net"] = dict_to_dataclass(
                HierarchicalNetConfig, nn_data["hierarchical_net"]
            )
        if "sparse_gated_net" in nn_data:
            nn_kwargs["sparse_gated_net"] = dict_to_dataclass(
                SparseGatedNetConfig, nn_data["sparse_gated_net"]
            )
        if "regional_expert" in nn_data:
            nn_kwargs["regional_expert"] = dict_to_dataclass(
                RegionalExpertConfig, nn_data["regional_expert"]
            )
        kwargs["neural_network"] = NeuralNetworkConfig(**nn_kwargs)

    if "ensemble" in data:
        ens_data = data["ensemble"]
        ens_kwargs = {}
        if "method" in ens_data:
            ens_kwargs["method"] = ens_data["method"]
        if "models" in ens_data:
            ens_kwargs["models"] = ens_data["models"]
        if "weighted" in ens_data:
            ens_kwargs["weighted"] = dict_to_dataclass(WeightedEnsembleConfig, ens_data["weighted"])
        if "bin_by_bin" in ens_data:
            ens_kwargs["bin_by_bin"] = dict_to_dataclass(BinByBinConfig, ens_data["bin_by_bin"])
        kwargs["ensemble"] = EnsembleConfig(**ens_kwargs)

    if "split" in data:
        kwargs["split"] = dict_to_dataclass(SplitConfig, data["split"])

    if "training" in data:
        kwargs["training"] = dict_to_dataclass(TrainingConfig, data["training"])

    return Settings(**kwargs)


# Global settings instance
settings = load_config()


def reload_config(config_path: Optional[Path] = None) -> Settings:
    """Reload configuration from file."""
    global settings
    settings = load_config(config_path)
    return settings
