"""Configuration dataclasses for SPEC2SMILES pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import yaml


@dataclass(frozen=True)
class SpectrumConfig:
    """Spectrum processing configuration."""

    n_bins: int = 500
    bin_width: float = 1.0
    max_mz: float = 500.0
    transform: str = "sqrt"  # sqrt, log, or none
    normalize: bool = True


@dataclass(frozen=True)
class DescriptorConfig:
    """Molecular descriptor configuration.

    Extended to 30 descriptors for 100% uniqueness on HPJ dataset.
    Original 12 descriptors had only 88.9% uniqueness.
    """

    names: Tuple[str, ...] = (
        # Original 12 descriptors
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
        # Extended descriptors for uniqueness
        "ExactMolWt",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumAromaticHeterocycles",
        "NumAromaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumAliphaticCarbocycles",
        "NumSaturatedHeterocycles",
        "NumSaturatedCarbocycles",
        "LabuteASA",
        "BalabanJ",
        "BertzCT",
        "Chi0",
        "Chi1",
        "Chi2n",
        "Chi3n",
        "Chi4n",
        "HallKierAlpha",
    )

    @property
    def n_descriptors(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class PartAConfig:
    """Part A (LightGBM) configuration."""

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
    verbose: int = -1

    def to_lgbm_params(self) -> dict:
        """Convert to LightGBM parameter dict."""
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "random_state": 42,
        }


@dataclass(frozen=True)
class PartBConfig:
    """Part B (VAE) configuration."""

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


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration."""

    n_candidates: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    """Training data split configuration."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    part_a: PartAConfig = field(default_factory=PartAConfig)
    part_b: PartBConfig = field(default_factory=PartBConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            spectrum=SpectrumConfig(**data.get("spectrum", {})),
            descriptors=DescriptorConfig(
                names=tuple(data.get("descriptors", {}).get("names", DescriptorConfig().names))
            ),
            part_a=PartAConfig(**data.get("part_a", {})),
            part_b=PartBConfig(**data.get("part_b", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            training=TrainingConfig(**data.get("training", {})),
            random_seed=data.get("random_seed", 42),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "spectrum": {
                "n_bins": self.spectrum.n_bins,
                "bin_width": self.spectrum.bin_width,
                "max_mz": self.spectrum.max_mz,
                "transform": self.spectrum.transform,
                "normalize": self.spectrum.normalize,
            },
            "descriptors": {"names": list(self.descriptors.names)},
            "part_a": {
                "n_estimators": self.part_a.n_estimators,
                "num_leaves": self.part_a.num_leaves,
                "learning_rate": self.part_a.learning_rate,
                "feature_fraction": self.part_a.feature_fraction,
                "bagging_fraction": self.part_a.bagging_fraction,
                "bagging_freq": self.part_a.bagging_freq,
                "reg_alpha": self.part_a.reg_alpha,
                "reg_lambda": self.part_a.reg_lambda,
                "early_stopping_rounds": self.part_a.early_stopping_rounds,
            },
            "part_b": {
                "latent_dim": self.part_b.latent_dim,
                "hidden_dim": self.part_b.hidden_dim,
                "n_layers": self.part_b.n_layers,
                "dropout": self.part_b.dropout,
                "max_seq_len": self.part_b.max_seq_len,
                "learning_rate": self.part_b.learning_rate,
                "n_epochs": self.part_b.n_epochs,
                "batch_size": self.part_b.batch_size,
                "kl_cycle_length": self.part_b.kl_cycle_length,
                "gradient_clip": self.part_b.gradient_clip,
            },
            "inference": {
                "n_candidates": self.inference.n_candidates,
                "temperature": self.inference.temperature,
                "top_p": self.inference.top_p,
            },
            "training": {
                "train_ratio": self.training.train_ratio,
                "val_ratio": self.training.val_ratio,
                "test_ratio": self.training.test_ratio,
            },
            "random_seed": self.random_seed,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
