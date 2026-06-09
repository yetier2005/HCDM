"""
Configuration management for HCDM.

Defines HCDMConfig as a dataclass that loads from YAML files,
with sensible defaults for all hyperparameter groups.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class ModelConfig:
    """Diffusion model configuration."""

    # DiT model name or path (diffusers format)
    dit_model_name: str = "facebook/DiT-XL-2-256"
    # VAE model name or path
    vae_model_name: str = "stabilityai/sd-vae-ft-mse"
    # Image size (square)
    image_size: int = 256
    # Latent channels from VAE
    latent_channels: int = 4
    # Latent spatial size (image_size // vae_scale_factor)
    latent_size: int = 32
    # VAE scale factor
    vae_scale_factor: int = 8
    # Dit hidden dimension (for feature extraction)
    hidden_dim: int = 1152
    # Number of transformer blocks
    num_blocks: int = 28
    # Patch size for DiT
    patch_size: int = 2
    # Number of attention heads
    num_heads: int = 16
    # Prediction type: "v_prediction" (DiT) or "epsilon"
    prediction_type: str = "v_prediction"
    # Whether to use FP16 for VAE decode
    use_fp16: bool = True


@dataclass
class DistillationConfig:
    """Distillation hyperparameters."""

    # Images per class
    ipc: int = 10
    # Number of DDIM sampling steps
    ddim_steps: int = 50
    # Training timesteps for original DiT (used for alpha schedule)
    num_train_timesteps: int = 1000
    # Classifier-free guidance scale
    cfg_scale: float = 1.5
    # Batch size for parallel generation within a class
    batch_size: int = 10
    # Whether to use deterministic DDIM (sigma=0)
    deterministic: bool = True
    # Seed for reproducibility
    seed: int = 42


@dataclass
class HCDMParams:
    """HCDM-specific hyperparameters."""

    # Layer group definitions for multi-level feature extraction
    # Keys are level names, values are [start_block_idx, end_block_idx]
    # Default: split 28 DiT blocks into 3 groups
    layer_groups: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "L1": (0, 8),        # Early blocks: fine local features
        "L2": (9, 18),       # Middle blocks: part-level structure
        "L3": (19, 27),      # Late blocks: global semantics
    })

    # Lambda for inter-class repulsion
    lambda_repel: float = 0.3
    # Lambda for intra-batch diversity regularization
    lambda_div: float = 0.1
    # Diversity margin: minimum MMD between synthetic samples in a batch
    div_margin: float = 0.05
    # Top-K hard negative classes for inter-class repulsion
    topk_hard_negatives: int = 5
    # Hinge margin for inter-class repulsion
    inter_margin: float = 0.5
    # MMD kernel type: "rbf", "linear", "poly"
    mmd_kernel: str = "rbf"
    # Whether to use median heuristic for MMD bandwidth
    mmd_median_heuristic: bool = True
    # Fixed MMD sigma (only used if median_heuristic=False)
    mmd_sigma: float = 1.0
    # Whether to L2-normalize features before MMD
    normalize_features: bool = True
    # Gradient clipping value for guidance
    grad_clip: float = 1.0
    # Step interval for applying HCDM guidance (1 = every step)
    guidance_every_n_steps: int = 1


@dataclass
class GuidanceSchedule:
    """Timestep-aware guidance scheduling parameters."""

    # Global guidance scale multiplier
    guidance_scale: float = 100.0
    # Beta distribution parameters for global schedule
    beta_alpha: float = 2.0
    beta_beta: float = 2.0
    # Level weight: (center, sigma) as fraction of total steps
    level_centers: Dict[str, float] = field(default_factory=lambda: {
        "L1": 0.10,  # Fine: peak at 10% progress (late in denoising)
        "L2": 0.40,  # Mid: peak at 40% progress
        "L3": 0.75,  # Coarse: peak at 75% progress (early in denoising)
    })
    level_sigmas: Dict[str, float] = field(default_factory=lambda: {
        "L1": 0.12,
        "L2": 0.18,
        "L3": 0.15,
    })
    # Minimum weight threshold (below this, skip the level)
    min_weight: float = 0.01


@dataclass
class DataConfig:
    """Dataset configuration."""

    # Dataset name: "cifar10", "cifar100", "imagenet1k", "imagewoof", "imagenette"
    dataset: str = "cifar10"
    # Path to dataset (auto-download if None)
    data_path: Optional[str] = None
    # Number of classes
    num_classes: int = 10
    # Number of real samples per class for feature pre-computation
    samples_per_class: int = 500
    # Batch size for feature pre-computation
    precompute_batch_size: int = 32
    # Number of workers for dataloader
    num_workers: int = 4
    # Image preprocessing: resize + center crop
    preprocess_resize: int = 256
    preprocess_crop: int = 256


@dataclass
class OutputConfig:
    """Output and logging configuration."""

    # Output directory
    output_dir: str = "./outputs"
    # Experiment name
    exp_name: str = "hcdm_default"
    # Whether to use wandb
    use_wandb: bool = False
    # Wandb project name
    wandb_project: str = "hcdm"
    # Checkpoint directory
    ckpt_dir: Optional[str] = None
    # Real features cache directory
    features_dir: Optional[str] = None
    # Logging interval (steps)
    log_interval: int = 10
    # Save intermediate images
    save_intermediate: bool = False
    # Save format for distilled data
    save_format: str = "pt"  # "pt" or "png"


@dataclass
class HCDMConfig:
    """Master configuration for HCDM.

    Loads from a YAML file and organizes all hyperparameters
    into logical groups.

    Usage:
        config = HCDMConfig.from_yaml("configs/cifar10.yaml")
        config = HCDMConfig.from_yaml("configs/cifar10.yaml", overrides={"ipc": 50})
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    hcdm: HCDMParams = field(default_factory=HCDMParams)
    guidance: GuidanceSchedule = field(default_factory=GuidanceSchedule)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Convenience aliases for frequently accessed params
    @property
    def ipc(self) -> int:
        return self.distillation.ipc

    @property
    def ddim_steps(self) -> int:
        return self.distillation.ddim_steps

    @property
    def lambda_repel(self) -> float:
        return self.hcdm.lambda_repel

    @property
    def lambda_div(self) -> float:
        return self.hcdm.lambda_div

    @property
    def guidance_scale(self) -> float:
        return self.guidance.guidance_scale

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], overrides: Optional[Dict] = None) -> "HCDMConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.
            overrides: Optional dict of dot-path overrides, e.g.,
                      {"distillation.ipc": 50, "hcdm.lambda_repel": 0.5}.

        Returns:
            HCDMConfig instance with parsed values.
        """
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        # Apply overrides
        if overrides:
            for key_path, value in overrides.items():
                cls._set_nested(raw, key_path.split("."), value)

        config = cls(
            model=ModelConfig(**raw.get("model", {})),
            distillation=DistillationConfig(**raw.get("distillation", {})),
            hcdm=HCDMParams(**raw.get("hcdm", {})),
            guidance=GuidanceSchedule(**raw.get("guidance", {})),
            data=DataConfig(**raw.get("data", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
        return config

    @staticmethod
    def _set_nested(d, keys, value):
        """Set a value in a nested dict using a list of keys."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def to_dict(self) -> Dict:
        """Convert config to a flat-ish dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "distillation": self.distillation.__dict__,
            "hcdm": self.hcdm.__dict__,
            "guidance": self.guidance.__dict__,
            "data": self.data.__dict__,
            "output": self.output.__dict__,
        }

    def __repr__(self) -> str:
        lines = ["HCDMConfig:"]
        for section_name in ["model", "distillation", "hcdm", "guidance", "data", "output"]:
            section = getattr(self, section_name)
            lines.append(f"  [{section_name}]")
            for key, value in section.__dict__.items():
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)
