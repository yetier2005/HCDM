"""
HCDM: Hierarchical Contrastive Distribution Matching
for Diffusion-Based Dataset Distillation.
"""

__version__ = "0.1.0"
__author__ = "HCDM Contributors"

from .config import HCDMConfig
from .distiller import HCDMDistiller
from .feature_extractor import FeatureExtractor
from .distribution_matching import (
    compute_mmd,
    compute_hcdm_loss,
    compute_intra_class_loss,
    compute_inter_class_loss,
    compute_diversity_loss,
)
from .guidance import (
    level_weight,
    guidance_schedule,
    inject_guidance,
)

__all__ = [
    "HCDMConfig",
    "HCDMDistiller",
    "FeatureExtractor",
    "compute_mmd",
    "compute_hcdm_loss",
    "compute_intra_class_loss",
    "compute_inter_class_loss",
    "compute_diversity_loss",
    "level_weight",
    "guidance_schedule",
    "inject_guidance",
]
