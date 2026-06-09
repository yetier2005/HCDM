"""
Feature extraction system for HCDM.

Provides a unified interface for extracting multi-level features
from different diffusion model backbones (DiT, UNet).

The key abstraction: any diffusion model that exposes internal
representations at different semantic depths can be used.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

from .models.dit_wrapper import DiTWrapper


class FeatureExtractor:
    """Multi-level feature extractor for diffusion models.

    Works with DiT (via transformer block hooks) and UNet
    (via decoder up_block hooks).

    Usage:
        extractor = FeatureExtractor(dit_wrapper, layer_groups)
        features = extractor.extract(z_latent, class_labels)
        # features = {'L1': Tensor[B, D1], 'L2': Tensor[B, D2], 'L3': Tensor[B, D3]}
    """

    def __init__(
        self,
        model_wrapper: DiTWrapper,
        layer_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        normalize_output: bool = False,
    ):
        """
        Args:
            model_wrapper: DiTWrapper instance (or compatible).
            layer_groups: Dict of {level_name: (start_block, end_block)}.
                          If None, uses DiTWrapper's default 3-level split.
            normalize_output: Whether to L2-normalize features after extraction.
        """
        self.model = model_wrapper
        self.layer_groups = layer_groups
        self.normalize_output = normalize_output

    def extract(
        self,
        z: torch.Tensor,
        class_labels: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        with_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-level features from a batch of latents.

        Args:
            z: Latents [B, C, H, W].
            class_labels: Class indices [B].
            t: Timesteps [B]. None = use t=0.
            with_grad: Preserve gradient flow if True (for guidance).

        Returns:
            Dict {level_name: Tensor[B, D]}. Features are mean-pooled.
        """
        features = self.model.extract_features(
            z=z,
            class_labels=class_labels,
            t=t,
            layer_groups=self.layer_groups,
            with_grad=with_grad,
        )

        if self.normalize_output:
            import torch.nn.functional as F
            features = {k: F.normalize(v, p=2, dim=-1) for k, v in features.items()}

        return features

    @torch.no_grad()
    def precompute_real_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        samples_per_class: int = 500,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Pre-compute multi-level features for real dataset.

        Iterates through the dataloader, encodes each image to latent space,
        extracts multi-level features, and organizes them by class.

        Args:
            dataloader: DataLoader yielding (images, labels) tuples.
                        Images in [-1, 1], shape [B, 3, H, W].
            samples_per_class: Max samples to store per class.
            device: Device to compute on (auto if None).
            verbose: Print progress.

        Returns:
            Nested dict: {class_idx: {level_name: Tensor[N_c, D]}}
            where N_c ≤ samples_per_class.
        """
        class_features = defaultdict(lambda: defaultdict(list))
        class_counts = defaultdict(int)

        if device is None:
            device = next(self.model.vae.parameters()).device if self.model.vae else "cuda"

        # Determine number of classes
        num_classes = max(1, len(class_counts))

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=samples_per_class * num_classes // dataloader.batch_size,
                       desc="Precomputing features")

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Check if all classes are full
            all_full = all(
                class_counts[c] >= samples_per_class
                for c in range(num_classes)
            )
            if all_full:
                break

            # Encode to latent
            z = self.model.encode(images)

            # Extract features at t≈0
            feats = self.extract(z, labels, t=torch.zeros(images.size(0), dtype=torch.long, device=device))

            # Store per class
            for i in range(images.size(0)):
                c = labels[i].item()
                if class_counts[c] >= samples_per_class:
                    continue
                for level, feat_tensor in feats.items():
                    class_features[c][level].append(feat_tensor[i].cpu())
                class_counts[c] += 1

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        # Stack features per class per level
        result = {}
        for c in sorted(class_features.keys()):
            result[c] = {}
            for level, feature_list in class_features[c].items():
                if len(feature_list) > 0:
                    result[c][level] = torch.stack(feature_list)

        if verbose:
            for c in result:
                counts = {l: result[c][l].size(0) for l in result[c]}
                print(f"  Class {c}: {counts}")

        return result

    def save_features(
        self,
        features: Dict[int, Dict[str, torch.Tensor]],
        path: str,
    ):
        """Save precomputed features to disk.

        Args:
            features: {class_idx: {level: Tensor[N, D]}}.
            path: File path (.pt extension).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, path)

    @staticmethod
    def load_features(path: str) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load precomputed features from disk.

        Args:
            path: File path.

        Returns:
            {class_idx: {level: Tensor[N, D]}}.
        """
        return torch.load(path, map_location="cpu")

    def __repr__(self) -> str:
        levels = list(self.layer_groups.keys()) if self.layer_groups else ["L1", "L2", "L3"]
        return f"FeatureExtractor(levels={levels}, normalize={self.normalize_output})"
