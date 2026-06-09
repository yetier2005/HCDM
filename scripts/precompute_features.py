#!/usr/bin/env python3
"""
Pre-compute real dataset multi-level features for HCDM.

This script:
1. Loads the DiT model and VAE
2. Loads the target dataset
3. Encodes all real images to latents
4. Extracts multi-level features (L1/L2/L3) from DiT
5. Organizes features by class
6. Saves to disk for use during distillation

Usage:
    python scripts/precompute_features.py --config configs/cifar10.yaml
    python scripts/precompute_features.py --config configs/imagenet1k.yaml --output ./features.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcdm.config import HCDMConfig
from hcdm.models.dit_wrapper import DiTWrapper
from hcdm.feature_extractor import FeatureExtractor


def build_dataloader(config: HCDMConfig):
    """Build a dataloader for the specified dataset."""
    dataset_name = config.data.dataset.lower()
    preprocess_resize = config.data.preprocess_resize
    preprocess_crop = config.data.preprocess_crop

    transform = transforms.Compose([
        transforms.Resize(preprocess_resize),
        transforms.CenterCrop(preprocess_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(
            root=config.data.data_path or "./data",
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        dataset = CIFAR100(
            root=config.data.data_path or "./data",
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name in ("imagenet1k", "imagenet"):
        from torchvision.datasets import ImageNet
        dataset = ImageNet(
            root=config.data.data_path or "./data/imagenet",
            split="train",
            transform=transform,
        )
    elif dataset_name in ("imagewoof", "imagenette"):
        # These are ImageNet subsets, use ImageFolder
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(
            root=config.data.data_path or f"./data/{dataset_name}/train",
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.precompute_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute real dataset features for HCDM"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for features (.pt file). Overrides config.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Max samples per class (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    args = parser.parse_args()

    # Load config
    config = HCDMConfig.from_yaml(args.config)
    if args.samples_per_class is not None:
        config.data.samples_per_class = args.samples_per_class

    print("=" * 60)
    print("HCDM Feature Pre-computation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Dataset: {config.data.dataset} ({config.data.num_classes} classes)")
    print(f"Samples per class: {config.data.samples_per_class}")
    print(f"Model: {config.model.dit_model_name}")
    print("=" * 60)

    # Load DiT model
    print("\n[1/4] Loading DiT model...")
    dit = DiTWrapper(
        dit_model_name=config.model.dit_model_name,
        vae_model_name=config.model.vae_model_name,
        image_size=config.model.image_size,
        num_train_timesteps=config.distillation.num_train_timesteps,
        latent_channels=config.model.latent_channels,
        hidden_dim=config.model.hidden_dim,
        num_blocks=config.model.num_blocks,
        patch_size=config.model.patch_size,
        prediction_type=config.model.prediction_type,
        use_fp16=config.model.use_fp16,
        device=args.device,
    )
    print(f"  DiT loaded on {dit.device_info}")

    # Initialize feature extractor
    extractor = FeatureExtractor(
        model_wrapper=dit,
        layer_groups=config.hcdm.layer_groups,
        normalize_output=False,
    )

    # Build dataloader
    print("\n[2/4] Loading dataset...")
    dataloader = build_dataloader(config)
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Batch size: {config.data.precompute_batch_size}")

    # Pre-compute features
    print("\n[3/4] Extracting features...")
    real_features = extractor.precompute_real_features(
        dataloader=dataloader,
        samples_per_class=config.data.samples_per_class,
        device=args.device,
        verbose=True,
    )

    # Verify
    print(f"\n[4/4] Verifying features...")
    total_samples = 0
    for c in sorted(real_features.keys()):
        for level in ["L1", "L2", "L3"]:
            if level in real_features[c]:
                n = real_features[c][level].size(0)
                d = real_features[c][level].size(1)
                total_samples += n
                print(f"  Class {c:3d} / {level}: {n:4d} samples × {d:4d} dims")

    # Save
    output_path = args.output or config.output.features_dir or f"./outputs/{config.data.dataset}_features.pt"
    extractor.save_features(real_features, output_path)
    print(f"\nFeatures saved to: {output_path}")
    print(f"Total feature vectors: {total_samples}")

    # Memory estimate
    file_size = Path(output_path).stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")
    print("\nDone! Use these features with:")
    print(f"  python scripts/distill.py --config {args.config} --features {output_path}")


if __name__ == "__main__":
    main()
