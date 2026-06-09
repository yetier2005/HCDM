#!/usr/bin/env python3
"""
HCDM Dataset Distillation — Main Entry Point.

This script runs the full distillation pipeline:
1. Loads config and model
2. Loads pre-computed real dataset features
3. For each class, generates IPC images via HCDM-guided diffusion
4. Saves the distilled dataset

Usage:
    # Step 1: Pre-compute features
    python scripts/precompute_features.py --config configs/cifar10.yaml

    # Step 2: Distill
    python scripts/distill.py --config configs/cifar10.yaml --ipc 10
    python scripts/distill.py --config configs/cifar10.yaml --ipc 1
    python scripts/distill.py --config configs/cifar10.yaml --ipc 50
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcdm.config import HCDMConfig
from hcdm.models.dit_wrapper import DiTWrapper
from hcdm.feature_extractor import FeatureExtractor
from hcdm.distiller import HCDMDistiller


def main():
    parser = argparse.ArgumentParser(
        description="HCDM: Hierarchical Contrastive Distribution Matching for Dataset Distillation"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--ipc",
        type=int,
        default=None,
        help="Images per class (overrides config)",
    )
    parser.add_argument(
        "--features", "-f",
        type=str,
        default=None,
        help="Path to pre-computed features (.pt). Overrides config.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class indices to distill (default: all)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    args = parser.parse_args()

    # Load config
    config = HCDMConfig.from_yaml(args.config)
    if args.ipc is not None:
        config.distillation.ipc = args.ipc

    print("=" * 70)
    print("HCDM: Hierarchical Contrastive Distribution Matching")
    print("Dataset Distillation via Diffusion Models")
    print("=" * 70)
    print(f"Config:    {args.config}")
    print(f"IPC:       {config.distillation.ipc}")
    print(f"DDIM steps:{config.distillation.ddim_steps}")
    print(f"Lambda_r:  {config.hcdm.lambda_repel}")
    print(f"Lambda_d:  {config.hcdm.lambda_div}")
    print(f"Guidance:  {config.guidance.guidance_scale}")
    print(f"Dataset:   {config.data.dataset} ({config.data.num_classes} classes)")
    print("=" * 70)

    # Load DiT model
    print("\n[1/5] Loading DiT model...")
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
    print(f"  Model: {config.model.dit_model_name}")
    print(f"  Device: {dit.device_info}")
    print(f"  Image size: {dit.image_size}×{dit.image_size}")
    print(f"  Latent size: {dit.latent_size}×{dit.latent_size}×{dit.latent_channels}")
    print(f"  Blocks: {dit.num_blocks}, Hidden: {dit.hidden_dim}D")

    # Load features
    features_path = args.features or config.output.features_dir
    if not features_path:
        features_path = f"./outputs/{config.data.dataset}_features.pt"

    print(f"\n[2/5] Loading pre-computed features from {features_path}...")
    try:
        real_features = FeatureExtractor.load_features(features_path)
    except FileNotFoundError:
        print(f"ERROR: Features file not found: {features_path}")
        print("Please run precompute_features.py first:")
        print(f"  python scripts/precompute_features.py --config {args.config}")
        sys.exit(1)

    num_loaded_classes = len(real_features)
    print(f"  Loaded features for {num_loaded_classes} classes")
    for level in ["L1", "L2", "L3"]:
        sample_class = list(real_features.keys())[0]
        if level in real_features[sample_class]:
            shape = real_features[sample_class][level].shape
            print(f"    {level}: {shape}")

    # Parse classes to distill
    if args.classes:
        target_classes = [int(c) for c in args.classes.split(",")]
        print(f"\n  Target classes: {target_classes}")
    else:
        target_classes = None

    # Check for resume
    if args.resume_from:
        print(f"\n  Resuming from {args.resume_from}")
        # Load existing results
        resume_path = Path(args.resume_from) / "distilled_data.pt"
        if resume_path.exists():
            existing = torch.load(resume_path, map_location="cpu")
            existing_classes = set(existing.get("class_indices", []))
            if target_classes:
                target_classes = [c for c in target_classes if c not in existing_classes]
            print(f"  Already completed: {len(existing_classes)} classes")
            print(f"  Remaining: {len(target_classes)} classes")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        model_wrapper=dit,
        layer_groups=config.hcdm.layer_groups,
        normalize_output=False,
    )

    # Initialize distiller
    print(f"\n[3/5] Initializing HCDM distiller...")
    distiller = HCDMDistiller(
        config=config,
        dit_wrapper=dit,
        real_features=real_features,
        feature_extractor=feature_extractor,
    )
    print(distiller)

    # Estimate memory and time
    print(f"\n[4/5] Distilling dataset...")
    print(f"  Total images to generate: {config.data.num_classes} × {config.ipc} = "
          f"{config.data.num_classes * config.ipc}")
    print(f"  Estimated time per class: ~{config.distillation.ddim_steps * 0.5:.0f}s")
    print(f"  Estimated total time: ~{config.data.num_classes * config.distillation.ddim_steps * 0.5 / 60:.0f} min")

    # Run distillation
    distilled_data = distiller.distill_all(
        classes=target_classes,
        verbose=args.verbose,
    )

    # Save results
    print(f"\n[5/5] Saving distilled dataset...")
    output_dir = args.output or config.output.output_dir or "./outputs"
    output_dir = Path(output_dir) / config.output.exp_name
    distiller.save_distilled_data(
        distilled_data,
        output_dir=str(output_dir),
        format=config.output.save_format,
    )

    # Final summary
    print(f"\n" + "=" * 70)
    print(f"Distillation Complete!")
    print(f"=" * 70)
    print(f"  IPC:        {config.ipc}")
    print(f"  Classes:    {len(distilled_data)}")
    print(f"  Total images: {sum(v.size(0) for v in distilled_data.values())}")
    print(f"  Output:     {output_dir}/distilled_data.pt")
    print(f"\n  Image stats (class 0):")
    sample = distilled_data[list(distilled_data.keys())[0]]
    print(f"    Shape:  {sample.shape}")
    print(f"    Range:  [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"    Mean:   {sample.mean():.4f}")
    print(f"    Std:    {sample.std():.4f}")
    print(f"\n  Next: train a classifier on the distilled data and evaluate!")
    print(f"  See: scripts/evaluate.py (coming soon)")


if __name__ == "__main__":
    main()
