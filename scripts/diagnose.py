#!/usr/bin/env python3
"""
Diagnose HCDM generation quality.
Checks: VAE encode/decode, plain DDIM vs HCDM, latent stats, image quality.

Usage:
    python scripts/diagnose.py --config configs/cifar10.yaml --distilled ./outputs/.../distilled_data.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))

from hcdm.config import HCDMConfig
from hcdm.models.dit_wrapper import DiTWrapper


def test_vae_roundtrip(dit: DiTWrapper, device):
    """Test VAE encode → decode on a random CIFAR-like image."""
    print("\n" + "=" * 60)
    print("[1] VAE Roundtrip Test")
    print("=" * 60)

    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = CIFAR10(root="./data", train=True, download=True, transform=transform)
    x_real = torch.stack([ds[i][0] for i in range(4)]).to(device)

    print(f"  Real image: shape={x_real.shape}, range=[{x_real.min():.3f},{x_real.max():.3f}]")

    # Encode
    z = dit.encode(x_real)
    print(f"  VAE latent: shape={z.shape}, range=[{z.min():.3f},{z.max():.3f}], "
          f"mean={z.mean():.4f}, std={z.std():.4f}")

    # Decode
    x_recon = dit.decode(z)
    print(f"  Reconstructed: shape={x_recon.shape}, range=[{x_recon.min():.3f},{x_recon.max():.3f}]")

    # Check reconstruction quality
    mse = ((x_real - x_recon) ** 2).mean().item()
    print(f"  Reconstruction MSE: {mse:.6f}")

    # Save comparison
    comparison = torch.cat([x_real[:2], x_recon[:2]], dim=0)
    save_image(comparison, "./outputs/diagnose_vae_roundtrip.png", normalize=True, nrow=2)
    print(f"  Saved: ./outputs/diagnose_vae_roundtrip.png")

    return mse < 0.1  # Good reconstruction


def test_plain_ddim(dit: DiTWrapper, device, num_classes=10):
    """Test plain DDIM generation (no HCDM guidance)."""
    print("\n" + "=" * 60)
    print("[2] Plain DDIM Generation (no HCDM guidance)")
    print("=" * 60)

    T = 50
    step_ratio = dit.num_train_timesteps // T
    timesteps = list(reversed(range(0, dit.num_train_timesteps, step_ratio)))[:T]

    all_images = []
    class_c = 0

    # Test with 4-channel latent (VAE standard)
    m = 4
    class_labels = torch.full((m,), class_c, dtype=torch.long, device=device)

    print(f"  Starting latent channels: {dit.latent_channels}")
    z_t = torch.randn(m, dit.latent_channels, dit.latent_size, dit.latent_size, device=device)

    print(f"  Initial z_T: shape={z_t.shape}, mean={z_t.mean():.4f}, std={z_t.std():.4f}")

    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            next_t = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1
            t_batch = torch.full((m,), t, dtype=torch.long, device=device)

            eps = dit.predict_noise(z_t, t_batch, class_labels, cfg_scale=1.5)
            z_t = dit.ddim_step(z_t, eps, t, next_t)

            if step_idx % 10 == 0:
                print(f"  t={t}: z shape={z_t.shape}, mean={z_t.mean():.4f}, std={z_t.std():.4f}")

    print(f"  Final z_0: shape={z_t.shape}, mean={z_t.mean():.4f}, std={z_t.std():.4f}")

    # Decode
    images = dit.decode(z_t)
    print(f"  Decoded images: shape={images.shape}, range=[{images.min():.3f},{images.max():.3f}]")

    save_image(images, "./outputs/diagnose_plain_ddim.png", normalize=True, nrow=2)
    print(f"  Saved: ./outputs/diagnose_plain_ddim.png")

    return images


def analyze_distilled(path: str):
    """Analyze the distilled dataset."""
    print("\n" + "=" * 60)
    print("[3] Distilled Dataset Analysis")
    print("=" * 60)

    data = torch.load(path, map_location="cpu")
    images = data["images"]
    labels = data["labels"]

    print(f"  Shape: {images.shape}")
    print(f"  Labels: {labels.unique().tolist()}")
    print(f"  Range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mean: {images.mean():.4f}")
    print(f"  Std: {images.std():.4f}")

    # Per-class stats
    for c in labels.unique().tolist():
        mask = labels == c
        c_imgs = images[mask]
        print(f"  Class {c}: {c_imgs.shape[0]} imgs, "
              f"range=[{c_imgs.min():.3f},{c_imgs.max():.3f}], "
              f"mean={c_imgs.mean():.4f}, std={c_imgs.std():.4f}")

    # Check if images look like noise
    # Normal images should have std ~0.3-0.5 (normalized to [-1,1])
    if images.std() < 0.1:
        print(f"\n  ⚠️  WARNING: Image std is very low ({images.std():.4f}) - images may be flat/gray!")
    elif images.std() > 1.0:
        print(f"\n  ⚠️  WARNING: Image std is very high ({images.std():.4f}) - images may be noisy!")

    # Save sample grid
    n_show = min(100, len(images))
    sample = images[:n_show]
    save_image(sample, "./outputs/diagnose_distilled_samples.png",
               normalize=True, nrow=10)
    print(f"\n  Saved sample grid: ./outputs/diagnose_distilled_samples.png")
    print(f"  Open and visually inspect this image!")


def analyze_features(path: str):
    """Analyze pre-computed features."""
    print("\n" + "=" * 60)
    print("[4] Feature Analysis")
    print("=" * 60)

    feats = torch.load(path, map_location="cpu")
    for c in sorted(feats.keys())[:3]:
        for level in sorted(feats[c].keys()):
            f = feats[c][level]
            print(f"  Class {c}, {level}: shape={f.shape}, "
                  f"mean={f.mean():.4f}, std={f.std():.4f}, "
                  f"norm={f.norm(dim=-1).mean():.2f}")


def diagnose_channel_mismatch(dit: DiTWrapper, device):
    """Deep dive into the 4→8 channel issue."""
    print("\n" + "=" * 60)
    print("[5] Channel Mismatch Deep Dive")
    print("=" * 60)

    # Test with 4-channel input
    z_4 = torch.randn(2, 4, dit.latent_size, dit.latent_size, device=device)
    t = torch.zeros(2, dtype=torch.long, device=device)
    y = torch.zeros(2, dtype=torch.long, device=device)

    with torch.no_grad():
        out = dit.transformer(z_4.to(dtype=dit.dtype), timestep=t, class_labels=y)
        out_tensor = out.sample if hasattr(out, 'sample') else out

    print(f"  Input 4ch: {z_4.shape}")
    print(f"  Output:     {out_tensor.shape}")

    # Check if output is structured
    # Compare first 4 vs last 4 channels
    first4 = out_tensor[:, :4, :, :]
    last4 = out_tensor[:, 4:, :, :]
    print(f"  First 4ch: mean={first4.mean():.4f}, std={first4.std():.4f}")
    print(f"  Last 4ch:  mean={last4.mean():.4f}, std={last4.std():.4f}")
    # Corr between first 4 and last 4 channel groups
    f4 = first4.flatten()
    l4 = last4.flatten()
    corr_4 = torch.corrcoef(torch.stack([f4, l4]))[0, 1]
    print(f"  Corr(first4, last4) = {corr_4:.4f}")

    # Check if output channels are paired somehow
    for i in range(4):
        ci = out_tensor[:, i, :, :].flatten()
        cj = out_tensor[:, i+4, :, :].flatten()
        corr = torch.corrcoef(torch.stack([ci, cj]))[0, 1]
        print(f"  Ch{i} ↔ Ch{i+4} correlation: {corr:.4f}")

    # Maybe the expected output shape is different
    # What if the model output is [B, 2*latent, H, W] ?
    # Test: use all 8 channels and try VAE decode
    print(f"\n  Testing: full 8ch as latent → VAE decode...")
    try:
        x_8 = dit.decode(z_4[:, :4])  # only 4ch works
        print(f"  4ch decode: OK, shape={x_8.shape}")
    except Exception as e:
        print(f"  4ch decode error: {e}")

    # Maybe this is a completely different thing
    print(f"\n  Model config details:")
    cfg = dit.transformer.config
    for attr in ['in_channels', 'out_channels', 'patch_size', 'sample_size',
                 'num_attention_heads', 'num_layers', 'hidden_size']:
        val = getattr(cfg, attr, 'N/A')
        print(f"    config.{attr}: {val}")

    # Try to inspect the raw model architecture
    print(f"\n  DiT blocks: {len(dit.transformer.transformer_blocks)}")
    last_block = dit.transformer.transformer_blocks[-1]
    for name, module in last_block.named_modules():
        if isinstance(module, nn.Linear):
            print(f"    {name}: weight {list(module.weight.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10.yaml")
    parser.add_argument("--distilled", default="./outputs/hcdm_cifar10/distilled_data.pt")
    parser.add_argument("--features", default="./outputs/cifar10_features.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = HCDMConfig.from_yaml(args.config)

    # 1. VAE roundtrip
    print("\nLoading DiT model...")
    dit = DiTWrapper(
        dit_model_name=config.model.dit_model_name,
        vae_model_name=config.model.vae_model_name,
        image_size=config.model.image_size,
        num_train_timesteps=config.distillation.num_train_timesteps,
        latent_channels=config.model.latent_channels,
        use_fp16=config.model.use_fp16,
        device=args.device,
    )

    test_vae_roundtrip(dit, device)

    # 2. Plain DDIM
    test_plain_ddim(dit, device, config.data.num_classes)

    # 3. Analyze distilled data
    if Path(args.distilled).exists():
        analyze_distilled(args.distilled)

    # 4. Features
    if Path(args.features).exists():
        analyze_features(args.features)

    # 5. Channel deep dive
    diagnose_channel_mismatch(dit, device)

    print("\n" + "=" * 60)
    print("Diagnosis complete. Check ./outputs/diagnose_*.png for visual results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
