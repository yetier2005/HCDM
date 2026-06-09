"""
Core HCDM distillation engine.

Orchestrates the full distillation pipeline:
1. For each class, generates m images via HCDM-guided reverse diffusion.
2. At each denoising step:
   a. Standard CFG noise prediction
   b. Estimate clean latent ẑ₀
   c. Multi-level feature extraction
   d. HCDM loss computation (intra + inter + diversity)
   e. Gradient injection into the reverse process
   f. DDIM step
3. VAE decode → final distilled images

Supports:
- Batch parallel generation within a class
- Checkpointing for long runs
- Mixed precision
- Configurable guidance frequency
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from .config import HCDMConfig
from .models.dit_wrapper import DiTWrapper
from .feature_extractor import FeatureExtractor
from .distribution_matching import compute_hcdm_loss
from .guidance import (
    level_weights_batch,
    guidance_schedule,
    inject_guidance,
)


class HCDMDistiller:
    """HCDM dataset distillation engine.

    Takes a pre-trained DiT model and pre-computed real features,
    generates a distilled dataset through HCDM-guided reverse diffusion.

    Usage:
        distiller = HCDMDistiller(config, dit_wrapper, real_features)
        distilled_data = distiller.distill_all()
        # distilled_data = {class_idx: Tensor[m, 3, H, W]}
    """

    def __init__(
        self,
        config: HCDMConfig,
        dit_wrapper: DiTWrapper,
        real_features: Dict[int, Dict[str, torch.Tensor]],
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        """
        Args:
            config: Full HCDM configuration.
            dit_wrapper: Initialized DiT model wrapper.
            real_features: Pre-computed real dataset features.
                {class_idx: {level_name: Tensor[N_c, D]}}
            feature_extractor: FeatureExtractor instance.
                Created automatically if None.
        """
        self.config = config
        self.dit = dit_wrapper
        self.real_features = real_features
        self.feature_extractor = feature_extractor or FeatureExtractor(
            dit_wrapper, config.hcdm.layer_groups
        )

        # Cache config values
        self.ipc = config.distillation.ipc
        self.T = config.distillation.ddim_steps
        self.num_train_steps = config.distillation.num_train_timesteps
        self.cfg_scale = config.distillation.cfg_scale
        self.batch_size = config.distillation.batch_size
        self.deterministic = config.distillation.deterministic
        self.seed = config.distillation.seed

        # HCDM params
        self.lambda_repel = config.hcdm.lambda_repel
        self.lambda_div = config.hcdm.lambda_div
        self.div_margin = config.hcdm.div_margin
        self.inter_margin = config.hcdm.inter_margin
        self.topk_hard_negatives = config.hcdm.topk_hard_negatives
        self.mmd_kernel = config.hcdm.mmd_kernel
        self.normalize_features = config.hcdm.normalize_features
        self.grad_clip = config.hcdm.grad_clip
        self.guidance_every_n = config.hcdm.guidance_every_n_steps

        # Guidance schedule params
        self.guidance_scale = config.guidance.guidance_scale
        self.beta_alpha = config.guidance.beta_alpha
        self.beta_beta = config.guidance.beta_beta
        self.level_centers = config.guidance.level_centers
        self.level_sigmas = config.guidance.level_sigmas
        self.min_weight = config.guidance.min_weight

        # Derived
        self.num_classes = len(real_features)
        self.device = dit_wrapper.device

        # Set seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def distill_class(
        self,
        class_c: int,
        verbose: bool = True,
        save_intermediate: bool = False,
    ) -> torch.Tensor:
        """Generate a distilled batch of images for a single class.

        Uses HCDM-guided reverse diffusion with DDIM steps.
        Generates m = ipc images in parallel.

        Args:
            class_c: Target class index.
            verbose: Print progress.
            save_intermediate: Whether to save intermediate latents.

        Returns:
            Generated images [m, 3, H, W] in [-1, 1].
        """
        m = self.ipc
        class_labels = torch.full((m,), class_c, dtype=torch.long, device=self.device)

        # Warmup: run a single forward pass to auto-detect DiT output channels
        # (the DiT in some environments outputs more channels than VAE latents)
        dummy_z = torch.randn(
            1, self.dit.latent_channels,
            self.dit.latent_size, self.dit.latent_size,
            device=self.device,
        )
        dummy_t = torch.zeros(1, dtype=torch.long, device=self.device)
        dummy_y = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            _ = self.dit.predict_noise(dummy_z, dummy_t, dummy_y, cfg_scale=1.0)

        # Move current class real features to device
        real_features_class_dev = {
            level: feat.to(self.device)
            for level, feat in self.real_features[class_c].items()
        }

        # Pre-compute class mean features on device for fast hard-negative screening
        class_means_dev = {}
        for c, feats in self.real_features.items():
            class_means_dev[c] = {
                level: feat.mean(dim=0, keepdim=True).to(self.device)
                for level, feat in feats.items()
            }

        # Initialize noise with VAE latent channels (model input, NOT output)
        z_t = torch.randn(m, self.dit.latent_channels,
                         self.dit.latent_size, self.dit.latent_size,
                         device=self.device)

        # DDIM timestep sequence
        step_ratio = self.num_train_steps // self.T
        timesteps = list(range(0, self.num_train_steps, step_ratio))[:self.T]
        timesteps = list(reversed(timesteps))  # Reverse: T-1 → 0

        if verbose:
            from tqdm import tqdm
            iterator = tqdm(list(enumerate(timesteps)), desc=f"Class {class_c}")
        else:
            iterator = enumerate(timesteps)

        intermediates = [] if save_intermediate else None

        for step_idx, t in iterator:
            next_t = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1

            # 1. CFG noise prediction
            t_batch = torch.full((m,), t, dtype=torch.long, device=self.device)
            eps = self.dit.predict_noise(z_t, t_batch, class_labels, cfg_scale=self.cfg_scale)

            # 2. HCDM guidance (every N steps)
            if self.guidance_every_n > 0 and step_idx % self.guidance_every_n == 0 and t > 0:
                # Compute global guidance strength
                s_t = guidance_schedule(
                    t, self.num_train_steps,
                    guidance_scale=self.guidance_scale,
                    beta_alpha=self.beta_alpha,
                    beta_beta=self.beta_beta,
                )

                if s_t > 0:
                    # Need z_t with grad for backward
                    z_t_grad = z_t.detach().clone()
                    z_t_grad.requires_grad_(True)

                    # Estimate z_0 using model's built-in converter
                    alpha_bar_t = self.dit.get_alpha_bar(t, as_tensor=True)
                    if self.dit.prediction_type == "v_prediction":
                        v_pred = eps.detach()
                        z_0_pred = self.dit._predict_x0_from_v(z_t_grad, v_pred, alpha_bar_t)
                    else:
                        eps_detached = eps.detach()
                        z_0_pred = self.dit._predict_x0_from_eps(z_t_grad, eps_detached, alpha_bar_t)

                    # Extract multi-level features WITH gradient tracking
                    synth_features = self.feature_extractor.extract(
                        z_0_pred,
                        class_labels,
                        t=torch.zeros(m, dtype=torch.long, device=self.device),
                        with_grad=True,
                    )

                    # Compute HCDM loss (real features moved to device inside)
                    real_features_class = real_features_class_dev

                    loss_hcdm = compute_hcdm_loss(
                        synth_features=synth_features,
                        real_features_class=real_features_class,
                        all_real_features=self.real_features,
                        class_c=class_c,
                        t=t,
                        T=self.num_train_steps,
                        lambda_repel=self.lambda_repel,
                        lambda_div=self.lambda_div,
                        div_margin=self.div_margin,
                        inter_margin=self.inter_margin,
                        topk_hard_negatives=self.topk_hard_negatives,
                        kernel=self.mmd_kernel,
                        normalize=self.normalize_features,
                        level_centers=self.level_centers,
                        level_sigmas=self.level_sigmas,
                        min_weight=self.min_weight,
                    )

                    # Inject guidance
                    eps = inject_guidance(
                        eps=eps,
                        z_t=z_t_grad,
                        loss=loss_hcdm,
                        s_t=s_t,
                        grad_clip=self.grad_clip,
                    )

            # 3. DDIM step
            z_t = self.dit.ddim_step(z_t, eps, t, next_t, eta=0.0 if self.deterministic else 0.1)

            if save_intermediate:
                intermediates.append(z_t.detach().cpu())

        # Decode final latent
        images = self.dit.decode(z_t)
        images = images.float()

        # Clamp to valid range
        images = torch.clamp(images, -1.0, 1.0)

        return images

    def distill_all(
        self,
        classes: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """Generate distilled dataset for all (or specified) classes.

        Args:
            classes: List of class indices. None = all classes.
            verbose: Print progress.

        Returns:
            Dict mapping class_idx → generated images [ipc, 3, H, W].
        """
        if classes is None:
            classes = sorted(self.real_features.keys())

        results = {}
        total_start = time.time()

        for class_c in classes:
            print(f"\n{'='*50}")
            print(f"Distilling class {class_c} ({classes.index(class_c)+1}/{len(classes)})")
            print(f"{'='*50}")

            start = time.time()
            images = self.distill_class(class_c, verbose=verbose)
            elapsed = time.time() - start

            results[class_c] = images.cpu()
            print(f"  Time: {elapsed:.1f}s | Images shape: {images.shape}")
            print(f"  Image stats: min={images.min():.3f}, max={images.max():.3f}, "
                  f"mean={images.mean():.3f}, std={images.std():.3f}")

        total_elapsed = time.time() - total_start
        print(f"\n{'='*50}")
        print(f"Distillation complete! Total time: {total_elapsed:.1f}s "
              f"({total_elapsed/len(classes):.1f}s per class)")
        print(f"{'='*50}")

        return results

    def save_distilled_data(
        self,
        distilled_data: Dict[int, torch.Tensor],
        output_dir: str,
        format: str = "pt",
    ):
        """Save distilled dataset to disk.

        Args:
            distilled_data: {class_idx: Tensor[ipc, 3, H, W]}.
            output_dir: Directory to save to.
            format: "pt" (single .pt file) or "png" (individual images).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "pt":
            # Save as single tensor
            # Organize: [num_classes * ipc, 3, H, W] with labels
            all_images = []
            all_labels = []
            for class_c in sorted(distilled_data.keys()):
                imgs = distilled_data[class_c]
                all_images.append(imgs)
                all_labels.append(torch.full((imgs.size(0),), class_c, dtype=torch.long))

            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            save_dict = {
                "images": all_images,
                "labels": all_labels,
                "ipc": self.ipc,
                "num_classes": self.num_classes,
                "config": self.config.to_dict(),
            }
            torch.save(save_dict, output_path / "distilled_data.pt")
            print(f"Saved distilled data to {output_path / 'distilled_data.pt'}")

        elif format == "png":
            from torchvision.utils import save_image

            for class_c in sorted(distilled_data.keys()):
                class_dir = output_path / f"class_{class_c:03d}"
                class_dir.mkdir(parents=True, exist_ok=True)

                imgs = distilled_data[class_c]
                for i in range(imgs.size(0)):
                    # Convert from [-1, 1] to [0, 1]
                    img = (imgs[i] + 1.0) / 2.0
                    save_image(img, class_dir / f"sample_{i:03d}.png")

            print(f"Saved images to {output_dir}/class_*/")

        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def from_config(
        cls,
        config: HCDMConfig,
        real_features_path: Optional[str] = None,
    ) -> "HCDMDistiller":
        """Factory method: create a distiller from config and (optionally) load features.

        Args:
            config: HCDMConfig instance.
            real_features_path: Path to precomputed features (.pt).
                               If None, uses config.output.features_dir.

        Returns:
            Initialized HCDMDistiller.
        """
        # Initialize DiT wrapper
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
        )

        # Load real features
        if real_features_path is None:
            real_features_path = config.output.features_dir
        if real_features_path:
            real_features = torch.load(real_features_path, map_location="cpu")
            print(f"Loaded real features from {real_features_path}")
        else:
            raise ValueError("real_features_path is required")

        return cls(config, dit, real_features)

    def __repr__(self) -> str:
        return (
            f"HCDMDistiller(\n"
            f"  ipc={self.ipc}, steps={self.T}, cfg_scale={self.cfg_scale},\n"
            f"  lambda_repel={self.lambda_repel}, lambda_div={self.lambda_div},\n"
            f"  guidance_scale={self.guidance_scale}, num_classes={self.num_classes},\n"
            f"  device={self.device}\n"
            f")"
        )
