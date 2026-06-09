"""
DiT (Diffusion Transformer) model wrapper for HCDM.

Encapsulates a pre-trained DiT model with VAE encoder/decoder
and provides a clean interface for:
- Encoding images to latent space
- Noise prediction with classifier-free guidance
- Multi-level feature extraction via transformer block hooks
- DDIM sampling steps
- Decoding latents back to pixel space

Supports both v-prediction (standard DiT) and epsilon-prediction modes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class DiTWrapper:
    """Wrapper around a DiT (Diffusion Transformer) model for HCDM.

    Handles:
    - VAE encoding/decoding
    - Noise scheduling (DDPM cosine schedule)
    - Noise prediction with optional CFG
    - Multi-level feature extraction from specified transformer blocks
    - DDIM reverse sampling

    Usage:
        dit = DiTWrapper(
            dit_model_name="facebook/DiT-XL-2-256",
            vae_model_name="stabilityai/sd-vae-ft-mse",
            image_size=256,
        )
        # Pre-compute features
        real_features = dit.extract_features(z_clean, class_labels)
        # During distillation
        eps = dit.predict_noise(z_t, t, class_labels, cfg_scale=1.5)
    """

    def __init__(
        self,
        dit_model_name: str = "facebook/DiT-XL-2-256",
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        image_size: int = 256,
        num_train_timesteps: int = 1000,
        latent_channels: int = 4,
        hidden_dim: int = 1152,
        num_blocks: int = 28,
        patch_size: int = 2,
        num_heads: int = 16,
        prediction_type: str = "v_prediction",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            dit_model_name: HuggingFace model ID or local path for DiT.
            vae_model_name: HuggingFace model ID or local path for VAE.
            image_size: Input image size (square).
            num_train_timesteps: Number of diffusion timesteps used in training.
            latent_channels: Number of VAE latent channels.
            hidden_dim: DiT transformer hidden dimension.
            num_blocks: Number of DiT transformer blocks.
            patch_size: DiT patch size.
            num_heads: Number of attention heads.
            prediction_type: 'v_prediction' or 'epsilon'.
            use_fp16: Use FP16 for VAE operations.
            device: Device string (auto-detect if None).
        """
        self.image_size = image_size
        self.num_train_timesteps = num_train_timesteps
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.prediction_type = prediction_type
        self.use_fp16 = use_fp16

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = torch.float16 if use_fp16 else torch.float32

        # DiT output channels (may differ from latent_channels in non-standard models)
        # None means "same as latent_channels" — set by _validate_output_shape()
        self.dit_output_channels = None

        # Latent size after VAE encoding
        self.vae_scale_factor = 8  # Standard for SD VAE
        self.latent_size = image_size // self.vae_scale_factor

        self._load_models(dit_model_name, vae_model_name)
        self._setup_scheduler()

    def _load_models(self, dit_model_name: str, vae_model_name: str):
        """Load DiT and VAE models."""
        try:
            from diffusers import (
                DiTTransformer2DModel,
                AutoencoderKL,
            )
        except ImportError:
            raise ImportError(
                "diffusers is required. Install with: pip install diffusers"
            )

        # Load VAE
        try:
            vae_load_kwargs = {}
            self.vae = AutoencoderKL.from_pretrained(
                vae_model_name, **vae_load_kwargs
            ).to(self.device, dtype=self.dtype)
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
            print(f"  VAE loaded: {type(self.vae).__name__}")
        except Exception as e:
            print(f"  Could not load VAE from {vae_model_name}: {e}")
            print("  VAE will need to be set manually via set_vae()")
            self.vae = None

        # Load DiT transformer
        # Try multiple loading strategies
        load_strategies = [
            {},                                          # direct load
            {"subfolder": "transformer"},                 # nested in transformer/
        ]
        for strategy in load_strategies:
            try:
                self.transformer = DiTTransformer2DModel.from_pretrained(
                    dit_model_name, **strategy
                ).to(self.device, dtype=self.dtype)
                self.transformer.eval()
                for param in self.transformer.parameters():
                    param.requires_grad = False
                print(f"  DiT loaded with strategy: {strategy if strategy else 'direct'}")
                break
            except Exception as e:
                last_error = e
                continue
        else:
            print(f"  Could not load DiT from {dit_model_name}: {last_error}")
            print("  DiT transformer will need to be set manually via set_transformer()")
            self.transformer = None

        # Determine actual parameters from loaded models
        if self.transformer is not None:
            config = self.transformer.config
            self.hidden_dim = getattr(config, 'hidden_size', self.hidden_dim)
            self.num_blocks = getattr(config, 'num_layers', self.num_blocks)
            self.num_heads = getattr(config, 'num_attention_heads', self.num_heads)
            self.patch_size = getattr(config, 'patch_size', self.patch_size)

            # IMPORTANT: detect actual in_channels from DiT config
            detected_in_channels = getattr(config, 'in_channels', None)
            if detected_in_channels is not None and detected_in_channels != self.latent_channels:
                print(f"  Detected DiT in_channels={detected_in_channels} "
                      f"(config had {self.latent_channels}, auto-correcting)")
                self.latent_channels = detected_in_channels

            # Validate actual output shape with a dummy forward pass
            self._validate_output_shape()

            print(f"  Effective config: in_channels={self.latent_channels}, "
                  f"hidden_dim={self.hidden_dim}, num_blocks={self.num_blocks}, "
                  f"patch_size={self.patch_size}")

    def _setup_scheduler(self):
        """Setup noise scheduler (cosine schedule used by DiT)."""
        # Cosine schedule following DiT paper
        betas = self._cosine_beta_schedule(self.num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]).to(self.device),
            self.alphas_cumprod[:-1]
        ])

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule as used in DiT.

        Args:
            timesteps: Number of timesteps.
            s: Small offset to prevent singularities.

        Returns:
            Beta values [timesteps].
        """
        import math
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, max=0.999)

    def _validate_output_shape(self):
        """Run a dummy forward pass to detect the actual output channels.

        The DiT config may report in_channels=4, but the actual loaded model
        (depending on diffusers version) may output a different number of
        channels. This method runs a minimal forward pass and corrects
        self.latent_channels to match reality.
        """
        if self.transformer is None:
            return

        try:
            with torch.no_grad():
                # Create dummy input matching current assumed shape
                dummy_z = torch.zeros(
                    1, self.latent_channels,
                    self.latent_size, self.latent_size,
                    device=self.device, dtype=self.dtype,
                )
                dummy_t = torch.zeros(1, dtype=torch.long, device=self.device)
                dummy_y = torch.zeros(1, dtype=torch.long, device=self.device)

                output = self.transformer(
                    dummy_z,
                    timestep=dummy_t,
                    class_labels=dummy_y,
                )

                # Handle both dict-style and tensor outputs
                if hasattr(output, 'sample'):
                    out_tensor = output.sample
                else:
                    out_tensor = output

                actual_channels = out_tensor.shape[1]

                if actual_channels != self.latent_channels:
                    print(f"  *** Shape mismatch detected! ***")
                    print(f"  Input channels (VAE latent):   {self.latent_channels}")
                    print(f"  Output channels (DiT predict): {actual_channels}")
                    print(f"  Model output shape: {tuple(out_tensor.shape)}")
                    # IMPORTANT: Do NOT change latent_channels (it must match VAE
                    # output for encoding/decoding). Instead, store the DiT output
                    # channels separately and slice during reverse diffusion.
                    self.dit_output_channels = actual_channels
                    print(f"  Stored dit_output_channels={self.dit_output_channels}, "
                          f"will slice predictions to match z_t during sampling")
                else:
                    self.dit_output_channels = actual_channels
                    print(f"  Shape validation passed: "
                          f"input={dummy_z.shape} → output={out_tensor.shape}")

        except Exception as e:
            print(f"  Shape validation skipped (could not run forward pass): {e}")
            print(f"  Assuming latent_channels={self.latent_channels}")

    def set_vae(self, vae: nn.Module):
        """Set VAE manually (useful if auto-loading fails)."""
        self.vae = vae.to(self.device, dtype=self.dtype)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def set_transformer(self, transformer: nn.Module):
        """Set DiT transformer manually."""
        self.transformer = transformer.to(self.device, dtype=self.dtype)
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            x: Images [B, 3, H, W] in [-1, 1].

        Returns:
            Latents [B, C_lat, H_lat, W_lat].
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call set_vae() first.")

        with torch.no_grad():
            # VAE encode
            posterior = self.vae.encode(x)
            if hasattr(posterior, 'latent_dist'):
                z = posterior.latent_dist.sample()
            else:
                z = posterior
            # Scale factor (standard for SD VAE)
            z = z * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents back to pixel space.

        Args:
            z: Latents [B, C_lat, H_lat, W_lat].

        Returns:
            Images [B, 3, H, W] in [-1, 1].
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")

        with torch.no_grad():
            z = z / self.vae.config.scaling_factor
            x = self.vae.decode(z)
            if hasattr(x, 'sample'):
                x = x.sample
        return x

    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Predict noise (or v) using DiT with classifier-free guidance.

        Args:
            z_t: Noisy latents [B, C, H, W].
            t: Timesteps [B] (int).
            class_labels: Class indices [B].
            cfg_scale: CFG scale (1.0 = no CFG, 1.5 = standard for DiT).

        Returns:
            Predicted noise/v [B, C_out, H, W].
        """
        if self.transformer is None:
            raise RuntimeError("DiT transformer not loaded.")

        def _call_transformer(z, t_val, y_val):
            """Call transformer and extract tensor from output."""
            raw = self.transformer(
                z.to(dtype=self.dtype),
                timestep=t_val,
                class_labels=y_val,
            )
            # Handle various output formats
            if hasattr(raw, 'sample') and raw.sample is not None:
                return raw.sample.float()
            elif isinstance(raw, torch.Tensor):
                return raw.float()
            elif isinstance(raw, dict):
                # Try common keys
                for key in ['sample', 'hidden_states', 'latent', 'output']:
                    if key in raw and raw[key] is not None:
                        return raw[key].float()
            raise ValueError(
                f"Unexpected transformer output type: {type(raw)}. "
                f"Keys: {dir(raw) if hasattr(raw, 'keys') else 'N/A'}"
            )

        if cfg_scale != 1.0:
            null_labels = torch.full_like(class_labels, 1000)
            noise_uncond = _call_transformer(z_t, t, null_labels)
            noise_cond = _call_transformer(z_t, t, class_labels)
            noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise = _call_transformer(z_t, t, class_labels)

        # Detect and handle shape mismatch between input and output
        if noise.shape[1] != z_t.shape[1]:
            if not hasattr(self, '_shape_warned'):
                print(f"  predict_noise: input z_t shape={tuple(z_t.shape)}, "
                      f"output noise shape={tuple(noise.shape)}")
                print(f"  Channel mismatch: z_t has {z_t.shape[1]}, "
                      f"DiT outputs {noise.shape[1]} channels")
                # Try first-4, then last-4 — pick the one with higher variance
                var_first4 = noise[:, :z_t.shape[1], :, :].var().item()
                var_last4 = noise[:, noise.shape[1]-z_t.shape[1]:, :, :].var().item()
                print(f"  First {z_t.shape[1]}ch var={var_first4:.4f}, "
                      f"Last {z_t.shape[1]}ch var={var_last4:.4f}")
                if var_last4 > var_first4:
                    print(f"  Using LAST {z_t.shape[1]} channels (higher variance)")
                    self._slice_offset = noise.shape[1] - z_t.shape[1]
                else:
                    print(f"  Using FIRST {z_t.shape[1]} channels")
                    self._slice_offset = 0
                self.dit_output_channels = noise.shape[1]
                self._shape_warned = True
            # Slice to match z_t channels
            offset = getattr(self, '_slice_offset', 0)
            noise = noise[:, offset:offset + z_t.shape[1], :, :]

        return noise

    def extract_features(
        self,
        z: torch.Tensor,
        class_labels: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        layer_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        with_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-level features from DiT transformer blocks.

        Registers forward hooks on specified block groups and runs a
        forward pass to collect intermediate representations.

        Args:
            z: Latents [B, C, H, W] (usually clean latents, t≈0).
            class_labels: Class indices [B].
            t: Timesteps [B]. If None, uses t=0 (clean).
            layer_groups: Dict of {level_name: (start_idx, end_idx)}.
            with_grad: If True, preserves gradient flow through the model.
                       Required for HCDM guidance gradient computation.

        Returns:
            Dict of {level_name: Tensor[B, D]} where D = hidden_dim.
            Features are mean-pooled over spatial (patch) dimensions.
        """
        if self.transformer is None:
            raise RuntimeError("DiT transformer not loaded.")

        if layer_groups is None:
            # Default 3-level split
            layer_groups = {
                "L1": (0, self.num_blocks // 3),
                "L2": (self.num_blocks // 3, 2 * self.num_blocks // 3),
                "L3": (2 * self.num_blocks // 3, self.num_blocks),
            }

        if t is None:
            t = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # Register hooks on the LAST block of each group
        features = {}
        hooks = []

        for level_name, (start, end) in layer_groups.items():
            hook_block_idx = end - 1
            if hook_block_idx >= len(self.transformer.transformer_blocks):
                hook_block_idx = len(self.transformer.transformer_blocks) - 1

            def make_hook(name, needs_grad):
                def hook_fn(module, input, output):
                    # output: hidden states [B, N_patches, hidden_dim]
                    if needs_grad:
                        features[name] = output  # preserve grad
                    else:
                        features[name] = output.detach()
                return hook_fn

            block = self.transformer.transformer_blocks[hook_block_idx]
            handle = block.register_forward_hook(make_hook(level_name, with_grad))
            hooks.append(handle)

        # Forward pass
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            self.transformer(
                z.to(dtype=self.dtype),
                timestep=t,
                class_labels=class_labels,
            )

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Mean-pool over patches [B, N_patches, D] → [B, D]
        pooled = {}
        for level_name, feat in features.items():
            pooled[level_name] = feat.float().mean(dim=1)

        return pooled

    def ddim_step(
        self,
        z_t: torch.Tensor,
        eps: torch.Tensor,
        t: int,
        next_t: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single deterministic DDIM reverse step.

        Args:
            z_t: Current latent [B, C, H, W].
            eps: Noise prediction [B, C, H, W].
            t: Current timestep index.
            next_t: Next timestep index.
            eta: Stochasticity (0 = deterministic).

        Returns:
            z_{t-1} [B, C, H, W].
        """
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_next = self.alphas_cumprod[next_t] if next_t >= 0 else torch.tensor(1.0, device=z_t.device)

        # Predict z_0
        if self.prediction_type == "v_prediction":
            z_0_pred = self._predict_x0_from_v(z_t, eps, alpha_bar_t)
        else:
            z_0_pred = self._predict_x0_from_eps(z_t, eps, alpha_bar_t)

        # DDIM direction
        direction = torch.sqrt(1.0 - alpha_bar_next - eta**2) * eps
        z_next = torch.sqrt(alpha_bar_next) * z_0_pred + direction

        if eta > 0:
            z_next = z_next + eta * torch.randn_like(z_t)

        return z_next

    def _predict_x0_from_v(self, z_t, v_pred, alpha_bar_t):
        """Estimate z_0 from v-prediction."""
        sqrt_alpha = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
        return sqrt_alpha * z_t - sqrt_one_minus * v_pred

    def _predict_x0_from_eps(self, z_t, eps_pred, alpha_bar_t):
        """Estimate z_0 from epsilon-prediction."""
        sqrt_alpha = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
        return (z_t - sqrt_one_minus * eps_pred) / sqrt_alpha

    def get_alpha_bar(self, t: int, as_tensor: bool = True) -> torch.Tensor:
        """Get cumulative alpha product at timestep t.

        Args:
            t: Timestep index.
            as_tensor: Return as tensor on self.device (True) or float (False).
        """
        val = self.alphas_cumprod[t]
        if as_tensor:
            return val.clone().detach().to(self.device)
        return val.item()

    @property
    def device_info(self) -> str:
        return str(self.device)

    def __repr__(self) -> str:
        return (
            f"DiTWrapper(\n"
            f"  image_size={self.image_size},\n"
            f"  latent_size={self.latent_size},\n"
            f"  latent_channels={self.latent_channels},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_blocks={self.num_blocks},\n"
            f"  prediction_type={self.prediction_type},\n"
            f"  device={self.device}\n"
            f")"
        )
