"""
Timestep-aware guidance scheduling for HCDM.

Implements:
- Level-specific Gaussian weight schedules
- Global Beta-distribution guidance strength schedule
- Gradient injection into the reverse diffusion process
"""

import torch
from typing import Dict, Optional


def level_weight(
    level: str,
    t: int,
    T: int,
    level_centers: Optional[Dict[str, float]] = None,
    level_sigmas: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the weight for a feature level at a given timestep.

    Each level has a Gaussian window centered at a specific phase
    of the denoising process.
    - L1 (Early/Fine layers): peak late in denoising (t small),
      capturing local textures and fine details.
    - L2 (Middle layers): peak mid-way, capturing part-level structure.
    - L3 (Late/Coarse layers): peak early in denoising (t large),
      capturing global semantics and class prototypes.

    Args:
        level: Level name ('L1', 'L2', 'L3').
        t: Current timestep (0 to T-1, where T = max noise).
        T: Total number of timesteps.
        level_centers: Dict of level → center position (fraction of T).
        level_sigmas: Dict of level → sigma (fraction of T).

    Returns:
        Weight in [0, 1], higher means this level is more important at timestep t.
    """
    if level_centers is None:
        level_centers = {
            "L1": 0.10,   # Fine details: peak when image is almost clean
            "L2": 0.40,   # Mid-level: peak at middle of denoising
            "L3": 0.75,   # Coarse semantics: peak early in denoising
        }
    if level_sigmas is None:
        level_sigmas = {
            "L1": 0.12,
            "L2": 0.18,
            "L3": 0.15,
        }

    center = level_centers.get(level, 0.40)
    sigma = level_sigmas.get(level, 0.18)

    t_normalized = t / max(T, 1)
    weight = float(torch.exp(torch.tensor(
        -((t_normalized - center) ** 2) / (2 * sigma ** 2)
    )).item())

    return weight


def level_weights_batch(
    levels: list,
    t: int,
    T: int,
    level_centers: Optional[Dict[str, float]] = None,
    level_sigmas: Optional[Dict[str, float]] = None,
    min_weight: float = 0.01,
) -> Dict[str, float]:
    """Compute normalized level weights for all levels at timestep t.

    Args:
        levels: List of level names.
        t: Current timestep.
        T: Total timesteps.
        level_centers: Dict of level → center fraction.
        level_sigmas: Dict of level → sigma fraction.
        min_weight: Minimum weight threshold for inclusion.

    Returns:
        Dict of level → normalized weight (sums to 1).
    """
    raw_weights = {}
    for level in levels:
        w = level_weight(level, t, T, level_centers, level_sigmas)
        if w >= min_weight:
            raw_weights[level] = w

    if len(raw_weights) == 0:
        # Fallback: equal weights
        raw_weights = {l: 1.0 for l in levels}

    # Normalize
    w_sum = sum(raw_weights.values())
    return {l: w / w_sum for l, w in raw_weights.items()}


def guidance_schedule(
    t: int,
    T: int,
    guidance_scale: float = 100.0,
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
) -> float:
    """Global guidance strength schedule.

    Uses a Beta(α, β) distribution shape:
    - Weak at beginning (t close to T): image is mostly noise,
      distribution matching signal is unreliable.
    - Strongest in the middle: meaningful structure emerges.
    - Weak at end (t close to 0): the image is almost clean,
      limited room for modification.

    Args:
        t: Current timestep (0 to T-1).
        T: Total timesteps.
        guidance_scale: Maximum guidance strength.
        beta_alpha: Alpha parameter for Beta distribution.
        beta_beta: Beta parameter for Beta distribution.

    Returns:
        Guidance strength at timestep t.
    """
    import math
    from scipy.special import beta as beta_func

    t_norm = t / max(T, 1)

    # Edge cases: no guidance at extremes
    if t_norm > 0.98 or t_norm < 0.02:
        return 0.0

    # Beta distribution PDF
    # pdf(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
    B = beta_func(beta_alpha, beta_beta)
    pdf = (t_norm ** (beta_alpha - 1) * (1 - t_norm) ** (beta_beta - 1)) / B

    # Normalize so that peak = guidance_scale
    # Peak of Beta(α,β) is at (α-1)/(α+β-2) when α,β > 1
    peak = (beta_alpha - 1) / (beta_alpha + beta_beta - 2)
    peak_pdf = (peak ** (beta_alpha - 1) * (1 - peak) ** (beta_beta - 1)) / B
    normalized = pdf / peak_pdf

    return normalized * guidance_scale


def inject_guidance(
    eps: torch.Tensor,
    z_t: torch.Tensor,
    loss: torch.Tensor,
    s_t: float,
    grad_clip: float = 1.0,
    eps_reg: float = 1e-8,
) -> torch.Tensor:
    """Inject HCDM distribution-matching guidance into the noise prediction.

    Modifies the noise prediction using the gradient of the HCDM loss
    with respect to the current noisy latent:

        ε_guided = ε + s(t) · ∇_{z_t} L_HCDM

    This steers the denoising trajectory toward generating images
    whose features better match the target data distribution.

    Args:
        eps: Original noise prediction [B, C, H, W] (after CFG).
        z_t: Current noisy latent (requires grad) [B, C, H, W].
        loss: HCDM loss scalar (must be result of z_t-dependent computation).
        s_t: Current guidance strength (from guidance_schedule).
        grad_clip: Maximum L2 norm of the gradient.
        eps_reg: Small constant to avoid division by zero.

    Returns:
        Guided noise prediction [B, C, H, W].
    """
    if s_t <= 0.0 or loss is None:
        return eps

    # Compute gradient of loss w.r.t. z_t
    grads = torch.autograd.grad(
        loss,
        z_t,
        grad_outputs=torch.ones_like(loss),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )

    if grads is None or grads[0] is None:
        return eps

    grad = grads[0]

    # Gradient clipping for stability
    grad_norm = grad.norm(p=2, dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(grad_clip / (grad_norm + eps_reg), max=1.0)
    grad = grad * scale

    # Inject guidance
    # Note: for v-prediction, the sign follows the same convention
    # as classifier guidance
    eps_guided = eps + s_t * grad

    return eps_guided


def predict_x0_from_v_pred(
    z_t: torch.Tensor,
    v_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """Estimate clean latent z_0 from v-prediction.

    v-prediction (used by DiT): v = √(ᾱ_t) * ε - √(1-ᾱ_t) * z_0
    Rearranging: z_0 = √(ᾱ_t) * z_t - √(1-ᾱ_t) * v

    Args:
        z_t: Noisy latent [B, C, H, W].
        v_pred: v-prediction from model [B, C, H, W].
        alpha_bar_t: ᾱ_t, cumulative product of alphas [B] or scalar.

    Returns:
        Estimated clean latent z_0 [B, C, H, W].
    """
    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    z_0 = sqrt_alpha_bar * z_t - sqrt_one_minus_alpha_bar * v_pred
    return z_0


def predict_x0_from_eps(
    z_t: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """Estimate clean latent z_0 from epsilon-prediction.

    ε-prediction: z_t = √(ᾱ_t) * z_0 + √(1-ᾱ_t) * ε
    Rearranging: z_0 = (z_t - √(1-ᾱ_t) * ε) / √(ᾱ_t)

    Args:
        z_t: Noisy latent [B, C, H, W].
        eps_pred: ε-prediction from model [B, C, H, W].
        alpha_bar_t: ᾱ_t [B] or scalar.

    Returns:
        Estimated clean latent z_0 [B, C, H, W].
    """
    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    z_0 = (z_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
    return z_0


def ddim_step(
    z_t: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    next_t: int,
    alpha_bars: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM reverse step.

    Args:
        z_t: Current latent [B, C, H, W].
        eps: Noise prediction (possibly guided) [B, C, H, W].
        t: Current timestep index.
        next_t: Next timestep index (t - Δt).
        alpha_bars: Pre-computed ᾱ_t for all timesteps [T].
        eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM-like).

    Returns:
        z_{t-1} [B, C, H, W].
    """
    alpha_bar_t = alpha_bars[t]
    alpha_bar_next = alpha_bars[next_t] if next_t >= 0 else torch.tensor(1.0)

    # Predict z_0
    z_0_pred = predict_x0_from_eps(z_t, eps, alpha_bar_t)

    # DDIM update
    sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
    pred_dir = torch.sqrt(1.0 - alpha_bar_next - eta ** 2) * eps
    z_next = sqrt_alpha_bar_next * z_0_pred + pred_dir

    if eta > 0:
        noise = torch.randn_like(z_t)
        z_next = z_next + eta * noise

    return z_next
