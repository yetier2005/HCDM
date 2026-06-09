"""
Distribution matching losses for HCDM.

Implements:
- Maximum Mean Discrepancy (MMD) with multi-kernel RBF
- Intra-class distribution attraction
- Inter-class distribution repulsion with hard negative mining
- Intra-batch diversity regularization
- Combined HCDM loss with timestep-aware level weighting
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def compute_mmd(
    A: torch.Tensor,
    B: torch.Tensor,
    sigma: Optional[float] = None,
    kernel: str = "rbf",
    normalize: bool = True,
) -> torch.Tensor:
    """Compute squared Maximum Mean Discrepancy (MMD²) between two feature sets.

    Uses RBF kernel with median heuristic bandwidth by default.

    Args:
        A: First feature set [n, d]
        B: Second feature set [m, d]
        sigma: RBF kernel bandwidth. If None, uses median heuristic.
        kernel: Kernel type ('rbf', 'linear', 'poly').
        normalize: Whether to L2-normalize features before computing MMD.

    Returns:
        MMD² scalar value (non-negative).

    Shape:
        - A: (n_samples_A, feature_dim)
        - B: (n_samples_B, feature_dim)
        - Output: scalar tensor
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got A: {A.dim()}D, B: {B.dim()}D")

    if A.size(0) == 0 or B.size(0) == 0:
        return torch.tensor(0.0, device=A.device)

    # L2 normalize for stability
    if normalize:
        A = F.normalize(A, p=2, dim=-1)
        B = F.normalize(B, p=2, dim=-1)

    if sigma is None:
        # Median heuristic
        combined = torch.cat([A, B], dim=0)
        # Use a subset to avoid OOM for large feature sets
        if combined.size(0) > 1000:
            idx = torch.randperm(combined.size(0))[:1000]
            combined = combined[idx]
        pairwise_dists = torch.pdist(combined, p=2)
        sigma = torch.median(pairwise_dists) / 2.0
        # Ensure sigma is positive
        sigma = sigma.clamp(min=1e-6)

    if kernel == "rbf":
        return _mmd_rbf(A, B, sigma)
    elif kernel == "linear":
        return _mmd_linear(A, B)
    elif kernel == "poly":
        return _mmd_poly(A, B, sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")


def _mmd_rbf(A: torch.Tensor, B: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """MMD² with RBF (Gaussian) kernel.

    k(x, y) = exp(-||x - y||² / (2 * sigma²))
    """
    # Pairwise squared distances
    # K_AA
    A_sqn = (A**2).sum(dim=1, keepdim=True)  # [n, 1]
    AA_dist = A_sqn + A_sqn.T - 2 * A @ A.T   # [n, n]
    K_AA = torch.exp(-AA_dist / (2 * sigma**2))

    # K_BB
    B_sqn = (B**2).sum(dim=1, keepdim=True)  # [m, 1]
    BB_dist = B_sqn + B_sqn.T - 2 * B @ B.T   # [m, m]
    K_BB = torch.exp(-BB_dist / (2 * sigma**2))

    # K_AB
    AB_dist = A_sqn + B_sqn.T - 2 * A @ B.T   # [n, m]
    K_AB = torch.exp(-AB_dist / (2 * sigma**2))

    # MMD² = E[k(A,A')] + E[k(B,B')] - 2E[k(A,B)]
    # Unbiased estimate
    n, m = A.size(0), B.size(0)

    # K_AA without diagonal
    K_AA_no_diag = K_AA - torch.diag(torch.diag(K_AA))
    K_BB_no_diag = K_BB - torch.diag(torch.diag(K_BB))

    term1 = K_AA_no_diag.sum() / (n * (n - 1)) if n > 1 else K_AA.mean()
    term2 = K_BB_no_diag.sum() / (m * (m - 1)) if m > 1 else K_BB.mean()
    term3 = K_AB.mean()

    mmd2 = term1 + term2 - 2 * term3
    # Clamp to non-negative (unbiased estimator can be slightly negative)
    return torch.clamp(mmd2, min=0.0)


def _mmd_linear(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """MMD² with linear kernel k(x,y) = x·y."""
    mean_A = A.mean(dim=0)
    mean_B = B.mean(dim=0)
    return ((mean_A - mean_B) ** 2).sum()


def _mmd_poly(A: torch.Tensor, B: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """MMD² with polynomial kernel k(x,y) = (x·y / sigma² + 1)³."""
    c = 1.0 / (sigma.item() ** 2)
    gamma = sigma

    def poly_kernel(X, Y):
        # (X·Y^T / sigma² + 1)^3
        return (X @ Y.T * c + 1.0) ** 3

    K_AA = poly_kernel(A, A)
    K_BB = poly_kernel(B, B)
    K_AB = poly_kernel(A, B)

    n, m = A.size(0), B.size(0)
    K_AA_no_diag = K_AA - torch.diag(torch.diag(K_AA))
    K_BB_no_diag = K_BB - torch.diag(torch.diag(K_BB))

    term1 = K_AA_no_diag.sum() / (n * (n - 1)) if n > 1 else K_AA.mean()
    term2 = K_BB_no_diag.sum() / (m * (m - 1)) if m > 1 else K_BB.mean()
    term3 = K_AB.mean()

    mmd2 = term1 + term2 - 2 * term3
    return torch.clamp(mmd2, min=0.0)


def compute_multi_kernel_mmd(
    A: torch.Tensor,
    B: torch.Tensor,
    sigmas: Optional[List[float]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """MMD² with mixture of RBF kernels at different bandwidths.

    Uses multiple sigma values to capture distribution differences
    at different scales.

    Args:
        A: First feature set [n, d].
        B: Second feature set [m, d].
        sigmas: List of RBF bandwidths. If None, uses [0.2s, 0.5s, 1.0s, 2.0s, 5.0s]
                where s is the median heuristic bandwidth.
        normalize: L2 normalize features.

    Returns:
        Multi-kernel MMD² scalar.
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(f"Expected 2D tensors")

    if A.size(0) == 0 or B.size(0) == 0:
        return torch.tensor(0.0, device=A.device)

    if normalize:
        A = F.normalize(A, p=2, dim=-1)
        B = F.normalize(B, p=2, dim=-1)

    # Base sigma via median heuristic
    combined = torch.cat([A, B], dim=0)
    if combined.size(0) > 1000:
        idx = torch.randperm(combined.size(0))[:1000]
        combined = combined[idx]
    base_sigma = torch.median(torch.pdist(combined, p=2)) / 2.0
    base_sigma = base_sigma.clamp(min=1e-6)

    if sigmas is None:
        sigmas = [0.2 * base_sigma.item(), 0.5 * base_sigma.item(),
                  base_sigma.item(), 2.0 * base_sigma.item(), 5.0 * base_sigma.item()]

    mmd_sum = 0.0
    for s in sigmas:
        mmd_sum += _mmd_rbf(A, B, torch.tensor(s, device=A.device))

    return mmd_sum / len(sigmas)


def compute_intra_class_loss(
    synth_features: Dict[str, torch.Tensor],
    real_features: Dict[str, torch.Tensor],
    level_weights: Optional[Dict[str, float]] = None,
    kernel: str = "rbf",
    normalize: bool = True,
) -> torch.Tensor:
    """Intra-class distribution attraction loss.

    Minimizes MMD² between synthetic and real features of the SAME class.
    Pulls synthetic distribution toward real distribution at each level.

    Args:
        synth_features: {level_name: Tensor[B, D]} synthetic features.
        real_features: {level_name: Tensor[N, D]} real features (same class).
        level_weights: Optional {level_name: float} weighting per level.
                       If None, equal weights.
        kernel: MMD kernel type.
        normalize: L2 normalize features.

    Returns:
        Scalar loss (lower = more similar distributions).
    """
    total_loss = 0.0
    levels = list(synth_features.keys())

    if level_weights is None:
        level_weights = {l: 1.0 / len(levels) for l in levels}

    for level in levels:
        if level not in real_features:
            continue
        f_s = synth_features[level]
        f_r = real_features[level]

        w = level_weights.get(level, 1.0)
        mmd_val = compute_mmd(f_s, f_r, kernel=kernel, normalize=normalize)
        total_loss += w * mmd_val

    return total_loss


def compute_inter_class_loss(
    synth_features: Dict[str, torch.Tensor],
    all_real_features: Dict[int, Dict[str, torch.Tensor]],
    class_c: int,
    topk: int = 5,
    margin: float = 0.5,
    level_weights: Optional[Dict[str, float]] = None,
    kernel: str = "rbf",
    normalize: bool = True,
) -> torch.Tensor:
    """Inter-class distribution repulsion loss.

    Maximizes MMD² between synthetic features and real features of DIFFERENT classes.
    Uses hard negative mining: only considers the top-k closest negative classes.
    Uses hinge loss: repulsion stops once margin is exceeded.

    Args:
        synth_features: {level_name: Tensor[B, D]} synthetic features.
        all_real_features: {class_idx: {level_name: Tensor[N, D]}} all real features.
        class_c: Current target class.
        topk: Number of hard negative classes to consider.
        margin: Hinge margin for repulsion.
        level_weights: Optional level weight dict.
        kernel: MMD kernel type.
        normalize: L2 normalize features.

    Returns:
        Scalar loss (negative = more separated from other classes).
    """
    total_loss = 0.0
    levels = list(synth_features.keys())

    if level_weights is None:
        level_weights = {l: 1.0 / len(levels) for l in levels}

    # Find hard negative classes: compute mean feature distance for each negative class
    # Use a middle-level feature for fast screening
    screen_level = "L2" if "L2" in levels else levels[0]
    f_s_mean = synth_features[screen_level].mean(dim=0, keepdim=True)  # [1, D]
    if normalize:
        f_s_mean = F.normalize(f_s_mean, p=2, dim=-1)

    class_distances = {}
    for c_neg, feats in all_real_features.items():
        if c_neg == class_c:
            continue
        f_r_mean = feats[screen_level].mean(dim=0, keepdim=True)
        if normalize:
            f_r_mean = F.normalize(f_r_mean, p=2, dim=-1)
        dist = torch.norm(f_s_mean - f_r_mean, dim=-1).item()
        class_distances[c_neg] = dist

    # Select top-k closest (hardest) negative classes
    hard_negatives = sorted(class_distances, key=class_distances.get)[:topk]

    for level in levels:
        f_s = synth_features[level]
        w = level_weights.get(level, 1.0)

        for c_neg in hard_negatives:
            f_r_neg = all_real_features[c_neg][level]
            mmd_val = compute_mmd(f_s, f_r_neg, kernel=kernel, normalize=normalize)

            # Hinge loss: L = max(0, margin - MMD²)
            # This pushes MMD² above `margin`, then stops
            total_loss += w * torch.relu(margin - mmd_val)

    # Normalize by number of negatives
    num_terms = len(levels) * len(hard_negatives)
    if num_terms > 0:
        total_loss = total_loss / num_terms

    return total_loss


def compute_diversity_loss(
    synth_features: Dict[str, torch.Tensor],
    margin: float = 0.05,
    level_weights: Optional[Dict[str, float]] = None,
    kernel: str = "rbf",
    normalize: bool = True,
) -> torch.Tensor:
    """Intra-batch diversity regularization.

    Encourages synthetic samples within a batch to be different from each other.
    Uses pairwise MMD with hinge: penalizes when samples are too similar.

    Args:
        synth_features: {level_name: Tensor[B, D]} synthetic features (batch of m samples).
        margin: Minimum desired MMD between samples.
        level_weights: Optional level weights.
        kernel: MMD kernel type.
        normalize: L2 normalize.

    Returns:
        Scalar diversity loss (lower = more diverse, countered by negative sign in total loss).
    """
    B = list(synth_features.values())[0].size(0)
    if B <= 1:
        return torch.tensor(0.0, device=list(synth_features.values())[0].device)

    levels = list(synth_features.keys())
    if level_weights is None:
        level_weights = {l: 1.0 / len(levels) for l in levels}

    total_loss = 0.0
    pair_count = 0

    for level in levels:
        f = synth_features[level]
        if normalize:
            f = F.normalize(f, p=2, dim=-1)
        w = level_weights.get(level, 1.0)

        for i in range(B):
            for j in range(i + 1, B):
                # MMD between individual samples (as 1-sample "distributions")
                fi = f[i:i+1]  # [1, D]
                fj = f[j:j+1]  # [1, D]
                # For single samples, we can use L2 distance as a proxy
                # since MMD with RBF reduces to kernel distance
                dist = torch.norm(fi - fj, dim=-1)
                # Hinge: penalize if distance < margin
                total_loss += w * torch.relu(margin - dist)
                pair_count += 1

    if pair_count > 0:
        total_loss = total_loss / pair_count

    return total_loss


def compute_hcdm_loss(
    synth_features: Dict[str, torch.Tensor],
    real_features_class: Dict[str, torch.Tensor],
    all_real_features: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    class_c: Optional[int] = None,
    t: Optional[int] = None,
    T: int = 50,
    lambda_repel: float = 0.3,
    lambda_div: float = 0.1,
    div_margin: float = 0.05,
    inter_margin: float = 0.5,
    topk_hard_negatives: int = 5,
    kernel: str = "rbf",
    normalize: bool = True,
    level_centers: Optional[Dict[str, float]] = None,
    level_sigmas: Optional[Dict[str, float]] = None,
    min_weight: float = 0.01,
) -> torch.Tensor:
    """Combined Hierarchical Contrastive Distribution Matching (HCDM) loss.

    L_HCDM = Σ_l w_l(t) · [MMD²(f_s^l, f_real^l(c)) - λ_repel · L_inter - λ_div · L_div]

    Where:
    - w_l(t) is the timestep-aware level weight (Gaussian schedule)
    - L_inter is the inter-class repulsion loss
    - L_div is the intra-batch diversity loss

    Args:
        synth_features: {level: Tensor[B, D]} current synthetic features.
        real_features_class: {level: Tensor[N, D]} real features for target class.
        all_real_features: {class: {level: Tensor[N, D]}} all classes' features (for inter-class).
        class_c: Current target class index.
        t: Current diffusion timestep (for level weighting).
        T: Total diffusion steps.
        lambda_repel: Weight for inter-class repulsion term.
        lambda_div: Weight for diversity term.
        div_margin: Diversity hinge margin.
        inter_margin: Inter-class repulsion hinge margin.
        topk_hard_negatives: Number of hard negative classes.
        kernel: MMD kernel type.
        normalize: L2 normalize features.
        level_centers: Dict of level → center (as fraction of T) for Gaussian weight.
        level_sigmas: Dict of level → sigma (as fraction of T) for Gaussian weight.
        min_weight: Minimum level weight to include in computation.

    Returns:
        Combined HCDM loss scalar.
    """
    levels = list(synth_features.keys())

    # Default level centers and sigmas
    if level_centers is None:
        level_centers = {"L1": 0.10, "L2": 0.40, "L3": 0.75}
    if level_sigmas is None:
        level_sigmas = {"L1": 0.12, "L2": 0.18, "L3": 0.15}

    # Compute timestep-aware level weights
    if t is not None:
        t_normalized = t / max(T, 1)
        computed_level_weights = {}
        for level in levels:
            center = level_centers.get(level, 0.4)
            sigma = level_sigmas.get(level, 0.18)
            w = torch.exp(torch.tensor(
                -((t_normalized - center) ** 2) / (2 * sigma ** 2)
            ))
            if w.item() >= min_weight:
                computed_level_weights[level] = w.item()
    else:
        computed_level_weights = {l: 1.0 / len(levels) for l in levels}

    if len(computed_level_weights) == 0:
        # Fallback: use all levels equally
        computed_level_weights = {l: 1.0 / len(levels) for l in levels}

    # Normalize weights
    w_sum = sum(computed_level_weights.values())
    level_weights = {l: w / w_sum for l, w in computed_level_weights.items()}

    # 1. Intra-class attraction (always computed)
    L_intra = compute_intra_class_loss(
        synth_features, real_features_class,
        level_weights=level_weights,
        kernel=kernel, normalize=normalize,
    )

    # 2. Inter-class repulsion (if other class features available)
    L_inter = torch.tensor(0.0, device=L_intra.device)
    if all_real_features is not None and class_c is not None:
        L_inter = compute_inter_class_loss(
            synth_features, all_real_features, class_c,
            topk=topk_hard_negatives,
            margin=inter_margin,
            level_weights=level_weights,
            kernel=kernel, normalize=normalize,
        )

    # 3. Intra-batch diversity
    L_div = torch.tensor(0.0, device=L_intra.device)
    B = list(synth_features.values())[0].size(0)
    if B > 1:
        L_div = compute_diversity_loss(
            synth_features,
            margin=div_margin,
            level_weights=level_weights,
            kernel=kernel, normalize=normalize,
        )

    # Combined loss
    # L_intra: pull close (minimize)
    # L_inter: push away from negatives (minimize L_inter → maximize MMD to negatives)
    # L_div: encourage within-batch diversity (minimize L_div → maximize pairwise distance)
    L_total = L_intra - lambda_repel * L_inter - lambda_div * L_div

    return L_total
