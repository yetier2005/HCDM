# HCDM: Hierarchical Contrastive Distribution Matching

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**HCDM** is a novel method for **diffusion-based dataset distillation** that uses **multi-level distribution discrepancy metrics** with **contrastive objectives** to guide the reverse diffusion process toward generating high-quality, highly representative distilled datasets.

## Overview

Dataset distillation aims to synthesize a compact yet information-rich dataset — where models trained on this small set achieve performance comparable to training on the full original dataset. HCDM advances this goal by:

1. **Multi-Level Feature Extraction**: Extracts features from different semantic depths of a diffusion model (DiT transformer blocks), capturing local textures, part-level structures, and global semantics.
2. **Contrastive Distribution Matching**: Minimizes intra-class MMD (pull same-class distributions close) while maximizing inter-class MMD (push different-class distributions apart).
3. **Timestep-Aware Guidance**: Injects distribution-matching gradients at different diffusion timesteps with Gaussian scheduling — coarse semantics early, fine details late.
4. **Diversity Regularization**: Encourages intra-batch diversity to prevent mode collapse.

## Architecture

```
Real Dataset ──→ [VAE Encoder] ──→ [DiT Feature Extraction] ──→ Real Features Cache
                                                                    │
                                                                    │ (pre-computed)
                                                                    ▼
z_T ~ N(0,I) ──→ ╔══════════════════════════════════════════╗ ──→ Distilled Images
                  ║   Reverse Diffusion with HCDM Guidance     ║
                  ║                                           ║
                  ║  for t = T..1:                            ║
                  ║    ε = CFG(z_t, t, c)                     ║
                  ║    ẑ₀ = predict_x0(z_t, ε)               ║
                  ║    f_s = ExtractFeatures(ẑ₀)              ║
                  ║    L = MMD²(f_s, f_real)                  ║
                  ║      - λ·MMD²(f_s, f_other_classes)       ║
                  ║      - β·Diversity(f_s_batch)             ║
                  ║    ε = ε + s(t)·∇_{z_t}L                 ║
                  ║    z_{t-1} = DDIM_step(z_t, ε)            ║
                  ╚══════════════════════════════════════════╝
```

## Installation

```bash
cd HCDM
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.1+
- diffusers 0.25+
- CUDA-capable GPU (recommended)

## Quick Start

### 1. Pre-compute Real Dataset Features

```bash
python scripts/precompute_features.py --config configs/cifar10.yaml
```

This extracts L1/L2/L3 features from all training images and saves them to `./outputs/cifar10_features.pt`.

### 2. Run Distillation

```bash
# Generate IPC=10 distilled CIFAR-10
python scripts/distill.py --config configs/cifar10.yaml --ipc 10

# Generate IPC=1 distilled CIFAR-10 (extreme compression)
python scripts/distill.py --config configs/cifar10.yaml --ipc 1

# Generate IPC=50 for ImageNet-1K
python scripts/distill.py --config configs/imagenet1k.yaml --ipc 50
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All hyperparameters are in YAML files under `configs/`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `distillation.ipc` | Images Per Class | 10 |
| `distillation.ddim_steps` | DDIM sampling steps | 50 |
| `distillation.cfg_scale` | Classifier-free guidance scale | 1.5 |
| `hcdm.lambda_repel` | Inter-class repulsion strength | 0.3 |
| `hcdm.lambda_div` | Diversity regularization strength | 0.1 |
| `hcdm.topk_hard_negatives` | Hard negative classes per step | 5 |
| `hcdm.normalize_features` | L2-normalize before MMD | true |
| `guidance.guidance_scale` | Peak guidance strength | 100.0 |
| `guidance.guidance_every_n_steps` | Apply HCDM every N steps | 1 |

## Project Structure

```
HCDM/
├── configs/                # YAML configuration files
│   ├── default.yaml        # Default config (all parameters)
│   ├── cifar10.yaml        # CIFAR-10 experiment config
│   └── imagenet1k.yaml     # ImageNet-1K experiment config
├── hcdm/                   # Core library
│   ├── config.py           # Configuration management
│   ├── distiller.py        # Main distillation engine
│   ├── feature_extractor.py # Multi-level feature extraction
│   ├── distribution_matching.py # MMD + contrastive losses
│   ├── guidance.py         # Timestep-aware guidance
│   └── models/
│       └── dit_wrapper.py  # DiT model wrapper
├── scripts/                # Executable scripts
│   ├── precompute_features.py # Feature pre-computation
│   └── distill.py          # Main distillation entry point
├── tests/                  # Unit tests
│   ├── test_distribution_matching.py
│   └── test_feature_extractor.py
└── requirements.txt        # Python dependencies
```

## Key Components

### `hcdm/distribution_matching.py`

Core distribution matching losses:
- `compute_mmd(A, B)` — MMD² with RBF kernel (median heuristic bandwidth)
- `compute_hcdm_loss(...)` — Combined HCDM loss with timestep-aware level weighting

### `hcdm/guidance.py`

Timestep-aware guidance scheduling:
- `level_weight(level, t, T)` — Gaussian window weight per level per timestep
- `guidance_schedule(t, T, scale)` — Beta distribution global guidance strength
- `inject_guidance(eps, z_t, loss, s_t)` — Gradient injection with clipping

### `hcdm/distiller.py`

Main distillation loop orchestrating:
1. Per-class parallel generation
2. CFG noise prediction
3. Multi-level feature extraction
4. HCDM loss computation
5. Gradient injection
6. DDIM step

## Method Details

### Multi-Level Feature Hierarchy

| Level | DiT Blocks | Semantic Content | Timestep Peak |
|-------|------------|------------------|---------------|
| **L1** (Fine) | 0–8 | Local textures, edges, colors | t/T ≈ 0.10 (late) |
| **L2** (Mid) | 9–18 | Part-level structure, shapes | t/T ≈ 0.40 (mid) |
| **L3** (Coarse) | 19–27 | Global semantics, class prototypes | t/T ≈ 0.75 (early) |

### Loss Function

$$\mathcal{L}_{\text{HCDM}} = \sum_{l} w_l(t) \left[ \text{MMD}^2(f_s^l, f_{\text{real}}^l(c)) - \lambda_{\text{repel}} \cdot \mathcal{L}_{\text{inter}} - \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}} \right]$$

Where:
- $w_l(t)$ is the timestep-aware Gaussian weight for level $l$
- $\mathcal{L}_{\text{inter}}$ uses hinge loss on MMD² to hard negative classes
- $\mathcal{L}_{\text{div}}$ encourages pairwise diversity within the synthetic batch

### DDIM with HCDM Guidance

$$\epsilon_{\text{guided}} = \epsilon_{\text{cfg}} + s(t) \cdot \nabla_{z_t} \mathcal{L}_{\text{HCDM}}$$

The global guidance strength $s(t)$ follows a Beta(2,2) distribution, peaking at the middle of denoising.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hcdm2026,
  title = {HCDM: Hierarchical Contrastive Distribution Matching for Dataset Distillation},
  year = {2026},
  url = {https://github.com/your-org/HCDM},
}
```

## License

MIT License — See LICENSE file for details.
