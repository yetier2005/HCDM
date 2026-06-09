"""
Unit tests for feature extraction and guidance scheduling.

Note: These tests use mock features (no DiT model needed).
"""

import torch
import pytest

torch.manual_seed(42)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcdm.guidance import (
    level_weight,
    level_weights_batch,
    guidance_schedule,
    inject_guidance,
    predict_x0_from_v_pred,
    predict_x0_from_eps,
)


class TestLevelWeight:
    """Test timestep-aware level weighting."""

    def test_l1_peaks_late(self):
        """L1 (fine) should peak at low t (late in denoising)."""
        T = 50
        w_early = level_weight("L1", t=40, T=T)  # Early in denoising
        w_late = level_weight("L1", t=5, T=T)    # Late in denoising
        assert w_late > w_early, f"L1 should peak late: w(5)={w_late:.4f} > w(40)={w_early:.4f}"

    def test_l3_peaks_early(self):
        """L3 (coarse) should peak at high t (early in denoising)."""
        T = 50
        w_early = level_weight("L3", t=40, T=T)  # Early in denoising
        w_late = level_weight("L3", t=5, T=T)    # Late in denoising
        assert w_early > w_late, f"L3 should peak early: w(40)={w_early:.4f} > w(5)={w_late:.4f}"

    def test_weight_range(self):
        """Weights should be in [0, 1]."""
        T = 100
        for level in ["L1", "L2", "L3"]:
            for t in range(0, T, 5):
                w = level_weight(level, t, T)
                assert 0.0 <= w <= 1.0, f"Level {level}, t={t}: weight={w}"

    def test_level_weights_batch(self):
        """Level weights batch should sum to 1."""
        from hcdm.guidance import level_weights_batch
        T = 50
        for t in range(0, T, 5):
            weights = level_weights_batch(["L1", "L2", "L3"], t, T)
            # Should have at least one level active
            assert len(weights) >= 1
            # Should sum to 1
            assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestGuidanceSchedule:
    """Test global guidance strength schedule."""

    def test_schedule_range(self):
        """Guidance should be in [0, guidance_scale]."""
        T = 1000
        max_scale = 100.0
        for t in range(0, T, 10):
            s = guidance_schedule(t, T, max_scale)
            assert 0.0 <= s <= max_scale, f"t={t}: s={s}"

    def test_schedule_zero_at_extremes(self):
        """Guidance should be 0 at the very beginning and end."""
        T = 1000
        assert guidance_schedule(0, T) == 0.0
        assert guidance_schedule(T - 1, T) == 0.0

    def test_schedule_symmetric_shape(self):
        """Beta(2,2) schedule should be symmetric."""
        T = 1000
        # Check symmetry around T/2
        s_left = guidance_schedule(T // 4, T, 100.0)
        s_right = guidance_schedule(3 * T // 4, T, 100.0)
        # Beta(2,2) is symmetric: pdf(x) = pdf(1-x)
        assert abs(s_left - s_right) < 1e-6, f"Schedule should be symmetric"

    def test_schedule_peak(self):
        """Peak should be at the expected position."""
        T = 1000
        # For Beta(2,2), peak is at 0.5
        s_peak = guidance_schedule(T // 2, T, 100.0)
        s_off_peak = guidance_schedule(T // 2 + 50, T, 100.0)
        assert s_peak >= s_off_peak, "Peak should be at the center"


class TestGuidanceInjection:
    """Test HCDM gradient injection into noise prediction."""

    def test_inject_guidance_basic(self):
        """Basic guidance injection with mock gradient."""
        B, C, H, W = 2, 4, 32, 32
        z_t = torch.randn(B, C, H, W, requires_grad=True)
        eps = torch.randn(B, C, H, W)

        # Create a simple loss
        loss = z_t.pow(2).mean()

        eps_guided = inject_guidance(
            eps=eps,
            z_t=z_t,
            loss=loss,
            s_t=10.0,
            grad_clip=1.0,
        )

        assert eps_guided.shape == eps.shape
        # Epsilon should be modified
        assert not torch.allclose(eps_guided, eps)

    def test_inject_guidance_zero_strength(self):
        """Zero guidance strength should not modify eps."""
        B, C, H, W = 1, 4, 32, 32
        z_t = torch.randn(B, C, H, W, requires_grad=True)
        eps = torch.randn(B, C, H, W)
        loss = z_t.pow(2).mean()

        eps_guided = inject_guidance(eps, z_t, loss, s_t=0.0)
        assert torch.allclose(eps_guided, eps)

    def test_gradient_clipping(self):
        """Gradient clipping should limit the gradient norm."""
        B, C, H, W = 1, 4, 32, 32
        eps = torch.randn(B, C, H, W)

        # First: no clipping (large grad_clip = effectively no clip)
        z_t1 = torch.randn(B, C, H, W, requires_grad=True)
        loss1 = z_t1.pow(2).sum() * 1000
        eps_no_clip = inject_guidance(eps, z_t1, loss1, s_t=1.0, grad_clip=999.0)

        # Second: strong clipping
        z_t2 = torch.randn(B, C, H, W, requires_grad=True)
        loss2 = z_t2.pow(2).sum() * 1000
        eps_clip = inject_guidance(eps, z_t2, loss2, s_t=1.0, grad_clip=0.01)

        diff_no_clip = (eps_no_clip - eps).norm().item()
        diff_clip = (eps_clip - eps).norm().item()

        assert diff_clip <= diff_no_clip, "Clipped gradient should produce smaller change"


class TestX0Prediction:
    """Test x0 estimation from noise predictions."""

    def test_v_pred_to_x0(self):
        """Test v-prediction to x0 conversion."""
        z_t = torch.randn(2, 4, 32, 32)
        v_pred = torch.randn(2, 4, 32, 32)
        alpha_bar = torch.tensor([0.8, 0.8])

        z_0 = predict_x0_from_v_pred(z_t, v_pred, alpha_bar)
        assert z_0.shape == z_t.shape
        assert not torch.isnan(z_0).any()

    def test_eps_pred_to_x0(self):
        """Test epsilon-prediction to x0 conversion."""
        z_t = torch.randn(2, 4, 32, 32)
        eps_pred = torch.randn(2, 4, 32, 32)
        alpha_bar = torch.tensor([0.5, 0.5])

        z_0 = predict_x0_from_eps(z_t, eps_pred, alpha_bar)
        assert z_0.shape == z_t.shape

    def test_x0_reconstruction(self):
        """Forward and inverse should be consistent."""
        B, C, H, W = 1, 4, 8, 8
        z_0 = torch.randn(B, C, H, W)
        alpha_bar = torch.tensor([0.7])

        # Forward: add noise
        eps = torch.randn(B, C, H, W)
        z_t = torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * eps

        # Inverse: estimate z_0
        z_0_est = predict_x0_from_eps(z_t, eps, alpha_bar)
        assert torch.allclose(z_0, z_0_est, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
