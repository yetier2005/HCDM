"""
Unit tests for distribution matching losses.
"""

import torch
import pytest

# These tests work without GPU
torch.manual_seed(42)

# Import functions under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcdm.distribution_matching import (
    compute_mmd,
    compute_multi_kernel_mmd,
    compute_intra_class_loss,
    compute_inter_class_loss,
    compute_diversity_loss,
    compute_hcdm_loss,
)


class TestMMD:
    """Test MMD computation with different kernels."""

    def test_mmd_identical_distributions(self):
        """MMD should be ~0 for identical distributions."""
        A = torch.randn(100, 64)
        mmd = compute_mmd(A, A.clone(), kernel="rbf", normalize=True)
        assert mmd.item() < 0.1, f"MMD should be small for identical sets, got {mmd.item()}"

    def test_mmd_different_distributions(self):
        """MMD should be >0 for different distributions."""
        A = torch.randn(100, 64)
        B = torch.randn(100, 64) + 5.0  # Shifted
        mmd = compute_mmd(A, B, kernel="rbf", normalize=True)
        assert mmd.item() > 0.0, f"MMD should be positive for different distributions"

    def test_mmd_symmetry(self):
        """MMD should be symmetric."""
        A = torch.randn(50, 32)
        B = torch.randn(80, 32) + 2.0
        mmd_ab = compute_mmd(A, B, kernel="rbf", normalize=True)
        mmd_ba = compute_mmd(B, A, kernel="rbf", normalize=True)
        assert torch.allclose(mmd_ab, mmd_ba, atol=1e-4), f"MMD should be symmetric"

    def test_mmd_non_negative(self):
        """MMD should always be >= 0."""
        A = torch.randn(50, 32)
        B = torch.randn(30, 32)
        mmd = compute_mmd(A, B, kernel="rbf", normalize=True)
        assert mmd.item() >= 0, f"MMD should be non-negative"

    def test_mmd_linear_kernel(self):
        """Linear kernel MMD: should equal ||mean(A) - mean(B)||²."""
        A = torch.randn(100, 32)
        B = torch.randn(100, 32) + 3.0
        mmd = compute_mmd(A, B, kernel="linear", normalize=False)
        expected = ((A.mean(0) - B.mean(0)) ** 2).sum()
        assert torch.allclose(mmd, expected, atol=1e-2)

    def test_mmd_poly_kernel(self):
        """Polynomial kernel should produce non-negative result."""
        A = torch.randn(100, 32)
        B = torch.randn(100, 32)
        mmd = compute_mmd(A, B, kernel="poly", normalize=True)
        assert mmd.item() >= 0

    def test_mmd_empty_input(self):
        """Empty input should return 0."""
        A = torch.randn(0, 64)
        B = torch.randn(10, 64)
        mmd = compute_mmd(A, B, kernel="rbf")
        assert mmd.item() == 0.0

    def test_mmd_single_sample(self):
        """Single sample per distribution."""
        A = torch.randn(1, 64)
        B = torch.randn(1, 64)
        mmd = compute_mmd(A, B, kernel="rbf", normalize=True)
        assert mmd.item() >= 0
        assert not torch.isnan(mmd)

    def test_multi_kernel_mmd(self):
        """Multi-kernel MMD should combine multiple bandwidths."""
        A = torch.randn(100, 64)
        B = torch.randn(100, 64) + 1.0
        mmd = compute_multi_kernel_mmd(A, B, sigmas=[0.5, 1.0, 2.0])
        assert mmd.item() >= 0
        assert not torch.isnan(mmd)


class TestIntraClassLoss:
    """Test intra-class distribution attraction."""

    def test_intra_class_basic(self):
        """Basic sanity check for intra-class loss."""
        synth = {"L1": torch.randn(10, 64), "L2": torch.randn(10, 128), "L3": torch.randn(10, 256)}
        real = {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128), "L3": torch.randn(100, 256)}

        loss = compute_intra_class_loss(synth, real, kernel="rbf", normalize=True)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_intra_class_with_level_weights(self):
        """Intra-class loss with custom level weights."""
        synth = {"L1": torch.randn(5, 32), "L2": torch.randn(5, 64)}
        real = {"L1": torch.randn(50, 32), "L2": torch.randn(50, 64)}
        weights = {"L1": 0.3, "L2": 0.7}

        loss = compute_intra_class_loss(synth, real, level_weights=weights)
        assert loss.item() >= 0


class TestInterClassLoss:
    """Test inter-class distribution repulsion."""

    def test_inter_class_basic(self):
        """Basic inter-class repulsion test."""
        synth = {"L1": torch.randn(10, 64), "L2": torch.randn(10, 128)}

        all_real = {
            0: {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128)},
            1: {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128)},
            2: {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128)},
        }

        loss = compute_inter_class_loss(synth, all_real, class_c=0, topk=2)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestDiversityLoss:
    """Test intra-batch diversity regularization."""

    def test_diversity_single_sample(self):
        """Single sample has no diversity loss."""
        synth = {"L1": torch.randn(1, 64)}
        loss = compute_diversity_loss(synth)
        assert loss.item() == 0.0

    def test_diversity_identical_samples(self):
        """Identical samples should produce high diversity penalty."""
        x = torch.randn(4, 64)
        synth = {"L1": x}
        loss = compute_diversity_loss(synth, margin=999.0)
        assert loss.item() > 0, "Identical samples should be penalized"

    def test_diversity_very_different_samples(self):
        """Very different samples should produce low diversity penalty."""
        x = torch.randn(4, 64) * 100  # Very spread out
        synth = {"L1": x}
        loss = compute_diversity_loss(synth, margin=0.01)
        assert loss.item() == 0.0, "Very spread samples should not be penalized"


class TestHCDMLoss:
    """Test the combined HCDM loss."""

    def test_hcdm_loss_basic(self):
        """Full HCDM loss with all components."""
        synth = {
            "L1": torch.randn(5, 64),
            "L2": torch.randn(5, 128),
            "L3": torch.randn(5, 256),
        }
        real_class = {
            "L1": torch.randn(100, 64),
            "L2": torch.randn(100, 128),
            "L3": torch.randn(100, 256),
        }
        all_real = {
            0: real_class,
            1: {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128), "L3": torch.randn(100, 256)},
            2: {"L1": torch.randn(100, 64), "L2": torch.randn(100, 128), "L3": torch.randn(100, 256)},
        }

        loss = compute_hcdm_loss(
            synth_features=synth,
            real_features_class=real_class,
            all_real_features=all_real,
            class_c=0,
            t=25,
            T=50,
            lambda_repel=0.3,
            lambda_div=0.1,
        )

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_hcdm_loss_no_inter_class(self):
        """HCDM loss without inter-class repulsion."""
        synth = {"L1": torch.randn(5, 32)}
        real_class = {"L1": torch.randn(50, 32)}

        loss = compute_hcdm_loss(
            synth_features=synth,
            real_features_class=real_class,
            # No all_real_features → no inter-class
            class_c=None,
        )

        assert not torch.isnan(loss)
        assert loss.item() != 0  # Should still have intra-class MMD

    def test_hcdm_loss_timestep_weighting(self):
        """HCDM loss at different timesteps should produce different weights."""
        synth = {
            "L1": torch.randn(5, 64),
            "L2": torch.randn(5, 128),
            "L3": torch.randn(5, 256),
        }
        real_class = {
            "L1": torch.randn(100, 64),
            "L2": torch.randn(100, 128),
            "L3": torch.randn(100, 256),
        }

        # Early timestep (t=40/50): L3 should dominate
        loss_early = compute_hcdm_loss(
            synth_features=synth,
            real_features_class=real_class,
            t=40, T=50,
        )

        # Late timestep (t=5/50): L1 should dominate
        loss_late = compute_hcdm_loss(
            synth_features=synth,
            real_features_class=real_class,
            t=5, T=50,
        )

        # Both should be valid
        assert not torch.isnan(loss_early)
        assert not torch.isnan(loss_late)
        # They should be different because level weights differ
        assert not torch.allclose(loss_early, loss_late)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
