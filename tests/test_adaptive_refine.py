"""Test mesh-free collocation with adaptive refinement."""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, adaptive_refine, hessian, sobol, train


class TestAdaptiveRefine:
    def test_shape_preserved(self):
        """n_add == n_remove should keep the same total count."""
        pts = sobol(100, [(0.0, 1.0)])
        res_sq = torch.rand(100)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)], n_add=20, n_remove=20)
        assert result.shape == (100, 1)

    def test_grows_with_more_add(self):
        pts = sobol(100, [(0.0, 1.0)])
        res_sq = torch.rand(100)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)], n_add=30, n_remove=10)
        assert result.shape[0] == 120  # 100 - 10 + 30

    def test_shrinks_with_more_remove(self):
        pts = sobol(100, [(0.0, 1.0)])
        res_sq = torch.rand(100)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)], n_add=0, n_remove=20)
        assert result.shape[0] == 80

    def test_points_within_bounds(self):
        pts = sobol(100, [(0.0, 1.0), (0.0, 2.0)])
        res_sq = torch.rand(100)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0), (0.0, 2.0)],
                                 n_add=30, n_remove=30, sigma=0.5)
        assert (result[:, 0] >= 0.0).all()
        assert (result[:, 0] <= 1.0).all()
        assert (result[:, 1] >= 0.0).all()
        assert (result[:, 1] <= 2.0).all()

    def test_new_points_near_high_residual(self):
        """New points should cluster near the high-residual seed points."""
        torch.manual_seed(42)
        pts = torch.linspace(0, 1, 100).unsqueeze(1)
        # high residual only at x=0.5 region
        res_sq = torch.zeros(100)
        res_sq[45:55] = 10.0

        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)],
                                 n_add=10, n_remove=10, sigma=0.02)
        new_pts = result[90:]  # the last 10 are the new ones
        # new points should be near 0.5 (all seeds are from the 45-55 range)
        assert (new_pts - 0.5).abs().mean() < 0.1

    def test_detached_output(self):
        pts = sobol(50, [(0.0, 1.0)]).requires_grad_(True)
        res_sq = torch.rand(50)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)], n_add=10, n_remove=10)
        assert not result.requires_grad

    def test_no_add_returns_kept_only(self):
        pts = sobol(50, [(0.0, 1.0)])
        res_sq = torch.rand(50)
        result = adaptive_refine(pts, res_sq, [(0.0, 1.0)], n_add=0, n_remove=10)
        assert result.shape[0] == 40


class TestAdaptiveRefineInTrain:
    def test_train_with_adaptive_refine(self):
        model = MLP([1, 16, 1])

        def pde(net, x):
            H = hessian(net, x)
            return H[0, 0, 0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0.0, 1.0)],
            adam_epochs=50, lbfgs_max_iter=5,
            n_interior=100, log_every=200,
            adaptive_refine_every=20,
            adaptive_refine_ratio=0.2,
            adaptive_refine_sigma=0.1,
        )
        assert len(losses) > 0
        assert all(math.isfinite(l) for l in losses)

    def test_combined_resample_and_refine(self):
        """Both resample_every and adaptive_refine_every can run together."""
        model = MLP([1, 16, 1])

        def pde(net, x):
            H = hessian(net, x)
            return H[0, 0, 0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0.0, 1.0)],
            adam_epochs=50, lbfgs_max_iter=5,
            n_interior=100, log_every=200,
            resample_every=25,
            adaptive_refine_every=10,
        )
        assert len(losses) > 0


@pytest.mark.slow
class TestAdaptiveRefineConvergence:
    def test_refine_improves_accuracy(self):
        """Adaptive refinement should help with accuracy on Poisson 1D."""
        from problems import poisson_1d
        prob = poisson_1d()

        # with adaptive refinement
        model_ref = MLP(prob["layers"])
        model_ref.train()
        train(
            model_ref, prob["pde"], prob["bc"], prob["domain"],
            adam_epochs=2000, lbfgs_max_iter=2000,
            n_interior=1000, log_every=1000, seed=42,
            adaptive_refine_every=200, adaptive_refine_ratio=0.15,
        )

        model_ref.eval()
        x_test = sobol(500, prob["domain"])
        with torch.no_grad():
            u_pred = model_ref(x_test)
            u_exact = prob["exact"](x_test)
        if u_exact.dim() == 1:
            u_exact = u_exact.unsqueeze(1)
        l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
        assert l2_rel < 0.15, f"Adaptive refine Poisson 1D L2={l2_rel:.4f} > 15%"
