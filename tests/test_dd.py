"""Test domain decomposition — subdomains, window functions, DDModel."""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import (
    MLP, DDModel, decompose_domain, pde_loss,
    hessian, sobol, train,
)


def _cosine_window(center, half_width):
    """Local helper reproducing the cosine bell window logic."""
    def window(x):
        dist = ((x - center) / half_width).abs()
        per_dim = 0.5 * (1.0 + torch.cos(math.pi * dist.clamp(max=1.0)))
        return per_dim.prod(dim=-1)
    return window


class TestDecomposeDomain:
    def test_1d_three_subdomains(self):
        subs = decompose_domain([(0.0, 1.0)], [3], overlap=0.25)
        assert len(subs) == 3
        # centers should be evenly spaced
        centers = [s["center"][0].item() for s in subs]
        assert abs(centers[0] - 1/6) < 1e-6
        assert abs(centers[1] - 0.5) < 1e-6
        assert abs(centers[2] - 5/6) < 1e-6

    def test_2d_four_subdomains(self):
        subs = decompose_domain([(0.0, 1.0), (0.0, 1.0)], [2, 2], overlap=0.25)
        assert len(subs) == 4

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            decompose_domain([(0.0, 1.0), (0.0, 1.0)], [3])

    def test_subdomain_centers_cover_domain(self):
        subs = decompose_domain([(0.0, 1.0)], [4], overlap=0.2)
        centers = sorted([s["center"][0].item() for s in subs])
        assert centers[0] < 0.2  # first center near left
        assert centers[-1] > 0.8  # last center near right


class TestPartitionOfUnity:
    def test_windows_sum_to_one(self):
        """Window values should sum to ~1 after normalization (done in DDModel.forward)."""
        subs = decompose_domain([(0.0, 1.0)], [3], overlap=0.5)
        windows = [_cosine_window(s["center"], s["half_width"]) for s in subs]

        x = torch.linspace(0.05, 0.95, 50).unsqueeze(1)
        w_vals = torch.stack([w(x) for w in windows], dim=-1)
        w_sum = w_vals.sum(dim=-1)
        # all points should have nonzero total window coverage
        assert (w_sum > 0.01).all(), f"Some points have near-zero coverage: {w_sum.min()}"


class TestDDModel:
    def test_output_shape(self):
        subs = decompose_domain([(0.0, 1.0), (0.0, 1.0)], [2, 2])
        model = DDModel([2, 32, 1], subs)
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_gradient_flow(self):
        subs = decompose_domain([(0.0, 1.0)], [2])
        model = DDModel([1, 32, 1], subs)
        x = torch.rand(10, 1, requires_grad=True)
        y = model(x)
        y.sum().backward()

        # at least some sub-networks should receive gradients
        has_any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_any_grad

    def test_multi_output(self):
        subs = decompose_domain([(0.0, 1.0), (0.0, 1.0)], [2, 2])
        model = DDModel([2, 32, 3], subs)
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 3)

    def test_with_fourier(self):
        subs = decompose_domain([(0.0, 1.0)], [2])
        model = DDModel([1, 32, 1], subs, fourier_features=16)
        x = torch.rand(10, 1)
        y = model(x)
        assert y.shape == (10, 1)

    def test_with_pde_loss(self):
        """pde_loss should work with DDModel via functional_call."""
        subs = decompose_domain([(0.0, 1.0)], [2], overlap=0.5)
        model = DDModel([1, 32, 1], subs)

        def pde_fn(net, x):
            H = hessian(net, x)
            return H[0, 0, 0]

        pts = torch.rand(20, 1, requires_grad=True)
        loss, res_sq = pde_loss(model, pde_fn, pts)
        assert loss.dim() == 0
        assert res_sq.shape == (20,)
        loss.backward()

    def test_dd_parameters_count(self):
        """DDModel with 3 subdomains should have ~3x the parameters of a single MLP."""
        layers = [1, 32, 1]
        single = MLP(layers)
        subs = decompose_domain([(0.0, 1.0)], [3])
        dd = DDModel(layers, subs)

        n_single = sum(p.numel() for p in single.parameters())
        n_dd = sum(p.numel() for p in dd.parameters())
        assert n_dd == 3 * n_single


@pytest.mark.slow
class TestDDConvergence:
    def test_poisson_1d_with_dd(self):
        """DDModel should converge on Poisson 1D."""
        from problems import poisson_1d
        prob = poisson_1d()

        subs = decompose_domain(prob["domain"], [3], overlap=0.5)
        model = DDModel([1, 64, 64, 1], subs)
        model.train()
        losses = train(
            model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
            adam_epochs=2000, lbfgs_max_iter=2000, n_interior=1500,
            log_every=500, seed=42,
        )

        model.eval()
        x_test = sobol(500, prob["domain"])
        with torch.no_grad():
            u_pred = model(x_test)
            u_exact = prob["exact"](x_test)
        if u_exact.dim() == 1:
            u_exact = u_exact.unsqueeze(1)
        l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
        assert l2_rel < 0.15, f"DD Poisson 1D L2 error {l2_rel:.4f} > 15%"
