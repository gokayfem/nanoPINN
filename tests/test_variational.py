"""Test variational / energy-based formulations."""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import (
    MLP, energy_loss, jacobian, sobol, train,
)


def _poisson_1d_energy_fn():
    """Energy density for -u'' = pi^2 sin(pi x): E = 0.5*(du/dx)^2 - f*u."""
    def energy(net, x):
        J = jacobian(net, x)
        u = net(x)
        f = (math.pi ** 2) * torch.sin(torch.tensor(math.pi) * x[0])
        return 0.5 * J[0, 0] ** 2 - f * u[0]
    return energy


class TestEnergyLoss:
    def test_output_shape(self):
        model = MLP([1, 16, 1])
        energy_fn = _poisson_1d_energy_fn()
        pts = sobol(50, [(0.0, 1.0)]).requires_grad_(True)

        loss, per_point = energy_loss(model, energy_fn, pts, [(0.0, 1.0)])
        assert loss.dim() == 0
        assert per_point.shape == (50,)

    def test_backward(self):
        model = MLP([1, 16, 1])
        energy_fn = _poisson_1d_energy_fn()
        pts = sobol(30, [(0.0, 1.0)]).requires_grad_(True)

        loss, _ = energy_loss(model, energy_fn, pts, [(0.0, 1.0)])
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_volume_scaling(self):
        """Loss should scale with domain volume."""
        model = MLP([1, 16, 1])
        energy_fn = _poisson_1d_energy_fn()
        pts = sobol(50, [(0.0, 1.0)]).requires_grad_(True)

        loss1, _ = energy_loss(model, energy_fn, pts, [(0.0, 1.0)])
        loss2, _ = energy_loss(model, energy_fn, pts, [(0.0, 2.0)])
        # loss2 should be ~2x loss1 (domain is 2x wider)
        assert abs(loss2.item() / loss1.item() - 2.0) < 0.01

    def test_per_point_detached(self):
        model = MLP([1, 16, 1])
        energy_fn = _poisson_1d_energy_fn()
        pts = sobol(20, [(0.0, 1.0)]).requires_grad_(True)

        _, per_point = energy_loss(model, energy_fn, pts, [(0.0, 1.0)])
        assert not per_point.requires_grad

    def test_2d_domain(self):
        """Energy loss should work for 2D domains."""
        model = MLP([2, 16, 1])

        def energy_2d(net, x):
            J = jacobian(net, x)
            return 0.5 * (J[0, 0] ** 2 + J[0, 1] ** 2)

        pts = sobol(30, [(0.0, 1.0), (0.0, 1.0)]).requires_grad_(True)
        loss, per_point = energy_loss(model, energy_2d, pts, [(0.0, 1.0), (0.0, 1.0)])
        assert loss.dim() == 0
        assert per_point.shape == (30,)
        loss.backward()


class TestEnergyInTrain:
    def test_train_with_energy_fn(self):
        """train() should accept energy_fn kwarg."""
        model = MLP([1, 16, 1])
        energy_fn = _poisson_1d_energy_fn()

        def bc(model, device):
            x0 = torch.zeros(10, 1, device=device)
            x1 = torch.ones(10, 1, device=device)
            return (model(x0) ** 2).mean() + (model(x1) ** 2).mean()

        losses = train(
            model, pde_fn=lambda net, x: net(x)[0], bc_fn=bc, domain=[(0.0, 1.0)],
            adam_epochs=20, lbfgs_max_iter=5,
            n_interior=50, log_every=100,
            energy_fn=energy_fn,
        )
        assert len(losses) > 0
        assert all(math.isfinite(l) for l in losses)

    def test_loss_decreases(self):
        """Energy loss should decrease during training."""
        model = MLP([1, 32, 32, 1])
        energy_fn = _poisson_1d_energy_fn()

        def bc(model, device):
            x0 = torch.zeros(50, 1, device=device)
            x1 = torch.ones(50, 1, device=device)
            return (model(x0) ** 2).mean() + (model(x1) ** 2).mean()

        losses = train(
            model, pde_fn=lambda net, x: net(x)[0], bc_fn=bc, domain=[(0.0, 1.0)],
            adam_epochs=100, lbfgs_max_iter=0,
            n_interior=200, log_every=200,
            energy_fn=energy_fn,
        )
        # loss should generally decrease (compare first 10 avg vs last 10 avg)
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early


@pytest.mark.slow
class TestVariationalConvergence:
    def test_poisson_1d_variational(self):
        """Variational Poisson 1D should converge."""
        from problems import poisson_1d_variational
        prob = poisson_1d_variational()

        model = MLP(prob["layers"])
        model.train()
        losses = train(
            model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
            adam_epochs=3000, lbfgs_max_iter=3000,
            n_interior=1000, log_every=1000, seed=42,
            energy_fn=prob["energy"],
        )

        model.eval()
        x_test = sobol(500, prob["domain"])
        with torch.no_grad():
            u_pred = model(x_test)
            u_exact = prob["exact"](x_test)
        if u_exact.dim() == 1:
            u_exact = u_exact.unsqueeze(1)
        l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
        assert l2_rel < 0.15, f"Variational Poisson 1D L2 error {l2_rel:.4f} > 15%"
