"""Convergence tests: train PINNs and compare against exact analytical solutions.

These are the "big tests" — they actually train networks and verify that
the PINN solution matches the known exact solution within tolerance.

Marked with @pytest.mark.slow for tests that take > 30 seconds.
Run all: pytest tests/test_convergence.py -v
Run fast only: pytest tests/test_convergence.py -v -m "not slow"
"""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, sobol, train
from problems import (
    poisson_1d,
    poisson_2d,
    heat_1d,
    harmonic_oscillator,
    helmholtz_1d,
    advection_1d,
)


def _l2_relative_error(model, prob, n_test=1000, device="cpu"):
    """Compute L2 relative error between PINN and exact solution."""
    model.eval()
    x_test = sobol(n_test, prob["domain"], device=device)
    with torch.no_grad():
        u_pred = model(x_test)
        u_exact = prob["exact"](x_test).to(device)
    if u_exact.dim() == 1:
        u_exact = u_exact.unsqueeze(1)
    return (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()


def _train_problem(prob, adam_epochs=2000, lbfgs_max_iter=2000, n_interior=2000, activation="tanh"):
    """Train a PINN on a problem and return model + losses."""
    device = "cpu"
    layers = prob["layers"]
    model = MLP(layers, activation=activation).to(device)
    model.train()
    losses = train(
        model,
        pde_fn=prob["pde"],
        bc_fn=prob["bc"],
        domain=prob["domain"],
        n_interior=n_interior,
        adam_epochs=adam_epochs,
        lbfgs_max_iter=lbfgs_max_iter,
        log_every=500,
        device=device,
        seed=42,
        resample_every=0,
    )
    return model, losses


# ─── Fast convergence tests (< 30s each) ────────────────────────────────────


class TestPoissonConvergence:
    """Poisson 1D should converge quickly — it's the simplest PDE."""

    def test_loss_decreases(self):
        prob = poisson_1d()
        model, losses = _train_problem(prob, adam_epochs=500, lbfgs_max_iter=500, n_interior=500)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_convergence(self):
        prob = poisson_1d()
        model, losses = _train_problem(prob, adam_epochs=1000, lbfgs_max_iter=1000, n_interior=1000)
        err = _l2_relative_error(model, prob)
        assert err < 0.05, f"Poisson 1D L2 error {err:.4f} > 5%"


class TestHarmonicOscillatorConvergence:
    def test_loss_decreases(self):
        prob = harmonic_oscillator()
        model, losses = _train_problem(prob, adam_epochs=500, lbfgs_max_iter=500, n_interior=500)
        assert losses[-1] < losses[0]

    def test_convergence(self):
        prob = harmonic_oscillator()
        model, losses = _train_problem(prob, adam_epochs=2000, lbfgs_max_iter=3000, n_interior=1500)
        err = _l2_relative_error(model, prob)
        assert err < 0.15, f"Harmonic oscillator L2 error {err:.4f} > 15%"


class TestHelmholtzConvergence:
    def test_convergence(self):
        prob = helmholtz_1d()
        model, losses = _train_problem(prob, adam_epochs=1000, lbfgs_max_iter=1000, n_interior=1000)
        err = _l2_relative_error(model, prob)
        assert err < 0.05, f"Helmholtz 1D L2 error {err:.4f} > 5%"


# ─── Slow convergence tests (> 30s each) ────────────────────────────────────


@pytest.mark.slow
class TestPoisson2DConvergence:
    def test_convergence(self):
        prob = poisson_2d()
        model, losses = _train_problem(prob, adam_epochs=3000, lbfgs_max_iter=3000, n_interior=3000)
        err = _l2_relative_error(model, prob)
        assert err < 0.10, f"Poisson 2D L2 error {err:.4f} > 10%"


@pytest.mark.slow
class TestHeat1DConvergence:
    def test_convergence(self):
        prob = heat_1d()
        model, losses = _train_problem(prob, adam_epochs=3000, lbfgs_max_iter=3000, n_interior=3000)
        err = _l2_relative_error(model, prob)
        assert err < 0.10, f"Heat 1D L2 error {err:.4f} > 10%"


@pytest.mark.slow
class TestAdvection1DConvergence:
    def test_convergence(self):
        prob = advection_1d()
        model, losses = _train_problem(prob, adam_epochs=3000, lbfgs_max_iter=3000, n_interior=3000)
        err = _l2_relative_error(model, prob)
        assert err < 0.15, f"Advection 1D L2 error {err:.4f} > 15%"


# ─── Architecture comparison tests ──────────────────────────────────────────


class TestSIRENActivation:
    """SIREN should also converge on Poisson 1D."""

    def test_siren_converges(self):
        prob = poisson_1d()
        layers = [1, 64, 64, 64, 1]
        prob_siren = {**prob, "layers": layers}
        model, losses = _train_problem(prob_siren, adam_epochs=2000, lbfgs_max_iter=3000,
                                       n_interior=1500, activation="siren")
        err = _l2_relative_error(model, prob)
        assert err < 0.15, f"SIREN Poisson 1D L2 error {err:.4f} > 15%"


# ─── Sanity checks ──────────────────────────────────────────────────────────


class TestTrainingBasics:
    def test_returns_losses(self):
        prob = poisson_1d()
        model, losses = _train_problem(prob, adam_epochs=10, lbfgs_max_iter=10, n_interior=100)
        assert len(losses) > 10  # at least Adam epochs
        assert all(isinstance(l, float) for l in losses)

    def test_model_is_modified(self):
        prob = poisson_1d()
        model = MLP(prob["layers"])
        w_before = model.net[0].weight.data.clone()
        model.train()
        train(model, prob["pde"], prob["bc"], prob["domain"],
              adam_epochs=10, lbfgs_max_iter=10, n_interior=100, log_every=100)
        w_after = model.net[0].weight.data
        assert not torch.equal(w_before, w_after), "Weights should change during training"
