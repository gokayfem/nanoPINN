"""Test causal training for time-dependent PDEs."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, causal_weights, causal_pde_loss, pde_loss, sobol, train
from nanopinn import jacobian


class TestCausalWeights:
    def test_shape(self):
        points = torch.rand(100, 2)  # (x, t)
        res_sq = torch.rand(100)
        w = causal_weights(points, res_sq, time_dim=1)
        assert w.shape == (100,)

    def test_weights_in_range(self):
        points = torch.rand(200, 2)
        res_sq = torch.rand(200)
        w = causal_weights(points, res_sq, time_dim=1)
        assert (w >= 0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_earlier_times_higher_weight(self):
        """When residuals grow with time, earlier points should get higher weight."""
        n = 200
        points = torch.zeros(n, 2)
        points[:, 0] = torch.rand(n)  # x
        points[:, 1] = torch.linspace(0, 1, n)  # t

        # residuals that increase with time
        res_sq = points[:, 1] ** 2

        w = causal_weights(points, res_sq, time_dim=1, epsilon=5.0)

        # early points (small t) should have higher weights than late points (large t)
        early_mask = points[:, 1] < 0.2
        late_mask = points[:, 1] > 0.8
        assert w[early_mask].mean() > w[late_mask].mean()

    def test_epsilon_zero_uniform_weights(self):
        """epsilon=0 should give all weights = 1."""
        points = torch.rand(100, 2)
        res_sq = torch.rand(100) * 10
        w = causal_weights(points, res_sq, time_dim=1, epsilon=0.0)
        torch.testing.assert_close(w, torch.ones(100), atol=1e-5, rtol=1e-5)

    def test_uniform_residual_near_uniform_weights(self):
        """Uniform residuals should produce nearly uniform weights."""
        points = torch.rand(100, 2)
        res_sq = torch.ones(100) * 0.01
        w = causal_weights(points, res_sq, time_dim=1, epsilon=1.0)
        # weights should be reasonably close to each other
        assert (w.max() - w.min()) < 0.5

    def test_constant_time_returns_ones(self):
        """If all points have the same time, weights should be 1."""
        points = torch.rand(50, 2)
        points[:, 1] = 0.5  # all same time
        res_sq = torch.rand(50)
        w = causal_weights(points, res_sq, time_dim=1)
        torch.testing.assert_close(w, torch.ones(50), atol=1e-5, rtol=1e-5)


class TestCausalPdeLoss:
    def test_returns_correct_shapes(self):
        model = MLP([2, 32, 1])

        def pde_fn(net, x):
            J = jacobian(net, x)
            return J[0, 0] + J[0, 1]

        points = torch.rand(50, 2, requires_grad=True)
        loss, res_sq = causal_pde_loss(model, pde_fn, points, time_dim=1)

        assert loss.dim() == 0  # scalar
        assert res_sq.shape == (50,)

    def test_backward_passes(self):
        model = MLP([2, 32, 1])

        def pde_fn(net, x):
            J = jacobian(net, x)
            return J[0, 0] + J[0, 1]

        points = torch.rand(50, 2, requires_grad=True)
        loss, _ = causal_pde_loss(model, pde_fn, points, time_dim=1)
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestCausalTrainIntegration:
    def test_causal_train_runs(self):
        """train() with causal=True should complete without error."""
        model = MLP([2, 32, 32, 1])

        def pde_fn(net, x):
            J = jacobian(net, x)
            return J[0, 0] + J[0, 1]

        def bc_fn(model, device):
            x = torch.zeros(10, 2, device=device)
            return (model(x) ** 2).mean()

        losses = train(
            model, pde_fn=pde_fn, bc_fn=bc_fn,
            domain=[(0, 1), (0, 1)],
            adam_epochs=20, lbfgs_max_iter=10,
            n_interior=100, log_every=100,
            causal=True, causal_time_dim=1, causal_epsilon=1.0,
        )
        assert len(losses) > 20
        assert all(isinstance(l, float) for l in losses)
