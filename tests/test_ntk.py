"""Test NTK-based adaptive loss weighting."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, ntk_weights, train, hessian


class TestNtkWeights:
    def test_returns_correct_length(self):
        model = MLP([1, 16, 1])
        x = torch.rand(5, 1, requires_grad=True)
        l1 = (model(x) ** 2).mean()
        l2 = (model(x) ** 2).sum()
        w = ntk_weights(model, [l1, l2])
        assert len(w) == 2

    def test_weights_sum_to_n(self):
        model = MLP([1, 16, 1])
        x = torch.rand(5, 1, requires_grad=True)
        l1 = (model(x) ** 2).mean()
        l2 = (model(x) ** 2).sum()
        w = ntk_weights(model, [l1, l2])
        assert abs(sum(w) - 2.0) < 1e-6

    def test_three_losses_sum_to_three(self):
        model = MLP([1, 16, 1])
        x = torch.rand(5, 1, requires_grad=True)
        losses = [
            (model(x) ** 2).mean(),
            (model(x) ** 2).sum(),
            (model(x[:1]) ** 2).mean(),
        ]
        w = ntk_weights(model, losses)
        assert len(w) == 3
        assert abs(sum(w) - 3.0) < 1e-5

    def test_equal_losses_equal_weights(self):
        """Identical losses should give weights close to 1.0 each."""
        model = MLP([1, 16, 1])
        x = torch.rand(5, 1, requires_grad=True)
        l1 = (model(x) ** 2).mean()
        l2 = (model(x) ** 2).mean()  # identical
        w = ntk_weights(model, [l1, l2])
        assert abs(w[0] - 1.0) < 0.1
        assert abs(w[1] - 1.0) < 0.1

    def test_single_loss(self):
        model = MLP([1, 16, 1])
        x = torch.rand(5, 1, requires_grad=True)
        loss = (model(x) ** 2).mean()
        w = ntk_weights(model, [loss])
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-6

    def test_zero_trace_handled(self):
        """All-zero gradients should not cause division by zero."""
        model = MLP([1, 16, 1])
        # loss = constant (no grad wrt model params)
        loss = torch.tensor(0.0, requires_grad=True)
        w = ntk_weights(model, [loss, loss])
        assert len(w) == 2
        assert all(torch.isfinite(torch.tensor(wi)) for wi in w)

    def test_imbalanced_traces(self):
        """Loss with much larger gradient norm should get smaller weight."""
        model = MLP([1, 32, 1])
        x = torch.rand(10, 1, requires_grad=True)
        l_small = (model(x) ** 2).mean()
        l_big = (model(x) ** 2).mean() * 1000  # artificially large scale

        w = ntk_weights(model, [l_small, l_big])
        # l_big has larger gradient norm -> should get smaller weight
        assert w[1] < w[0], f"Big loss should get smaller weight: {w}"


class TestNtkTrainIntegration:
    def test_ntk_train_runs(self):
        """train() with ntk_weighting=True should complete."""
        model = MLP([1, 16, 1])

        def pde(net, x):
            return net(x)[0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0, 1)],
            adam_epochs=20, lbfgs_max_iter=5,
            n_interior=50, log_every=100,
            ntk_weighting=True, ntk_every=5,
        )
        assert len(losses) > 20
        assert all(isinstance(l, float) for l in losses)

    def test_ntk_disabled_default(self):
        """ntk_weighting=False is default, no overhead."""
        model = MLP([1, 16, 1])

        def pde(net, x):
            return net(x)[0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0, 1)],
            adam_epochs=10, lbfgs_max_iter=5,
            n_interior=50, log_every=100,
        )
        assert len(losses) > 10
