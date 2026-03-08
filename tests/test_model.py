"""Test MLP architectures — shapes, activations, gradient flow."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP


class TestMLPShapes:
    @pytest.mark.parametrize("layers", [
        [1, 32, 1],
        [2, 64, 64, 1],
        [3, 128, 128, 128, 7],
    ])
    def test_output_shape(self, layers):
        model = MLP(layers)
        x = torch.rand(10, layers[0])
        y = model(x)
        assert y.shape == (10, layers[-1])

    def test_single_point(self):
        model = MLP([2, 32, 1])
        x = torch.rand(1, 2)
        y = model(x)
        assert y.shape == (1, 1)


class TestActivations:
    @pytest.mark.parametrize("act", ["tanh", "gelu", "swish", "siren"])
    def test_activation_runs(self, act):
        model = MLP([2, 32, 32, 1], activation=act)
        x = torch.rand(5, 2)
        y = model(x)
        assert y.shape == (5, 1)
        assert torch.isfinite(y).all()

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP([2, 32, 1], activation="invalid")


class TestGradientFlow:
    @pytest.mark.parametrize("act", ["tanh", "siren"])
    def test_gradients_flow(self, act):
        model = MLP([2, 64, 64, 1], activation=act)
        x = torch.rand(10, 2, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        has_grad = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_siren_bounded_output(self):
        """SIREN outputs should be bounded due to sin activations."""
        model = MLP([2, 64, 64, 1], activation="siren")
        x = torch.rand(100, 2) * 10
        with torch.no_grad():
            y = model(x)
        # SIREN outputs are bounded by the linear output layer, but intermediate
        # activations are bounded by [-1, 1]. Output should be reasonable.
        assert y.abs().max() < 100.0
