"""Test MLP architectures — shapes, activations, gradient flow, Fourier features, normalization."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, FourierFeatures, jacobian, hessian
from torch.func import functional_call, vmap


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
        assert y.abs().max() < 100.0


class TestFourierFeatures:
    def test_output_shape(self):
        ff = FourierFeatures(2, 32)
        x = torch.rand(10, 2)
        y = ff(x)
        assert y.shape == (10, 64)  # 2 * mapping_size

    def test_mlp_with_fourier_shape(self):
        model = MLP([2, 64, 1], fourier_features=32)
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_fourier_deterministic_with_seed(self):
        torch.manual_seed(42)
        ff1 = FourierFeatures(2, 16)
        torch.manual_seed(42)
        ff2 = FourierFeatures(2, 16)
        assert torch.equal(ff1.B, ff2.B)

    def test_fourier_sigma_scales_frequencies(self):
        torch.manual_seed(0)
        ff_low = FourierFeatures(2, 32, sigma=1.0)
        torch.manual_seed(0)
        ff_high = FourierFeatures(2, 32, sigma=10.0)
        assert ff_high.B.abs().mean() > ff_low.B.abs().mean()

    def test_derivatives_with_fourier(self):
        """jacobian/hessian must work through Fourier encoding."""
        model = MLP([2, 32, 1], fourier_features=16)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        x = torch.tensor([0.3, 0.7])
        J = jacobian(fwd, x)
        assert J.shape == (1, 2)
        assert torch.isfinite(J).all()

        H = hessian(fwd, x)
        assert H.shape == (1, 2, 2)
        assert torch.isfinite(H).all()

    def test_gradient_flow_with_fourier(self):
        model = MLP([2, 64, 1], fourier_features=32)
        x = torch.rand(10, 2, requires_grad=True)
        y = model(x)
        y.sum().backward()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable_params)
        assert has_grad


class TestNormalization:
    @pytest.mark.parametrize("norm_type", ["weight", "spectral"])
    def test_norm_runs(self, norm_type):
        model = MLP([2, 32, 32, 1], norm=norm_type)
        x = torch.rand(5, 2)
        y = model(x)
        assert y.shape == (5, 1)
        assert torch.isfinite(y).all()

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="Unknown norm"):
            MLP([2, 32, 1], norm="invalid")

    def test_gradient_flow_with_norm(self):
        model = MLP([2, 64, 64, 1], norm="spectral")
        x = torch.rand(10, 2, requires_grad=True)
        y = model(x)
        y.sum().backward()
        has_grad = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_derivatives_with_spectral_norm(self):
        """jacobian must work through spectral-normalized layers."""
        model = MLP([2, 32, 1], norm="spectral")
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        x = torch.tensor([0.3, 0.7])
        J = jacobian(fwd, x)
        assert J.shape == (1, 2)
        assert torch.isfinite(J).all()

    def test_derivatives_with_weight_norm(self):
        """jacobian must work through weight-normalized layers."""
        model = MLP([2, 32, 1], norm="weight")
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        x = torch.tensor([0.3, 0.7])
        J = jacobian(fwd, x)
        assert J.shape == (1, 2)
        assert torch.isfinite(J).all()

    def test_siren_with_norm(self):
        """SIREN + normalization should not crash."""
        model = MLP([2, 32, 1], activation="siren", norm="spectral")
        x = torch.rand(5, 2)
        y = model(x)
        assert y.shape == (5, 1)
        assert torch.isfinite(y).all()
