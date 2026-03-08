"""Test ResNet architecture with skip connections."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import ResNet, hessian, jacobian, pde_loss, sobol, train


class TestResNetShape:
    def test_1d_output(self):
        model = ResNet([1, 32, 32, 1])
        x = torch.rand(10, 1)
        y = model(x)
        assert y.shape == (10, 1)

    def test_2d_output(self):
        model = ResNet([2, 64, 64, 64, 1])
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_multi_output(self):
        model = ResNet([2, 64, 64, 3])
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 3)


class TestResNetGradient:
    def test_gradient_flow(self):
        model = ResNet([1, 32, 32, 1])
        x = torch.rand(10, 1, requires_grad=True)
        y = model(x)
        y.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_skip_connection_effect(self):
        """ResNet with 1 hidden layer should differ from MLP due to skip connection."""
        from nanopinn import MLP
        torch.manual_seed(42)
        resnet = ResNet([1, 32, 32, 1])
        torch.manual_seed(42)
        mlp = MLP([1, 32, 32, 1])
        x = torch.rand(5, 1)
        y_res = resnet(x)
        y_mlp = mlp(x)
        assert y_res.shape == y_mlp.shape
        assert not torch.allclose(y_res, y_mlp), "ResNet and MLP should produce different outputs"


class TestResNetFeatures:
    def test_tune_beta(self):
        model = ResNet([1, 32, 32, 1], tune_beta=True)
        # beta0 and betas should be nn.Parameter when tune_beta=True
        assert isinstance(model.beta0, torch.nn.Parameter)
        assert isinstance(model.betas, torch.nn.Parameter)
        assert model.beta0.requires_grad
        assert model.betas.requires_grad

    def test_no_tune_beta(self):
        model = ResNet([1, 32, 32, 1], tune_beta=False)
        # betas should be buffers (not parameters)
        param_names = [n for n, _ in model.named_parameters()]
        assert "beta0" not in param_names
        assert "betas" not in param_names

    def test_with_fourier(self):
        model = ResNet([2, 32, 32, 1], fourier_features=16)
        x = torch.rand(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_with_norm(self):
        model = ResNet([1, 32, 32, 1], norm="spectral")
        x = torch.rand(10, 1)
        y = model(x)
        assert y.shape == (10, 1)

    def test_activations(self):
        for act in ["tanh", "gelu", "swish", "sigmoid"]:
            model = ResNet([1, 32, 32, 1], activation=act)
            y = model(torch.rand(5, 1))
            assert y.shape == (5, 1)

    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            ResNet([1, 32, 32, 1], activation="invalid")

    def test_siren_not_supported(self):
        with pytest.raises(ValueError, match="SIREN"):
            ResNet([1, 32, 32, 1], activation="siren")

    def test_minimum_layers(self):
        with pytest.raises(ValueError, match="at least 3"):
            ResNet([1, 1])

    def test_unequal_hidden_dims(self):
        with pytest.raises(ValueError, match="equal"):
            ResNet([1, 32, 64, 1])

    def test_tune_beta_gradient_flow(self):
        model = ResNet([1, 32, 32, 1], tune_beta=True)
        x = torch.rand(10, 1)
        y = model(x)
        y.sum().backward()
        assert model.beta0.grad is not None and model.beta0.grad.abs().sum() > 0
        assert model.betas.grad is not None and model.betas.grad.abs().sum() > 0


class TestResNetVmap:
    def test_pde_loss_compatible(self):
        """ResNet must work with vmap-based pde_loss."""
        model = ResNet([1, 32, 32, 1])

        def pde_fn(net, x):
            H = hessian(net, x)
            return H[0, 0, 0]

        pts = torch.rand(20, 1, requires_grad=True)
        loss, res_sq = pde_loss(model, pde_fn, pts)
        assert loss.dim() == 0
        assert res_sq.shape == (20,)
        loss.backward()

    def test_train_integration(self):
        model = ResNet([1, 16, 16, 1])

        def pde(net, x):
            H = hessian(net, x)
            return H[0, 0, 0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(model, pde, bc, [(0.0, 1.0)],
                       adam_epochs=20, lbfgs_max_iter=5,
                       n_interior=50, log_every=100)
        assert len(losses) > 0


@pytest.mark.slow
class TestResNetConvergence:
    def test_poisson_1d(self):
        """ResNet should converge on Poisson 1D."""
        from problems import poisson_1d
        prob = poisson_1d()

        model = ResNet(prob["layers"])
        model.train()
        train(model, prob["pde"], prob["bc"], prob["domain"],
              adam_epochs=2000, lbfgs_max_iter=2000,
              n_interior=1000, log_every=1000, seed=42)

        model.eval()
        x_test = sobol(500, prob["domain"])
        with torch.no_grad():
            u_pred = model(x_test)
            u_exact = prob["exact"](x_test)
        if u_exact.dim() == 1:
            u_exact = u_exact.unsqueeze(1)
        l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
        assert l2_rel < 0.15, f"ResNet Poisson 1D L2={l2_rel:.4f} > 15%"
