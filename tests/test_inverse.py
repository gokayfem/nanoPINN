"""Test inverse problems — InverseParams, observation_loss, parameter recovery."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, InverseParams, observation_loss, hessian, sobol, train


class TestInverseParams:
    def test_creates_parameters(self):
        inv = InverseParams(k=2.0, alpha=0.5)
        assert isinstance(inv.k, torch.nn.Parameter)
        assert isinstance(inv.alpha, torch.nn.Parameter)
        assert inv.k.item() == pytest.approx(2.0)
        assert inv.alpha.item() == pytest.approx(0.5)

    def test_getitem(self):
        inv = InverseParams(k=3.0)
        assert inv["k"] is inv.k

    def test_as_dict(self):
        inv = InverseParams(a=1.0, b=2.0)
        d = inv.as_dict()
        assert d == {"a": pytest.approx(1.0), "b": pytest.approx(2.0)}

    def test_to_device(self):
        inv = InverseParams(k=1.0)
        inv = inv.to("cpu")  # should not error
        assert inv.k.device.type == "cpu"

    def test_in_optimizer(self):
        inv = InverseParams(k=1.0)
        opt = torch.optim.SGD(inv.parameters(), lr=0.1)
        # create a simple loss that depends on k
        loss = (inv.k - 5.0) ** 2
        loss.backward()
        opt.step()
        assert inv.k.item() != pytest.approx(1.0), "k should have been updated"

    def test_gradient_through_pde(self):
        """InverseParams should receive gradients through pde_fn closure."""
        inv = InverseParams(k=1.0)
        model = MLP([1, 16, 1])

        def pde(net, x):
            u = net(x)
            return inv.k * u[0]

        from nanopinn import pde_loss
        pts = torch.rand(10, 1, requires_grad=True)
        loss, _ = pde_loss(model, pde, pts)
        loss.backward()
        assert inv.k.grad is not None, "k should have gradient"


class TestObservationLoss:
    def test_perfect_match_zero(self):
        model = MLP([1, 16, 1])
        x = torch.rand(10, 1)
        with torch.no_grad():
            u = model(x)
        loss = observation_loss(model, x, u)
        assert loss.item() < 1e-10

    def test_shape_mismatch_handled(self):
        """(N,) obs should work with (N, 1) model output."""
        model = MLP([1, 16, 1])
        x = torch.rand(10, 1)
        u_flat = torch.rand(10)
        loss = observation_loss(model, x, u_flat)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        model = MLP([1, 16, 1])
        x = torch.rand(10, 1)
        u = torch.rand(10, 1)
        loss = observation_loss(model, x, u)
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestExtraParamsInTrain:
    def test_extra_params_module(self):
        """train() with extra_params=InverseParams should complete."""
        inv = InverseParams(k=1.0)
        model = MLP([1, 16, 1])

        def pde(net, x):
            u = net(x)
            return inv.k * u[0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0, 1)],
            extra_params=inv, adam_epochs=10, lbfgs_max_iter=5,
            n_interior=50, log_every=100,
        )
        assert len(losses) > 10

    def test_extra_params_list(self):
        """train() with extra_params as list of Parameters."""
        k = torch.nn.Parameter(torch.tensor(1.0))
        model = MLP([1, 16, 1])

        def pde(net, x):
            return k * net(x)[0]

        def bc(model, device):
            return (model(torch.zeros(5, 1, device=device)) ** 2).mean()

        losses = train(
            model, pde, bc, [(0, 1)],
            extra_params=[k], adam_epochs=10, lbfgs_max_iter=5,
            n_interior=50, log_every=100,
        )
        assert len(losses) > 10

    def test_extra_params_none_unchanged(self):
        """Default behavior (no extra_params) should work as before."""
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


@pytest.mark.slow
class TestInverseConvergence:
    def test_helmholtz_k_recovery(self):
        """Recover wavenumber k from noisy observations."""
        from problems import helmholtz_1d_inverse
        prob = helmholtz_1d_inverse(true_k=2.0, noise_std=0.01)
        inv = prob["inv_params"]

        model = MLP(prob["layers"])
        model.train()
        train(
            model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
            extra_params=inv, adam_epochs=3000, lbfgs_max_iter=3000,
            n_interior=2000, log_every=1000, seed=42,
        )

        recovered_k = inv.k.item()
        true_k = prob["true_params"]["k"]
        rel_err = abs(recovered_k - true_k) / true_k
        assert rel_err < 0.25, f"Recovered k={recovered_k:.3f}, true={true_k}, error={rel_err:.1%}"
