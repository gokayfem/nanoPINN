"""Test derivative operators against torch.autograd reference."""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, jacobian, hessian, laplacian


def _autograd_jacobian(model, x_batch):
    """Reference Jacobian using torch.autograd.grad (batched)."""
    x_batch = x_batch.requires_grad_(True)
    y = model(x_batch)
    m = y.shape[1]
    d = x_batch.shape[1]
    J = torch.zeros(x_batch.shape[0], m, d)
    for i in range(m):
        grads = torch.autograd.grad(y[:, i].sum(), x_batch, create_graph=True)[0]
        J[:, i, :] = grads
    return J


def _autograd_hessian(model, x_single):
    """Reference Hessian using torch.autograd.grad for a single point."""
    x = x_single.unsqueeze(0).requires_grad_(True)
    y = model(x)
    m = y.shape[1]
    d = x.shape[1]
    H = torch.zeros(m, d, d)
    for i in range(m):
        grads = torch.autograd.grad(y[0, i], x, create_graph=True)[0]
        for j in range(d):
            g2 = torch.autograd.grad(grads[0, j], x, retain_graph=True)[0]
            H[i, j, :] = g2[0]
    return H


class TestJacobian:
    def test_1d_to_1d(self):
        """Jacobian of 1D->1D net matches autograd."""
        model = MLP([1, 32, 1])
        x = torch.rand(5, 1)
        J_ref = _autograd_jacobian(model, x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call, vmap

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        # For 1D input, vmap over (5, 1) keeping the inner dim
        J_func = vmap(lambda xi: jacobian(fwd, xi))(x)
        torch.testing.assert_close(J_func, J_ref, atol=1e-5, rtol=1e-4)

    def test_2d_to_1d(self):
        """Jacobian of 2D->1D net matches autograd."""
        model = MLP([2, 32, 1])
        x = torch.rand(5, 2)
        J_ref = _autograd_jacobian(model, x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call, vmap

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        J_func = vmap(lambda xi: jacobian(fwd, xi))(x)
        torch.testing.assert_close(J_func, J_ref, atol=1e-5, rtol=1e-4)

    def test_2d_to_3d(self):
        """Jacobian of 2D->3D net matches autograd."""
        model = MLP([2, 32, 3])
        x = torch.rand(5, 2)
        J_ref = _autograd_jacobian(model, x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call, vmap

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        J_func = vmap(lambda xi: jacobian(fwd, xi))(x)
        torch.testing.assert_close(J_func, J_ref, atol=1e-5, rtol=1e-4)


class TestHessian:
    def test_1d_to_1d(self):
        """Hessian of 1D->1D net matches autograd."""
        model = MLP([1, 32, 1])
        x = torch.tensor([0.5])
        H_ref = _autograd_hessian(model, x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        H_func = hessian(fwd, x)
        torch.testing.assert_close(H_func, H_ref, atol=1e-4, rtol=1e-3)

    def test_2d_to_1d(self):
        """Hessian of 2D->1D net matches autograd."""
        model = MLP([2, 32, 1])
        x = torch.tensor([0.3, 0.7])
        H_ref = _autograd_hessian(model, x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        H_func = hessian(fwd, x)
        torch.testing.assert_close(H_func, H_ref, atol=1e-4, rtol=1e-3)

    def test_symmetry(self):
        """Hessian should be symmetric: H[i,j,k] == H[i,k,j]."""
        model = MLP([2, 32, 1])
        x = torch.tensor([0.5, 0.5])

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        from torch.func import functional_call

        def fwd(xi):
            return functional_call(model, (params, buffers), xi.unsqueeze(0)).squeeze(0)

        H = hessian(fwd, x)
        torch.testing.assert_close(H[0, 0, 1], H[0, 1, 0], atol=1e-5, rtol=1e-4)


class TestLaplacian:
    def test_known_function(self):
        """Laplacian of sin(pi*x)*sin(pi*y) should be -2*pi^2*sin(pi*x)*sin(pi*y)."""
        def f(x):
            return (torch.sin(math.pi * x[0]) * torch.sin(math.pi * x[1])).unsqueeze(0)

        x = torch.tensor([0.3, 0.7])
        lap = laplacian(f, x, out_idx=0)
        expected = -2.0 * math.pi ** 2 * math.sin(math.pi * 0.3) * math.sin(math.pi * 0.7)
        torch.testing.assert_close(lap, torch.tensor(expected), atol=1e-5, rtol=1e-4)
