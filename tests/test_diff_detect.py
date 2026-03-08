"""Test automatic differentiation order detection."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import detect_diff_order, hessian, jacobian, laplacian


class TestDetectDiffOrder:
    def test_hessian_only(self):
        def pde(net, x):
            H = hessian(net, x)
            return -H[0, 0, 0]

        info = detect_diff_order(pde)
        assert info["max_order"] == 2
        assert info["uses_hessian"] is True
        assert info["uses_jacobian"] is False

    def test_jacobian_only(self):
        def pde(net, x):
            J = jacobian(net, x)
            return J[0, 0] + J[0, 1]

        info = detect_diff_order(pde)
        assert info["max_order"] == 1
        assert info["uses_jacobian"] is True
        assert info["uses_hessian"] is False

    def test_both_jacobian_and_hessian(self):
        def pde(net, x):
            J = jacobian(net, x)
            H = hessian(net, x)
            return J[0, 1] - H[0, 0, 0]

        info = detect_diff_order(pde)
        assert info["max_order"] == 2
        assert info["uses_jacobian"] is True
        assert info["uses_hessian"] is True

    def test_laplacian(self):
        def pde(net, x):
            return -laplacian(net, x)

        info = detect_diff_order(pde)
        assert info["max_order"] == 2
        assert info["uses_laplacian"] is True

    def test_no_derivatives(self):
        def pde(net, x):
            return net(x)[0]

        info = detect_diff_order(pde)
        assert info["max_order"] == 0
        assert info["uses_jacobian"] is False
        assert info["uses_hessian"] is False

    def test_builtin_fallback(self):
        """Uninspectable callables should fall back to max order."""
        info = detect_diff_order(len)
        assert info["max_order"] == 2
        assert info["uses_jacobian"] is True

    def test_poisson_1d(self):
        from problems import poisson_1d
        prob = poisson_1d()
        info = detect_diff_order(prob["pde"])
        assert info["max_order"] == 2
        assert info["uses_hessian"] is True

    def test_advection_1d(self):
        from problems import advection_1d
        prob = advection_1d()
        info = detect_diff_order(prob["pde"])
        assert info["max_order"] == 1
        assert info["uses_jacobian"] is True
        assert info["uses_hessian"] is False

    def test_heat_1d(self):
        from problems import heat_1d
        prob = heat_1d()
        info = detect_diff_order(prob["pde"])
        assert info["max_order"] == 2
        assert info["uses_jacobian"] is True
        assert info["uses_hessian"] is True

    def test_stokes_2d(self):
        from problems import stokes_2d
        prob = stokes_2d()
        info = detect_diff_order(prob["pde"])
        assert info["max_order"] == 2
        assert info["uses_jacobian"] is True
        assert info["uses_hessian"] is True
