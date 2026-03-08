"""Test sampling functions — bounds, shapes, distributions."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import sobol, uniform, boundary


class TestSobol:
    def test_shape(self):
        pts = sobol(100, [(0, 1), (0, 1)])
        assert pts.shape == (100, 2)

    def test_bounds(self):
        pts = sobol(1000, [(-1, 1), (0, 2)])
        assert pts[:, 0].min() >= -1.0
        assert pts[:, 0].max() <= 1.0
        assert pts[:, 1].min() >= 0.0
        assert pts[:, 1].max() <= 2.0

    def test_1d(self):
        pts = sobol(50, [(0, 1)])
        assert pts.shape == (50, 1)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_coverage(self):
        """Sobol should cover the domain more uniformly than random."""
        pts = sobol(256, [(0, 1), (0, 1)])
        # Check all 4 quadrants have points
        assert (pts[:, 0] < 0.5).any()
        assert (pts[:, 0] >= 0.5).any()
        assert (pts[:, 1] < 0.5).any()
        assert (pts[:, 1] >= 0.5).any()


class TestUniform:
    def test_shape(self):
        pts = uniform(100, [(0, 1), (0, 1)])
        assert pts.shape == (100, 2)

    def test_bounds(self):
        pts = uniform(10000, [(-2, 3), (0, 10)])
        assert pts[:, 0].min() >= -2.0
        assert pts[:, 0].max() <= 3.0
        assert pts[:, 1].min() >= 0.0
        assert pts[:, 1].max() <= 10.0


class TestBoundary:
    def test_shape_2d(self):
        pts = boundary(400, [(0, 1), (0, 1)])
        # 4 faces, 100 per face
        assert pts.shape == (400, 2)

    def test_on_boundary(self):
        pts = boundary(400, [(0, 1), (0, 1)])
        # Each point should have at least one coordinate at 0 or 1
        on_bnd = (
            (pts[:, 0] == 0) | (pts[:, 0] == 1) |
            (pts[:, 1] == 0) | (pts[:, 1] == 1)
        )
        assert on_bnd.all()

    def test_1d(self):
        pts = boundary(10, [(0, 1)])
        # In 1D: points at x=0 and x=1
        assert (pts == 0.0).any()
        assert (pts == 1.0).any()
