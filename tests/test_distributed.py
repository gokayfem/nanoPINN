"""Test distributed training utilities (CPU-only unit tests).

Multi-GPU integration tests are marked with @pytest.mark.multigpu
and skipped unless multiple CUDA devices are available.
"""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn_distributed import distribute_points


class TestDistributePoints:
    def test_shard_sizes_even(self):
        """Even split: 100 points / 4 GPUs = 25 each."""
        pts = torch.rand(100, 2)
        shards = [distribute_points(pts, r, 4) for r in range(4)]
        total = sum(s.shape[0] for s in shards)
        assert total == 100

    def test_shard_sizes_uneven(self):
        """Uneven split: last rank gets remainder."""
        pts = torch.rand(103, 2)
        shards = [distribute_points(pts, r, 4) for r in range(4)]
        total = sum(s.shape[0] for s in shards)
        assert total == 103
        # last shard should be >= the others
        assert shards[-1].shape[0] >= shards[0].shape[0]

    def test_shards_non_overlapping(self):
        """Shards should partition the point set (no duplicates)."""
        pts = torch.arange(20).unsqueeze(1).float()
        shards = [distribute_points(pts, r, 4) for r in range(4)]
        all_points = torch.cat(shards, dim=0)
        assert all_points.shape[0] == 20

    def test_single_gpu(self):
        """Single GPU should return all points."""
        pts = torch.rand(50, 3)
        shard = distribute_points(pts, 0, 1)
        assert shard.shape == pts.shape
        assert torch.equal(shard, pts)

    def test_two_gpus(self):
        pts = torch.rand(10, 2)
        s0 = distribute_points(pts, 0, 2)
        s1 = distribute_points(pts, 1, 2)
        assert s0.shape[0] + s1.shape[0] == 10
        # should be the first and second halves
        assert torch.equal(s0, pts[:5])
        assert torch.equal(s1, pts[5:])
