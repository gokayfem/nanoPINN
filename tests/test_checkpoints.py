"""Test checkpoint save/load functionality."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanopinn import MLP, save_checkpoint, load_checkpoint


class TestCheckpoints:
    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve weights exactly."""
        model = MLP([2, 32, 1])
        path = str(tmp_path / "test.pt")

        # save
        losses = [1.0, 0.5, 0.1]
        config = {"lr": 1e-3, "epochs": 100}
        save_checkpoint(path, model, losses, config)

        # load into fresh model
        model2 = MLP([2, 32, 1])
        ckpt = load_checkpoint(path, model2)

        # weights should match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Param {n1} mismatch after load"

    def test_config_preserved(self, tmp_path):
        model = MLP([1, 16, 1])
        path = str(tmp_path / "test.pt")
        config = {"problem": "poisson_1d", "lr": 0.001, "seed": 42}
        save_checkpoint(path, model, [1.0], config)

        ckpt = load_checkpoint(path)
        assert ckpt["config"] == config

    def test_losses_preserved(self, tmp_path):
        model = MLP([1, 16, 1])
        path = str(tmp_path / "test.pt")
        losses = [1.0, 0.8, 0.5, 0.2, 0.1]
        save_checkpoint(path, model, losses, {})

        ckpt = load_checkpoint(path)
        assert ckpt["losses"] == losses

    def test_load_without_model(self, tmp_path):
        """Load checkpoint without passing a model — just get the dict."""
        model = MLP([2, 32, 1])
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, model, [0.5], {"key": "val"})

        ckpt = load_checkpoint(path)
        assert "model_state" in ckpt
        assert "config" in ckpt
        assert "losses" in ckpt
        assert "metadata" in ckpt

    def test_metadata(self, tmp_path):
        model = MLP([1, 16, 1])
        path = str(tmp_path / "test.pt")
        meta = {"l2_error": 0.001, "training_time": 42.5}
        save_checkpoint(path, model, [], {}, metadata=meta)

        ckpt = load_checkpoint(path)
        assert ckpt["metadata"] == meta

    def test_default_metadata_empty(self, tmp_path):
        model = MLP([1, 16, 1])
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, model, [], {})

        ckpt = load_checkpoint(path)
        assert ckpt["metadata"] == {}

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            load_checkpoint("/nonexistent/path/model.pt")

    def test_fourier_model_roundtrip(self, tmp_path):
        """Checkpoint roundtrip works with Fourier features."""
        model = MLP([2, 32, 1], fourier_features=16, fourier_sigma=5.0)
        path = str(tmp_path / "fourier.pt")
        save_checkpoint(path, model, [0.5], {"fourier_features": 16})

        model2 = MLP([2, 32, 1], fourier_features=16, fourier_sigma=5.0)
        load_checkpoint(path, model2)

        x = torch.rand(5, 2)
        with torch.no_grad():
            assert torch.equal(model(x), model2(x))
