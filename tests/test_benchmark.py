"""Test benchmark framework — result structure, timing, output format."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# monkey-patch benchmark settings for fast tests
import benchmark
benchmark.adam_epochs = 10
benchmark.lbfgs_max_iter = 10
benchmark.n_interior = 100
benchmark.log_every = 100


class TestRunBenchmark:
    def test_returns_dict(self):
        cfg = {
            "activation": "tanh",
            "fourier_features": 0,
            "norm": "none",
            "causal": False,
            "use_dd": False,
        }
        result = benchmark.run_benchmark("poisson_1d", "baseline", cfg)
        assert isinstance(result, dict)
        assert "problem" in result
        assert "config" in result
        assert result["problem"] == "poisson_1d"
        assert result["config"] == "baseline"

    def test_result_has_metrics(self):
        cfg = {
            "activation": "tanh",
            "fourier_features": 0,
            "norm": "none",
            "causal": False,
            "use_dd": False,
        }
        result = benchmark.run_benchmark("poisson_1d", "baseline", cfg)
        assert not result.get("skipped")
        assert "l2_error" in result
        assert "time_s" in result
        assert "peak_memory_mb" in result
        assert "n_params" in result

    def test_timing_positive(self):
        cfg = {
            "activation": "tanh",
            "fourier_features": 0,
            "norm": "none",
            "causal": False,
            "use_dd": False,
        }
        result = benchmark.run_benchmark("poisson_1d", "baseline", cfg)
        assert result["time_s"] > 0

    def test_skip_causal_for_non_temporal(self):
        """Causal config should be skipped for non-time-dependent problems."""
        cfg = {
            "activation": "tanh",
            "fourier_features": 0,
            "norm": "none",
            "causal": True,
            "use_dd": False,
        }
        result = benchmark.run_benchmark("poisson_1d", "causal", cfg)
        assert result.get("skipped")


class TestPrintResults:
    def test_no_crash(self):
        """print_results_table should not crash with sample data."""
        results = [
            {
                "problem": "poisson_1d",
                "config": "baseline",
                "l2_error": 0.001,
                "time_s": 10.5,
                "peak_memory_mb": 50.0,
                "n_params": 5000,
                "final_loss": 1e-5,
                "skipped": False,
            },
            {
                "problem": "poisson_1d",
                "config": "causal",
                "skipped": True,
                "reason": "no time_dim",
            },
        ]
        benchmark.print_results_table(results)  # should not raise
