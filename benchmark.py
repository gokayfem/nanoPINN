"""
nanoPINN benchmark suite — nanoGPT style.

Runs all problem x config combinations and produces a comparison table.
Measures L2 error, training time, and peak memory.

Usage:
    python benchmark.py
    python benchmark.py --problems=poisson_1d,heat_1d
    python benchmark.py --configs=baseline,fourier
    python benchmark.py --adam_epochs=1000 --lbfgs_max_iter=1000
"""

import json
import os
import sys
import time
import tracemalloc

import torch

from nanopinn import MLP, DDModel, decompose_domain, save_checkpoint, sobol, train
from problems import PROBLEMS

# ─── config ──────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
problems = "all"  # or comma-separated: "poisson_1d,heat_1d"
configs = "all"  # or comma-separated: "baseline,fourier,spectral,causal,dd"
output_dir = "benchmark_results"
adam_epochs = 3000
lbfgs_max_iter = 3000
n_interior = 3000
hidden_dim = 64
num_hidden = 3
seed = 42
log_every = 1000

# ─── poor man's configurator ────────────────────────────────────────────────

config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    key, val = arg.lstrip("-").split("=", 1)
    if key in globals():
        old = globals()[key]
        if isinstance(old, bool):
            globals()[key] = val.lower() in ("true", "1", "yes")
        elif isinstance(old, int):
            globals()[key] = int(val)
        elif isinstance(old, float):
            globals()[key] = float(val)
        else:
            globals()[key] = val


# ─── benchmark configs ──────────────────────────────────────────────────────

CONFIGS = {
    "baseline": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
    },
    "fourier": {
        "activation": "tanh",
        "fourier_features": 64,
        "fourier_sigma": 10.0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
    },
    "spectral": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "spectral",
        "causal": False,
        "use_dd": False,
    },
    "causal": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": True,
        "causal_epsilon": 1.0,
        "use_dd": False,
    },
    "dd": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": True,
        "dd_n_sub": 2,
        "dd_overlap": 0.25,
    },
    "siren": {
        "activation": "siren",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
    },
    "ntk": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
        "ntk_weighting": True,
        "ntk_every": 100,
    },
    "variational": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
        "use_energy": True,
    },
    "adaptive": {
        "activation": "tanh",
        "fourier_features": 0,
        "norm": "none",
        "causal": False,
        "use_dd": False,
        "adaptive_refine_every": 200,
        "adaptive_refine_ratio": 0.15,
    },
}


def _build_model(prob, cfg):
    """Build a model from problem + config."""
    input_dim = len(prob["domain"])
    output_dim = prob["layers"][-1]
    layers = [input_dim] + [hidden_dim] * num_hidden + [output_dim]

    act = cfg.get("activation", "tanh")
    ff = cfg.get("fourier_features", 0)
    fs = cfg.get("fourier_sigma", 1.0)
    nm = cfg.get("norm", "none")

    if cfg.get("use_dd", False):
        nsub = [cfg.get("dd_n_sub", 2)] * input_dim
        subdomains = decompose_domain(prob["domain"], nsub, cfg.get("dd_overlap", 0.25))
        return DDModel(layers, subdomains, activation=act,
                       fourier_features=ff, fourier_sigma=fs, norm=nm)
    return MLP(layers, activation=act, fourier_features=ff, fourier_sigma=fs, norm=nm)


def run_benchmark(problem_name: str, config_name: str, cfg: dict) -> dict:
    """Run a single benchmark. Returns result dict."""
    prob = PROBLEMS[problem_name]()

    # skip causal config for non-time-dependent problems
    if cfg.get("causal") and "time_dim" not in prob:
        return {
            "problem": problem_name,
            "config": config_name,
            "skipped": True,
            "reason": "causal requires time_dim",
        }

    # skip variational config for problems without energy function
    if cfg.get("use_energy") and "energy" not in prob:
        return {
            "problem": problem_name,
            "config": config_name,
            "skipped": True,
            "reason": "variational requires energy function",
        }

    model = _build_model(prob, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # measure memory
    if device == "cpu":
        tracemalloc.start()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model.train()

    causal_kwargs = {}
    if cfg.get("causal"):
        causal_kwargs = {
            "causal": True,
            "causal_epsilon": cfg.get("causal_epsilon", 1.0),
            "causal_time_dim": prob.get("time_dim", -1),
        }

    ntk_kwargs = {}
    if cfg.get("ntk_weighting"):
        ntk_kwargs = {
            "ntk_weighting": True,
            "ntk_every": cfg.get("ntk_every", 100),
        }

    energy_kwargs = {}
    if cfg.get("use_energy") and "energy" in prob:
        energy_kwargs = {"energy_fn": prob["energy"]}

    refine_kwargs = {}
    if cfg.get("adaptive_refine_every", 0) > 0:
        refine_kwargs = {
            "adaptive_refine_every": cfg["adaptive_refine_every"],
            "adaptive_refine_ratio": cfg.get("adaptive_refine_ratio", 0.1),
            "adaptive_refine_sigma": cfg.get("adaptive_refine_sigma", 0.05),
        }

    try:
        losses = train(
            model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
            n_interior=n_interior, adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_max_iter,
            log_every=log_every, device=device, seed=seed,
            **causal_kwargs, **ntk_kwargs, **energy_kwargs, **refine_kwargs,
        )
    except Exception as e:
        return {
            "problem": problem_name,
            "config": config_name,
            "skipped": True,
            "reason": str(e),
        }

    elapsed = time.perf_counter() - t0

    # peak memory
    if device == "cpu":
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / 1024 / 1024
    elif torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mb = 0.0

    # evaluate
    model.eval()
    x_test = sobol(1000, prob["domain"], device=device)
    with torch.no_grad():
        u_pred = model(x_test)
        u_exact = prob["exact"](x_test).to(device)
    if u_exact.dim() == 1:
        u_exact = u_exact.unsqueeze(1)
    l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()

    return {
        "problem": problem_name,
        "config": config_name,
        "l2_error": l2_rel,
        "time_s": elapsed,
        "peak_memory_mb": peak_mb,
        "n_params": n_params,
        "final_loss": losses[-1] if losses else float("nan"),
        "skipped": False,
    }


def print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    active = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    if active:
        header = f"{'Problem':<22} {'Config':<12} {'L2 Error':>10} {'Time (s)':>10} {'Mem (MB)':>10} {'Params':>8}"
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for r in active:
            print(
                f"{r['problem']:<22} {r['config']:<12} "
                f"{r['l2_error']:>10.4e} {r['time_s']:>10.1f} "
                f"{r['peak_memory_mb']:>10.1f} {r['n_params']:>8,}"
            )
        print("=" * len(header))

    if skipped:
        print(f"\nSkipped {len(skipped)} combinations:")
        for r in skipped:
            print(f"  {r['problem']} + {r['config']}: {r.get('reason', 'unknown')}")


def run_all_benchmarks() -> list[dict]:
    """Run all problem x config combinations."""
    prob_names = list(PROBLEMS.keys()) if problems == "all" else [p.strip() for p in problems.split(",")]
    cfg_names = list(CONFIGS.keys()) if configs == "all" else [c.strip() for c in configs.split(",")]

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total = len(prob_names) * len(cfg_names)
    idx = 0
    for pname in prob_names:
        for cname in cfg_names:
            idx += 1
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] {pname} + {cname}")
            print(f"{'='*60}")
            result = run_benchmark(pname, cname, CONFIGS[cname])
            results.append(result)

            if not result.get("skipped"):
                # save individual result
                rpath = os.path.join(output_dir, f"{pname}_{cname}.json")
                with open(rpath, "w") as f:
                    json.dump(result, f, indent=2)

    # save all results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results_table(results)
