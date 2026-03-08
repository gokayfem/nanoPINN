"""
Generate pre-trained checkpoints for all built-in problems.

Run once to create checkpoints/:
    python generate_checkpoints.py
    python generate_checkpoints.py --adam_epochs=10000
"""

import os
import sys

import torch

from nanopinn import MLP, save_checkpoint, sobol, train
from problems import PROBLEMS

# ─── config ──────────────────────────────────────────────────────────────────

adam_epochs = 5000
lbfgs_max_iter = 5000
n_interior = 5000
hidden_dim = 64
num_hidden = 3
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "checkpoints"

# ─── poor man's configurator ────────────────────────────────────────────────

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

# ─── generate ────────────────────────────────────────────────────────────────

os.makedirs(output_dir, exist_ok=True)

# skip multi-output problems for simple checkpoint generation
skip = {"stokes_2d"}

for name, prob_fn in PROBLEMS.items():
    if name in skip:
        continue

    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    prob = prob_fn()
    input_dim = len(prob["domain"])
    output_dim = prob["layers"][-1]
    layers = [input_dim] + [hidden_dim] * num_hidden + [output_dim]

    model = MLP(layers).to(device)
    model.train()
    losses = train(
        model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
        n_interior=n_interior, adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_max_iter,
        device=device, seed=seed,
    )

    # evaluate
    model.eval()
    x_test = sobol(1000, prob["domain"], device=device)
    with torch.no_grad():
        u_pred = model(x_test)
        u_exact = prob["exact"](x_test).to(device)
    if u_exact.dim() == 1:
        u_exact = u_exact.unsqueeze(1)
    l2_rel = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
    print(f"L2 relative error: {l2_rel:.4e}")

    config = {
        "layers": layers,
        "activation": "tanh",
        "adam_epochs": adam_epochs,
        "lbfgs_max_iter": lbfgs_max_iter,
        "n_interior": n_interior,
    }
    path = os.path.join(output_dir, f"{name}.pt")
    save_checkpoint(path, model, losses, config, metadata={"l2_error": l2_rel})
    print(f"Saved {path}")

print(f"\nAll checkpoints saved to {output_dir}/")
