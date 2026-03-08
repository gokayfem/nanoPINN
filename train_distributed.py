"""
nanoPINN distributed training script.

Usage:
    torchrun --nproc_per_node=2 train_distributed.py --problem=poisson_2d
    torchrun --nproc_per_node=4 train_distributed.py --problem=heat_1d --causal=True
"""

import sys
import torch

from nanopinn import MLP, save_checkpoint, sobol
from nanopinn_distributed import setup, cleanup, train_distributed
from problems import PROBLEMS

# ─── config ──────────────────────────────────────────────────────────────────

problem = "poisson_1d"
activation = "tanh"
hidden_dim = 64
num_hidden = 3
n_interior = 5000
adam_lr = 1e-3
adam_epochs = 5000
log_every = 100
seed = 42
causal = False
causal_epsilon = 1.0
causal_n_bins = 20
ntk_weighting = False
ntk_every = 100
resample_every = 500

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

# ─── setup ───────────────────────────────────────────────────────────────────

rank, world_size = setup()
device = f"cuda:{rank}"

prob = PROBLEMS[problem]()
input_dim = len(prob["domain"])
output_dim = prob["layers"][-1]
layers = [input_dim] + [hidden_dim] * num_hidden + [output_dim]
model = MLP(layers, activation=activation)

if rank == 0:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Problem: {prob['name']}, Model: {layers}, {n_params:,} params")
    print(f"DDP: {world_size} GPUs")

extra_params = prob.get("inv_params")
causal_time_dim = prob.get("time_dim", -1)

# ─── train ───────────────────────────────────────────────────────────────────

losses = train_distributed(
    model, pde_fn=prob["pde"], bc_fn=prob["bc"], domain=prob["domain"],
    extra_params=extra_params, n_interior=n_interior, adam_lr=adam_lr,
    adam_epochs=adam_epochs, log_every=log_every, seed=seed,
    causal=causal, causal_epsilon=causal_epsilon,
    causal_time_dim=causal_time_dim, causal_n_bins=causal_n_bins,
    ntk_weighting=ntk_weighting, ntk_every=ntk_every,
    resample_every=resample_every,
)

# ─── evaluate (rank 0 only) ─────────────────────────────────────────────────

if rank == 0:
    model.eval()
    x_test = sobol(1000, prob["domain"], device=device)
    with torch.no_grad():
        u_pred = model(x_test)
        u_exact = prob["exact"](x_test).to(device)
    if u_exact.dim() == 1:
        u_exact = u_exact.unsqueeze(1)
    l2_rel = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    print(f"\nL2 relative error: {l2_rel.item():.4e}")

    save_checkpoint(f"{problem}_ddp.pt", model, losses, {k: globals()[k] for k in config_keys})
    print(f"Saved {problem}_ddp.pt")

cleanup()
