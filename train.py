"""
nanoPINN training script — nanoGPT style.

Single-file, config via globals, override from CLI:
    python train.py --problem=poisson_2d --adam_epochs=10000
    python train.py --activation=siren --n_interior=8000
"""

import sys
import torch

from nanopinn import MLP, train
from problems import PROBLEMS

# ─── config (override any of these from command line) ────────────────────────

# problem
problem = "poisson_1d"

# model
activation = "tanh"  # 'tanh' | 'siren' | 'gelu' | 'swish'
hidden_dim = 64
num_hidden = 3
omega_0 = 30.0  # SIREN frequency (only used if activation='siren')

# sampling
n_interior = 5000
resample_every = 500

# training
adam_lr = 1e-3
adam_epochs = 5000
lbfgs_max_iter = 5000
do_compile = False
seed = 42
log_every = 100

# system
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── poor man's configurator (same as nanoGPT) ──────────────────────────────

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

if problem not in PROBLEMS:
    raise ValueError(f"Unknown problem '{problem}'. Choose from: {list(PROBLEMS.keys())}")

prob = PROBLEMS[problem]()
print(f"Problem: {prob['name']}")
print(f"Domain: {prob['domain']}")
print(f"Device: {device}")

# build model: [input_dim, hidden, ..., hidden, output_dim]
input_dim = len(prob["domain"])
output_dim = prob["layers"][-1]
layers = [input_dim] + [hidden_dim] * num_hidden + [output_dim]
model = MLP(layers, activation=activation, omega_0=omega_0).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {layers}, {activation}, {n_params:,} params")

# ─── train ───────────────────────────────────────────────────────────────────

model.train()
losses = train(
    model,
    pde_fn=prob["pde"],
    bc_fn=prob["bc"],
    domain=prob["domain"],
    n_interior=n_interior,
    adam_lr=adam_lr,
    adam_epochs=adam_epochs,
    lbfgs_max_iter=lbfgs_max_iter,
    do_compile=do_compile,
    resample_every=resample_every,
    log_every=log_every,
    device=device,
    seed=seed,
)

# ─── evaluate ────────────────────────────────────────────────────────────────

model.eval()
from nanopinn import sobol

x_test = sobol(1000, prob["domain"], device=device)
with torch.no_grad():
    u_pred = model(x_test)
    u_exact = prob["exact"](x_test).to(device)

# handle shape mismatch (1D problems return flat, model returns (N, 1))
if u_exact.dim() == 1:
    u_exact = u_exact.unsqueeze(1)

l2_rel = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print(f"\nL2 relative error: {l2_rel.item():.4e}")

# ─── save ────────────────────────────────────────────────────────────────────

torch.save({"model": model.state_dict(), "losses": losses, "config": {k: globals()[k] for k in config_keys}}, f"{problem}.pt")
print(f"Saved {problem}.pt")
