"""
nanoPINN training script — nanoGPT style.

Single-file, config via globals, override from CLI:
    python train.py --problem=poisson_2d --adam_epochs=10000
    python train.py --activation=siren --n_interior=8000
    python train.py --fourier_features=64 --fourier_sigma=10.0
    python train.py --norm=spectral --causal=True
    python train.py --use_dd=True --dd_subdomains=3,3
    python train.py --problem=helmholtz_1d_inverse  # inverse problem
    python train.py --ntk_weighting=True            # NTK loss balancing
    python train.py --transfer_from=helmholtz_1d.pt # transfer learning
"""

import sys
import torch

from nanopinn import (
    MLP, DDModel, decompose_domain, save_checkpoint, load_checkpoint,
    train, sobol, freeze_layers, unfreeze_all,
)
from problems import PROBLEMS

# ─── config (override any of these from command line) ────────────────────────

# problem
problem = "poisson_1d"

# model
activation = "tanh"  # 'tanh' | 'siren' | 'gelu' | 'swish'
hidden_dim = 64
num_hidden = 3
omega_0 = 30.0  # SIREN frequency (only used if activation='siren')

# fourier features
fourier_features = 0  # 0 to disable, e.g. 64 for RFF encoding
fourier_sigma = 1.0

# normalization
norm = "none"  # 'none' | 'weight' | 'spectral'

# domain decomposition
use_dd = False
dd_subdomains = ""  # comma-separated, e.g. "3" for 1D or "2,2" for 2D
dd_overlap = 0.25

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

# causal training
causal = False
causal_epsilon = 1.0
causal_n_bins = 20

# NTK adaptive weighting
ntk_weighting = False
ntk_every = 100

# transfer learning
transfer_from = ""  # path to source checkpoint
transfer_keep_last = 1  # layers to keep trainable during transfer
unfreeze_after = 0  # unfreeze all after this many Adam epochs (0=no freeze)

# checkpoints
load_from = ""  # path to checkpoint to load
save_to = ""  # path to save (default: {problem}.pt)

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

# build model
input_dim = len(prob["domain"])
output_dim = prob["layers"][-1]
layers = [input_dim] + [hidden_dim] * num_hidden + [output_dim]

if use_dd:
    nsub = [int(x) for x in dd_subdomains.split(",") if x] if dd_subdomains else [2] * input_dim
    subdomains = decompose_domain(prob["domain"], nsub, dd_overlap)
    model = DDModel(
        layers, subdomains, activation=activation,
        fourier_features=fourier_features, fourier_sigma=fourier_sigma, norm=norm,
    ).to(device)
    print(f"DDModel: {len(subdomains)} subdomains, {nsub}")
else:
    model = MLP(
        layers, activation=activation, omega_0=omega_0,
        fourier_features=fourier_features, fourier_sigma=fourier_sigma, norm=norm,
    ).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {layers}, {activation}, {n_params:,} params")
if fourier_features > 0:
    print(f"Fourier: {fourier_features} features, sigma={fourier_sigma}")
if norm != "none":
    print(f"Norm: {norm}")

# load checkpoint or transfer
if transfer_from:
    ckpt = load_checkpoint(transfer_from, model, device, strict=False)
    print(f"Transfer from {transfer_from}")
    if unfreeze_after > 0:
        freeze_layers(model, keep_last=transfer_keep_last)
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"Frozen {frozen_count} params, will unfreeze after epoch {unfreeze_after}")
elif load_from:
    ckpt = load_checkpoint(load_from, model, device)
    print(f"Loaded checkpoint from {load_from}")

# inverse problem params
extra_params = prob.get("inv_params")
if extra_params is not None:
    extra_params = extra_params.to(device)
    print(f"Inverse params: {extra_params.as_dict()}")
    print(f"True params: {prob['true_params']}")

# ─── train ───────────────────────────────────────────────────────────────────

# auto-detect causal time_dim from problem
causal_time_dim = prob.get("time_dim", -1)

# callback for transfer learning unfreezing
_callback = None
if unfreeze_after > 0:
    def _callback(epoch, loss):
        if epoch == unfreeze_after:
            unfreeze_all(model)
            print(f"[epoch {epoch}] Unfroze all layers")

model.train()
losses = train(
    model,
    pde_fn=prob["pde"],
    bc_fn=prob["bc"],
    domain=prob["domain"],
    extra_params=extra_params,
    n_interior=n_interior,
    adam_lr=adam_lr,
    adam_epochs=adam_epochs,
    lbfgs_max_iter=lbfgs_max_iter,
    do_compile=do_compile,
    resample_every=resample_every,
    log_every=log_every,
    device=device,
    seed=seed,
    causal=causal,
    causal_epsilon=causal_epsilon,
    causal_time_dim=causal_time_dim,
    causal_n_bins=causal_n_bins,
    ntk_weighting=ntk_weighting,
    ntk_every=ntk_every,
    callback=_callback,
)

# ─── evaluate ────────────────────────────────────────────────────────────────

model.eval()
x_test = sobol(1000, prob["domain"], device=device)
with torch.no_grad():
    u_pred = model(x_test)
    u_exact = prob["exact"](x_test).to(device)

if u_exact.dim() == 1:
    u_exact = u_exact.unsqueeze(1)

l2_rel = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print(f"\nL2 relative error: {l2_rel.item():.4e}")

# report inverse params
if extra_params is not None:
    recovered = extra_params.as_dict()
    true_vals = prob["true_params"]
    print("Recovered parameters:")
    for k in recovered:
        rel_err = abs(recovered[k] - true_vals[k]) / abs(true_vals[k])
        print(f"  {k}: {recovered[k]:.4f} (true: {true_vals[k]:.4f}, error: {rel_err:.1%})")

# ─── save ────────────────────────────────────────────────────────────────────

save_path = save_to if save_to else f"{problem}.pt"
pde_params = extra_params.as_dict() if extra_params is not None else None
save_checkpoint(save_path, model, losses, {k: globals()[k] for k in config_keys}, pde_params=pde_params)
print(f"Saved {save_path}")
