# nanoPINN

The simplest, fastest repository for training Physics-Informed Neural Networks. It is a rewrite of the PINN paradigm that prioritizes teeth over education, inspired by [nanoGPT](https://github.com/karpathy/nanoGPT). The file `nanopinn.py` is a ~550-line library that gives you `jacobian`, `hessian`, `MLP`, Fourier features, domain decomposition, causal training, and a hybrid Adam→L-BFGS training loop. The file `train.py` is a ~140-line nanoGPT-style training script with config-as-globals. That's it.

Because the code is so simple, it is very easy to hack to your needs: define any PDE as a plain Python function, pick an activation, and train.

## install

```
pip install torch numpy
```

Dependencies:

- [pytorch](https://pytorch.org) >=2.0 <3
- [numpy](https://numpy.org/install/) <3

That's all you need. No SymPy, no config frameworks, no class hierarchies.

## quick start

The fastest way to get started is to solve the 1D Poisson equation. Just run:

```
python train.py
```

This trains a 3-layer Tanh MLP on `-u''(x) = π²sin(πx)` with `u(0) = u(1) = 0` (exact solution: `u = sin(πx)`). On a CPU this takes about 30 seconds and you should see:

```
Problem: poisson_1d
Domain: [(0.0, 1.0)]
Device: cpu
Model: [1, 64, 64, 64, 1], tanh, 8,769 params
[Adam   100/5000] loss=2.8431e+00 lr=1.00e-03 (0.5s)
[Adam   200/5000] loss=5.3197e-01 lr=9.98e-04 (0.9s)
...
Adam done in 21.4s, loss=5.1203e-05
L-BFGS done in 5.2s (1423 iters), loss=2.6810e-08
Total: 26.6s

L2 relative error: 1.2847e-04
Saved poisson_1d.pt
```

**I have a GPU.** Great, just add `--device=cuda`:

```
python train.py --device=cuda
```

**I want a different PDE.** Pick any built-in problem:

```
python train.py --problem=poisson_2d
python train.py --problem=heat_1d
python train.py --problem=helmholtz_1d
python train.py --problem=harmonic_oscillator
python train.py --problem=advection_1d
python train.py --problem=stokes_2d          # multi-output system (u, v, p)
```

**I want Fourier features for high-frequency solutions.**

```
python train.py --fourier_features=64 --fourier_sigma=10.0
```

**I want spectral normalization for stable training.**

```
python train.py --norm=spectral
```

**I want causal training for time-dependent PDEs.**

```
python train.py --problem=heat_1d --causal=True
```

**I want domain decomposition (FBPINNs-style).**

```
python train.py --problem=poisson_2d --use_dd=True --dd_subdomains=3,3
```

**I want to use SIREN.** Switch the activation:

```
python train.py --problem=poisson_1d --activation=siren --adam_epochs=2000 --lbfgs_max_iter=3000
```

**I want more collocation points and a bigger network.** Override any config:

```
python train.py --problem=poisson_2d --n_interior=10000 --hidden_dim=128 --num_hidden=4 --adam_epochs=10000
```

All config variables are plain Python globals at the top of `train.py`. Override from the command line with `--key=value`, exactly like nanoGPT.

## define your own PDE

A PDE is just a Python function. Here's the complete code to solve Poisson 1D from scratch:

```python
import math
import torch
from nanopinn import MLP, hessian, sobol, train

# PDE residual: -u''(x) = π²sin(πx)
def pde(net, x):
    H = hessian(net, x)                                    # (1, 1, 1) Hessian
    f = (math.pi ** 2) * torch.sin(math.pi * x[0])         # source term
    return -H[0, 0, 0] - f                                 # scalar residual

# Boundary conditions: u(0) = u(1) = 0
def bc(model, device):
    x0 = torch.zeros(200, 1, device=device)
    x1 = torch.ones(200, 1, device=device)
    return (model(x0) ** 2).mean() + (model(x1) ** 2).mean()

model = MLP([1, 64, 64, 64, 1], activation='tanh')
losses = train(model, pde_fn=pde, bc_fn=bc, domain=[(0.0, 1.0)])
```

The key insight: `pde(net, x)` receives a callable `net` mapping a single point `(d,)` → `(m,)`, and a single point `x` of shape `(d,)`. You use `jacobian(net, x)` and `hessian(net, x)` to get derivatives. The library automatically `vmap`s this over all collocation points for batched, efficient computation.

**Multi-output systems** (Navier-Stokes, elasticity): return a vector instead of a scalar:

```python
# Stokes: network outputs (u, v, p), PDE returns (3,) residual
def stokes_pde(net, x):
    J = jacobian(net, x)   # (3, 2): d[u,v,p]/d[x,y]
    H = hessian(net, x)    # (3, 2, 2)
    r_mom_x = -J[2, 0] + (H[0, 0, 0] + H[0, 1, 1]) - f_x(x)
    r_mom_y = -J[2, 1] + (H[1, 0, 0] + H[1, 1, 1]) - f_y(x)
    r_cont  = J[0, 0] + J[1, 1]
    return torch.stack([r_mom_x, r_mom_y, r_cont])

model = MLP([2, 64, 64, 64, 3])  # 3 outputs: u, v, p
```

## features

### Fourier feature encoding

Random Fourier Features (RFF) help networks learn high-frequency solutions by mapping inputs through `[sin(2πxB), cos(2πxB)]` with a random frequency matrix B:

```python
model = MLP([2, 64, 1], fourier_features=64, fourier_sigma=10.0)
```

### Weight / spectral normalization

Stabilize training and improve convergence:

```python
model = MLP([2, 64, 1], norm="spectral")  # or norm="weight"
```

### Causal training

For time-dependent PDEs, weight residuals so earlier times are prioritized (Wang et al. 2022):

```python
train(model, pde, bc, domain, causal=True, causal_epsilon=1.0)
```

### Domain decomposition

FBPINNs-style: split domain into overlapping subdomains with separate sub-networks blended by cosine windows:

```python
from nanopinn import DDModel, decompose_domain

subs = decompose_domain([(0, 1), (0, 1)], n_subdomains=[3, 3], overlap=0.25)
model = DDModel([2, 64, 1], subs)
train(model, pde, bc, domain)
```

### Checkpoints

Save and load trained models:

```python
from nanopinn import save_checkpoint, load_checkpoint

save_checkpoint("model.pt", model, losses, config)
ckpt = load_checkpoint("model.pt", model)  # loads weights into model
```

## how it works

nanoPINN uses three ideas from modern PyTorch to make PINNs fast:

1. **`torch.func.jacrev`** — reverse-mode automatic Jacobians and Hessians, composable with vmap
2. **`torch.func.vmap`** — vectorized map that batches per-point derivative computation into a single GPU call (replaces 35+ sequential `autograd.grad` calls in typical PINN code)
3. **Hybrid Adam → L-BFGS** — Adam with cosine annealing for broad exploration, then L-BFGS with strong Wolfe line search for fine convergence

The training loop also supports:
- **Sobol quasi-random sampling** for better domain coverage than uniform random
- **Adaptive residual-based resampling** — keeps high-residual points, replaces low-residual with fresh samples
- **`torch.compile`** on CUDA for kernel fusion (opt-in via `--do_compile=True`)

## built-in problems

Seven PDEs with exact analytical solutions, verified by the test suite:

| Problem | Equation | Domain | Exact Solution | Outputs |
|---------|----------|--------|----------------|---------|
| `poisson_1d` | −u″ = π²sin(πx) | [0, 1] | sin(πx) | 1 |
| `poisson_2d` | −Δu = 2π²sin(πx)sin(πy) | [0,1]² | sin(πx)sin(πy) | 1 |
| `heat_1d` | u_t = u_xx | [0,1]×[0,1] | e^(−π²t)sin(πx) | 1 |
| `harmonic_oscillator` | u″ + u = 0 | [0, 2π] | cos(x) | 1 |
| `helmholtz_1d` | −u″ − k²u = f | [0, 1] | sin(πx) | 1 |
| `advection_1d` | u_t + u_x = 0 | [0,2π]×[0,1] | sin(x − t) | 1 |
| `stokes_2d` | Stokes flow | [0,1]² | manufactured | 3 (u,v,p) |

## benchmarks

Run the full benchmark suite comparing all features across all problems:

```
python benchmark.py
python benchmark.py --problems=poisson_1d,heat_1d --configs=baseline,fourier
python benchmark.py --adam_epochs=1000  # fast mode
```

Generates a comparison table with L2 error, training time, and peak memory for each problem × config combination. Results saved to `benchmark_results/`.

## pre-trained checkpoints

Generate checkpoints for all built-in problems:

```
python generate_checkpoints.py
```

Load a pre-trained model:

```python
from nanopinn import MLP, load_checkpoint

model = MLP([1, 64, 64, 64, 1])
ckpt = load_checkpoint("checkpoints/poisson_1d.pt", model)
# model is now ready for evaluation or fine-tuning
```

## API reference

The entire library is in `nanopinn.py`. Here's everything it exports:

```python
# ─── Derivatives (designed for use inside pde_fn) ───
jacobian(f, x)              # f:(d,)→(m,), x:(d,) → (m, d)
hessian(f, x)               # f:(d,)→(m,), x:(d,) → (m, d, d)
laplacian(f, x, out_idx=0)  # sum of d²f/dx_i²  → scalar

# ─── Networks ───
FourierFeatures(in_features, mapping_size, sigma=1.0)
MLP(layers, activation='tanh', omega_0=30.0,
    fourier_features=0, fourier_sigma=1.0, norm='none')
DDModel(layers, subdomains, activation='tanh', ...)

# ─── Sampling ───
sobol(n, bounds, device)     # Sobol quasi-random  → (n, d)
uniform(n, bounds, device)   # uniform random      → (n, d)
boundary(n, bounds, device)  # points on faces     → (n, d)

# ─── Loss ───
pde_loss(model, pde_fn, points)     # vmapped PDE residual loss
causal_pde_loss(model, pde_fn, points, time_dim, epsilon, n_bins)
causal_weights(points, residuals_sq, time_dim, epsilon, n_bins)

# ─── Domain decomposition ───
cosine_window(center, half_width)    # window function factory
decompose_domain(bounds, n_subdomains, overlap)

# ─── Checkpoints ───
save_checkpoint(path, model, losses, config, metadata=None)
load_checkpoint(path, model=None, device='cpu')

# ─── Training ───
train(model, pde_fn, bc_fn, domain,
      n_interior=5000, adam_lr=1e-3, adam_epochs=5000,
      lbfgs_max_iter=5000, do_compile=False,
      resample_every=500, log_every=100,
      device='cpu', seed=42,
      causal=False, causal_epsilon=1.0,
      causal_time_dim=-1, causal_n_bins=20)
# Returns: list[float] (loss history)
```

## tests

93 tests covering derivatives, models, sampling, convergence, causal training, domain decomposition, checkpoints, and benchmarks:

```
pip install pytest
python -m pytest tests/ -v
```

Fast tests only (~60s):

```
python -m pytest tests/ -v -m "not slow"
```

The convergence tests actually train PINNs and verify the L2 relative error against the analytical solution:

| Test | Tolerance |
|------|-----------|
| Poisson 1D | < 5% |
| Poisson 2D | < 10% |
| Heat 1D | < 10% |
| Helmholtz 1D | < 5% |
| Harmonic Oscillator | < 15% |
| Advection 1D | < 15% |
| Stokes 2D | < 25% |
| SIREN on Poisson 1D | < 15% |
| DD on Poisson 1D | < 15% |

## file structure

```
nanopinn/
├── nanopinn.py              # ~550 lines. the entire library.
├── problems.py              # ~300 lines. 7 PDEs with exact solutions.
├── train.py                 # ~140 lines. nanoGPT-style training script.
├── benchmark.py             # ~200 lines. benchmark suite.
├── generate_checkpoints.py  # ~70 lines. checkpoint generator.
├── pytest.ini
└── tests/
    ├── test_derivatives.py    # jacobian/hessian vs autograd reference
    ├── test_model.py          # MLP, Fourier, normalization
    ├── test_sampling.py       # bounds, coverage, shapes
    ├── test_convergence.py    # convergence to exact solutions + multi-output
    ├── test_causal.py         # causal weighting + training
    ├── test_dd.py             # domain decomposition + DDModel
    ├── test_checkpoints.py    # save/load roundtrip
    └── test_benchmark.py      # benchmark framework
```

## design philosophy

Stolen from the best:

| Source | What we took |
|--------|-------------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Single-file library, config-as-globals, no frameworks |
| [DeepXDE](https://github.com/lululxvi/deepxde) | PDE-as-a-Python-function returning a residual |
| [FBPINNs](https://github.com/benmoseley/FBPINNs) | `vmap` for batched per-point derivatives, domain decomposition |
| [PyDEns](https://github.com/analysiscenter/pydens) | Minimal API surface — `D(f, x)` style operators |
| [Wang et al. 2022](https://arxiv.org/abs/2203.07404) | Causal training for time-dependent PDEs |
| [Tancik et al. 2020](https://arxiv.org/abs/2006.10739) | Fourier feature encoding for high-frequency solutions |

What we deliberately left out:
- No YAML/Hydra config files
- No PyTorch Lightning
- No class hierarchies for problems or solvers
- No SymPy dependency
- No multi-backend abstraction

## todos

- [x] Fourier feature input encoding
- [x] Weight normalization / spectral normalization
- [x] Multi-output system support (Stokes 2D)
- [x] Domain decomposition (FBPINNs-style)
- [x] Causal training for time-dependent PDEs
- [x] Benchmark suite with timing comparisons
- [x] Pre-trained checkpoints for common problems
- [ ] Inverse problems
- [ ] Transfer learning across PDE parameters
- [ ] Multi-GPU training
- [ ] Adaptive loss weighting (NTK-based)
