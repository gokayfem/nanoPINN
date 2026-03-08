# nanoPINN

The simplest, fastest repository for training Physics-Informed Neural Networks. It is a rewrite of the PINN paradigm that prioritizes teeth over education, inspired by [nanoGPT](https://github.com/karpathy/nanoGPT). The file `nanopinn.py` is a ~300-line library that gives you `jacobian`, `hessian`, `MLP`, Sobol sampling, and a hybrid Adam→L-BFGS training loop. The file `train.py` is a ~100-line nanoGPT-style training script with config-as-globals. That's it.

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

For time-dependent PDEs, time is just another input dimension:

```python
# Heat equation: u_t = u_xx
def heat_pde(net, x):
    J = jacobian(net, x)   # (1, 2) — [du/dx, du/dt]
    H = hessian(net, x)    # (1, 2, 2)
    u_t = J[0, 1]          # du/dt
    u_xx = H[0, 0, 0]      # d²u/dx²
    return u_t - u_xx

model = MLP([2, 64, 64, 64, 1])
train(model, heat_pde, my_bc, domain=[(0, 1), (0, 1)])  # x in [0,1], t in [0,1]
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

Six PDEs with exact analytical solutions, verified by the test suite:

| Problem | Equation | Domain | Exact Solution |
|---------|----------|--------|----------------|
| `poisson_1d` | −u″ = π²sin(πx) | [0, 1] | sin(πx) |
| `poisson_2d` | −Δu = 2π²sin(πx)sin(πy) | [0,1]² | sin(πx)sin(πy) |
| `heat_1d` | u_t = u_xx | [0,1]×[0,1] | e^(−π²t)sin(πx) |
| `harmonic_oscillator` | u″ + u = 0 | [0, 2π] | cos(x) |
| `helmholtz_1d` | −u″ − k²u = f | [0, 1] | sin(πx) |
| `advection_1d` | u_t + u_x = 0 | [0,2π]×[0,1] | sin(x − t) |

## API reference

The entire library is in `nanopinn.py`. Here's everything it exports:

```python
# ─── Derivatives (designed for use inside pde_fn) ───
jacobian(f, x)              # f:(d,)→(m,), x:(d,) → (m, d)
hessian(f, x)               # f:(d,)→(m,), x:(d,) → (m, d, d)
laplacian(f, x, out_idx=0)  # sum of d²f/dx_i²  → scalar

# ─── Network ───
MLP(layers, activation='tanh', omega_0=30.0)
# layers: [input_dim, hidden, ..., hidden, output_dim]
# activation: 'tanh' | 'siren' | 'gelu' | 'swish'

# ─── Sampling ───
sobol(n, bounds, device)     # Sobol quasi-random  → (n, d)
uniform(n, bounds, device)   # uniform random      → (n, d)
boundary(n, bounds, device)  # points on faces     → (n, d)

# ─── Training ───
train(model, pde_fn, bc_fn, domain,
      n_interior=5000, adam_lr=1e-3, adam_epochs=5000,
      lbfgs_max_iter=5000, do_compile=False,
      resample_every=500, log_every=100,
      device='cpu', seed=42)
# Returns: list[float] (loss history)
```

## tests

39 tests covering derivatives, models, sampling, and convergence against exact solutions:

```
pip install pytest
python -m pytest tests/ -v
```

Fast tests only (~30s):

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
| SIREN on Poisson 1D | < 15% |

## file structure

```
nanopinn/
├── nanopinn.py     # ~300 lines. the entire library.
├── problems.py     # ~250 lines. 6 PDEs with exact solutions.
├── train.py        # ~100 lines. nanoGPT-style training script.
├── pytest.ini
└── tests/
    ├── test_derivatives.py    # jacobian/hessian vs autograd reference
    ├── test_model.py          # MLP shapes, activations, gradient flow
    ├── test_sampling.py       # bounds, coverage, shapes
    └── test_convergence.py    # convergence to exact analytical solutions
```

## design philosophy

Stolen from the best:

| Source | What we took |
|--------|-------------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Single-file library, config-as-globals, no frameworks |
| [DeepXDE](https://github.com/lululxvi/deepxde) | PDE-as-a-Python-function returning a residual |
| [FBPINNs](https://github.com/benmoseley/FBPINNs) | `vmap` for batched per-point derivative computation |
| [PyDEns](https://github.com/analysiscenter/pydens) | Minimal API surface — `D(f, x)` style operators |

What we deliberately left out:
- No YAML/Hydra config files
- No PyTorch Lightning
- No class hierarchies for problems or solvers
- No SymPy dependency
- No multi-backend abstraction

## todos

- [ ] Fourier feature input encoding
- [ ] Weight normalization / spectral normalization
- [ ] Multi-output system support (Navier-Stokes, elasticity)
- [ ] Domain decomposition (FBPINNs-style)
- [ ] Causal training for time-dependent PDEs
- [ ] Benchmark suite with timing comparisons
- [ ] Pre-trained checkpoints for common problems
