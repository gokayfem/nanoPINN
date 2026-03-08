"""
nanoPINN — a minimal, fast Physics-Informed Neural Network library.

The entire library in one file. Inspired by nanoGPT's philosophy:
everything readable, hackable, and fast. ~350 lines of PyTorch.

Usage:
    from nanopinn import jacobian, hessian, laplacian, MLP, sobol, train

    def poisson(net, x):
        H = hessian(net, x)
        f = pi**2 * sin(pi * x[0])
        return -H[0, 0, 0] - f

    model = MLP([1, 64, 64, 1])
    train(model, pde_fn=poisson, bc_fn=my_bc, domain=[(0, 1)])
"""

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap, functional_call

# ─── Derivative operators ────────────────────────────────────────────────────
# These are thin wrappers around torch.func.jacrev, designed to be vmapped.
# f: (d,) -> (m,) is the network forward for a SINGLE point.
# x: (d,) is a single input point.


def jacobian(f: Callable, x: torch.Tensor) -> torch.Tensor:
    """Jacobian df_i/dx_j. f: (d,)->(m,), x: (d,). Returns (m, d)."""
    return jacrev(f)(x)


def hessian(f: Callable, x: torch.Tensor) -> torch.Tensor:
    """Hessian d²f_i/(dx_j dx_k). f: (d,)->(m,), x: (d,). Returns (m, d, d)."""
    return jacrev(jacrev(f))(x)


def laplacian(f: Callable, x: torch.Tensor, out_idx: int = 0) -> torch.Tensor:
    """Laplacian: sum_j d²f[out_idx]/dx_j². Scalar output."""
    H = hessian(f, x)
    return torch.stack([H[out_idx, j, j] for j in range(x.shape[0])]).sum()


# ─── Network architectures ──────────────────────────────────────────────────


class SirenLayer(nn.Module):
    """Sinusoidal activation layer (SIREN)."""

    def __init__(self, in_f: int, out_f: int, omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_f, out_f)
        bound = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class MLP(nn.Module):
    """Simple feedforward network. The only network you need.

    Args:
        layers: list of layer sizes, e.g. [2, 64, 64, 64, 1]
        activation: 'tanh' | 'siren' | 'gelu' | 'swish'
        omega_0: frequency for SIREN (only used if activation='siren')
    """

    def __init__(self, layers: list[int], activation: str = "tanh", omega_0: float = 30.0):
        super().__init__()
        acts = {"tanh": nn.Tanh, "gelu": nn.GELU, "swish": nn.SiLU, "siren": None}
        if activation not in acts:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(acts.keys())}")

        net: list[nn.Module] = []
        for i in range(len(layers) - 1):
            if activation == "siren":
                net.append(SirenLayer(layers[i], layers[i + 1], omega_0, is_first=(i == 0)))
            else:
                net.append(nn.Linear(layers[i], layers[i + 1]))
                if i < len(layers) - 2:
                    net.append(acts[activation]())
        self.net = nn.Sequential(*net)

        if activation != "siren":
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Sampling ────────────────────────────────────────────────────────────────


def sobol(n: int, bounds: list[tuple[float, float]], device: str = "cpu") -> torch.Tensor:
    """Sobol quasi-random points. bounds: [(lo, hi), ...]. Returns (n, d)."""
    d = len(bounds)
    engine = torch.quasirandom.SobolEngine(d, scramble=True)
    pts = engine.draw(n).to(device)
    lo = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
    hi = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)
    return pts * (hi - lo) + lo


def uniform(n: int, bounds: list[tuple[float, float]], device: str = "cpu") -> torch.Tensor:
    """Uniform random points. bounds: [(lo, hi), ...]. Returns (n, d)."""
    d = len(bounds)
    lo = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
    hi = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)
    return torch.rand(n, d, device=device) * (hi - lo) + lo


def boundary(n: int, bounds: list[tuple[float, float]], device: str = "cpu") -> torch.Tensor:
    """Points on each face of the domain. Returns (n_actual, d)."""
    d = len(bounds)
    n_per_face = max(1, n // (2 * d))
    all_pts = []
    for dim in range(d):
        for side in [0, 1]:
            pts = uniform(n_per_face, bounds, device)
            pts = pts.clone()
            pts[:, dim] = bounds[dim][side]
            all_pts.append(pts)
    return torch.cat(all_pts, dim=0)


# ─── Core: vmapped PDE loss ─────────────────────────────────────────────────


def _make_single_forward(model: nn.Module, params: dict, buffers: dict):
    """Create a single-point forward function for use inside vmap."""

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return functional_call(model, (params, buffers), x.unsqueeze(0)).squeeze(0)

    return fwd


def pde_loss(
    model: nn.Module,
    pde_fn: Callable,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean squared PDE residual via vmap.

    Args:
        model: neural network
        pde_fn: user-defined PDE residual function(net, x) -> scalar
        points: collocation points (N, d)

    Returns:
        (scalar_loss, per_point_residuals_squared)
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def single_residual(x: torch.Tensor) -> torch.Tensor:
        net = _make_single_forward(model, params, buffers)
        return pde_fn(net, x)

    residuals = vmap(single_residual)(points)
    sq = residuals ** 2
    return sq.mean(), sq.detach()


# ─── Adaptive resampling ────────────────────────────────────────────────────


def resample(
    points: torch.Tensor,
    residuals_sq: torch.Tensor,
    bounds: list[tuple[float, float]],
    ratio: float = 0.3,
) -> torch.Tensor:
    """Replace lowest-residual points with fresh Sobol samples."""
    n = points.shape[0]
    n_keep = int(n * (1.0 - ratio))
    _, idx = torch.sort(residuals_sq, descending=True)
    kept = points[idx[:n_keep]].detach()
    n_new = n - n_keep
    fresh = sobol(n_new, bounds, device=points.device)
    return torch.cat([kept, fresh], dim=0)


# ─── Training loop ──────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    pde_fn: Callable,
    bc_fn: Callable,
    domain: list[tuple[float, float]],
    *,
    n_interior: int = 5000,
    adam_lr: float = 1e-3,
    adam_epochs: int = 5000,
    lbfgs_max_iter: int = 5000,
    do_compile: bool = False,
    resample_every: int = 500,
    log_every: int = 100,
    device: str = "cpu",
    seed: int = 42,
) -> list[float]:
    """Train PINN with Adam → L-BFGS. Returns loss history.

    Args:
        model: neural network (already on device)
        pde_fn: function(net, x) -> scalar residual (single point)
        bc_fn: function(model, device) -> scalar BC loss
        domain: list of (lo, hi) bounds for each input dimension
        n_interior: number of collocation points
        adam_lr: Adam learning rate
        adam_epochs: number of Adam epochs
        lbfgs_max_iter: max L-BFGS iterations
        do_compile: whether to torch.compile the loss (CUDA only)
        resample_every: adaptive resampling interval (0 to disable)
        log_every: print interval
        device: compute device
        seed: random seed
    """
    torch.manual_seed(seed)
    losses: list[float] = []

    # sample interior points
    pts = sobol(n_interior, domain, device=device).requires_grad_(True)

    # optionally compile
    _pde_loss = pde_loss
    if do_compile and device != "cpu":
        try:
            _pde_loss = torch.compile(pde_loss, mode="reduce-overhead")
        except Exception:
            _pde_loss = pde_loss

    def total_loss() -> tuple[torch.Tensor, torch.Tensor]:
        pl, res_sq = _pde_loss(model, pde_fn, pts)
        bl = bc_fn(model, device)
        return pl + bl, res_sq

    # ── Phase 1: Adam ──
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)

    t0 = time.perf_counter()
    for epoch in range(1, adam_epochs + 1):
        optimizer.zero_grad()
        loss, res_sq = total_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        losses.append(lv)

        if epoch % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"[Adam {epoch:>5d}/{adam_epochs}] loss={lv:.4e} lr={optimizer.param_groups[0]['lr']:.2e} ({dt:.1f}s)")

        # adaptive resampling
        if resample_every > 0 and epoch % resample_every == 0:
            with torch.no_grad():
                pts_new = resample(pts, res_sq.squeeze(), domain)
                pts = pts_new.requires_grad_(True)

    dt_adam = time.perf_counter() - t0
    print(f"Adam done in {dt_adam:.1f}s, loss={losses[-1]:.4e}")

    # ── Phase 2: L-BFGS ──
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_max_iter,
        max_eval=lbfgs_max_iter,
        history_size=100,
        tolerance_grad=1e-7,
        tolerance_change=0.5 * np.finfo(np.float64).eps,
        line_search_fn="strong_wolfe",
    )

    it = 0

    def closure() -> torch.Tensor:
        nonlocal it
        lbfgs.zero_grad()
        loss, _ = total_loss()
        loss.backward()
        lv = loss.item()
        losses.append(lv)
        it += 1
        if it % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"[L-BFGS {it:>5d}] loss={lv:.4e} ({dt:.1f}s)")
        return loss

    t1 = time.perf_counter()
    lbfgs.step(closure)
    dt_lbfgs = time.perf_counter() - t1
    print(f"L-BFGS done in {dt_lbfgs:.1f}s ({it} iters), loss={losses[-1]:.4e}")
    print(f"Total: {time.perf_counter() - t0:.1f}s")

    return losses
