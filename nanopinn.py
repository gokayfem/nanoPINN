"""
nanoPINN — a minimal, fast Physics-Informed Neural Network library.

The entire library in one file. Inspired by nanoGPT's philosophy:
everything readable, hackable, and fast.

Usage:
    from nanopinn import jacobian, hessian, laplacian, MLP, sobol, train

    def poisson(net, x):
        H = hessian(net, x)
        f = pi**2 * sin(pi * x[0])
        return -H[0, 0, 0] - f

    model = MLP([1, 64, 64, 1])
    train(model, pde_fn=poisson, bc_fn=my_bc, domain=[(0, 1)])
"""

import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap, functional_call
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

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


class FourierFeatures(nn.Module):
    """Random Fourier Feature encoding for high-frequency solutions.

    Maps (N, d) -> (N, 2*mapping_size) via:
        [sin(2*pi*x @ B), cos(2*pi*x @ B)]
    where B is a fixed random matrix scaled by sigma.
    """

    def __init__(self, in_features: int, mapping_size: int, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(in_features, mapping_size) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


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


def _apply_norm(module: nn.Module, norm_type: str) -> nn.Module:
    """Apply weight or spectral normalization to all Linear layers."""
    if norm_type == "none":
        return module
    if norm_type not in ("weight", "spectral"):
        raise ValueError(f"Unknown norm '{norm_type}'. Choose from: none, weight, spectral")
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if norm_type == "weight":
                setattr(module, name, weight_norm(child))
            else:
                setattr(module, name, spectral_norm(child))
        elif isinstance(child, SirenLayer):
            if norm_type == "weight":
                child.linear = weight_norm(child.linear)
            else:
                child.linear = spectral_norm(child.linear)
        else:
            _apply_norm(child, norm_type)
    return module


class MLP(nn.Module):
    """Simple feedforward network. The only network you need.

    Args:
        layers: list of layer sizes, e.g. [2, 64, 64, 64, 1]
        activation: 'tanh' | 'siren' | 'gelu' | 'swish'
        omega_0: frequency for SIREN (only used if activation='siren')
        fourier_features: number of Fourier features (0 to disable)
        fourier_sigma: frequency scale for Fourier features
        norm: 'none' | 'weight' | 'spectral'
    """

    def __init__(
        self,
        layers: list[int],
        activation: str = "tanh",
        omega_0: float = 30.0,
        fourier_features: int = 0,
        fourier_sigma: float = 1.0,
        norm: str = "none",
    ):
        super().__init__()
        acts = {"tanh": nn.Tanh, "gelu": nn.GELU, "swish": nn.SiLU, "siren": None}
        if activation not in acts:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(acts.keys())}")

        # Fourier feature encoding
        self.fourier = None
        effective_layers = list(layers)
        if fourier_features > 0:
            self.fourier = FourierFeatures(layers[0], fourier_features, fourier_sigma)
            effective_layers[0] = 2 * fourier_features

        net: list[nn.Module] = []
        for i in range(len(effective_layers) - 1):
            if activation == "siren":
                net.append(SirenLayer(effective_layers[i], effective_layers[i + 1], omega_0, is_first=(i == 0)))
            else:
                net.append(nn.Linear(effective_layers[i], effective_layers[i + 1]))
                if i < len(effective_layers) - 2:
                    net.append(acts[activation]())
        self.net = nn.Sequential(*net)

        if activation != "siren":
            self._init_weights()

        if norm != "none":
            _apply_norm(self.net, norm)

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            x = self.fourier(x)
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


def _vmapped_residuals(
    model: nn.Module, pde_fn: Callable, points: torch.Tensor
) -> torch.Tensor:
    """Compute per-point PDE residuals via vmap + functional_call."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def single_residual(x: torch.Tensor) -> torch.Tensor:
        net = _make_single_forward(model, params, buffers)
        return pde_fn(net, x)

    return vmap(single_residual)(points)


def pde_loss(
    model: nn.Module,
    pde_fn: Callable,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean squared PDE residual via vmap.

    Supports both scalar and vector residuals (multi-output systems).

    Args:
        model: neural network
        pde_fn: function(net, x) -> scalar or (R,) vector residual
        points: collocation points (N, d)

    Returns:
        (scalar_loss, per_point_residuals_squared (N,))
    """
    residuals = _vmapped_residuals(model, pde_fn, points)
    sq = residuals ** 2
    if sq.dim() == 1:
        # scalar residual: (N,)
        return sq.mean(), sq.detach()
    # vector residual: (N, R) -> reduce to (N,) for resampling
    return sq.mean(), sq.sum(dim=1).detach()


# ─── Causal training ────────────────────────────────────────────────────────


def causal_weights(
    points: torch.Tensor,
    residuals_sq: torch.Tensor,
    time_dim: int = -1,
    epsilon: float = 1.0,
    n_bins: int = 20,
) -> torch.Tensor:
    """Compute causal weights for time-dependent PDEs (Wang et al. 2022).

    Earlier time slices get higher weight. Points at times where the
    cumulative residual is large get down-weighted.

    Args:
        points: (N, d) collocation points
        residuals_sq: (N,) per-point squared residuals
        time_dim: which input dimension is time
        epsilon: causality strength (higher = more aggressive)
        n_bins: number of temporal bins

    Returns:
        (N,) tensor of weights in [0, 1]
    """
    t = points[:, time_dim].detach()
    t_min, t_max = t.min(), t.max()
    if t_max - t_min < 1e-10:
        return torch.ones(points.shape[0], device=points.device)

    # bin edges
    edges = torch.linspace(t_min.item(), t_max.item(), n_bins + 1, device=points.device)
    bin_idx = torch.bucketize(t, edges[1:-1])  # (N,) in [0, n_bins-1]

    # mean residual per bin
    bin_means = torch.zeros(n_bins, device=points.device)
    bin_counts = torch.zeros(n_bins, device=points.device)
    bin_counts.scatter_add_(0, bin_idx, torch.ones_like(t))
    bin_means.scatter_add_(0, bin_idx, residuals_sq.detach())
    safe_counts = bin_counts.clamp(min=1)
    bin_means = bin_means / safe_counts

    # cumulative residual and weights
    cumulative = torch.cumsum(bin_means, dim=0)
    # shift so first bin has weight 1 (no penalty for earliest time)
    cumulative = torch.cat([torch.zeros(1, device=points.device), cumulative[:-1]])
    bin_weights = torch.exp(-epsilon * cumulative.clamp(max=20.0))

    return bin_weights[bin_idx]


def causal_pde_loss(
    model: nn.Module,
    pde_fn: Callable,
    points: torch.Tensor,
    time_dim: int = -1,
    epsilon: float = 1.0,
    n_bins: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PDE loss with causal weighting for time-dependent problems.

    Drop-in replacement for pde_loss that prioritizes earlier times.

    Returns:
        (weighted_loss, per_point_residuals_squared (N,))
    """
    residuals = _vmapped_residuals(model, pde_fn, points)
    sq = residuals ** 2
    if sq.dim() == 1:
        res_sq_detached = sq.detach()
        res_sq_live = sq
    else:
        res_sq_detached = sq.sum(dim=1).detach()
        res_sq_live = sq.sum(dim=1)

    weights = causal_weights(points, res_sq_detached, time_dim, epsilon, n_bins)
    return (weights * res_sq_live).mean(), res_sq_detached


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


# ─── Domain decomposition (FBPINNs-style) ──────────────────────────────────


def cosine_window(center: torch.Tensor, half_width: torch.Tensor) -> Callable:
    """Create a cosine bell window function.

    Returns a function (N, d) -> (N,) with values in [0, 1],
    peaking at center and decaying to 0 at center +/- half_width.
    """
    def window(x: torch.Tensor) -> torch.Tensor:
        # normalized distance in each dimension: 0 at center, 1 at edge
        dist = ((x - center) / half_width).abs()
        # clamp to [0, 1] and apply cosine taper per dimension
        per_dim = 0.5 * (1.0 + torch.cos(math.pi * dist.clamp(max=1.0)))
        return per_dim.prod(dim=-1)  # product over dimensions

    return window


def decompose_domain(
    bounds: list[tuple[float, float]],
    n_subdomains: list[int],
    overlap: float = 0.25,
) -> list[dict]:
    """Split domain into overlapping rectangular subdomains.

    Args:
        bounds: [(lo, hi), ...] domain bounds
        n_subdomains: number of splits along each dimension
        overlap: fraction of subdomain width to overlap with neighbors

    Returns:
        list of {"center": (d,), "half_width": (d,), "bounds": [(lo, hi), ...]}
    """
    d = len(bounds)
    if len(n_subdomains) != d:
        raise ValueError(f"n_subdomains length {len(n_subdomains)} != domain dims {d}")

    # build 1D grids per dimension
    grids = []
    for dim in range(d):
        lo, hi = bounds[dim]
        n = n_subdomains[dim]
        width = (hi - lo) / n
        centers = [lo + (i + 0.5) * width for i in range(n)]
        hw = width * (0.5 + overlap)
        grids.append([(c, hw) for c in centers])

    # cartesian product of all dimensions
    subdomains = []
    for combo in itertools.product(*grids):
        center = torch.tensor([c for c, _ in combo])
        half_width = torch.tensor([hw for _, hw in combo])
        sub_bounds = [
            (max(bounds[i][0], combo[i][0] - combo[i][1]),
             min(bounds[i][1], combo[i][0] + combo[i][1]))
            for i in range(d)
        ]
        subdomains.append({
            "center": center,
            "half_width": half_width,
            "bounds": sub_bounds,
        })
    return subdomains


class DDModel(nn.Module):
    """Domain decomposition model with overlapping sub-networks.

    Creates one MLP per subdomain, blends outputs using cosine windows
    (partition of unity).

    Args:
        layers: network architecture (same for all sub-networks)
        subdomains: list from decompose_domain()
        activation: activation function
        fourier_features: Fourier encoding size (0 to disable)
        fourier_sigma: Fourier frequency scale
        norm: normalization type
    """

    def __init__(
        self,
        layers: list[int],
        subdomains: list[dict],
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_sigma: float = 1.0,
        norm: str = "none",
    ):
        super().__init__()
        self.subnets = nn.ModuleList([
            MLP(layers, activation, fourier_features=fourier_features,
                fourier_sigma=fourier_sigma, norm=norm)
            for _ in subdomains
        ])
        # register window geometry as buffers so they move with .to(device)
        centers = torch.stack([sd["center"].float() for sd in subdomains])
        half_widths = torch.stack([sd["half_width"].float() for sd in subdomains])
        self.register_buffer("_centers", centers)      # (n_sub, d)
        self.register_buffer("_half_widths", half_widths)  # (n_sub, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute all window values inline using registered buffers
        w_list = []
        for i in range(self._centers.shape[0]):
            dist = ((x - self._centers[i]) / self._half_widths[i]).abs()
            per_dim = 0.5 * (1.0 + torch.cos(math.pi * dist.clamp(max=1.0)))
            w_list.append(per_dim.prod(dim=-1))
        weights = torch.stack(w_list, dim=-1)  # (N, n_sub)
        # normalize to partition of unity
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # evaluate each sub-network
        outputs = torch.stack([net(x) for net in self.subnets], dim=-1)  # (N, out, n_sub)
        # blend: weighted sum over subdomains
        weighted = outputs * weights.unsqueeze(1)  # (N, out, n_sub)
        return weighted.sum(dim=-1)  # (N, out)


# ─── Checkpoints ────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    model: nn.Module,
    losses: list[float],
    config: dict,
    metadata: dict | None = None,
) -> None:
    """Save model checkpoint with config and training history."""
    torch.save({
        "model_state": model.state_dict(),
        "losses": losses,
        "config": config,
        "metadata": metadata or {},
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module | None = None,
    device: str = "cpu",
    weights_only: bool = True,
) -> dict:
    """Load checkpoint. Optionally load weights into a model.

    Returns the full checkpoint dict (config, losses, metadata, model_state).
    """
    ckpt = torch.load(path, map_location=device, weights_only=weights_only)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    return ckpt


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
    causal: bool = False,
    causal_epsilon: float = 1.0,
    causal_time_dim: int = -1,
    causal_n_bins: int = 20,
) -> list[float]:
    """Train PINN with Adam -> L-BFGS. Returns loss history.

    Args:
        model: neural network (already on device)
        pde_fn: function(net, x) -> scalar or vector residual (single point)
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
        causal: enable causal weighting for time-dependent PDEs
        causal_epsilon: causality strength parameter
        causal_time_dim: which input dimension is time
        causal_n_bins: number of temporal bins for causal weighting
    """
    torch.manual_seed(seed)
    losses: list[float] = []

    # sample interior points
    pts = sobol(n_interior, domain, device=device).requires_grad_(True)

    # select loss function
    if causal:
        def _loss_fn(model_, pde_fn_, pts_):
            return causal_pde_loss(model_, pde_fn_, pts_, causal_time_dim, causal_epsilon, causal_n_bins)
    else:
        _loss_fn = pde_loss

    # optionally compile
    if do_compile and device != "cpu":
        try:
            _loss_fn = torch.compile(_loss_fn, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile failed, using eager mode: {e}")

    def total_loss() -> tuple[torch.Tensor, torch.Tensor]:
        pl, res_sq = _loss_fn(model, pde_fn, pts)
        bl = bc_fn(model, device)
        return pl + bl, res_sq

    # -- Phase 1: Adam --
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

    # -- Phase 2: L-BFGS --
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
