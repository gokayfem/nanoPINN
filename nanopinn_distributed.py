"""
nanoPINN distributed training — multi-GPU via DDP.

Splits collocation points across GPUs, aggregates losses.
Adam-only (L-BFGS is incompatible with DDP gradient sync).

Usage:
    torchrun --nproc_per_node=2 train_distributed.py --problem=poisson_2d
    torchrun --nproc_per_node=4 train_distributed.py --problem=heat_1d --causal=True
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nanopinn import pde_loss, causal_pde_loss, ntk_weights, sobol, resample


def setup(backend: str = "nccl") -> tuple[int, int]:
    """Initialize distributed process group. Returns (rank, world_size)."""
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup() -> None:
    """Destroy distributed process group."""
    dist.destroy_process_group()


def distribute_points(
    points: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """Split collocation points across GPUs. Returns local shard."""
    n = points.shape[0]
    chunk = n // world_size
    start = rank * chunk
    end = start + chunk if rank < world_size - 1 else n
    return points[start:end]


def train_distributed(
    model: nn.Module,
    pde_fn,
    bc_fn,
    domain: list[tuple[float, float]],
    *,
    extra_params: nn.Module | list[torch.nn.Parameter] | None = None,
    n_interior: int = 5000,
    adam_lr: float = 1e-3,
    adam_epochs: int = 5000,
    log_every: int = 100,
    seed: int = 42,
    causal: bool = False,
    causal_epsilon: float = 1.0,
    causal_time_dim: int = -1,
    causal_n_bins: int = 20,
    ntk_weighting: bool = False,
    ntk_every: int = 100,
    resample_every: int = 500,
) -> list[float]:
    """DDP training loop. Each rank gets a shard of collocation points.

    Must be called after setup(). Only rank 0 logs.
    Does not include L-BFGS (incompatible with DDP).
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{rank}"

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    base_model = ddp_model.module  # unwrapped for pde_loss/functional_call

    # collect all trainable parameters
    all_params = list(ddp_model.parameters())
    if extra_params is not None:
        if isinstance(extra_params, nn.Module):
            extra_params = extra_params.to(device)
            all_params = all_params + list(extra_params.parameters())
        else:
            all_params = all_params + list(extra_params)

    # all ranks generate same points, then shard
    torch.manual_seed(seed)
    all_pts = sobol(n_interior, domain, device=device)
    local_pts = distribute_points(all_pts, rank, world_size).requires_grad_(True)

    _loss_fn = causal_pde_loss if causal else pde_loss
    _ntk_w = [1.0, 1.0]

    optimizer = torch.optim.Adam(all_params, lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    losses: list[float] = []

    t0 = time.perf_counter()
    for epoch in range(1, adam_epochs + 1):
        optimizer.zero_grad()

        # PDE loss on local shard
        if causal:
            pl, res_sq = _loss_fn(
                base_model, pde_fn, local_pts,
                causal_time_dim, causal_epsilon, causal_n_bins,
            )
        else:
            pl, res_sq = _loss_fn(base_model, pde_fn, local_pts)

        # BC loss on all GPUs (cheap, needs full boundary)
        bl = bc_fn(ddp_model, device)
        loss = _ntk_w[0] * pl + _ntk_w[1] * bl
        loss.backward()
        optimizer.step()
        scheduler.step()

        # all-reduce loss for logging
        loss_val = loss.detach().clone()
        dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
        lv = (loss_val / world_size).item()
        losses.append(lv)

        if rank == 0 and epoch % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"[DDP Adam {epoch:>5d}/{adam_epochs}] loss={lv:.4e} ({dt:.1f}s)")

        # NTK rebalancing (rank 0 computes, broadcasts)
        if ntk_weighting and epoch % ntk_every == 0:
            if rank == 0:
                pl_ntk, _ = _loss_fn(base_model, pde_fn, local_pts)
                bl_ntk = bc_fn(ddp_model, device)
                _ntk_w = ntk_weights(base_model, [pl_ntk, bl_ntk])
            w_tensor = torch.tensor(_ntk_w, device=device)
            dist.broadcast(w_tensor, src=0)
            _ntk_w = w_tensor.tolist()

        # resample
        if resample_every > 0 and epoch % resample_every == 0:
            torch.manual_seed(seed + epoch)
            all_pts = sobol(n_interior, domain, device=device)
            local_pts = distribute_points(all_pts, rank, world_size).requires_grad_(True)

    if rank == 0:
        dt = time.perf_counter() - t0
        print(f"DDP Adam done in {dt:.1f}s, loss={losses[-1]:.4e}")

    return losses
