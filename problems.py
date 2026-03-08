"""
Built-in PDE problems with exact analytical solutions.

Each problem is a function that returns a dict:
    pde:    function(net, x) -> scalar residual (for a single point)
    bc:     function(model, device) -> scalar BC loss
    domain: list of (lo, hi) bounds
    exact:  function(x_tensor) -> exact solution tensor
    layers: suggested network architecture
    name:   human-readable name

Usage:
    from problems import poisson_1d
    prob = poisson_1d()
    model = MLP(prob['layers'])
    train(model, prob['pde'], prob['bc'], prob['domain'])
"""

import math

import torch

from nanopinn import hessian, jacobian, sobol


# ─── 1. Poisson 1D ──────────────────────────────────────────────────────────
# -u''(x) = pi^2 * sin(pi*x) on (0, 1)
# u(0) = u(1) = 0
# Exact: u(x) = sin(pi*x)

def poisson_1d():
    domain = [(0.0, 1.0)]
    layers = [1, 64, 64, 64, 1]

    def pde(net, x):
        H = hessian(net, x)
        f = (math.pi ** 2) * torch.sin(torch.tensor(math.pi) * x[0])
        return -H[0, 0, 0] - f

    def bc(model, device):
        n = 200
        x0 = torch.zeros(n, 1, device=device)
        x1 = torch.ones(n, 1, device=device)
        return (model(x0) ** 2).mean() + (model(x1) ** 2).mean()

    def exact(x):
        return torch.sin(math.pi * x)

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="poisson_1d")


# ─── 2. Poisson 2D ──────────────────────────────────────────────────────────
# -Laplacian(u) = 2*pi^2 * sin(pi*x)*sin(pi*y) on (0,1)^2
# u = 0 on boundary
# Exact: u(x,y) = sin(pi*x)*sin(pi*y)

def poisson_2d():
    domain = [(0.0, 1.0), (0.0, 1.0)]
    layers = [2, 64, 64, 64, 1]

    def pde(net, x):
        H = hessian(net, x)
        u_xx = H[0, 0, 0]
        u_yy = H[0, 1, 1]
        pi_t = torch.tensor(math.pi)
        f = 2.0 * (math.pi ** 2) * torch.sin(pi_t * x[0]) * torch.sin(pi_t * x[1])
        return -(u_xx + u_yy) - f

    def bc(model, device):
        n = 100
        pts = []
        for dim in [0, 1]:
            for val in [0.0, 1.0]:
                p = sobol(n, domain, device=device)
                p = p.clone()
                p[:, dim] = val
                pts.append(p)
        all_pts = torch.cat(pts, dim=0)
        return (model(all_pts) ** 2).mean()

    def exact(x):
        return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="poisson_2d")


# ─── 3. Heat 1D ─────────────────────────────────────────────────────────────
# u_t = u_xx on (0,1) x (0,1)
# IC: u(x, 0) = sin(pi*x)
# BC: u(0, t) = u(1, t) = 0
# Exact: u(x, t) = exp(-pi^2 * t) * sin(pi*x)
# Input: x = [x_space, t]

def heat_1d():
    domain = [(0.0, 1.0), (0.0, 1.0)]
    layers = [2, 64, 64, 64, 1]

    def pde(net, x):
        J = jacobian(net, x)
        H = hessian(net, x)
        u_t = J[0, 1]
        u_xx = H[0, 0, 0]
        return u_t - u_xx

    def bc(model, device):
        n = 200
        pi_t = torch.tensor(math.pi)

        # IC: u(x, 0) = sin(pi*x)
        x_ic = sobol(n, [(0.0, 1.0)], device=device)
        t_ic = torch.zeros(n, 1, device=device)
        pts_ic = torch.cat([x_ic, t_ic], dim=1)
        target_ic = torch.sin(pi_t * x_ic)
        loss_ic = ((model(pts_ic) - target_ic) ** 2).mean()

        # BC: u(0,t) = u(1,t) = 0
        t_bc = sobol(n, [(0.0, 1.0)], device=device)
        pts_0 = torch.cat([torch.zeros(n, 1, device=device), t_bc], dim=1)
        pts_1 = torch.cat([torch.ones(n, 1, device=device), t_bc], dim=1)
        loss_bc = (model(pts_0) ** 2).mean() + (model(pts_1) ** 2).mean()

        return loss_ic + loss_bc

    def exact(x):
        return torch.exp(-(math.pi ** 2) * x[:, 1:2]) * torch.sin(math.pi * x[:, 0:1])

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="heat_1d")


# ─── 4. Harmonic Oscillator ─────────────────────────────────────────────────
# u''(x) + u(x) = 0 on (0, 2*pi)
# u(0) = 1, u'(0) = 0
# Exact: u(x) = cos(x)

def harmonic_oscillator():
    domain = [(0.0, 2.0 * math.pi)]
    layers = [1, 64, 64, 64, 1]

    def pde(net, x):
        u = net(x)
        H = hessian(net, x)
        return H[0, 0, 0] + u[0]

    def bc(model, device):
        n = 100
        # u(0) = 1
        x0 = torch.zeros(n, 1, device=device)
        loss_u0 = ((model(x0) - 1.0) ** 2).mean()

        # u'(0) = 0 via autograd
        x0_g = torch.zeros(1, 1, device=device, requires_grad=True)
        u0 = model(x0_g)
        du = torch.autograd.grad(u0, x0_g, create_graph=True)[0]
        loss_du0 = (du ** 2).mean()

        # u(2*pi) = 1  (extra pin for stability)
        x_end = torch.full((n, 1), 2.0 * math.pi, device=device)
        loss_end = ((model(x_end) - 1.0) ** 2).mean()

        return loss_u0 + loss_du0 + loss_end

    def exact(x):
        return torch.cos(x)

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="harmonic_oscillator")


# ─── 5. Helmholtz 1D ────────────────────────────────────────────────────────
# -u''(x) - k^2 * u(x) = f(x) on (0, 1)
# u(0) = u(1) = 0
# With k=1, u_exact = sin(pi*x): f = (pi^2 - 1)*sin(pi*x)
# Exact: u(x) = sin(pi*x)

def helmholtz_1d(k: float = 1.0):
    domain = [(0.0, 1.0)]
    layers = [1, 64, 64, 64, 1]
    f_coeff = math.pi ** 2 - k ** 2

    def pde(net, x):
        u = net(x)
        H = hessian(net, x)
        f = f_coeff * torch.sin(torch.tensor(math.pi) * x[0])
        return -H[0, 0, 0] - (k ** 2) * u[0] - f

    def bc(model, device):
        n = 200
        x0 = torch.zeros(n, 1, device=device)
        x1 = torch.ones(n, 1, device=device)
        return (model(x0) ** 2).mean() + (model(x1) ** 2).mean()

    def exact(x):
        return torch.sin(math.pi * x)

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="helmholtz_1d")


# ─── 6. Advection 1D ────────────────────────────────────────────────────────
# u_t + u_x = 0 on (0, 2*pi) x (0, 1)
# IC: u(x, 0) = sin(x)
# Exact: u(x, t) = sin(x - t)
# Input: x = [x_space, t]

def advection_1d():
    domain = [(0.0, 2.0 * math.pi), (0.0, 1.0)]
    layers = [2, 64, 64, 64, 1]

    def pde(net, x):
        J = jacobian(net, x)
        u_x = J[0, 0]
        u_t = J[0, 1]
        return u_t + u_x

    def bc(model, device):
        n = 300
        # IC: u(x, 0) = sin(x)
        x_ic = sobol(n, [(0.0, 2.0 * math.pi)], device=device)
        t_ic = torch.zeros(n, 1, device=device)
        pts_ic = torch.cat([x_ic, t_ic], dim=1)
        target_ic = torch.sin(x_ic)
        loss_ic = ((model(pts_ic) - target_ic) ** 2).mean()

        # Periodic BC: u(0, t) = u(2*pi, t)
        t_bc = sobol(n, [(0.0, 1.0)], device=device)
        pts_left = torch.cat([torch.zeros(n, 1, device=device), t_bc], dim=1)
        pts_right = torch.cat([torch.full((n, 1), 2.0 * math.pi, device=device), t_bc], dim=1)
        loss_periodic = ((model(pts_left) - model(pts_right)) ** 2).mean()

        return loss_ic + loss_periodic

    def exact(x):
        return torch.sin(x[:, 0:1] - x[:, 1:2])

    return dict(pde=pde, bc=bc, domain=domain, exact=exact, layers=layers, name="advection_1d")


# ─── Registry ────────────────────────────────────────────────────────────────

PROBLEMS = {
    "poisson_1d": poisson_1d,
    "poisson_2d": poisson_2d,
    "heat_1d": heat_1d,
    "harmonic_oscillator": harmonic_oscillator,
    "helmholtz_1d": helmholtz_1d,
    "advection_1d": advection_1d,
}
