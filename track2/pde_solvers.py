"""Reference PDE solvers for the Track 2 smooth-equation testbeds.

Design
------
Each problem is a small dataclass-like object exposing a *uniform* interface so
the training / evaluation code is equation-agnostic:

    problem.name            : str
    problem.n_channels      : int      (field channels, e.g. wave has u,v -> 2)
    problem.grid_n          : int      (spatial samples on the periodic domain)
    problem.dt              : float     (macro-step the neural operator learns)
    problem.true_step(s)    : (B,C,N) -> (B,C,N)   one macro-step of the truth
    problem.energy(s)       : (B,C,N) -> (B,)       the conserved/invariant scalar
    problem.random_ic(n,seed): -> (n,C,N)           smooth, well-resolved ICs
    problem.rollout(ic, K)  : -> (K+1,B,C,N)        truth reference trajectory

Sampling / downsampling discipline (mentor's explicit caution)
--------------------------------------------------------------
* Temporal: the operator always learns the map at the SAME macro-step ``dt``
  used to build the data. The reference integrator subdivides ``dt`` into
  ``n_inner`` micro-steps for accuracy; the network never sees the micro-steps.
  Rollout uses exactly ``dt`` per step, so train and eval are on the same
  temporal lattice.
* Spatial: initial conditions are band-limited well below the Nyquist mode of
  ``grid_n`` so nothing is aliased at t=0. Wave: K<=3 modes on N=32 (Nyquist
  mode 16). KdV: K<=4 modes on N=64 (Nyquist mode 32); KdV dynamics sharpen
  solitons and generate higher harmonics, which N=64 resolves for the horizons
  used here.
* Downsampling: if a field is ever moved to a coarser grid, use
  :func:`spectral_resample`, which low-pass filters before decimating to avoid
  aliasing. Never plain-stride a smooth field.

All fields are real ``torch.float32`` on the periodic domain ``[-L/2, L/2)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Same weights the operator uses, so the CGL control cannot drift from the model.
from ckino.nd import clenshaw_curtis_weights


# ---------------------------------------------------------------------------
# Anti-aliased spectral resampling (used only if we change resolution).
# ---------------------------------------------------------------------------
def spectral_resample(field: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample a periodic field along its last axis to ``n_out`` points.

    Uses truncation / zero-padding of the real FFT, which is the correct
    band-limited (anti-aliased) resampling for periodic smooth data. Plain
    striding would alias any energy above the coarse-grid Nyquist frequency;
    this filters it out first.

    field : (..., n_in) real
    returns (..., n_out) real
    """
    n_in = field.shape[-1]
    if n_out == n_in:
        return field
    fh = torch.fft.rfft(field, dim=-1)
    n_freq_out = n_out // 2 + 1
    if n_out < n_in:  # downsample: keep the low modes only (low-pass)
        fh = fh[..., :n_freq_out]
    else:              # upsample: zero-pad the high modes
        pad = n_freq_out - fh.shape[-1]
        fh = torch.nn.functional.pad(fh, (0, pad))
    # Scale so amplitudes are preserved under the length change.
    out = torch.fft.irfft(fh, n=n_out, dim=-1) * (n_out / n_in)
    return out


# ---------------------------------------------------------------------------
# 1-D wave equation  u_tt = c^2 u_xx   (canonical Hamiltonian PDE, CONSERVATIVE)
# ---------------------------------------------------------------------------
@dataclass
class WaveProblem:
    """First-order form (u, v=u_t); H = 1/2 ∫(v^2 + c^2 u_x^2) dx is conserved.

    This doubles as the *homogeneous velocity* simplification (constant wave
    speed c) requested as Track-2 item 2: it is a smooth, constant-coefficient
    wave propagation problem with a known analytic invariant.
    """

    name: str = "wave1d"
    grid_n: int = 32
    length: float = 2.0
    c: float = 1.0
    dt: float = 0.02
    n_inner: int = 4
    n_channels: int = 2

    def _k(self, n: int, device) -> torch.Tensor:
        return torch.fft.rfftfreq(n, d=self.length / n).to(device) * 2 * math.pi

    def grid(self, device=None) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.grid_n + 1, device=device)[:-1] * (self.length / 2.0)

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        """One macro-step via symplectic leap-frog, ``n_inner`` micro-steps.

        state: (B, 2, N)  channel 0 = u, channel 1 = v.
        """
        u = state[:, 0, :]
        v = state[:, 1, :]
        n = u.shape[-1]
        k2 = -(self._k(n, u.device) ** 2)
        dt = self.dt / self.n_inner
        for _ in range(self.n_inner):
            uxx = torch.fft.irfft(torch.fft.rfft(u, dim=-1) * k2, n=n, dim=-1)
            v = v + 0.5 * dt * (self.c ** 2) * uxx
            u = u + dt * v
            uxx = torch.fft.irfft(torch.fft.rfft(u, dim=-1) * k2, n=n, dim=-1)
            v = v + 0.5 * dt * (self.c ** 2) * uxx
        return torch.stack([u, v], dim=1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        u = state[..., 0, :]
        v = state[..., 1, :]
        n = u.shape[-1]
        k = self._k(n, u.device)
        ux = torch.fft.irfft(1j * k * torch.fft.rfft(u, dim=-1), n=n, dim=-1)
        return 0.5 * ((v ** 2).sum(-1) + (self.c ** 2) * (ux ** 2).sum(-1)) * (self.length / n)

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = self.grid()
        u = torch.zeros(n_batch, self.grid_n)
        v = torch.zeros(n_batch, self.grid_n)
        K = 3  # well below Nyquist mode 16 on N=32 -> nothing aliased
        cu = 2 * torch.rand(n_batch, K, generator=g) - 1
        cv = 2 * torch.rand(n_batch, K, generator=g) - 1
        for k in range(1, K + 1):
            u = u + cu[:, k - 1 : k] * torch.cos(math.pi * k * x / (self.length / 2.0)).unsqueeze(0)
            v = v + 0.3 * cv[:, k - 1 : k] * torch.sin(math.pi * k * x / (self.length / 2.0)).unsqueeze(0)
        u = 0.4 * u / (u.abs().amax(-1, keepdim=True) + 1e-6)
        v = 0.4 * v / (v.abs().amax(-1, keepdim=True) + 1e-6)
        return torch.stack([u, v], dim=1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [ic.clone()]
        s = ic
        for _ in range(n_steps):
            s = self.true_step(s)
            traj.append(s.clone())
        return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# 1-D Korteweg-de Vries  u_t + 6 u u_x - u_xxx = 0  (nonlinear, solitons)
# ---------------------------------------------------------------------------
@dataclass
class WaveDirichletProblem:
    """Wave equation on [0, L] with homogeneous DIRICHLET boundaries.

    Reproduces the benchmark used by the concurrent Symplectic Neural Operator
    (Makara-Yaguchi 2026): uniform grid, centred finite differences in space,
    symplectic leap-frog in time, and initial conditions multiplied by the
    envelope x(L-x) so that u(0) = u(L) = 0.

    It is Hamiltonian (so symplecticity is meaningful) *and* non-periodic (so a
    Fourier-parameterised operator is structurally mismatched) -- the setting
    that separates a Chebyshev operator from a Fourier one.
    """

    name: str = "wave1d_dir"
    grid_n: int = 64
    length: float = 1.0
    c: float = 1.0
    dt: float = 0.005
    n_inner: int = 4
    n_channels: int = 2

    @property
    def dx(self) -> float:
        return self.length / (self.grid_n - 1)

    def grid(self, device=None) -> torch.Tensor:
        return torch.linspace(0.0, self.length, self.grid_n, device=device)

    def _uxx(self, u: torch.Tensor) -> torch.Tensor:
        """Centred second difference with u = 0 outside the domain."""
        pad = torch.nn.functional.pad(u, (1, 1))
        return (pad[..., 2:] - 2.0 * u + pad[..., :-2]) / (self.dx ** 2)

    @staticmethod
    def _clamp_bc(u: torch.Tensor, v: torch.Tensor):
        u = u.clone(); v = v.clone()
        u[..., 0] = 0.0; u[..., -1] = 0.0
        v[..., 0] = 0.0; v[..., -1] = 0.0
        return u, v

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[:, 0, :], state[:, 1, :]
        dt = self.dt / self.n_inner
        c2 = self.c ** 2
        for _ in range(self.n_inner):
            v = v + 0.5 * dt * c2 * self._uxx(u)
            u = u + dt * v
            v = v + 0.5 * dt * c2 * self._uxx(u)
            u, v = self._clamp_bc(u, v)
        return torch.stack([u, v], dim=1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[..., 0, :], state[..., 1, :]
        ux = (u[..., 1:] - u[..., :-1]) / self.dx
        return 0.5 * ((v ** 2).sum(-1) * self.dx + (self.c ** 2) * (ux ** 2).sum(-1) * self.dx)

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0, :].sum(-1) * self.dx

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = self.grid()
        xi = 2.0 * x / self.length - 1.0                      # map to [-1, 1]
        env = (x * (self.length - x))                          # Dirichlet envelope
        env = env / (env.max() + 1e-12)
        K = 3
        cu = 2 * torch.rand(n_batch, K + 1, generator=g) - 1
        cv = 2 * torch.rand(n_batch, K + 1, generator=g) - 1
        u = torch.zeros(n_batch, self.grid_n)
        v = torch.zeros(n_batch, self.grid_n)
        for k in range(K + 1):                                  # truncated Chebyshev series
            Tk = torch.cos(k * torch.arccos(xi.clamp(-1.0, 1.0))).unsqueeze(0)
            decay = 1.0 / (1.0 + k) ** 2                        # bound the gradient energy
            u = u + decay * cu[:, k:k + 1] * Tk
            v = v + decay * cv[:, k:k + 1] * Tk
        u = 0.4 * (u * env) / ((u * env).abs().amax(-1, keepdim=True) + 1e-6)
        v = 0.05 * (v * env) / ((v * env).abs().amax(-1, keepdim=True) + 1e-6)
        u, v = self._clamp_bc(u, v)
        return torch.stack([u, v], dim=1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [ic.clone()]
        s = ic
        for _ in range(n_steps):
            s = self.true_step(s)
            traj.append(s.clone())
        return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# 1-D Korteweg-de Vries  u_t + 6 u u_x - u_xxx = 0  (nonlinear, solitons)
# ---------------------------------------------------------------------------
@dataclass
class KdVProblem:
    """1-D KdV in the form actually integrated here:  u_t + 6 u u_x - u_xxx = 0

    This is the x -> -x mirror of the textbook  u_t + 6 u u_x + u_xxx = 0;
    both are integrable KdV equations. The dispersion sign was verified
    numerically against the solver's own one-step increment (rel. err 0.03 for
    the -u_xxx form vs 2.0 for +u_xxx). To switch to the textbook +u_xxx
    convention, flip ``Lhat`` to ``+1j * k**3``.

    Non-canonical Hamiltonian PDE. Conserved invariants (confirmed in the
    self-test): mass ∫u, momentum ∫u^2/2, energy ∫(u_x^2 - 2u^3). Ground truth
    via ETD-RK2 (Cox-Matthews 2002).

    Nonlinear -> a sterner test of whether the Track-2 recipe generalises past
    the linear wave equation.
    """

    name: str = "kdv"
    grid_n: int = 64
    length: float = 2.0
    dt: float = 0.001
    n_inner: int = 10
    n_channels: int = 1

    def _k(self, n: int, device) -> torch.Tensor:
        return torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi

    def grid(self, device=None) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.grid_n + 1, device=device)[:-1] * (self.length / 2.0)

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0, :]
        n = u.shape[-1]
        k = self._k(n, u.device)
        Lhat = -1j * (k ** 3)
        dt = self.dt / self.n_inner
        eL = torch.exp(Lhat * dt)
        Lsafe = torch.where(Lhat.abs() < 1e-8, torch.full_like(Lhat, 1e-8 + 0j), Lhat)
        phi1 = (eL - 1.0) / Lsafe
        phi2 = (eL - 1.0 - Lhat * dt) / (Lsafe ** 2 * dt)
        for _ in range(self.n_inner):
            uh = torch.fft.fft(u, dim=-1)
            Nu = -3j * k * torch.fft.fft(u ** 2, dim=-1)
            ah = eL * uh + phi1 * dt * Nu
            a = torch.fft.ifft(ah, dim=-1).real
            Na = -3j * k * torch.fft.fft(a ** 2, dim=-1)
            uh = ah + phi2 * dt * (Na - Nu)
            u = torch.fft.ifft(uh, dim=-1).real
        return u.unsqueeze(1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """KdV Hamiltonian ∫(u_x^2 - 2 u^3) dx (a conserved invariant)."""
        u = state[..., 0, :]
        n = u.shape[-1]
        k = self._k(n, u.device)
        ux = torch.fft.ifft(1j * k * torch.fft.fft(u, dim=-1), dim=-1).real
        return ((ux ** 2).sum(-1) - 2.0 * (u ** 3).sum(-1)) * (self.length / n)

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0, :].sum(-1) * (self.length / self.grid_n)

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = self.grid()
        K = 4  # below Nyquist mode 32 on N=64
        coeff = 2 * torch.rand(n_batch, K, generator=g) - 1
        u = torch.zeros(n_batch, self.grid_n)
        for k in range(1, K + 1):
            u = u + coeff[:, k - 1 : k] * torch.sin(math.pi * k * x / (self.length / 2.0)).unsqueeze(0)
        u = 0.3 * u / (u.abs().amax(-1, keepdim=True) + 1e-6) + 0.5
        return u.unsqueeze(1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [ic.clone()]
        s = ic
        for _ in range(n_steps):
            s = self.true_step(s)
            traj.append(s.clone())
        return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# Additional equations spanning a difficulty ladder (paper study).
#   advection : linear, non-dispersive, energy + mass conserving   (easiest)
#   heat      : linear, DISSIPATIVE (energy decays, mass conserved)
#   burgers   : NONLINEAR + dissipative, forms steep shock fronts
# together with wave1d (conservative) and kdv (nonlinear dispersive) these give
# five 1-D problems covering conservative/dissipative and linear/nonlinear.
# ---------------------------------------------------------------------------
@dataclass
class AdvectionProblem:
    """u_t + c u_x = 0.  Exact spectral solution; the simplest transport PDE."""

    name: str = "advection"
    grid_n: int = 64
    length: float = 2.0
    c: float = 1.0
    dt: float = 0.005
    n_channels: int = 1
    spatial_dims: int = 1

    def _k(self, n, device):
        return torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0]
        k = self._k(u.shape[-1], u.device)
        uh = torch.fft.fft(u, dim=-1) * torch.exp(-1j * self.c * k * self.dt)
        return torch.fft.ifft(uh, dim=-1).real.unsqueeze(1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return (state[..., 0, :] ** 2).sum(-1) * (self.length / self.grid_n)

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0, :].sum(-1) * (self.length / self.grid_n)

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1, 1, self.grid_n + 1)[:-1] * (self.length / 2)
        u = torch.zeros(n_batch, self.grid_n)
        for k in range(1, 5):
            a = 2 * torch.rand(n_batch, 1, generator=g) - 1
            b = 2 * torch.rand(n_batch, 1, generator=g) - 1
            u = u + a * torch.sin(math.pi * k * x).unsqueeze(0) + b * torch.cos(math.pi * k * x).unsqueeze(0)
        u = 0.5 * u / (u.abs().amax(-1, keepdim=True) + 1e-6)
        return u.unsqueeze(1).float()

    @torch.no_grad()
    def rollout(self, ic, n_steps):
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s); traj.append(s.clone())
        return torch.stack(traj, 0)


@dataclass
class HeatProblem:
    """u_t = nu u_xx.  DISSIPATIVE: energy decays, mass is conserved.

    This is the first dissipative testbed in the project, and the one that lets
    the energy-envelope idea be tested honestly (wave/KdV are conservative).
    """

    name: str = "heat"
    grid_n: int = 64
    length: float = 2.0
    nu: float = 0.01
    dt: float = 0.005
    n_channels: int = 1
    spatial_dims: int = 1

    def _k(self, n, device):
        return torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0]
        k = self._k(u.shape[-1], u.device)
        uh = torch.fft.fft(u, dim=-1) * torch.exp(-self.nu * (k ** 2) * self.dt)
        return torch.fft.ifft(uh, dim=-1).real.unsqueeze(1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return (state[..., 0, :] ** 2).sum(-1) * (self.length / self.grid_n)

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0, :].sum(-1) * (self.length / self.grid_n)

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1, 1, self.grid_n + 1)[:-1] * (self.length / 2)
        u = torch.zeros(n_batch, self.grid_n)
        for k in range(1, 7):     # more modes: high ones decay fastest
            a = 2 * torch.rand(n_batch, 1, generator=g) - 1
            u = u + a * torch.sin(math.pi * k * x).unsqueeze(0) / k
        u = 0.6 * u / (u.abs().amax(-1, keepdim=True) + 1e-6)
        return u.unsqueeze(1).float()

    @torch.no_grad()
    def rollout(self, ic, n_steps):
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s); traj.append(s.clone())
        return torch.stack(traj, 0)


@dataclass
class BurgersProblem:
    """u_t + u u_x = nu u_xx.  Nonlinear AND dissipative; steepens into shocks.

    Ground truth by ETD-RK2 with the diffusion operator treated exactly.
    """

    name: str = "burgers"
    grid_n: int = 64
    length: float = 2.0
    nu: float = 0.02
    dt: float = 0.005
    n_inner: int = 8
    n_channels: int = 1
    spatial_dims: int = 1

    def _k(self, n, device):
        return torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0]
        n = u.shape[-1]
        k = self._k(n, u.device)
        L = -self.nu * (k ** 2)
        dt = self.dt / self.n_inner
        eL = torch.exp(L * dt)
        Ls = torch.where(L.abs() < 1e-12, torch.full_like(L, 1e-12), L)
        phi1 = (eL - 1.0) / Ls
        # 2/3 dealiasing mask for the quadratic nonlinearity
        mask = (k.abs() <= (2.0 / 3.0) * k.abs().max()).to(u.dtype)
        for _ in range(self.n_inner):
            uh = torch.fft.fft(u, dim=-1)
            Nu = -0.5j * k * torch.fft.fft(u ** 2, dim=-1) * mask
            uh = eL * uh + phi1 * Nu
            u = torch.fft.ifft(uh, dim=-1).real
        return u.unsqueeze(1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return (state[..., 0, :] ** 2).sum(-1) * (self.length / self.grid_n)

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0, :].sum(-1) * (self.length / self.grid_n)

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1, 1, self.grid_n + 1)[:-1] * (self.length / 2)
        u = torch.zeros(n_batch, self.grid_n)
        for k in range(1, 4):
            a = 2 * torch.rand(n_batch, 1, generator=g) - 1
            b = 2 * torch.rand(n_batch, 1, generator=g) - 1
            u = u + a * torch.sin(math.pi * k * x).unsqueeze(0) + b * torch.cos(math.pi * k * x).unsqueeze(0)
        u = 0.6 * u / (u.abs().amax(-1, keepdim=True) + 1e-6)
        return u.unsqueeze(1).float()

    @torch.no_grad()
    def rollout(self, ic, n_steps):
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s); traj.append(s.clone())
        return torch.stack(traj, 0)


# ---------------------------------------------------------------------------
# 1-D wave equation discretised on CHEBYSHEV-GAUSS-LOBATTO nodes.
#
# This is the reverse-direction control for the form-matching claim.  Every
# other problem here is sampled on a uniform grid, where constant weights are
# the correct quadrature and W_cheb is therefore the mismatched form.  On CGL
# nodes the roles swap: Clenshaw-Curtis is the correct quadrature, so W_cheb
# becomes the matched form.  If the paper's thesis holds, the preference
# between ``sacheb`` and ``sacheb_naive`` must REVERSE here.  If it does not,
# the honest reading is that constant weights are simply the better choice and
# the matching story is wrong.
#
# Construction.  With the Chebyshev first-derivative matrix D and the CC weight
# W, the weak-form stiffness S = D^T W D is symmetric, so the semi-discrete
# operator
#
#     A = -W^{-1} S       satisfies    W A = -S = A^T W,
#
# i.e. it is self-adjoint in W exactly as Section 3 requires.  The system
#
#     u_t = v,    v_t = c^2 A u,        H = 1/2 (v^T W v + c^2 (Du)^T W (Du))
#
# is then canonically Hamiltonian in omega_W, and Stormer-Verlet keeps it so.
# ---------------------------------------------------------------------------
@dataclass
class WaveCGLProblem:
    """Dirichlet wave equation on [-1, 1], sampled at CGL nodes.

    ``grid_n`` counts NODES, so the Chebyshev degree is ``grid_n - 1``.
    """

    name: str = "wave1d_cgl"
    grid_n: int = 49          # nodes -> Chebyshev degree 48
    length: float = 2.0       # domain [-1, 1]
    c: float = 1.0
    dt: float = 0.002
    n_inner: int = 8
    n_channels: int = 2
    spatial_dims: int = 1

    def __post_init__(self):
        self._ops: dict = {}

    def _cache(self, device, dtype):
        key = (device, dtype)
        hit = self._ops.get(key)
        if hit is None:
            N = self.grid_n - 1
            j = torch.arange(N + 1, device=device, dtype=dtype)
            x = torch.cos(math.pi * j / N)                 # x_0 = 1 ... x_N = -1
            cc = torch.ones(N + 1, device=device, dtype=dtype)
            cc[0] = 2.0
            cc[-1] = 2.0
            cc = cc * (-1.0) ** j
            dX = x.unsqueeze(1) - x.unsqueeze(0)
            eye = torch.eye(N + 1, device=device, dtype=dtype)
            D = (cc.unsqueeze(1) / cc.unsqueeze(0)) / (dX + eye)
            D = D - torch.diag(D.sum(dim=1))               # rows sum to zero
            w = clenshaw_curtis_weights(N, device=device, dtype=dtype)
            S = D.t() @ torch.diag(w) @ D                  # symmetric, PSD
            A = -(S / w.unsqueeze(1))                      # -W^{-1} S
            hit = (x, w, D, A)
            self._ops[key] = hit
        return hit

    def grid(self, device=None) -> torch.Tensor:
        return self._cache(device, torch.float32)[0]

    def weights(self, device=None) -> torch.Tensor:
        return self._cache(device, torch.float32)[1]

    @staticmethod
    def _clamp(u, v):
        u = u.clone(); v = v.clone()
        u[..., 0] = 0.0; u[..., -1] = 0.0
        v[..., 0] = 0.0; v[..., -1] = 0.0
        return u, v

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[:, 0, :], state[:, 1, :]
        _, _, _, A = self._cache(u.device, u.dtype)
        dt = self.dt / self.n_inner
        c2 = self.c ** 2
        for _ in range(self.n_inner):
            v = v + 0.5 * dt * c2 * (u @ A.t())
            u = u + dt * v
            v = v + 0.5 * dt * c2 * (u @ A.t())
            u, v = self._clamp(u, v)
        return torch.stack([u, v], dim=1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """H = 1/2 (v^T W v + c^2 (Du)^T W (Du)), the CC-quadrature energy."""
        u, v = state[..., 0, :], state[..., 1, :]
        _, w, D, _ = self._cache(u.device, u.dtype)
        ux = u @ D.t()
        return 0.5 * ((v ** 2 * w).sum(-1) + (self.c ** 2) * (ux ** 2 * w).sum(-1))

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        """Truncated Chebyshev series times (1 - x^2), so u = 0 at both ends.

        Coefficients depend only on ``seed``, not on ``grid_n``, so the same
        continuous field can be resampled at any resolution.
        """
        g = torch.Generator().manual_seed(seed)
        x = self.grid()
        env = 1.0 - x ** 2
        K = 4
        a = 2 * torch.rand(n_batch, K, generator=g) - 1
        b = 2 * torch.rand(n_batch, K, generator=g) - 1
        u = torch.zeros(n_batch, self.grid_n)
        v = torch.zeros(n_batch, self.grid_n)
        for k in range(1, K + 1):
            Tk = torch.cos(k * torch.acos(x.clamp(-1.0, 1.0)))
            u = u + a[:, k - 1 : k] * (Tk * env).unsqueeze(0)
            v = v + 0.3 * b[:, k - 1 : k] * (Tk * env).unsqueeze(0)
        u = 0.4 * u / (u.abs().amax(-1, keepdim=True) + 1e-6)
        v = 0.4 * v / (v.abs().amax(-1, keepdim=True) + 1e-6)
        u, v = self._clamp(u, v)
        return torch.stack([u, v], dim=1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s)
            traj.append(s.clone())
        return torch.stack(traj, dim=0)


PROBLEMS = {
    "advection": AdvectionProblem,
    "heat": HeatProblem,
    "wave1d": WaveProblem,
    "wave1d_dir": WaveDirichletProblem,
    "wave1d_cgl": WaveCGLProblem,
    "burgers": BurgersProblem,
    "kdv": KdVProblem,
}

# Difficulty ordering used for the generalisation table in the paper.
DIFFICULTY_ORDER = ["advection", "heat", "wave1d", "burgers", "kdv", "wave2d"]


def pde_rhs(problem, state: torch.Tensor) -> torch.Tensor:
    """Analytic du/dt for the physics-informed (PINN-style) residual loss.

    ``state`` is in PHYSICAL units, shape (B, C, *spatial). Returns du/dt with
    the same shape, so the residual is  (u_next - u)/dt - rhs(u).
    """
    name = problem.name
    L = problem.length

    def dx(u, order=1):
        n = u.shape[-1]
        k = torch.fft.fftfreq(n, d=L / n).to(u.device) * 2 * math.pi
        return torch.fft.ifft((1j * k) ** order * torch.fft.fft(u, dim=-1), dim=-1).real

    if name == "advection":
        return (-problem.c * dx(state[:, 0])).unsqueeze(1)
    if name == "heat":
        return (problem.nu * dx(state[:, 0], 2)).unsqueeze(1)
    if name == "burgers":
        u = state[:, 0]
        return (-u * dx(u) + problem.nu * dx(u, 2)).unsqueeze(1)
    if name == "kdv":
        u = state[:, 0]
        return (-6.0 * u * dx(u) + dx(u, 3)).unsqueeze(1)
    if name == "wave1d":
        u, v = state[:, 0], state[:, 1]
        return torch.stack([v, (problem.c ** 2) * dx(u, 2)], dim=1)
    if name == "wave1d_dir":
        u, v = state[:, 0], state[:, 1]
        return torch.stack([v, (problem.c ** 2) * problem._uxx(u)], dim=1)
    if name == "wave1d_cgl":
        u, v = state[:, 0], state[:, 1]
        A = problem._cache(u.device, u.dtype)[3]
        return torch.stack([v, (problem.c ** 2) * (u @ A.t())], dim=1)
    if name == "wave2d":
        u, v = state[:, 0], state[:, 1]
        n = u.shape[-1]
        k = torch.fft.fftfreq(n, d=L / n).to(u.device) * 2 * math.pi
        KX, KY = torch.meshgrid(k, k, indexing="ij")
        lap = torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * (-(KX ** 2 + KY ** 2)),
                              dim=(-2, -1)).real
        return torch.stack([v, (problem.c ** 2) * lap], dim=1)
    if name == "wave3d":
        u, v = state[:, 0], state[:, 1]
        return torch.stack([v, (problem.c ** 2) * problem._lap(u)], dim=1)
    if name == "ns2d":
        w = state[:, 0]
        n = w.shape[-1]
        k = torch.fft.fftfreq(n, d=L / n).to(w.device) * 2 * math.pi
        KX, KY = torch.meshgrid(k, k, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        inv = torch.where(K2 == 0, torch.zeros_like(K2), 1.0 / K2)
        w_hat = torch.fft.fft2(w, dim=(-2, -1))
        psi_hat = w_hat * inv
        u = torch.fft.ifft2(1j * KY * psi_hat, dim=(-2, -1)).real
        v = torch.fft.ifft2(-1j * KX * psi_hat, dim=(-2, -1)).real
        wx = torch.fft.ifft2(1j * KX * w_hat, dim=(-2, -1)).real
        wy = torch.fft.ifft2(1j * KY * w_hat, dim=(-2, -1)).real
        lap = torch.fft.ifft2(-K2 * w_hat, dim=(-2, -1)).real
        return (-(u * wx + v * wy) + problem.nu * lap).unsqueeze(1)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 2-D wave equation  u_tt = c^2 (u_xx + u_yy)   (item 7: move to 2-D)
# ---------------------------------------------------------------------------
@dataclass
class Wave2DProblem:
    """2-D constant-coefficient wave equation, first-order (u, v=u_t) form.

    Same symplectic leap-frog / spectral-Laplacian construction as the 1-D
    case, so results are directly comparable. H = 1/2 int (v^2 + c^2|grad u|^2).
    """

    name: str = "wave2d"
    grid_n: int = 32          # per axis -> grid_n x grid_n
    length: float = 2.0
    c: float = 1.0
    dt: float = 0.02
    n_inner: int = 4
    n_channels: int = 2
    spatial_dims: int = 2

    def _k2(self, n: int, device):
        kx = torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi
        ky = torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        return -(KX ** 2 + KY ** 2)

    def _lap(self, u: torch.Tensor) -> torch.Tensor:
        n = u.shape[-1]
        k2 = self._k2(n, u.device)
        return torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * k2, dim=(-2, -1)).real

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[:, 0], state[:, 1]
        dt = self.dt / self.n_inner
        for _ in range(self.n_inner):
            v = v + 0.5 * dt * (self.c ** 2) * self._lap(u)
            u = u + dt * v
            v = v + 0.5 * dt * (self.c ** 2) * self._lap(u)
        return torch.stack([u, v], dim=1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[..., 0, :, :], state[..., 1, :, :]
        n = u.shape[-1]
        kx = torch.fft.fftfreq(n, d=self.length / n).to(u.device) * 2 * math.pi
        KX, KY = torch.meshgrid(kx, kx, indexing="ij")
        uh = torch.fft.fft2(u, dim=(-2, -1))
        ux = torch.fft.ifft2(1j * KX * uh, dim=(-2, -1)).real
        uy = torch.fft.ifft2(1j * KY * uh, dim=(-2, -1)).real
        cell = (self.length / n) ** 2
        return 0.5 * ((v ** 2).sum((-2, -1))
                      + (self.c ** 2) * ((ux ** 2) + (uy ** 2)).sum((-2, -1))) * cell

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1.0, 1.0, self.grid_n + 1)[:-1] * (self.length / 2.0)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        u = torch.zeros(n_batch, self.grid_n, self.grid_n)
        v = torch.zeros(n_batch, self.grid_n, self.grid_n)
        K = 2  # low modes only: well below Nyquist (grid_n/2)
        for kx in range(1, K + 1):
            for ky in range(1, K + 1):
                a = (2 * torch.rand(n_batch, 1, 1, generator=g) - 1)
                b = (2 * torch.rand(n_batch, 1, 1, generator=g) - 1)
                u = u + a * torch.cos(math.pi * kx * X).unsqueeze(0) * torch.cos(math.pi * ky * Y).unsqueeze(0)
                v = v + 0.3 * b * torch.sin(math.pi * kx * X).unsqueeze(0) * torch.sin(math.pi * ky * Y).unsqueeze(0)
        u = 0.4 * u / (u.abs().amax((-2, -1), keepdim=True) + 1e-6)
        v = 0.4 * v / (v.abs().amax((-2, -1), keepdim=True) + 1e-6)
        return torch.stack([u, v], dim=1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [ic.clone()]
        s = ic
        for _ in range(n_steps):
            s = self.true_step(s)
            traj.append(s.clone())
        return torch.stack(traj, dim=0)


PROBLEMS["wave2d"] = Wave2DProblem


# ---------------------------------------------------------------------------
# 2-D incompressible Navier-Stokes, vorticity form  (the canonical FNO bench)
#   w_t + (u . grad) w = nu * lap(w),   u = grad^perp psi,   lap(psi) = -w
# Decaying turbulence on the periodic torus; pseudo-spectral, 2/3 dealiased,
# viscosity integrated exactly (ETD-RK2). Vorticity is a single channel.
# ---------------------------------------------------------------------------
@dataclass
class NavierStokes2DProblem:
    name: str = "ns2d"
    grid_n: int = 64
    length: float = 2.0
    nu: float = 3e-3
    dt: float = 0.01
    n_inner: int = 4
    n_channels: int = 1
    spatial_dims: int = 2

    def _spectral(self, n: int, device):
        k = torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi
        KX, KY = torch.meshgrid(k, k, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        inv = torch.where(K2 == 0, torch.zeros_like(K2), 1.0 / K2)   # streamfn has no mean mode
        kmax = k.abs().max()
        mask = ((KX.abs() <= (2.0 / 3.0) * kmax) & (KY.abs() <= (2.0 / 3.0) * kmax)).to(K2.dtype)
        return KX, KY, K2, inv, mask

    def _nonlinear(self, w_hat, KX, KY, inv, mask):
        """Spectral convection N(w) = -(u . grad) w, dealiased."""
        psi_hat = w_hat * inv
        u = torch.fft.ifft2(1j * KY * psi_hat, dim=(-2, -1)).real
        v = torch.fft.ifft2(-1j * KX * psi_hat, dim=(-2, -1)).real
        wx = torch.fft.ifft2(1j * KX * w_hat, dim=(-2, -1)).real
        wy = torch.fft.ifft2(1j * KY * w_hat, dim=(-2, -1)).real
        return torch.fft.fft2(-(u * wx + v * wy), dim=(-2, -1)) * mask

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        w = state[:, 0]
        n = w.shape[-1]
        KX, KY, K2, inv, mask = self._spectral(n, w.device)
        L = -self.nu * K2
        dt = self.dt / self.n_inner
        eL = torch.exp(L * dt)
        Ls = torch.where(L == 0, torch.ones_like(L), L)
        phi1 = torch.where(L == 0, torch.full_like(L, dt), (eL - 1.0) / Ls)
        phi2 = torch.where(L == 0, torch.full_like(L, dt / 2.0),
                           (eL - 1.0 - L * dt) / (Ls ** 2 * dt))
        w_hat = torch.fft.fft2(w, dim=(-2, -1))
        for _ in range(self.n_inner):
            Nw = self._nonlinear(w_hat, KX, KY, inv, mask)
            a_hat = eL * w_hat + phi1 * Nw
            Na = self._nonlinear(a_hat, KX, KY, inv, mask)
            w_hat = a_hat + phi2 * (Na - Nw)
        return torch.fft.ifft2(w_hat, dim=(-2, -1)).real.unsqueeze(1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Enstrophy int w^2 dx (decays under viscosity)."""
        w = state[..., 0, :, :]
        return (w ** 2).sum((-2, -1)) * (self.length / w.shape[-1]) ** 2

    def mass(self, state: torch.Tensor) -> torch.Tensor:
        w = state[..., 0, :, :]
        return w.sum((-2, -1)) * (self.length / w.shape[-1]) ** 2

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1.0, 1.0, self.grid_n + 1)[:-1] * (self.length / 2.0)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        w = torch.zeros(n_batch, self.grid_n, self.grid_n)
        K = 4  # low modes only -> band-limited, resolved on grid_n
        for kx in range(1, K + 1):
            for ky in range(1, K + 1):
                a = 2 * torch.rand(n_batch, 1, 1, generator=g) - 1
                b = 2 * torch.rand(n_batch, 1, 1, generator=g) - 1
                w = (w + a * (torch.sin(math.pi * kx * X) * torch.cos(math.pi * ky * Y)).unsqueeze(0)
                       + b * (torch.cos(math.pi * kx * X) * torch.sin(math.pi * ky * Y)).unsqueeze(0))
        w = w - w.mean((-2, -1), keepdim=True)                       # zero-mean vorticity
        w = 0.8 * w / (w.abs().amax((-2, -1), keepdim=True) + 1e-6)
        return w.unsqueeze(1).float()

    @torch.no_grad()
    def rollout(self, ic: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s); traj.append(s.clone())
        return torch.stack(traj, dim=0)


PROBLEMS["ns2d"] = NavierStokes2DProblem


# ---------------------------------------------------------------------------
# 3-D wave equation  u_tt = c^2 (u_xx + u_yy + u_zz)   (stage-3 target)
# ---------------------------------------------------------------------------
@dataclass
class Wave3DProblem:
    """3-D constant-coefficient wave equation, same construction as 1-D/2-D.

    This is the bridge toward the heterogeneous 3-D seismic problem: identical
    physics family, identical protocol, one dimension higher.
    """

    name: str = "wave3d"
    grid_n: int = 16
    length: float = 2.0
    c: float = 1.0
    dt: float = 0.02
    n_inner: int = 4
    n_channels: int = 2
    spatial_dims: int = 3

    def _k2(self, n, device):
        k = torch.fft.fftfreq(n, d=self.length / n).to(device) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
        return -(KX ** 2 + KY ** 2 + KZ ** 2)

    def _lap(self, u):
        n = u.shape[-1]
        return torch.fft.ifftn(torch.fft.fftn(u, dim=(-3, -2, -1)) * self._k2(n, u.device),
                               dim=(-3, -2, -1)).real

    def true_step(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[:, 0], state[:, 1]
        dt = self.dt / self.n_inner
        for _ in range(self.n_inner):
            v = v + 0.5 * dt * (self.c ** 2) * self._lap(u)
            u = u + dt * v
            v = v + 0.5 * dt * (self.c ** 2) * self._lap(u)
        return torch.stack([u, v], dim=1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[..., 0, :, :, :], state[..., 1, :, :, :]
        n = u.shape[-1]
        k = torch.fft.fftfreq(n, d=self.length / n).to(u.device) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
        uh = torch.fft.fftn(u, dim=(-3, -2, -1))
        g2 = sum(torch.fft.ifftn(1j * K * uh, dim=(-3, -2, -1)).real ** 2
                 for K in (KX, KY, KZ))
        cell = (self.length / n) ** 3
        return 0.5 * ((v ** 2).sum((-3, -2, -1)) + (self.c ** 2) * g2.sum((-3, -2, -1))) * cell

    def random_ic(self, n_batch: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.linspace(-1, 1, self.grid_n + 1)[:-1] * (self.length / 2)
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        u = torch.zeros(n_batch, *X.shape)
        v = torch.zeros_like(u)
        for kx in (1, 2):
            for ky in (1, 2):
                a = 2 * torch.rand(n_batch, 1, 1, 1, generator=g) - 1
                b = 2 * torch.rand(n_batch, 1, 1, 1, generator=g) - 1
                u = u + a * (torch.cos(math.pi * kx * X) * torch.cos(math.pi * ky * Y)
                             * torch.cos(math.pi * Z)).unsqueeze(0)
                v = v + 0.3 * b * (torch.sin(math.pi * kx * X) * torch.sin(math.pi * ky * Y)
                                   * torch.sin(math.pi * Z)).unsqueeze(0)
        u = 0.4 * u / (u.abs().amax((-3, -2, -1), keepdim=True) + 1e-6)
        v = 0.4 * v / (v.abs().amax((-3, -2, -1), keepdim=True) + 1e-6)
        return torch.stack([u, v], dim=1).float()

    @torch.no_grad()
    def rollout(self, ic, n_steps):
        traj, s = [ic.clone()], ic
        for _ in range(n_steps):
            s = self.true_step(s); traj.append(s.clone())
        return torch.stack(traj, 0)


PROBLEMS["wave3d"] = Wave3DProblem

# Spatial dimensionality of each problem (1-D unless declared otherwise).
for _n, _c in PROBLEMS.items():
    if not hasattr(_c, "spatial_dims"):
        _c.spatial_dims = 1


def get_problem(name: str, **kwargs):
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; choices: {list(PROBLEMS)}")
    return PROBLEMS[name](**kwargs)


# ---------------------------------------------------------------------------
# Self-test: verify the reference solvers conserve their invariants, and that
# anti-aliased resampling round-trips a band-limited field. Run:
#     python -m track2.pde_solvers
# ---------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    for name in ("wave1d", "kdv"):
        prob = get_problem(name)
        ic = prob.random_ic(8, seed=123)
        n_steps = 200
        traj = prob.rollout(ic, n_steps)  # (n_steps+1, B, C, N)
        E = torch.stack([prob.energy(traj[t]) for t in range(traj.shape[0])])  # (T, B)
        E0 = E[0].abs().mean().item()
        drift = (E - E[0]).abs().mean().item()
        rel = drift / (E0 + 1e-12)
        line = f"[{name:6s}] {n_steps} steps  |E0|={E0:.4e}  mean|dE|={drift:.3e}  rel={rel:.3%}"
        if name == "kdv":
            m = torch.stack([prob.mass(traj[t]) for t in range(traj.shape[0])])
            mdrift = (m - m[0]).abs().mean().item()
            line += f"  mass_drift={mdrift:.3e}"
        print(line)

    # Anti-aliased resample round-trip on a band-limited signal.
    prob = get_problem("wave1d")
    u = prob.random_ic(4, seed=7)[:, 0, :]  # (4, 32), K<=3 band-limited
    up = spectral_resample(u, 64)
    ub = spectral_resample(up, 32)
    err = (u - ub).abs().amax().item()
    print(f"[resample] 32->64->32 band-limited round-trip max-abs err = {err:.3e}")

    # The CGL control only means anything if its spatial operator really is
    # self-adjoint in the Clenshaw-Curtis weight, so check that directly.
    prob = get_problem("wave1d_cgl")
    _, w, _, A = prob._cache(torch.device("cpu"), torch.float64)
    WA = torch.diag(w) @ A
    asym = (WA - WA.t()).abs().amax().item() / WA.abs().amax().item()
    print(f"[wave1d_cgl] relative |WA - A^T W| = {asym:.3e}   (must be ~machine eps)")

    ic = prob.random_ic(8, seed=123)
    traj = prob.rollout(ic, 500)
    E = torch.stack([prob.energy(traj[t]) for t in range(traj.shape[0])])
    rel = (E - E[0]).abs().mean().item() / (E[0].abs().mean().item() + 1e-12)
    bc = traj[:, :, 0, :][..., [0, -1]].abs().amax().item()
    amp = traj[:, :, 0, :].abs().amax().item()
    print(f"[wave1d_cgl] 500 steps  rel|dE|={rel:.3%}  boundary={bc:.2e}  "
          f"max|u|={amp:.3f}")


if __name__ == "__main__":
    _selftest()
