#!/usr/bin/env python3
"""
3D elastic lattice simulator for training-data generation.

State convention for ML
-----------------------
q = displacement
p = momentum = mass * velocity

Saved training pairs
--------------------
q_t
p_t
q_t_plus_1
p_t_plus_1

Here is my training pipeline. I'm see a big difference in the loss between q and p. Suggest changes to normalise this. Check for other improvements in the pipeline to help training. Check that the physics includes the dissipative factor. For now, do not worry about the amount fo data or batches or epochs, I am generating more data. I just want to test the pipeline and logic now. 

"""

import os

# ---------------------------------------------------------------------
# Thread settings
# ---------------------------------------------------------------------
# IMPORTANT:
# This code is stencil / neighbour-loop dominated, not BLAS dominated.
# For best performance we use Numba threads for the main evolution work
# and force BLAS libraries to a single thread to avoid oversubscription.
#
# Adjust this number to the number of CPU threads you want Numba to use.
NUMBA_THREADS = 8

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads, get_num_threads

# Set Numba threading after import
set_num_threads(NUMBA_THREADS)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CFG = {
    # Grid
    "run_id" : 0,
    
    "nx": 20,
    "ny": 15,
    "nz": 15,

    # Time
    "dx": 20.0,
    "dt": 1.0e-3,
    "n_steps": 1000,

    # Material
    "rho0": 2500.0,
    "vp0": 3000.0,
    "vs0": 1800.0,

    # Neighbour damping coefficient
    "eta": 1.0e7,

    # Velocity model
    # -------------------------------------------------------------
    # existing choices: "homogeneous", "layered", "sphere", "custom"
    "velocity_model_type": "fractal_depth",

    # Reproducibility
    "model_seed": None,    # set to an integer for repeatable models

    # Depth-dependent vp background
    "vp_surface": 2200.0,
    "vp_bottom": 4000.0,

    # Fractal heterogeneity strength
    # This is a fractional perturbation applied to vp background.
    # Example 0.15 means about +/-15 percent style variation after scaling.
    "fractal_strength": 0.15,

    # Fractal roughness control
    # Larger beta means smoother long-wavelength dominated structure.
    # Common useful range is about 2.0 to 4.0.
    "fractal_beta": 3.0,

    # Optional clipping for vp
    "vp_min": 2200.0,
    "vp_max": 4000.0,

    # S-wave relation
    # "ratio"       -> vs = vs_vp_ratio * vp
    # "poisson"     -> use a fixed ratio equivalent
    "vs_model_type": "ratio",
    "vs_vp_ratio": 0.58,

    # Density relation
    # rho = rho_intercept + rho_slope * vp
    "rho_intercept": 1400.0,
    "rho_slope": 0.28,

    # -------------------------------------------------------------
    # One-time QA plot for the velocity model
    # -------------------------------------------------------------
    "plot_velocity_model_once": True,
    "velocity_plot_field": "vp",   # "vp", "vs", or "rho"

    # Layered model parameters
    "layer_z_fraction": 0.45,
    "vp_layer2": 3800.0,
    "vs_layer2": 2200.0,
    "rho_layer2": 2700.0,

    # Sphere model parameters
    "sphere_center_fraction": (0.5, 0.5, 0.55),
    "sphere_radius_fraction": 0.18,
    "vp_sphere": 4200.0,
    "vs_sphere": 2400.0,
    "rho_sphere": 2850.0,

    # Optional simple topography mask
    "use_topography": True,
    "topo_seed": None,   # set to int for repeatable terrain
    # smoothing strength (bigger = smoother hills)
    "topo_smooth_iters": 20,
    # height range (fraction of nz)
    "topo_min_frac": 0.75,
    "topo_max_frac": 0.95,

    # Boundary condition
    # Options: "free", "fixed", "periodic"
    "boundary": "periodic",

    # Absorbing taper
    "use_absorbing": True,
    "absorb_width": 6,
    "absorb_lambda": 0.0,

    # Source
    "use_source": True,
    "source_type": "moment",   # "force", "moment", "none"
    "source_direction": 2,     # for source_type="force": 0=x, 1=y, 2=z
    "source_scale": 1.0e12,
    "source_xyz": None,        # if None, place automatically
    "source_peak_frequency": 10.0,
    "source_time_shift": 0.08,
    
    # Source control
    "use_random_source": True,
    "source_margin": 5,
    # reproducibility (optional)
    "source_seed": None,

    # Saving
    "save_every": 2,
    "store_dtype": "float32",
    "output_npz": "output2/small_run",
    "output_metadata_json": "output2/small_metadata",

    # Plotting
    "enable_live_plot": True,
    "plot_every": 200,
    "plot_component": 0,   # "magnitude", 0, 1, or 2
}

# ---------------------------------------------------------------------
# 18-neighbour offsets from the original lattice
# ---------------------------------------------------------------------
NEIGHBOUR_OFFSETS = np.array([
    [ 0,  1,  0],
    [ 1,  1,  0],
    [ 1,  0,  0],
    [ 1, -1,  0],
    [ 0, -1,  0],
    [-1, -1,  0],
    [-1,  0,  0],
    [-1,  1,  0],
    [ 0,  1, -1],
    [ 0,  0, -1],
    [ 0, -1, -1],
    [ 0, -1,  1],
    [ 0,  0,  1],
    [ 0,  1,  1],
    [ 1,  0, -1],
    [-1,  0, -1],
    [-1,  0,  1],
    [ 1,  0,  1],
], dtype=np.int64)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def make_ricker_wavelet(n_steps, dt, f0, t0):
    t = np.arange(n_steps, dtype=np.float64) * dt
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)


def default_source_location(cfg):
    return (
        cfg["nx"] // 2,
        cfg["ny"] // 2,
        max(2, cfg["nz"] // 2),
    )


def make_rest_positions(cfg):
    nx, ny, nz, dx = cfg["nx"], cfg["ny"], cfg["nz"], cfg["dx"]
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dx
    z = np.arange(nz, dtype=np.float64) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    eq = np.empty((nx, ny, nz, 3), dtype=np.float64)
    eq[..., 0] = X
    eq[..., 1] = Y
    eq[..., 2] = Z
    return eq

def make_fractal_field_3d(nx, ny, nz, beta=3.0, seed=None):
    """
    Make a smooth fractal 3D random field using spectral shaping.

    The returned field is normalised to zero mean and unit standard deviation.

    Parameters
    ----------
    nx, ny, nz : int
        Grid size
    beta : float
        Power-law exponent controlling roughness.
        Larger beta gives smoother fields.
    seed : int or None
        RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Random complex spectrum
    noise = rng.normal(size=(nx, ny, nz)) + 1j * rng.normal(size=(nx, ny, nz))

    # Wavenumbers
    kx = np.fft.fftfreq(nx).reshape(nx, 1, 1)
    ky = np.fft.fftfreq(ny).reshape(1, ny, 1)
    kz = np.fft.fftfreq(nz).reshape(1, 1, nz)

    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # avoid divide by zero at zero frequency

    # Power-law spectral amplitude
    amp = k2 ** (-beta / 4.0)

    field_hat = noise * amp
    field = np.fft.ifftn(field_hat).real

    # Normalise
    field -= field.mean()
    field /= (field.std() + 1e-12)

    return field

def make_material_fields(cfg):
    """
    Create spatially varying vp, vs, rho, and an active mask.

    Supported model types:
    - homogeneous
    - layered
    - sphere
    - custom
    - fractal_depth

    active=False means the cell is outside the medium
    and should not contribute to dynamics.
    """
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    vp = np.full((nx, ny, nz), cfg["vp0"], dtype=np.float64)
    vs = np.full((nx, ny, nz), cfg["vs0"], dtype=np.float64)
    rho = np.full((nx, ny, nz), cfg["rho0"], dtype=np.float64)
    active = np.ones((nx, ny, nz), dtype=np.uint8)

    model_type = cfg["velocity_model_type"]

    if model_type == "homogeneous":
        pass

    elif model_type == "layered":
        z_cut = int(cfg["layer_z_fraction"] * nz)
        vp[:, :, z_cut:] = cfg["vp_layer2"]
        vs[:, :, z_cut:] = cfg["vs_layer2"]
        rho[:, :, z_cut:] = cfg["rho_layer2"]

    elif model_type == "sphere":
        x = np.arange(nx)[:, None, None]
        y = np.arange(ny)[None, :, None]
        z = np.arange(nz)[None, None, :]

        cx = cfg["sphere_center_fraction"][0] * (nx - 1)
        cy = cfg["sphere_center_fraction"][1] * (ny - 1)
        cz = cfg["sphere_center_fraction"][2] * (nz - 1)
        r = cfg["sphere_radius_fraction"] * min(nx, ny, nz)

        mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2
        vp[mask] = cfg["vp_sphere"]
        vs[mask] = cfg["vs_sphere"]
        rho[mask] = cfg["rho_sphere"]

    elif model_type == "custom":
        # Example custom model. Edit this however you like.
        x = np.linspace(0.0, 1.0, nx)[:, None, None]
        vp[:] = cfg["vp0"] * (1.0 + 0.25 * x)
        vs[:] = 0.6 * vp
        rho[:] = cfg["rho0"] + 250.0 * x

    elif model_type == "fractal_depth":
        # ---------------------------------------------------------
        # 1. depth-dependent vp background
        # ---------------------------------------------------------
        zf = np.linspace(0.0, 1.0, nz, dtype=np.float64)
        vp_bg_1d = cfg["vp_bottom"] + (cfg["vp_surface"] - cfg["vp_bottom"]) * zf
        vp_bg = np.broadcast_to(vp_bg_1d[None, None, :], (nx, ny, nz)).copy()

        # ---------------------------------------------------------
        # 2. fractal perturbation field
        # ---------------------------------------------------------
        frac = make_fractal_field_3d(
            nx, ny, nz,
            beta=cfg["fractal_beta"],
            seed=cfg.get("model_seed", None)
        )

        # ---------------------------------------------------------
        # 3. apply perturbation multiplicatively
        # ---------------------------------------------------------
        vp = vp_bg * (1.0 + cfg["fractal_strength"] * frac)

        # clip to a physical range
        vp = np.clip(vp, cfg["vp_min"], cfg["vp_max"])

        # ---------------------------------------------------------
        # 4. derive vs from vp
        # ---------------------------------------------------------
        if cfg["vs_model_type"] == "ratio":
            vs = cfg["vs_vp_ratio"] * vp
        elif cfg["vs_model_type"] == "poisson":
            vs = 0.57735026919 * vp
        else:
            raise ValueError(f"Unknown vs_model_type: {cfg['vs_model_type']}")

        # ---------------------------------------------------------
        # 5. derive rho from vp
        # ---------------------------------------------------------
        rho = cfg["rho_intercept"] + cfg["rho_slope"] * vp

    else:
        raise ValueError(f"Unknown velocity_model_type: {model_type}")

    # -------------------------------------------------------------
    # Optional topography mask
    # -------------------------------------------------------------
    if cfg["use_topography"]:

        rng = np.random.default_rng(cfg.get("topo_seed", None))
        topo = rng.normal(size=(nx, ny))

        for _ in range(cfg.get("topo_smooth_iters", 20)):
            topo = (
                topo
                + np.roll(topo, 1, axis=0)
                + np.roll(topo, -1, axis=0)
                + np.roll(topo, 1, axis=1)
                + np.roll(topo, -1, axis=1)
            ) / 5.0

        topo -= topo.min()
        topo /= topo.max() + 1e-12

        min_frac = cfg.get("topo_min_frac", 0.6)
        max_frac = cfg.get("topo_max_frac", 0.9)

        topo_height = (min_frac + (max_frac - min_frac) * topo) * nz
        topo_height = topo_height.astype(np.int32)

        for i in range(nx):
            for j in range(ny):
                k0 = int(np.clip(topo_height[i, j], 0, nz))
                active[i, j, k0:] = 0

    # Zero vp/vs in inactive cells
    vp[active == 0] = 0.0
    vs[active == 0] = 0.0

    return vp, vs, rho, active

def old_make_material_fields(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    vp = np.full((nx, ny, nz), cfg["vp0"], dtype=np.float64)
    vs = np.full((nx, ny, nz), cfg["vs0"], dtype=np.float64)
    rho = np.full((nx, ny, nz), cfg["rho0"], dtype=np.float64)
    active = np.ones((nx, ny, nz), dtype=np.uint8)

    model_type = cfg["velocity_model_type"]

    if model_type == "homogeneous":
        pass

    elif model_type == "layered":
        z_cut = int(cfg["layer_z_fraction"] * nz)
        vp[:, :, z_cut:] = cfg["vp_layer2"]
        vs[:, :, z_cut:] = cfg["vs_layer2"]
        rho[:, :, z_cut:] = cfg["rho_layer2"]

    elif model_type == "sphere":
        x = np.arange(nx)[:, None, None]
        y = np.arange(ny)[None, :, None]
        z = np.arange(nz)[None, None, :]
        cx = cfg["sphere_center_fraction"][0] * (nx - 1)
        cy = cfg["sphere_center_fraction"][1] * (ny - 1)
        cz = cfg["sphere_center_fraction"][2] * (nz - 1)
        r = cfg["sphere_radius_fraction"] * min(nx, ny, nz)
        mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2
        vp[mask] = cfg["vp_sphere"]
        vs[mask] = cfg["vs_sphere"]
        rho[mask] = cfg["rho_sphere"]

    elif model_type == "custom":
        # Example custom model. Edit freely.
        x = np.linspace(0.0, 1.0, nx)[:, None, None]
        vp[:] = cfg["vp0"] * (1.0 + 0.25 * x)
        vs[:] = 0.6 * vp
        rho[:] = cfg["rho0"] + 250.0 * x

    else:
        raise ValueError(f"Unknown velocity_model_type: {model_type}")
    
    if cfg["use_topography"]:
    
        # -----------------------------------------------------------
        # 1. random field
        # -----------------------------------------------------------
        rng = np.random.default_rng(cfg.get("topo_seed", None))
    
        topo = rng.normal(size=(nx, ny))
    
        # -----------------------------------------------------------
        # 2. smooth it (creates hills instead of noise)
        # -----------------------------------------------------------
        for _ in range(cfg.get("topo_smooth_iters", 20)):
            topo = (
                topo +
                np.roll(topo, 1, axis=0) +
                np.roll(topo, -1, axis=0) +
                np.roll(topo, 1, axis=1) +
                np.roll(topo, -1, axis=1)
            ) / 5.0
    
        # -----------------------------------------------------------
        # 3. normalise to [0,1]
        # -----------------------------------------------------------
        topo -= topo.min()
        topo /= topo.max() + 1e-12
    
        # -----------------------------------------------------------
        # 4. scale to height range
        # -----------------------------------------------------------
        min_frac = cfg.get("topo_min_frac", 0.6)
        max_frac = cfg.get("topo_max_frac", 0.9)
    
        topo_height = (min_frac + (max_frac - min_frac) * topo) * nz
        topo_height = topo_height.astype(np.int32)
    
        # -----------------------------------------------------------
        # 5. build mask
        # -----------------------------------------------------------
        for i in range(nx):
            for j in range(ny):
                k0 = topo_height[i, j]
                k0 = min(max(k0, 0), nz)
                active[i, j, k0:] = 0
    '''            
    if cfg["use_topography"]:
        xx = np.arange(nx)[:, None]
        yy = np.arange(ny)[None, :]
        top = (
            0.72 * nz
            + 0.07 * nz * np.sin(2.0 * np.pi * xx / max(nx, 1))
            + 0.05 * nz * np.cos(2.0 * np.pi * yy / max(ny, 1))
        ).astype(int)
        for i in range(nx):
            for j in range(ny):
                k0 = int(np.clip(top[i, j], 0, nz))
                active[i, j, k0:] = 0
    '''            
    vp[active == 0] = 0.0
    vs[active == 0] = 0.0

    return vp, vs, rho, active

def random_source_location(cfg, active):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    margin = cfg.get("source_margin", 5)

    # RNG (reproducible if desired)
    seed = cfg.get("source_seed", None)
    if seed is None:
        seed = cfg.get("run_id", None)
    rng = np.random.default_rng(seed)

    # find all valid candidate cells
    candidates = np.argwhere(active == 1)

    # filter out boundary regions
    valid = []
    for i, j, k in candidates:
        if (
            i > margin and i < nx - margin and
            j > margin and j < ny - margin and
            k > margin and k < nz - margin
        ):
            valid.append((i, j, k))

    if len(valid) == 0:
        raise RuntimeError("No valid source locations found")

    # pick one at random
    idx = rng.integers(0, len(valid))
    return np.array(valid[idx], dtype=np.int64)

def make_absorbing_profile(cfg,active):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    width = cfg["absorb_width"]
    lam = cfg["absorb_lambda"]

    if (not cfg["use_absorbing"]) or width <= 0:
        return np.ones((nx, ny, nz), dtype=np.float64)

    ax = np.ones(nx, dtype=np.float64)
    ay = np.ones(ny, dtype=np.float64)
    az = np.ones(nz, dtype=np.float64)

    for i in range(nx):
        if i < width:
            ax[i] = math.exp(-lam * (width - i) ** 2)
        elif i > nx - 1 - width:
            ax[i] = math.exp(-lam * (i - (nx - 1 - width)) ** 2)

    for j in range(ny):
        if j < width:
            ay[j] = math.exp(-lam * (width - j) ** 2)
        elif j > ny - 1 - width:
            ay[j] = math.exp(-lam * (j - (ny - 1 - width)) ** 2)

    for k in range(nz):
        if k < width:
            az[k] = math.exp(-lam * (width - k) ** 2)
        elif k > nz - 1 - width and cfg["use_topography"]==False:
            az[k] = math.exp(-lam * (k - (nz - 1 - width)) ** 2)


    # -----------------------------
    absorb = ax[:, None, None] * ay[None, :, None] * az[None, None, :]
    absorb = np.where(active == 1, absorb, 1.0)
    return absorb
    #return ax[:, None, None] * ay[None, :, None] * az[None, None, :]


def compute_bond_parameters(cfg, vp, vs, rho, active, offsets):
    """
    One-time precomputation of spring and bond coefficients.
    This is not the runtime hotspot, so plain NumPy is fine here.
    """
    dx = cfg["dx"]
    boundary = cfg["boundary"]
    nx, ny, nz = vp.shape
    n_neigh = offsets.shape[0]

    kspring = np.zeros((nx, ny, nz, n_neigh), dtype=np.float64)
    cbond = np.zeros((nx, ny, nz, n_neigh), dtype=np.float64)

    for n in range(n_neigh):
        ox, oy, oz = offsets[n]

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if active[i, j, k] == 0:
                        continue

                    ii = i + ox
                    jj = j + oy
                    kk = k + oz

                    if boundary == "periodic":
                        ii %= nx
                        jj %= ny
                        kk %= nz
                    else:
                        if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                            continue

                    if active[ii, jj, kk] == 0:
                        continue

                    vp_avg = 0.5 * (vp[i, j, k] + vp[ii, jj, kk])
                    vs_avg = 0.5 * (vs[i, j, k] + vs[ii, jj, kk])
                    rho_avg = 0.5 * (rho[i, j, k] + rho[ii, jj, kk])

                    kspring[i, j, k, n] = dx * (vp_avg * vp_avg - vs_avg * vs_avg) * (rho_avg / 2.0)
                    cbond[i, j, k, n] = (dx ** 3 * rho_avg / 2.0) * (-vp_avg * vp_avg + 3.0 * vs_avg * vs_avg)

    rest_vec = offsets.astype(np.float64) * dx
    r0 = np.sqrt(np.sum(rest_vec * rest_vec, axis=1))

    return kspring, cbond, r0, rest_vec


# ---------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------
@njit(cache=True)
def _wrap_index(idx, n):
    if idx < 0:
        return idx + n
    if idx >= n:
        return idx - n
    return idx


@njit(parallel=True, fastmath=True, cache=True)
def compute_internal_force_numba(
    q,
    v,
    active,
    offsets,
    rest_vec,
    r0,
    kspring,
    cbond,
    eta,
    boundary_mode,
):
    """
    boundary_mode:
        0 = free
        1 = fixed
        2 = periodic
    """
    nx, ny, nz, _ = q.shape
    n_neigh = offsets.shape[0]
    force = np.zeros_like(q)
    root_half = math.sqrt(0.5)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if active[i, j, k] == 0:
                    continue

                qx = q[i, j, k, 0]
                qy = q[i, j, k, 1]
                qz = q[i, j, k, 2]

                vx = v[i, j, k, 0]
                vy = v[i, j, k, 1]
                vz = v[i, j, k, 2]

                fx = 0.0
                fy = 0.0
                fz = 0.0

                for n in range(n_neigh):
                    ox = offsets[n, 0]
                    oy = offsets[n, 1]
                    oz = offsets[n, 2]

                    ii = i + ox
                    jj = j + oy
                    kk = k + oz

                    if boundary_mode == 2:
                        ii = _wrap_index(ii, nx)
                        jj = _wrap_index(jj, ny)
                        kk = _wrap_index(kk, nz)
                    else:
                        if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                            continue

                    if active[ii, jj, kk] == 0:
                        continue

                    d0 = rest_vec[n, 0] + (q[ii, jj, kk, 0] - qx)
                    d1 = rest_vec[n, 1] + (q[ii, jj, kk, 1] - qy)
                    d2 = rest_vec[n, 2] + (q[ii, jj, kk, 2] - qz)

                    dist = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                    if dist < 1.0e-14:
                        continue

                    inv_dist = 1.0 / dist
                    ux = d0 * inv_dist
                    uy = d1 * inv_dist
                    uz = d2 * inv_dist

                    spring_term = kspring[i, j, k, n] * (dist - r0[n])

                    # d - rest_vec = q_neigh - q_here
                    dq0 = d0 - rest_vec[n, 0]
                    dq1 = d1 - rest_vec[n, 1]
                    dq2 = d2 - rest_vec[n, 2]

                    coeff = root_half * cbond[i, j, k, n] / (r0[n] * r0[n])

                    bx = coeff * dq0
                    by = coeff * dq1
                    bz = coeff * dq2

                    dvx = v[ii, jj, kk, 0] - vx
                    dvy = v[ii, jj, kk, 1] - vy
                    dvz = v[ii, jj, kk, 2] - vz

                    fx += spring_term * ux + bx + eta * dvx
                    fy += spring_term * uy + by + eta * dvy
                    fz += spring_term * uz + bz + eta * dvz

                force[i, j, k, 0] = fx
                force[i, j, k, 1] = fy
                force[i, j, k, 2] = fz

    return force


@njit(cache=True)
def add_source_numba(force, source_xyz, source_value, source_scale, source_type_code, source_direction):
    """
    source_type_code:
        0 = none
        1 = force
        2 = moment
    """
    nx, ny, nz, _ = force.shape
    i, j, k = source_xyz
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return

    s = source_scale * source_value

    if source_type_code == 1:
        d = source_direction
        force[i, j, k, d] += s

    elif source_type_code == 2:
        # Simple symmetric moment-like source
        neigh = (
            ( i,   j,   k + 1,  0.0,  0.0,  1.0),
            ( i,   j,   k - 1,  0.0,  0.0, -1.0),
            ( i,   j + 1, k,    0.0,  1.0,  0.0),
            ( i,   j - 1, k,    0.0, -1.0,  0.0),
            ( i + 1, j,   k,    1.0,  0.0,  0.0),
            ( i - 1, j,   k,   -1.0,  0.0,  0.0),
        )
        for ii, jj, kk, ax, ay, az in neigh:
            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                force[ii, jj, kk, 0] += s * ax
                force[ii, jj, kk, 1] += s * ay
                force[ii, jj, kk, 2] += s * az


@njit(parallel=True, fastmath=True, cache=True)
def update_q_and_half_v_numba(q, v, force, mass, active, dt):
    nx, ny, nz, _ = q.shape
    q_new = np.empty_like(q)
    v_half = np.empty_like(v)
    half_dt = 0.5 * dt
    half_dt2 = 0.5 * dt * dt

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if active[i, j, k] == 0:
                    q_new[i, j, k, 0] = 0.0
                    q_new[i, j, k, 1] = 0.0
                    q_new[i, j, k, 2] = 0.0
                    v_half[i, j, k, 0] = 0.0
                    v_half[i, j, k, 1] = 0.0
                    v_half[i, j, k, 2] = 0.0
                    continue

                inv_m = 1.0 / mass[i, j, k]

                ax = force[i, j, k, 0] * inv_m
                ay = force[i, j, k, 1] * inv_m
                az = force[i, j, k, 2] * inv_m

                q_new[i, j, k, 0] = q[i, j, k, 0] + dt * v[i, j, k, 0] + half_dt2 * ax
                q_new[i, j, k, 1] = q[i, j, k, 1] + dt * v[i, j, k, 1] + half_dt2 * ay
                q_new[i, j, k, 2] = q[i, j, k, 2] + dt * v[i, j, k, 2] + half_dt2 * az

                v_half[i, j, k, 0] = v[i, j, k, 0] + half_dt * ax
                v_half[i, j, k, 1] = v[i, j, k, 1] + half_dt * ay
                v_half[i, j, k, 2] = v[i, j, k, 2] + half_dt * az

    return q_new, v_half


@njit(parallel=True, fastmath=True, cache=True)
def complete_v_numba(v_half, force_new, mass, active, dt):
    nx, ny, nz, _ = v_half.shape
    v_new = np.empty_like(v_half)
    half_dt = 0.5 * dt

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if active[i, j, k] == 0:
                    v_new[i, j, k, 0] = 0.0
                    v_new[i, j, k, 1] = 0.0
                    v_new[i, j, k, 2] = 0.0
                    continue

                inv_m = 1.0 / mass[i, j, k]
                v_new[i, j, k, 0] = v_half[i, j, k, 0] + half_dt * force_new[i, j, k, 0] * inv_m
                v_new[i, j, k, 1] = v_half[i, j, k, 1] + half_dt * force_new[i, j, k, 1] * inv_m
                v_new[i, j, k, 2] = v_half[i, j, k, 2] + half_dt * force_new[i, j, k, 2] * inv_m

    return v_new


@njit(parallel=True, fastmath=True, cache=True)
def apply_absorb_numba(v, absorb_profile):
    nx, ny, nz, _ = v.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                a = absorb_profile[i, j, k]
                v[i, j, k, 0] *= a
                v[i, j, k, 1] *= a
                v[i, j, k, 2] *= a


@njit(parallel=True, fastmath=True, cache=True)
def apply_fixed_boundaries_numba(q, v):
    nx, ny, nz, _ = q.shape

    for j in prange(ny):
        for k in range(nz):
            q[0, j, k, :] = 0.0
            q[nx - 1, j, k, :] = 0.0
            v[0, j, k, :] = 0.0
            v[nx - 1, j, k, :] = 0.0

    for i in prange(nx):
        for k in range(nz):
            q[i, 0, k, :] = 0.0
            q[i, ny - 1, k, :] = 0.0
            v[i, 0, k, :] = 0.0
            v[i, ny - 1, k, :] = 0.0

    for i in prange(nx):
        for j in range(ny):
            q[i, j, 0, :] = 0.0
            q[i, j, nz - 1, :] = 0.0
            v[i, j, 0, :] = 0.0
            v[i, j, nz - 1, :] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def momentum_from_velocity_numba(v, mass):
    nx, ny, nz, _ = v.shape
    p = np.empty_like(v)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                m = mass[i, j, k]
                p[i, j, k, 0] = m * v[i, j, k, 0]
                p[i, j, k, 1] = m * v[i, j, k, 1]
                p[i, j, k, 2] = m * v[i, j, k, 2]
    return p

def save_material_fields_npz(vp, vs, rho, cfg, out_dir="output2"):
    """
    Save vp, vs, rho into a single NPZ file with run_id label.

    File format:
        materials_run_<run_id>.npz

    Contents:
        vp, vs, rho arrays
    """
    os.makedirs(out_dir, exist_ok=True)

    run_id = cfg.get("run_id", "unknown")
    filename = os.path.join(out_dir, f"materials_run_{run_id}.npz")

    np.savez(filename, vp=vp, vs=vs, rho=rho)

    print(f"[INFO] Saved material fields to: {filename}")
    
# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def displacement_to_plot(q, cfg):
    comp = cfg["plot_component"]
    if comp == "magnitude":
        return np.sqrt(np.sum(q * q, axis=-1))
    elif comp in (0, 1, 2):
        return q[..., comp]
    else:
        raise ValueError("plot_component must be 'magnitude', 0, 1, or 2")

def plot_model_slices_once(field, name="vp"):
    """
    Plot three orthogonal slices through a 3D model field using figure num=10.
    This is intended as a one-time QA check before the time loop.
    """
    nx, ny, nz = field.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    xy = field[:, :, iz].T
    xz = field[:, iy, :].T
    yz = field[ix, :, :].T

    fig = plt.figure(num=9, figsize=(12, 4))
    fig.clf()

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(xy, origin="lower", aspect="auto", cmap="viridis")
    ax1.set_title(f"{name} XY slice, z={iz}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(xz, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_title(f"{name} XZ slice, y={iy}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(yz, origin="lower", aspect="auto", cmap="viridis")
    ax3.set_title(f"{name} YZ slice, x={ix}")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
def update_live_plot(q, step, cfg):
    field = displacement_to_plot(q, cfg)
    nx, ny, nz = field.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    xy = field[:, :, iz].T
    xz = field[:, iy, :].T
    yz = field[ix, :, :].T

    fig = plt.figure(num=10, figsize=(12, 4))
    fig.clf()

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(xy, origin="lower", aspect="auto", cmap="seismic")
    ax1.set_title(f"XY slice, z={iz}, step={step}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(xz, origin="lower", aspect="auto", cmap="seismic")
    ax2.set_title(f"XZ slice, y={iy}, step={step}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(yz, origin="lower", aspect="auto", cmap="seismic")
    ax3.set_title(f"YZ slice, x={ix}, step={step}")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    fig.tight_layout()
    plt.pause(0.001)


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------
def run_simulation(cfg):
    print(f"Numba threads in use: {get_num_threads()}")

    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    # Material and geometry
    vp, vs, rho, active = make_material_fields(cfg)
    eq = make_rest_positions(cfg)
    absorb_profile = make_absorbing_profile(cfg,active)

    save_material_fields_npz(vp, vs, rho, cfg)

    # ---------------------------------------------------------
    # QA plot of the velocity/density model
    if cfg.get("plot_velocity_model_once", False):
        field_name = cfg.get("velocity_plot_field", "vp")
        print('plotting model')
        if field_name == "vp":
            plot_model_slices_once(vp, name="vp")
        elif field_name == "vs":
            plot_model_slices_once(vs, name="vs")
        elif field_name == "rho":
            plot_model_slices_once(rho, name="rho")
        else:
            raise ValueError("velocity_plot_field must be 'vp', 'vs', or 'rho'")


    # Precompute bond coefficients
    kspring, cbond, r0, rest_vec = compute_bond_parameters(
        cfg, vp, vs, rho, active, NEIGHBOUR_OFFSETS
    )

    # State
    q = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    v = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    mass = rho * (cfg["dx"] ** 3)

    # Source
    #source_xyz = cfg["source_xyz"] if cfg["source_xyz"] is not None else default_source_location(cfg)
    if cfg.get("use_random_source", False):
        source_xyz = random_source_location(cfg, active)
    else:
        source_xyz = (
            cfg["source_xyz"]
            if cfg["source_xyz"] is not None
            else default_source_location(cfg)
        )
    source_xyz = np.array(source_xyz, dtype=np.int64)
    print(f"Source location: {source_xyz}")

    source_series = make_ricker_wavelet(
        cfg["n_steps"],
        cfg["dt"],
        cfg["source_peak_frequency"],
        cfg["source_time_shift"],
    )

    source_type_map = {"none": 0, "force": 1, "moment": 2}
    source_type_code = source_type_map[cfg["source_type"]]

    boundary_map = {"free": 0, "fixed": 1, "periodic": 2}
    boundary_mode = boundary_map[cfg["boundary"]]

    # Initial internal force
    force = compute_internal_force_numba(
        q, v, active, NEIGHBOUR_OFFSETS, rest_vec, r0, kspring, cbond, cfg["eta"], boundary_mode
    )

    # Preallocate training outputs
    save_every = cfg["save_every"]
    n_saved = (cfg["n_steps"] + save_every - 1) // save_every
    store_dtype = np.float32 if cfg["store_dtype"] == "float32" else np.float64

    q_t = np.empty((n_saved, nx, ny, nz, 3), dtype=store_dtype)
    p_t = np.empty((n_saved, nx, ny, nz, 3), dtype=store_dtype)
    q_tp1 = np.empty((n_saved, nx, ny, nz, 3), dtype=store_dtype)
    p_tp1 = np.empty((n_saved, nx, ny, nz, 3), dtype=store_dtype)

    # Warm-up compile. This avoids timing the first JIT compile during progress.
    force_test = compute_internal_force_numba(
        q, v, active, NEIGHBOUR_OFFSETS, rest_vec, r0, kspring, cbond, cfg["eta"], boundary_mode
    )
    q_test, v_half_test = update_q_and_half_v_numba(q, v, force_test, mass, active, cfg["dt"])
    _ = complete_v_numba(v_half_test, force_test, mass, active, cfg["dt"])
    _ = momentum_from_velocity_numba(v, mass)

    if cfg["enable_live_plot"]:
        plt.ion()
        update_live_plot(q, 0, cfg)

    save_idx = 0

    for step in range(cfg["n_steps"]):
        # Save old state for ML pair
        q_old = q.copy()
        p_old = momentum_from_velocity_numba(v, mass)

        # First half of velocity-Verlet
        q_new, v_half = update_q_and_half_v_numba(q, v, force, mass, active, cfg["dt"])

        if cfg["boundary"] == "fixed":
            apply_fixed_boundaries_numba(q_new, v_half)

        # Recompute internal force at updated state
        force_new = compute_internal_force_numba(
            q_new,
            v_half,
            active,
            NEIGHBOUR_OFFSETS,
            rest_vec,
            r0,
            kspring,
            cbond,
            cfg["eta"],
            boundary_mode,
        )

        # Add source to the new force
        if cfg["use_source"] and source_type_code != 0:
            add_source_numba(
                force_new,
                source_xyz,
                float(source_series[step]),
                float(cfg["source_scale"]),
                source_type_code,
                int(cfg["source_direction"]),
            )

        # Complete velocity update
        v_new = complete_v_numba(v_half, force_new, mass, active, cfg["dt"])

        # Absorbing taper
        if cfg["use_absorbing"]:
            apply_absorb_numba(v_new, absorb_profile)

        if cfg["boundary"] == "fixed":
            apply_fixed_boundaries_numba(q_new, v_new)

        q = q_new
        v = v_new
        force = force_new

        p_new = momentum_from_velocity_numba(v, mass)

        if step % save_every == 0:
            q_t[save_idx] = q_old.astype(store_dtype, copy=False)
            p_t[save_idx] = p_old.astype(store_dtype, copy=False)
            q_tp1[save_idx] = q.astype(store_dtype, copy=False)
            p_tp1[save_idx] = p_new.astype(store_dtype, copy=False)
            save_idx += 1

        if cfg["enable_live_plot"] and (step % cfg["plot_every"] == 0):
            update_live_plot(q, step, cfg)

        if step % 50 == 0:
            max_disp = np.max(np.sqrt(np.sum(q * q, axis=-1)))
            print(f"step={step:6d}   max|q|={max_disp:.6e}")

    metadata = {
        "description": "3D elastic lattice training pairs",
        "state_meaning": {
            "q_t": "displacement field at time t, shape [samples, nx, ny, nz, 3]",
            "p_t": "momentum field at time t, p = rho * dx^3 * velocity",
            "q_t_plus_1": "displacement field at time t+1",
            "p_t_plus_1": "momentum field at time t+1",
        },
        "grid": {
            "nx": cfg["nx"],
            "ny": cfg["ny"],
            "nz": cfg["nz"],
            "dx": cfg["dx"],
            "dt": cfg["dt"],
        },
        "material_model": cfg["velocity_model_type"],
        "boundary": cfg["boundary"],
        "use_absorbing": cfg["use_absorbing"],
        "source_type": cfg["source_type"],
        "source_xyz": [int(source_xyz[0]), int(source_xyz[1]), int(source_xyz[2])],
        "save_every": cfg["save_every"],
        "dtype": cfg["store_dtype"],
        "numba_threads": int(get_num_threads()),
    }
    
    run_id = cfg.get("run_id", 0)
    
    npz_name = f"{cfg['output_npz']}_{run_id}.npz"  # f"small_run_{run_id}.npz" 
    meta_name =f"{cfg['output_metadata_json']}_{run_id}.json" # f"small_run_{run_id}.json"
    
    np.savez_compressed(
        npz_name,
        q_t=q_t,
        p_t=p_t,
        q_t_plus_1=q_tp1,
        p_t_plus_1=p_tp1,
        vp=vp.astype(store_dtype),
        vs=vs.astype(store_dtype),
        rho=rho.astype(store_dtype),
        active=active.astype(np.uint8),
        eq=eq.astype(store_dtype),
        absorb_profile=absorb_profile.astype(store_dtype),
        source_series=source_series.astype(store_dtype),
        metadata_json=json.dumps(metadata),
    )

    with open(meta_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

   
    print("\nSaved files:")
    print(npz_name)
    print(meta_name)

    print("\nSaved arrays:")
    print("  q_t         : displacement at time t")
    print("  p_t         : momentum at time t")
    print("  q_t_plus_1  : displacement at time t+1")
    print("  p_t_plus_1  : momentum at time t+1")
    print("  vp, vs, rho : material fields")
    print("  active      : active medium mask")
    print("  eq          : equilibrium positions")

    if cfg["enable_live_plot"]:
        plt.ioff()
        plt.figure(num=10)
        plt.show()

    return {
        "q_t": q_t,
        "p_t": p_t,
        "q_t_plus_1": q_tp1,
        "p_t_plus_1": p_tp1,
        "vp": vp,
        "vs": vs,
        "rho": rho,
        "active": active,
        "eq": eq,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_simulation(CFG)

