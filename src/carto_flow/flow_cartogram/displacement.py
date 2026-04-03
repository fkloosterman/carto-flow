"""
Displacement computation utilities for flow-based cartography.

Functions
---------
displace_coords_numba
    Numba-compiled coordinate displacement with bilinear interpolation on uniform grids.
max_velocity_magnitude
    Single-pass max(sqrt(vx²+vy²)) without temporary array allocation.
max_abs_velocity
    Single-pass max(|vx|, |vy|) without temporary array allocation.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Module-level exports - Public API
__all__ = [
    "displace_coords_numba",
    "max_abs_velocity",
    "max_velocity_magnitude",
]


@njit(cache=True, fastmath=True)
def max_velocity_magnitude(vx: np.ndarray, vy: np.ndarray) -> float:
    """Return max(sqrt(vx² + vy²)) over all grid cells in a single pass.

    Equivalent to ``np.nanmax(np.sqrt(vx**2 + vy**2))`` but avoids allocating
    the intermediate arrays, cutting memory bandwidth by ~3x at large grid sizes.

    Parameters
    ----------
    vx, vy : np.ndarray, shape (sy, sx)
        Velocity component arrays.

    Returns
    -------
    float
        Maximum velocity magnitude.
    """
    ny = vx.shape[0]
    nx = vx.shape[1]
    m = 0.0
    for i in range(ny):
        for j in range(nx):
            sq = vx[i, j] * vx[i, j] + vy[i, j] * vy[i, j]
            if sq > m:
                m = sq
    return np.sqrt(m)


@njit(cache=True, fastmath=True)
def max_abs_velocity(vx: np.ndarray, vy: np.ndarray) -> float:
    """Return max(max|vx|, max|vy|) over all grid cells in a single pass.

    Equivalent to ``max(np.max(np.abs(vx)), np.max(np.abs(vy)))`` but avoids
    allocating the two intermediate abs arrays.

    Parameters
    ----------
    vx, vy : np.ndarray, shape (sy, sx)
        Velocity component arrays.

    Returns
    -------
    float
        Maximum absolute velocity in either direction.
    """
    ny = vx.shape[0]
    nx = vx.shape[1]
    mx = 0.0
    my = 0.0
    for i in range(ny):
        for j in range(nx):
            ax = abs(vx[i, j])
            ay = abs(vy[i, j])
            if ax > mx:
                mx = ax
            if ay > my:
                my = ay
    return max(mx, my)


# Numba JIT compilation with uniform grid
@njit(parallel=True, fastmath=True, cache=True)
def _displace_coords_parallel(
    coords: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Numba-optimized version with parallel execution for uniform grids
    dx, dy: constant grid spacing
    """
    n_points = coords.shape[0]
    new_coords = coords.copy()

    # Precompute reciprocals (division is slower than multiplication)
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy

    x_min = x_coords[0]
    y_min = y_coords[0]
    max_ix = vx.shape[1] - 2
    max_iy = vy.shape[0] - 2

    for i in prange(n_points):  # Parallel loop
        x, y = coords[i, 0], coords[i, 1]

        # Direct index calculation (much faster than searchsorted)
        ix = int((x - x_min) * dx_inv)
        iy = int((y - y_min) * dy_inv)

        # Clipping
        ix = max(0, min(ix, max_ix))
        iy = max(0, min(iy, max_iy))

        # Interpolation weights (simplified with uniform grid)
        tx = (x - x_coords[ix]) * dx_inv
        ty = (y - y_coords[iy]) * dy_inv

        # Precompute weight complements
        tx1 = 1.0 - tx
        ty1 = 1.0 - ty

        # Bilinear interpolation
        w00 = tx1 * ty1
        w10 = tx * ty1
        w01 = tx1 * ty
        w11 = tx * ty

        vx_interp = w00 * vx[iy, ix] + w10 * vx[iy, ix + 1] + w01 * vx[iy + 1, ix] + w11 * vx[iy + 1, ix + 1]

        vy_interp = w00 * vy[iy, ix] + w10 * vy[iy, ix + 1] + w01 * vy[iy + 1, ix] + w11 * vy[iy + 1, ix + 1]

        # Apply displacement
        new_coords[i, 0] += vx_interp * dt
        new_coords[i, 1] += vy_interp * dt

    return new_coords


# Re-JIT the same Python function without parallel=True.
# Numba treats prange as regular range when parallel is not set,
# so this produces an identical but single-threaded kernel with no code duplication.
_displace_coords_serial = njit(fastmath=True, cache=True, parallel=False)(_displace_coords_parallel.py_func)


def displace_coords_numba(
    coords: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    use_parallel: bool = True,
) -> np.ndarray:
    """Dispatch to parallel or serial JIT variant based on use_parallel."""
    if use_parallel:
        return _displace_coords_parallel(coords, x_coords, y_coords, vx, vy, dt, dx, dy)
    return _displace_coords_serial(coords, x_coords, y_coords, vx, vy, dt, dx, dy)
