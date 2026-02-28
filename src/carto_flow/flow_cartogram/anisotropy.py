"""
Anisotropy tensor utilities for velocity field modulation.

This module provides functions for applying and constructing anisotropy tensors,
which are used to modulate velocity fields in cartogram algorithms. Anisotropy
tensors allow for direction-dependent scaling and rotation of velocity vectors,
enabling more sophisticated flow field control.

Functions
---------
apply_anisotropy_tensor
    Apply anisotropy tensor to velocity field components.
build_axis_aligned_tensor
    Construct axis-aligned anisotropy field with optional rotation.

Examples
--------
>>> from carto_flow.flow_cartogram.anisotropy import (
...     apply_anisotropy_tensor,
...     build_axis_aligned_tensor,
... )
>>> import numpy as np
>>> vx = np.ones((10, 10)) * 0.1
>>> vy = np.zeros((10, 10))
>>> Axx, Axy, Ayx, Ayy = build_axis_aligned_tensor(10, 10, Dx=2.0, Dy=0.5)
>>> vx_new, vy_new = apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy)
"""

import numpy as np

# Module-level exports - Public API
__all__ = [
    "apply_anisotropy_tensor",
    "build_axis_aligned_tensor",
]


def apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy):
    """
    Apply anisotropy tensor to velocity field components.

    Axx etc are (ny,nx) arrays. Compute:
      vx' = Axx*vx + Axy*vy
      vy' = Ayx*vx + Ayy*vy

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity field components with shape (ny, nx)
    Axx, Axy, Ayx, Ayy : np.ndarray
        Anisotropy tensor components with shape (ny, nx)

    Returns
    -------
    vx_new, vy_new : np.ndarray
        Transformed velocity field components
    """
    vx_new = Axx * vx + Axy * vy
    vy_new = Ayx * vx + Ayy * vy
    return vx_new, vy_new


def build_axis_aligned_tensor(nx, ny, Dx=1.0, Dy=1.0, theta_field=None):
    """
    Build axis-aligned anisotropy field (scalar Dx, Dy) optionally rotated by theta(x,y).

    Returns Axx, Axy, Ayx, Ayy arrays. If theta_field is None, axis-aligned (no rotation).
    theta_field shape (ny,nx) in radians if provided.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (x, y)
    Dx, Dy : float, default=1.0
        Anisotropy scaling factors in x and y directions
    theta_field : np.ndarray, optional
        Rotation field with shape (ny, nx) in radians

    Returns
    -------
    Axx, Axy, Ayx, Ayy : np.ndarray
        Anisotropy tensor components
    """
    if theta_field is None:
        Axx = np.full((ny, nx), Dx)
        Ayy = np.full((ny, nx), Dy)
        Axy = Ayx = np.zeros((ny, nx))
        return Axx, Axy, Ayx, Ayy
    else:
        # rotated tensors
        c = np.cos(theta_field)
        s = np.sin(theta_field)
        Axx = Dx * c * c + Dy * s * s
        Ayy = Dx * s * s + Dy * c * c
        Axy = (Dx - Dy) * c * s
        Ayx = Axy
        return Axx, Axy, Ayx, Ayy
