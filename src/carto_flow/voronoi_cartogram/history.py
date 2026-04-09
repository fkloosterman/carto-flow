"""Snapshot types for the Voronoi cartogram iteration history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carto_flow._history import BaseSnapshot

__all__ = ["VoronoiSnapshot"]


@dataclass
class VoronoiSnapshot(BaseSnapshot):
    """Snapshot of Voronoi cartogram state at one iteration.

    Attributes
    ----------
    iteration : int
        The iteration number this snapshot represents.
    positions : np.ndarray, shape (G, 2)
        Centroid positions after this iteration.
    area_cv : float
        Coefficient of variation of Voronoi cell areas at this iteration
        (0 = perfect equal-area distribution).
    cells : np.ndarray of shapely.Geometry or None
        Clipped Voronoi cell polygons, shape (G,).  ``None`` when
        ``VoronoiOptions.record_cells=False`` (default) to save memory.
    """

    iteration: int
    positions: np.ndarray
    area_cv: float
    cells: np.ndarray | None = None

    def __repr__(self) -> str:
        cells_info = f"cells={len(self.cells)}" if self.cells is not None else "cells=None"
        return (
            f"VoronoiSnapshot(iter={self.iteration}, n={len(self.positions)}, area_cv={self.area_cv:.4f}, {cells_info})"
        )
