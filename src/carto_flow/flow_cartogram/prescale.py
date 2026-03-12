"""
Pre-scaling of connected geometry components for flow cartogram generation.

Before running the morphing algorithm, each connected component (group of
geometrically adjacent polygons) can be uniformly scaled to its target total
area. This reduces outer-boundary drift because the morphing only needs to
redistribute density *within* each component rather than also changing the
component's overall size.

Functions
---------
prescale_connected_components
    Pre-scale connected components to their target total area.

Examples
--------
>>> from shapely.geometry import box
>>> from carto_flow.flow_cartogram.prescale import prescale_connected_components
>>>
>>> # Two adjacent boxes (one component) and one isolated box (second component)
>>> geometries = [box(0, 0, 2, 1), box(2, 0, 4, 1), box(10, 10, 11, 11)]
>>> values = [100.0, 200.0, 50.0]
>>> target_density = 100.0
>>> scaled = prescale_connected_components(geometries, values, target_density)
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from shapely import affinity

from ..geo_utils.adjacency import find_adjacent_pairs

__all__ = ["prescale_connected_components"]


# ============================================================================
# Connected-Component Detection (Union-Find)
# ============================================================================


def _find_connected_components(n: int, pairs: list[tuple[int, int, float]]) -> list[list[int]]:
    """Identify connected components via Union-Find with path compression.

    Parameters
    ----------
    n : int
        Total number of nodes.
    pairs : list of (i, j, weight)
        Edges; only i and j are used, weight is ignored.

    Returns
    -------
    list of list of int
        Each inner list contains indices belonging to one component.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j, _ in pairs:
        union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    return list(components.values())


# ============================================================================
# Pre-Scaling
# ============================================================================


def prescale_connected_components(
    geometries: list,
    values: np.ndarray,
    target_density: float,
    distance_tolerance: float | None = None,
) -> list:
    """Pre-scale each connected component to its target total area.

    Uniformly scales each group of geometrically adjacent (touching or
    overlapping) polygons so that the component's total area equals the area
    implied by the global target density. This reduces outer-boundary drift
    during the subsequent morphing iterations.

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Input polygon geometries.
    values : array-like
        Data values (e.g., population) for each geometry. Must be the same
        length as ``geometries``.
    target_density : float
        Target equilibrium density (values per area unit, unscaled). Used to
        compute each component's target total area as
        ``sum(component_values) / target_density``.
    distance_tolerance : float, optional
        Buffer distance for adjacency detection (passed to
        :func:`~carto_flow.geo_utils.adjacency.find_adjacent_pairs`).
        If ``None``, auto-computed as 0.1% of the average geometry diameter.

    Returns
    -------
    list of shapely.Geometry
        Scaled geometries in the same order as the input.

    Notes
    -----
    Each component is scaled uniformly around its area-weighted centroid, so
    the shape is preserved and the centroid stays in place. Components whose
    current area or target area is zero (or negligible) are left unchanged.
    """
    values_array = np.asarray(values, dtype=float)
    n = len(geometries)

    # Find adjacent pairs and group into connected components
    pairs = find_adjacent_pairs(geometries, distance_tolerance)
    components = _find_connected_components(n, pairs)

    result = list(geometries)

    for component_indices in components:
        component_geoms = [geometries[i] for i in component_indices]
        component_values = values_array[component_indices]

        current_area = sum(g.area for g in component_geoms)
        target_area = float(np.sum(component_values)) / target_density

        if current_area <= 0 or target_area <= 0:
            continue  # degenerate component, skip

        scale_factor = math.sqrt(target_area / current_area)

        if abs(scale_factor - 1.0) < 1e-8:
            continue  # no meaningful scaling needed

        # Area-weighted centroid as the fixed point of the scaling
        cx = sum(g.centroid.x * g.area for g in component_geoms) / current_area
        cy = sum(g.centroid.y * g.area for g in component_geoms) / current_area
        origin = (cx, cy)

        for idx, geom in zip(component_indices, component_geoms, strict=False):
            result[idx] = affinity.scale(geom, scale_factor, scale_factor, origin=origin)

    return result
