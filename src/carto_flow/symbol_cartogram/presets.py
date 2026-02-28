"""Preset configurations for common cartogram styles.

Each preset returns a dict of keyword arguments for
``create_symbol_cartogram``. Presets are the only place that
references concrete layout and options types — this coupling
is intentional and appropriate for a convenience layer.

Usage::

    from carto_flow.symbol_cartogram import create_symbol_cartogram
    from carto_flow.symbol_cartogram.presets import preset_dorling

    result = create_symbol_cartogram(gdf, "population", **preset_dorling())
"""

from __future__ import annotations

from typing import Any

from .layout import CirclePackingLayout, CirclePhysicsLayout, GridBasedLayout
from .options import (
    CirclePackingLayoutOptions,
    CirclePhysicsLayoutOptions,
    GridBasedLayoutOptions,
)


def preset_dorling() -> dict[str, Any]:
    """Classic Dorling cartogram: proportional circles, free placement."""
    return {
        "layout": CirclePhysicsLayout(CirclePhysicsLayoutOptions(topology_weight=0.5)),
        "styling": {"symbol": "circle"},
    }


def preset_topology_preserving() -> dict[str, Any]:
    """Topology-preserving cartogram with contact-constrained physics.

    Uses a two-stage algorithm that better preserves the angular
    relationships between neighboring regions.
    """
    return {
        "layout": CirclePackingLayout(CirclePackingLayoutOptions(topology_weight=0.5)),
        "styling": {"symbol": "circle"},
    }


def preset_demers() -> dict[str, Any]:
    """Demers-style cartogram: proportional squares on grid."""
    return {
        "layout": GridBasedLayout(GridBasedLayoutOptions(tiling="square")),
        "styling": {"symbol": "square"},
    }


def preset_tile_map() -> dict[str, Any]:
    """Tile map: uniform hexagons on grid.

    Note: This preset does not include ``value_column``, so symbols
    will be uniformly sized when used with ``create_symbol_cartogram``.
    """
    return {
        "layout": GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon")),
        "styling": {"symbol": "hexagon"},
    }


def preset_fast() -> dict[str, Any]:
    """Fast preview: fewer iterations, looser convergence."""
    return {
        "layout": CirclePhysicsLayout(
            CirclePhysicsLayoutOptions(
                max_iterations=100,
                convergence_tolerance=1e-3,
            ),
        ),
    }


def preset_quality() -> dict[str, Any]:
    """High quality: more iterations, tighter convergence."""
    return {
        "layout": CirclePhysicsLayout(
            CirclePhysicsLayoutOptions(
                max_iterations=1000,
                convergence_tolerance=1e-5,
                topology_weight=0.5,
            ),
        ),
    }
