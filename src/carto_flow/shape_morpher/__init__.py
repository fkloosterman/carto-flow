"""
Shape morphing algorithms for flow-based cartogram generation.

This module provides comprehensive cartogram generation algorithms that create
flow-based cartograms by iteratively deforming polygons based on density-driven
velocity fields. These are the primary entry points for creating cartograms
from geospatial data.

Main Components
---------------
Core Functions

    multiresolution_morph: Multi-resolution morphing with progressive refinement
    morph_gdf: GeoDataFrame-based interface for cartogram generation
    morph_geometries: Low-level interface working directly with shapely geometries

Object-oriented interface

    MorphComputer: Stateful cartogram generation with refinement support
    MorphOptions: Configuration options for morphing algorithms
    MorphResult: Complete results container with metadata

Examples
--------
Basic GeoDataFrame interface:

    >>> from carto_flow.shape_morpher import morph_gdf, MorphComputer, MorphOptions
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Set up computation
    >>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
    >>>
    >>> # Create cartogram with GeoDataFrame interface
    >>> result = morph_gdf(gdf, 'population', options=MorphOptions(grid=grid))
    >>> cartogram = result.geometries
    >>> history = result.history

Object-oriented approach with refinement:

    >>> computer = MorphComputer(gdf, 'population', options=MorphOptions(grid=grid))
    >>> result, history = computer.morph()

Multi-resolution cartogram for better convergence:

    >>> cartogram_multi, grids, histories, landmarks = multiresolution_morph(
    ...     gdf, 'population', resolution=512, levels=3
    ... )
"""

# Core morphing functions
# Import sub-modules
from . import anisotropy, density, displacement, grid, history, velocity

# Object-oriented interface classes
from .shape_morpher import (
    MorphComputer,
    MorphOptions,
    MorphResult,
    morph_gdf,
    morph_geometries,
    multiresolution_morph,
)

# Define public API for explicit control over what is exported
__all__ = [
    # Object-oriented interface
    "MorphComputer",
    "MorphOptions",
    "MorphResult",
    # Sub-modules
    "anisotropy",
    "density",
    "displacement",
    "grid",
    "history",
    "morph_gdf",
    "morph_geometries",
    # Core morphing functions
    "multiresolution_morph",
    "velocity",
]
