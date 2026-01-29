"""
Geometry optimization utilities for efficient polygon processing.

This package provides high-performance functions for polygon area computation,
geometry unpacking/reconstruction, and coordinate manipulation using Numba JIT
compilation and vectorized operations.

Key Features
------------
- Numba-optimized area computation for polygons with holes
- Efficient geometry serialization/deserialization
- Vectorized coordinate transformations
- Parallel processing support for large datasets

Main Components
---------------
- Area computation functions (compute_polygon_area_numba, compute_complex_polygon_areas_numba)
- Geometry processing functions (unpack_geometry, reconstruct_geometry, unpack_geometries)
- Data structures (GeometryCoordinateInfo)

Examples
--------
    >>> from carto_flow.optimizations import unpack_geometries, GeometryCoordinateInfo
    >>> from shapely.geometry import Polygon
    >>>
    >>> # Process multiple polygons efficiently
    >>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    >>> coord_info = unpack_geometries(polygons, precompute_ring_info=True)
    >>>
    >>> # Transform coordinates in-place
    >>> coord_info.coords += displacement_vector
    >>>
    >>> # Compute areas efficiently
    >>> areas = coord_info.compute_areas(use_parallel=True)
    >>>
    >>> # Reconstruct geometries only when needed
    >>> final_polygons = reconstruct_geometries(coord_info)
"""

# Import and re-export main functions and classes from geometry module
from .geometry import (
    # Data structures
    GeometryCoordinateInfo,
    compute_complex_polygon_areas_numba,
    # Area computation functions
    compute_polygon_area_numba,
    reconstruct_geometries,
    reconstruct_geometry,
    unpack_geometries,
    # Geometry processing functions
    unpack_geometry,
)

# Define public API for explicit control over what is exported
__all__ = [
    # Data structures
    "GeometryCoordinateInfo",
    "compute_complex_polygon_areas_numba",
    # Area computation functions
    "compute_polygon_area_numba",
    "reconstruct_geometries",
    "reconstruct_geometry",
    "unpack_geometries",
    # Geometry processing functions
    "unpack_geometry",
]
