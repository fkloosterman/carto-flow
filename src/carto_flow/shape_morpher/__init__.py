"""
Core cartogram generation algorithms for flow-based shape deformation.

This module provides algorithms for creating contiguous cartograms where region
areas are proportional to a data variable. It uses a diffusion-based flow
algorithm that smoothly deforms geometries while preserving topology and
spatial relationships.

Functions
---------
morph_gdf
    Primary interface for cartogram generation from GeoDataFrames.
morph_geometries
    Low-level interface working directly with Shapely geometries.
multiresolution_morph
    Progressive multi-resolution morphing for better convergence.

Classes
-------
MorphComputer
    Stateful object-oriented interface for iterative refinement workflows.
MorphOptions
    Configuration dataclass with validation and quality presets.
MorphResult
    Result container with morphed geometries and convergence metrics.
MorphStatus
    Enum indicating morphing outcome (CONVERGED, STALLED, COMPLETED, FAILED).

Notes
-----
**Quality Presets**

MorphOptions provides convenience presets for common use cases:

- ``MorphOptions.preset_fast()`` - Fewer iterations, faster execution
- ``MorphOptions.preset_balanced()`` - Good quality/speed trade-off
- ``MorphOptions.preset_high_quality()`` - More iterations, tighter tolerance

**MorphResult Attributes**

The result object contains:

- ``geometries`` - Morphed GeoDataFrame or geometry list
- ``status`` - MorphStatus enum (CONVERGED, STALLED, COMPLETED, FAILED)
- ``iterations_completed`` - Number of iterations run
- ``final_mean_error`` - Mean log2 area error
- ``final_max_error`` - Max log2 area error
- ``displacement_field`` - Displacement field (if save_internals=True)
- ``history`` - Iteration history (if save_history=True)

**Sub-modules**

- ``velocity`` - Velocity field computation using FFTW
- ``density`` - Density field computation from geometries
- ``displacement`` - Coordinate displacement with numba acceleration
- ``grid`` - Grid utilities for spatial discretization
- ``history`` - Iteration history and snapshot management
- ``anisotropy`` - Anisotropy tensor utilities for velocity modulation
- ``animation`` - Animation generation utilities
- ``visualization`` - Visualization tools
- ``serialization`` - Result serialization
- ``metrics`` - Computation metrics
- ``comparison`` - Result comparison utilities

Examples
--------
Create a population cartogram:

>>> import geopandas as gpd
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> gdf = gpd.read_file("regions.geojson")
>>> result = morph_gdf(gdf, "population", options=MorphOptions.preset_balanced())
>>> cartogram = result.geometries
>>> print(f"Status: {result.status}")
>>> print(f"Iterations: {result.iterations_completed}")

Multi-resolution morphing for better convergence:

>>> from carto_flow.shape_morpher import multiresolution_morph
>>> result = multiresolution_morph(gdf, "population", levels=3, resolution=512)

Custom configuration:

>>> options = MorphOptions(
...     grid_size=512,
...     n_iter=100,
...     dt=0.5,
...     mean_tol=0.01,
...     max_tol=0.1,
...     save_history=True,
...     show_progress=True,
... )

Using MorphComputer for iterative refinement:

>>> from carto_flow.shape_morpher import MorphComputer
>>> computer = MorphComputer(gdf, column="population")
>>> computer.set_computation(n_iter=50, dt=0.5)
>>> computer.set_grid(grid_size=256)
>>> result = computer.morph()  # Initial morphing
>>> result = computer.morph()  # Further refinement
>>> computer.rollback()  # Undo last refinement
>>> computer.reset()  # Reset to original geometries

Accessing sub-modules:

>>> from carto_flow.shape_morpher import velocity, density, grid
>>> from carto_flow.shape_morpher.velocity import VelocityComputerFFTW

See Also
--------
[carto_flow.shape_splitter][] : Shape splitting and manipulation algorithms.
"""

# Import sub-modules
from . import (
    animation,
    anisotropy,
    comparison,
    density,
    displacement,
    grid,
    history,
    metrics,
    serialization,
    velocity,
    visualization,
)

# Core morphing functions and classes (from split modules)
from .algorithm import morph_geometries
from .api import morph_gdf, multiresolution_morph
from .computer import MorphComputer, RefinementRun
from .options import (
    MorphOptions,
    MorphOptionsConsistencyError,
    MorphOptionsError,
    MorphOptionsValidationError,
    MorphStatus,
)
from .result import MorphResult

# Define public API for explicit control over what is exported
__all__ = [
    "MorphComputer",
    "MorphOptions",
    "MorphOptionsConsistencyError",
    "MorphOptionsError",
    "MorphOptionsValidationError",
    "MorphResult",
    "MorphStatus",
    "RefinementRun",
    "animation",
    "anisotropy",
    "comparison",
    "density",
    "displacement",
    "grid",
    "history",
    "metrics",
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
    "serialization",
    "velocity",
    "visualization",
]
