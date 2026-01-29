"""
Flow-based cartogram algorithms for shape deformation and analysis.

This module provides comprehensive cartogram generation algorithms that create
flow-based cartograms by iteratively deforming polygons based on density-driven
velocity fields. These are the primary entry points for creating cartograms
from geospatial data.

Main Components
---------------
Core Functions
    flow_morph: Core single-resolution cartogram algorithm with full control
    morph_gdf: GeoDataFrame-based interface for cartogram generation
    morph_geometries: Low-level interface working directly with shapely geometries

Multi-resolution algorithms
    multires_morph: Multi-resolution morphing with progressive refinement
    multiresolution_morph: Enhanced multi-resolution using MorphComputer infrastructure

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

# Standard library imports
import copy
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Union

# Third-party imports
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

# Local imports - Geometry optimization functions
from ..optimizations.geometry import (
    reconstruct_geometries,
    unpack_geometries,
)

# Local imports - Core dependencies
from .density import compute_density_field_from_geometries
from .displacement import (
    displace_coords_numba,
)
from .grid import Grid, build_multilevel_grids
from .history import (
    CartogramInternalsSnapshot,
    CartogramSnapshot,
    History,
)
from .velocity import VelocityComputerFFTW

# Module-level exports - Public API
__all__ = [
    "MorphComputer",
    "MorphOptions",
    "MorphResult",
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
]


def multiresolution_morph(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    resolution: int = 512,
    levels: int = 3,
    margin: float = 0.5,
    square: bool = True,
    options: Union["MorphOptions", list["MorphOptions"], None] = None,
    return_computer: bool = False,
    displacement_coords: Any = None,
) -> Union[
    tuple[Any, list[Grid], list[History], Any, Any], tuple["MorphComputer", list[Grid], list[History], Any, Any]
]:
    """Multi-resolution morphing using MorphComputer infrastructure.

    This function creates cartograms at multiple resolution levels, using the
    MorphComputer class for better state management and refinement capabilities.
    Each resolution level refines the results from the previous level.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values
    column : str
        Name of the column containing data values for cartogram generation
    landmarks : Any, optional
        Optional landmark geometries for tracking reference points
    resolution : int, default=512
        Base resolution for the highest level grid
    levels : int, default=3
        Number of resolution levels to use
    margin : float, default=0.5
        Margin around the data bounds for grid creation
    square : bool, default=True
        Whether to create a square grid or rectangular
    options : MorphOptions, List[MorphOptions], or None, optional
        Configuration options for the morphing process:
        - None: Use default options for all levels
        - MorphOptions: Apply same options to all levels
        - List[MorphOptions]: Different options for each level
    return_computer : bool, default=False
        If True, also return the MorphComputer instance for advanced usage
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.

    Returns
    -------
    gdf_result : Any
        GeoDataFrame with morphed cartogram geometries
    grids : List[Grid]
        List of grids used for each resolution level
    histories : List[History]
        List of computation histories for each resolution level
    landmarks_result : Any
        Morphed landmark geometries (if landmarks provided)
    displacement_result : Any
        Final displacement field result (if displacement_coords provided)
    computer : MorphComputer, optional
        MorphComputer instance (only if return_computer=True)

    Examples
    --------
    >>> # Basic usage with default options
    >>> result, grids, histories, landmarks = multiresolution_morph(
    ...     gdf, 'population'
    ... )

    >>> # Single options object for all levels
    >>> opts = MorphOptions(n_iter=100, dt=0.2, density_smooth=1.0)
    >>> result, grids, histories, landmarks = multiresolution_morph(
    ...     gdf, 'population', options=opts
    ... )

    >>> # Different options per level
    >>> level_opts = [
    ...     MorphOptions(n_iter=50, dt=0.3, mean_tol=0.1),
    ...     MorphOptions(n_iter=100, dt=0.2, mean_tol=0.05),
    ...     MorphOptions(n_iter=150, dt=0.1, mean_tol=0.02)
    ... ]
    >>> result, grids, histories, landmarks = multiresolution_morph(
    ...     gdf, 'population', options=level_opts
    ... )

    >>> # Access to MorphComputer for advanced refinement
    >>> computer, grids, histories, landmarks, displacement = multiresolution_morph(
    ...     gdf, 'population', options=opts, return_computer=True
    ... )
    >>> computer.set_computation(mean_tol=0.01)
    >>> computer.morph()

    >>> # Multi-resolution cartogram with displacement field
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run multi-resolution morphing with displacement field computation
    >>> result, grids, histories, landmarks, displacement_field = multiresolution_morph(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,
    ...     levels=3
    ... )
    >>>
    >>> # Access displacement field results
    >>> print(f"Displacement field computed: {displacement_field is not None}")
    >>>
    >>> # With MorphComputer for advanced displacement field workflows
    >>> computer, grids, histories, landmarks, displacement = multiresolution_morph(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,
    ...     return_computer=True
    ... )
    >>> print(f"MorphComputer has displacement field: {'displacement' in str(computer)}")
    """

    # 1. Build multi-resolution grids with desired maximum resolution and number of levels
    grids = build_multilevel_grids(gdf.total_bounds, resolution, levels, margin=margin, square=square)

    # 2. Normalize options input
    if options is None:
        options = [MorphOptions()]
    elif not isinstance(options, (tuple, list)):
        options = [options]

    # For single option or None, cycle it for all levels
    # For multiple options, validate length matches levels
    if len(options) == 1:
        # Simple cycle implementation for single option
        options_cycle = (options[0] for _ in range(levels))
    else:
        if len(options) != levels:
            raise ValueError(f"Length of options ({len(options)}) must match levels ({levels})")
        options_cycle = options

    # 3. Initialize MorphComputer (options will be set in the loop)
    computer = MorphComputer(
        gdf=gdf,
        column=column,
        landmarks=landmarks,
        displacement_coords=displacement_coords,
        options=MorphOptions(),  # Basic options, will be overridden in loop
    )

    # 4. Progressive refinement through resolution levels
    histories = []

    for level, (grid, opts) in enumerate(zip(grids, options_cycle)):
        # Create options with grid and custom message only for levels we actually compute
        level_opts = copy.deepcopy(opts)
        level_opts.grid = grid
        level_opts.progress_message = f"{'Refining' if level > 0 else 'Morphing'} with {grid.sx}x{grid.sy} grid"

        # Update options for current level
        computer.set_options(level_opts)

        # Run morphing for this level
        gdf_result, history = computer.morph()
        histories.append(history)

        # Check for convergence using the status from the last refinement run
        last_run = computer.get_run_info()
        if last_run and last_run.status == "converged":
            break

    # 5. Get final landmarks if provided
    landmarks_result = computer.get_landmarks_result()

    # 6. Get final displacement field results if provided
    displacement_result = computer.get_displacement_field_result()

    # 7. Return results
    if return_computer:
        return computer, grids[: level + 1], histories, landmarks_result, displacement_result
    else:
        return gdf_result, grids[: level + 1], histories, landmarks_result, displacement_result


# ============================================================================
# Custom Exceptions
# ============================================================================


class MorphOptionsError(ValueError):
    """Base exception for MorphOptions validation errors."""

    pass


class MorphOptionsValidationError(MorphOptionsError):
    """Field validation errors (type, range, format)."""

    pass


class MorphOptionsConsistencyError(MorphOptionsError):
    """Logical consistency errors between related fields."""

    pass


# ============================================================================
# Configuration and Result Data Classes
# ============================================================================


@dataclass
class MorphOptions:
    """Configuration options for morphing algorithms with comprehensive validation.

    This class validates all options to ensure numerical stability and logical
    consistency for cartogram generation algorithms. Validation occurs during
    construction, direct attribute assignment, and when creating modified copies.

    Grid specification (in order of precedence):
        1. grid: Pre-constructed Grid object (highest precedence)
        2. grid_size + grid_margin + grid_square: Construction parameters
        3. Default: 100x100 grid if nothing specified

    Note
    ----
    When 'grid' is provided, 'grid_size', 'grid_margin', and 'grid_square' are ignored.

    Validation
    ----------
    - All numeric parameters are validated for type and range constraints
    - Logical consistency between related parameters is enforced
    - Invalid values raise MorphOptionsValidationError with detailed messages
    - Validation occurs on construction, direct assignment, and copy operations
    """

    # Grid construction parameters
    grid: Optional[Grid] = None  # For advanced users
    grid_size: Union[int, tuple[Optional[int], Optional[int]], None] = 256
    grid_margin: float = 0.5
    grid_square: bool = False

    # Computation options
    density_smooth: Optional[float] = None
    dt: float = 0.2
    n_iter: int = 100
    recompute_every: Optional[int] = None
    snapshot_every: Optional[int] = None
    mean_tol: float = 0.05
    max_tol: float = 0.1

    # Anisotropy options
    Dx: float = 1.0
    Dy: float = 1.0
    anisotropy: Optional[Callable[[Grid], tuple[np.ndarray, np.ndarray]]] = None

    # Smoothing options
    vsmooth: Optional[float] = None

    # Output options
    save_history: bool = True
    save_internals: bool = False
    show_progress: bool = True
    progress_message: Optional[str] = None

    def get_grid(self, bounds: tuple[float, float, float, float]) -> Grid:
        """Get the Grid object for computation with clear precedence.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            Bounding box as (min_x, min_y, max_x, max_y)

        Returns
        -------
        Grid
            Grid object for spatial discretization
        """
        if isinstance(self.grid, Grid):
            # Grid object provided - use it, ignore other grid params
            return self.grid
        elif self.grid_size is not None:
            # Use grid construction parameters
            return Grid.from_bounds(bounds, self.grid_size, self.grid_margin, self.grid_square)
        else:
            # Default grid
            return Grid.from_bounds(bounds, size=100)

    def copy_with(self, **kwargs) -> "MorphOptions":
        """Create a copy of this MorphOptions instance with specified options overridden.

        Parameters
        ----------
        **kwargs
            Keyword arguments corresponding to MorphOptions fields to override

        Returns
        -------
        MorphOptions
            New MorphOptions instance with the specified options overridden

        Examples
        --------
        Create new options with different iteration count and time step:

        >>> new_opts = opts.copy_with(n_iter=200, dt=0.15)

        Create options with different tolerance values:

        >>> strict_opts = opts.copy_with(mean_tol=0.01, max_tol=0.02)

        Chain multiple modifications:

        >>> final_opts = opts.copy_with(n_iter=150).with_options(dt=0.1)
        """
        # Validate that all provided kwargs are valid MorphOptions fields
        valid_fields = {field.name for field in self.__dataclass_fields__.values()}
        invalid_fields = set(kwargs.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid field(s): {invalid_fields}. Valid fields are: {valid_fields}")

        # Use dataclasses.replace to create a new instance with overridden fields
        # This will automatically trigger validation via __post_init__
        try:
            new_options = replace(self, **kwargs)
            return new_options
        except Exception as e:
            # Re-raise validation errors with additional context about which fields were being modified
            field_list = ", ".join(kwargs.keys())
            if isinstance(e, MorphOptionsValidationError):
                raise MorphOptionsValidationError(f"Failed to create options with {field_list}: {e!s}")
            else:
                # Re-raise other exceptions (like dataclass field errors) as-is
                raise

    def __post_init__(self):
        """Validate options after dataclass initialization."""
        self._validate()
        # Mark that validation has been initialized
        super().__setattr__("_validation_initialized", True)

    def __setattr__(self, name: str, value) -> None:
        """Validate individual fields when assigned directly."""
        # Only validate after full initialization to avoid recursion during __init__
        # Check if we're fully initialized by looking for a marker attribute
        if (
            hasattr(self, "_validation_initialized") and name != "_validation_initialized" and not name.startswith("_")
        ):  # Skip private attributes
            # Validate the new value before setting it
            self._validate_single_field(name, value)

        # Set the attribute (only if validation passed or during initialization)
        super().__setattr__(name, value)

    def _validate(self) -> None:
        """Comprehensive validation of all fields and their consistency."""
        errors = []

        # Field-level validation
        field_errors = self._validate_all_fields()
        errors.extend(field_errors)

        # Logical consistency validation
        consistency_errors = self._validate_consistency()
        errors.extend(consistency_errors)

        if errors:
            raise MorphOptionsValidationError("; ".join(errors))

    def _validate_all_fields(self) -> list[str]:
        """Validate all individual fields and return list of error messages."""
        errors = []

        # Validate each field using the single validation method
        field_names = [
            "grid_size",
            "grid_margin",
            "grid_square",
            "density_smooth",
            "dt",
            "n_iter",
            "recompute_every",
            "snapshot_every",
            "mean_tol",
            "max_tol",
            "Dx",
            "Dy",
            "anisotropy",
            "vsmooth",
            "save_history",
            "save_internals",
            "show_progress",
            "progress_message",
        ]

        for field_name in field_names:
            field_error = self._validate_field_value(field_name, getattr(self, field_name))
            if field_error:
                errors.append(field_error)

        return errors

    def _validate_field_value(self, field_name: str, value) -> str:
        """Single source of truth for field validation logic."""
        # Grid parameters
        if field_name == "grid_size":
            if value is not None:
                return self._validate_grid_size(value)
        elif field_name == "grid_margin":
            if not isinstance(value, (int, float)) or value < 0:
                return "grid_margin must be a non-negative number"
        elif field_name == "grid_square":
            if not isinstance(value, bool):
                return "grid_square must be a boolean"

        # Computation parameters
        elif field_name == "density_smooth":
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                return "density_smooth must be a positive number"
        elif field_name == "dt":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "dt must be > 0 and ≤ 1.0"
        elif field_name == "n_iter":
            if not isinstance(value, int) or value <= 0:
                return "n_iter must be a positive integer"
        elif field_name == "recompute_every":
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    return "recompute_every must be a positive integer"
                elif hasattr(self, "n_iter") and value > self.n_iter:
                    return "recompute_every cannot be greater than n_iter"
        elif field_name == "snapshot_every":
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    return "snapshot_every must be a positive integer"
                elif hasattr(self, "n_iter") and value > self.n_iter:
                    return "snapshot_every cannot be greater than n_iter"
        elif field_name == "mean_tol":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "mean_tol must be > 0 and ≤ 1.0"
        elif field_name == "max_tol":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "max_tol must be > 0 and ≤ 1.0"

        # Anisotropy parameters
        elif field_name == "Dx":
            if not isinstance(value, (int, float)) or value <= 0:
                return "Dx must be a positive number"
        elif field_name == "Dy":
            if not isinstance(value, (int, float)) or value <= 0:
                return "Dy must be a positive number"
        elif field_name == "anisotropy":
            if value is not None and not callable(value):
                return "anisotropy must be a callable function or None"

        # Smoothing parameters
        elif field_name == "vsmooth":
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                return "vsmooth must be a non-negative number"

        # Output parameters
        elif field_name == "save_history":
            if not isinstance(value, bool):
                return "save_history must be a boolean"
        elif field_name == "save_internals":
            if not isinstance(value, bool):
                return "save_internals must be a boolean"
        elif field_name == "show_progress":
            if not isinstance(value, bool):
                return "show_progress must be a boolean"
        elif field_name == "progress_message" and value is not None and not isinstance(value, str):
            return "progress_message must be a string or None"

        return ""  # No errors

    def _validate_consistency(self) -> list[str]:
        """Validate logical consistency between related fields."""
        errors = []

        # Tolerance ordering
        if self.mean_tol > self.max_tol:
            errors.append(f"mean_tol ({self.mean_tol}) must be ≤ max_tol ({self.max_tol})")

        # Anisotropy consistency
        if self.anisotropy is not None and (self.Dx != 1.0 or self.Dy != 1.0):
            errors.append("cannot combine anisotropy function with Dx/Dy ≠ 1.0")

        return errors

    def _validate_grid_size(self, grid_size) -> Optional[str]:
        """Validate grid_size parameter format and values."""
        if isinstance(grid_size, int):
            if grid_size <= 0:
                return "grid_size as int must be positive"
        elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
            for i, size in enumerate(grid_size):
                if size is not None and (not isinstance(size, int) or size <= 0):
                    return f"grid_size[{i}] must be a positive integer or None"
        else:
            return "grid_size must be a positive integer or tuple of (int, int)"

        return None

    def _validate_single_field(self, field_name: str, value) -> None:
        """Validate a single field and its consistency with other fields."""
        errors = []

        # Validate the specific field using single source of truth
        field_errors = self._validate_field_value(field_name, value)
        if field_errors:
            errors.append(field_errors)

        # Check logical consistency with current object state
        # Create a temporary copy to check consistency without modifying self
        temp_kwargs = {field_name: value}
        temp_obj = replace(self, **temp_kwargs)
        consistency_errors = temp_obj._validate_consistency()
        if consistency_errors:
            errors.extend(consistency_errors)

        if errors:
            raise MorphOptionsValidationError(f"Invalid {field_name}={value}: {', '.join(errors)}")

    @staticmethod
    def _calculate_bounds_from_geometries(geometries) -> tuple[float, float, float, float]:
        """
        Calculate bounding box from a list of geometries.

        This helper method computes the overall bounds (min_x, min_y, max_x, max_y)
        that encompass all provided geometries. Used internally for automatic grid
        creation when no explicit bounds are provided.

        Parameters
        ----------
        geometries : iterable
            Collection of shapely geometries with bounds attribute

        Returns
        -------
        Tuple[float, float, float, float]
            Bounding box as (min_x, min_y, max_x, max_y)

        Raises
        ------
        AttributeError
            If geometries don't have bounds attribute
        """

        # Calculate bounds from all geometries
        bounds = None
        for geom in geometries:
            if bounds is None:
                bounds = geom.bounds
            else:
                bounds = (
                    min(bounds[0], geom.bounds[0]),
                    min(bounds[1], geom.bounds[1]),
                    max(bounds[2], geom.bounds[2]),
                    max(bounds[3], geom.bounds[3]),
                )
        return bounds


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class MorphResult:
    """Complete results from a morphing operation.

    This class contains all results from a morphing operation including
    geometries, computation history, convergence status, and optional
    displacement field data.

    Attributes
    ----------
    geometries : Any
        The morphed geometries (GeoDataFrame or list of geometries)
    history : History
        Computation history with snapshots of the algorithm state
    landmarks : Any, optional
        Morphed landmark geometries if landmarks were provided
    status : str, default="completed"
        Computation status ("completed", "converged", "stalled", etc.)
    history_internals : History, optional
        Internal computation data if save_internals=True
    iterations_completed : int, optional
        Number of iterations completed before termination
    final_mean_error : float, optional
        Final mean area error at convergence
    final_max_error : float, optional
        Final maximum area error at convergence
    displacement_field : Union[Tuple[np.ndarray, np.ndarray], np.ndarray], optional
        Displacement field in same format as input coordinates
    displaced_coords : np.ndarray, optional
        Final displaced coordinates for refinement workflows
    """

    # Core results
    geometries: Any
    history: History
    landmarks: Optional[Any] = None
    status: str = "completed"

    # Optional computation metadata
    history_internals: Optional[History] = None
    iterations_completed: Optional[int] = None
    final_mean_error: Optional[float] = None
    final_max_error: Optional[float] = None

    # Displacement field results
    displacement_field: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None
    displaced_coords: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """Return concise string representation for terminal display.

        Returns
        -------
        str
            String representation showing key result information
        """
        # Basic info
        geom_count = self._get_geometry_count()
        snapshot_count = len(self.history.snapshots) if self.history else 0

        # Status and metrics
        status_str = f"status={self.status}"
        if self.iterations_completed is not None:
            status_str += f", iter={self.iterations_completed}"
        if self.final_mean_error is not None:
            status_str += f", mean_error={self.final_mean_error:.4f}"
        if self.final_max_error is not None:
            status_str += f", max_error={self.final_max_error:.4f}"

        # Landmark info
        landmark_str = ", landmarks" if self.landmarks is not None else ""

        return f"MorphResult(geoms={geom_count}, snapshots={snapshot_count}, {status_str}{landmark_str})"

    def _get_geometry_count(self) -> int:
        """
        Get the number of geometries in the result.

        This helper method provides a robust way to count geometries that works
        with different input types (GeoDataFrame, list of geometries, single geometry).

        Returns
        -------
        int
            Number of geometries in the result
        """
        try:
            if hasattr(self.geometries, "__len__"):
                return len(self.geometries)
            elif hasattr(self.geometries, "shape"):
                return self.geometries.shape[0]
            else:
                return 1  # Single geometry
        except:
            return 1  # Fallback

    def get_result_tuple(self) -> tuple[Any, History]:
        """Get backward-compatible tuple format: (geometries, history)"""
        return self.geometries, self.history

    def get_full_tuple(self) -> tuple[Any, History, Optional[Any]]:
        """Get full tuple format: (geometries, history, landmarks)"""
        if self.landmarks is not None:
            return self.geometries, self.history, self.landmarks
        else:
            return self.geometries, self.history


# ============================================================================
# Core Algorithm Functions
# ============================================================================


def _detect_coordinate_format(coords):
    """
    Automatically detect the format of displacement coordinates.

    Parameters
    ----------
    coords : array-like
        Input coordinates to analyze

    Returns
    -------
    str
        Detected format: 'points', 'grid', or 'mesh'

    Raises
    ------
    ValueError
        If format cannot be determined or is ambiguous
    """
    coords_array = np.asarray(coords)

    # Check for points format: (N, 2)
    if coords_array.ndim == 2 and coords_array.shape[1] == 2:
        # Additional check: ensure values look like coordinates (not too large)
        if np.any(np.abs(coords_array) > 1e10):
            raise ValueError("Coordinates appear to contain unreasonably large values")
        return "points"

    # Check for grid format: tuple/list of (X, Y) arrays
    elif isinstance(coords, (tuple, list)) and len(coords) == 2:
        try:
            X, Y = coords
            X_array = np.asarray(X)
            Y_array = np.asarray(Y)

            # Check if they have the same shape
            if X_array.shape != Y_array.shape:
                raise ValueError("X and Y arrays in grid format must have the same shape")

            # Check for reasonable dimensions (not too large)
            if X_array.size > 1e6:
                raise ValueError("Grid format coordinates appear to be too large (>1M points)")

            return "grid"
        except (ValueError, TypeError):
            pass

    # Check for mesh format: (M, N, 2)
    elif coords_array.ndim == 3 and coords_array.shape[2] == 2:
        # Check for reasonable size
        if coords_array.size > 2e6:  # 1M coordinates * 2 values
            raise ValueError("Mesh format coordinates appear to be too large (>1M points)")
        return "mesh"

    # If we get here, format is unclear
    raise ValueError(
        "Could not determine displacement_coords format. "
        "Expected: (N, 2) points, (X, Y) grid tuple, or (M, N, 2) mesh. "
        f"Got shape: {coords_array.shape if hasattr(coords_array, 'shape') else type(coords)}"
    )


def _convert_displacement_to_input_format(displacement_field, original_coords, detected_format):
    """
    Convert displacement field back to the same format as input coordinates.

    Parameters
    ----------
    displacement_field : (N,2) array
        (Ux, Uy) displacement arrays as (N,2) array
    original_coords : array-like
        Original input coordinates to match format
    detected_format : str
        The detected format of input coordinates

    Returns
    -------
    array-like
        Displacement field in same format as input coordinates
    """
    # Ux, Uy = displacement_field

    if detected_format == "points":
        # Input was (N, 2), return (N, 2) displacement
        # return np.column_stack([Ux, Uy])
        return displacement_field
    elif detected_format == "grid":
        # Input was (X, Y) tuple, return (X, Y) displacement tuple
        X, Y = original_coords
        return (displacement_field[:, 0].reshape(X.shape), displacement_field[:, 1].reshape(Y.shape))
    elif detected_format == "mesh":
        # Input was (M, N, 2), return (M, N, 2) displacement
        coords_array = np.asarray(original_coords)
        return displacement_field.reshape(coords_array.shape)
    else:
        raise ValueError(f"Unknown format: {detected_format}")


def _normalize_coordinates(coords, format_type=None):
    """
    Normalize displacement field coordinates to (N, 2) format.

    Parameters
    ----------
    coords : array-like
        Input coordinates in various formats
    format_type : str, optional
        Format of input coordinates. If None, will be auto-detected:
        - 'points': (N, 2) array of [x, y] coordinates
        - 'grid': Tuple of (X, Y) meshgrid arrays
        - 'mesh': (M, N, 2) array of coordinates

    Returns
    -------
    np.ndarray
        Normalized coordinates as (N, 2) array
    """
    # Auto-detect format if not provided
    if format_type is None:
        format_type = _detect_coordinate_format(coords)

    if format_type == "points":
        # Already in correct format
        return np.asarray(coords)
    elif format_type == "grid":
        # Convert meshgrid to point coordinates
        X, Y = coords
        return np.column_stack([X.ravel(), Y.ravel()])
    elif format_type == "mesh":
        # Convert mesh format to points
        coords_array = np.asarray(coords)
        return coords_array.reshape(-1, 2)
    else:
        raise ValueError(f"Unknown displacement_coords_format: {format_type}")


def morph_geometries(
    geometries,
    column_values,
    original_areas=None,
    landmarks=None,
    # New options parameter for simplified API
    options: Optional[MorphOptions] = None,
    # Displacement field computation
    displacement_coords=None,
    previous_displaced_coords=None,
) -> MorphResult:
    """
    Core morphing algorithm working with shapely geometries.

    This function implements the fundamental cartogram algorithm that works
    directly with shapely geometries, making it suitable for users who want
    to work without dataframes.

    Parameters
    ----------
    geometries : List[Geometry]
        Shapely geometries to morph
    column_values : np.ndarray
        Data values driving the morphing (e.g., population values)
    original_areas : np.ndarray, optional
        Original areas for refinement mode. If None, treats as initial morphing
    landmarks : Any, optional
        Optional landmarks GeoDataFrame for tracking reference points
    options : MorphOptions, optional
        Algorithm options (dt, n_iter, density_smooth, etc.)
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.
    previous_displaced_coords : np.ndarray, optional
        Previously displaced coordinates for refinement mode

    Returns
    -------
    result : MorphResult
        Complete morphing results containing:
        - geometries: List of morphed geometries
        - landmarks: List of morphed landmark geometries (if provided)
        - history: List of snapshot data for each saved iteration
        - status: Computation status ("converged", "stalled", or "completed")
        - displacement_field: Displacement field in same format as input coordinates (if displacement_coords provided)
        - displaced_coords: Final displaced coordinates for refinement (if displacement_coords provided)

    Examples
    --------
    >>> from carto_flow.shape_morpher import morph_geometries, MorphResult
    >>> from shapely.geometry import Polygon
    >>>
    >>> # Simple morphing
    >>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    >>> values = [100]
    >>>
    >>> # Get complete result object
    >>> result = morph_geometries(polygons, values, options=options)
    >>> morphed = result.geometries
    >>> snapshots = result.history
    >>> print(f"Status: {result.status}")
    >>> print(f"Converged in {len(snapshots)} iterations")

    >>> # Compute displacement field
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run morphing with displacement field computation (auto-detects format)
    >>> result = morph_geometries(
    ...     polygons, values,
    ...     displacement_coords=displacement_coords,
    ...     options=options
    ... )
    >>>
    >>> # Access displacement field results (same format as input)
    >>> displacement_field = result.displacement_field  # Format matches displacement_coords
    >>> displaced_coords = result.displaced_coords      # For refinement
    >>>
    >>> # Refine with displacement field (format auto-detected)
    >>> refined_result = morph_geometries(
    ...     result.geometries, values,
    ...     displacement_coords=displacement_coords,
    ...     previous_displaced_coords=result.displaced_coords,
    ...     options=options
    ... )
    """

    # --- Handle options ---
    if options is None:
        # Create options from explicit parameters (backward compatibility)
        options = MorphOptions()

    # 1. Setup and validation
    # Handle different input types for geometries
    if hasattr(geometries, "values"):
        # pandas Series or numpy array
        geom_list = geometries.values
    else:
        # List of geometries
        geom_list = geometries

    if len(geom_list) != len(column_values):
        raise ValueError("geometries and column_values must have same length")

    # 2. Compute areas and targets
    current_areas = np.array([g.area for g in geom_list])

    if original_areas is None:
        original_areas = current_areas

    # 3. Compute algorithm inputs
    column_values_array = np.asarray(column_values)
    mean_density = float(np.sum(column_values_array) / np.sum(original_areas))
    target_areas = np.sum(original_areas) * column_values_array / np.sum(column_values_array)

    # 4. Initialize algorithm state
    flat_geoms = unpack_geometries(geom_list)
    # Resolve grid using options
    grid = options.get_grid(options._calculate_bounds_from_geometries(geom_list))
    velocity_computer = VelocityComputerFFTW(grid, Dx=options.Dx, Dy=options.Dy)

    # Handle landmarks
    flat_landmarks_geoms = unpack_geometries(landmarks) if landmarks is not None else None

    # Handle displacement field coordinates
    displacement_coords_original = None
    displacement_coords_current = None
    detected_format = None
    if displacement_coords is not None:
        # Detect and normalize input format to (N, 2) array
        detected_format = _detect_coordinate_format(displacement_coords)
        displacement_coords_original = _normalize_coordinates(displacement_coords)

        if previous_displaced_coords is not None:
            # Validate that displacement coordinates match in size
            original_size = len(displacement_coords_original)
            previous_size = len(previous_displaced_coords)
            if original_size != previous_size:
                raise ValueError(
                    f"displacement_coords and previous_displaced_coords must have the same number of points. "
                    f"Got {original_size} and {previous_size} points respectively."
                )

            # Refinement mode - use previously displaced coordinates
            displacement_coords_current = previous_displaced_coords.copy()
        else:
            # Initial mode - start with original coordinates
            displacement_coords_current = displacement_coords_original.copy()

    # 5. Main algorithm loop with snapshotting
    history = History() if options.save_history else None
    history_internals = History() if options.save_internals else None
    # snapshots = []
    # snapshots_internal = []

    # --- Initial snapshot ---
    if options.save_history:
        current_areas = flat_geoms.compute_areas(use_parallel=True)
        errors = (current_areas - target_areas) / target_areas
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))

        snapshot = CartogramSnapshot(
            iteration=0,
            geometry=geom_list,
            area_errors=errors.copy(),
            mean_error=mean_error,
            max_error=max_error,
        )
        history.add_snapshot(snapshot)

    last_mean_error = np.inf
    stalled_acc = 0

    msg = "Morph geometries" if options.progress_message is None else options.progress_message
    pbar = tqdm.trange(options.n_iter, desc=msg, disable=not options.show_progress, miniters=1, mininterval=0)

    for step in pbar:
        # 1. Compute density field
        if (options.recompute_every is not None and step % options.recompute_every == 0) or step == 0:
            # Compute density field directly from geometries (no dataframe dependency)
            current_geoms = reconstruct_geometries(flat_geoms)
            rho = compute_density_field_from_geometries(
                current_geoms, column_values, grid, mean_density, options.density_smooth
            )

            # 2. Compute baseline velocity field
            vx, vy = velocity_computer.compute(rho)

            # 3. Velocity modulation
            if options.anisotropy is not None:
                fx, fy = options.anisotropy(grid)
                vx_mod = vx * fx
                vy_mod = vy * fy
            else:
                vx_mod, vy_mod = vx, vy

            # 4. Normalize for stability
            vmax = np.nanmax(np.sqrt(vx_mod**2 + vy_mod**2))
            if vmax > 1e-12:
                vx_mod /= vmax
                vy_mod /= vmax

            # Smooth velocity
            if options.vsmooth is not None and options.vsmooth > 0.0:
                vx_mod = gaussian_filter(vx_mod, sigma=options.vsmooth, mode="reflect")
                vy_mod = gaussian_filter(vy_mod, sigma=options.vsmooth, mode="reflect")

            if options.save_internals:
                snapshot = CartogramInternalsSnapshot(
                    iteration=step + 1,
                    rho=rho.copy(),
                    vx=vx.copy(),
                    vy=vy.copy(),
                    vx_mod=vx_mod.copy() if options.anisotropy is not None else None,
                    vy_mod=vy_mod.copy() if options.anisotropy is not None else None,
                )
                history_internals.add_snapshot(snapshot)

        # 5. Displace geometries
        max_v = max(np.max(np.abs(vx_mod)), np.max(np.abs(vy_mod)), 1e-8)
        dt_prime = options.dt * min(grid.dx, grid.dy) / max_v

        flat_geoms.coords = displace_coords_numba(
            flat_geoms.coords, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
        )
        flat_geoms.invalidate_cache()

        if landmarks is not None:
            flat_landmarks_geoms.coords = displace_coords_numba(
                flat_landmarks_geoms.coords, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
            )
            # no need to invalidate cache, because we do not compute the areas

        # Displace displacement field coordinates
        if displacement_coords is not None:
            displacement_coords_current[:] = displace_coords_numba(
                displacement_coords_current, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
            )

        # 6. Convergence stats
        current_areas = flat_geoms.compute_areas(use_parallel=True)
        errors = (current_areas - target_areas) / target_areas
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))

        converged = mean_error < options.mean_tol and max_error < options.max_tol
        stalled_acc += mean_error > last_mean_error
        stalled = stalled_acc > 5

        last_mean_error = mean_error

        status = (
            "converged"
            if converged
            else "stalled"
            if stalled
            else "completed"
            if (step + 1) == options.n_iter
            else "running"
        )

        # 7. Create snapshot data at appropriate iterations
        if options.save_history and (
            options.snapshot_every is None
            or step % options.snapshot_every == 0
            or converged
            or stalled
            or (step + 1) == options.n_iter
        ):
            snapshot_data = CartogramSnapshot(
                iteration=step + 1,
                geometry=reconstruct_geometries(flat_geoms),
                area_errors=errors.copy(),
                mean_error=mean_error,
                max_error=max_error,
            )
            history.add_snapshot(snapshot_data)

        pbar.set_postfix_str(f"max={100 * max_error:.1f}%, mean={100 * mean_error:.1f}% - {status}")

        if converged or stalled:
            pbar.update()
            pbar.close()
            break

    # 8. Return final geometries and snapshot data
    final_geometries = reconstruct_geometries(flat_geoms)

    # Handle landmarks if provided
    if landmarks is not None:
        landmarks = reconstruct_geometries(flat_landmarks_geoms)

    # Compute displacement field results
    displacement_field = None
    displaced_coords_result = None
    if displacement_coords is not None:
        # Compute displacement field (displaced - original)
        displacement_deltas = displacement_coords_current - displacement_coords_original

        # Convert displacement field back to input format
        displacement_field = _convert_displacement_to_input_format(
            displacement_deltas, displacement_coords, detected_format
        )
        # Return displaced coordinates for future refinement (no copy needed)
        displaced_coords_result = displacement_coords_current

    # Create result object
    result = MorphResult(
        geometries=final_geometries,
        landmarks=landmarks,
        history=history,
        status=status,
        history_internals=history_internals,
        displacement_field=displacement_field,
        displaced_coords=displaced_coords_result,
    )
    return result


def morph_gdf(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    # New options parameter for simplified API
    options: Optional[MorphOptions] = None,
    # Displacement field computation
    displacement_coords=None,
    previous_displaced_coords=None,
) -> MorphResult:
    """
    Generate flow-based cartogram using dataframe interface with morph_polygons core algorithm.

    This function provides a GeoDataFrame-based interface that handles dataframe validation,
    metadata management, and target area computation, while delegating the core morphing
    algorithm to the morph_polygons function.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values.
        Must have 'geometry' column with Polygon/MultiPolygon objects and
        the specified column with numeric data values.
    column : str
        Name of the column in gdf containing the data values that drive the
        cartogram deformation (e.g., 'population', 'income', 'density').
    landmarks : Any, optional
        Optional landmarks GeoDataFrame for tracking reference points
    options : MorphOptions, optional
        Algorithm options (dt, n_iter, density_smooth, etc.)
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.
    previous_displaced_coords : np.ndarray, optional
        Previously displaced coordinates for refinement mode

    Returns
    -------
    result : MorphResult
        Complete morphing results containing:
        - geometries: GeoDataFrame with deformed geometries
        - history: History object with computation snapshots
        - status: Computation status ("completed", "converged", etc.)
        - Optional: landmarks, history_internals, performance metrics
        - displacement_field: Displacement field in same format as input coordinates (if displacement_coords provided)
        - displaced_coords: Final displaced coordinates for refinement (if displacement_coords provided)

        For backward compatibility, also returns tuple: (gdf_result, history)

    Examples
    --------
    >>> import geopandas as gpd
    >>> from carto_flow.shape_morpher import morph_gdf, MorphResult
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Load geographic data
    >>> gdf = gpd.read_file('regions.geojson')
    >>> grid = Grid.from_bounds(gdf.total_bounds, size=100)
    >>>
    >>> # Create population-based cartogram (returns MorphResult)
    >>> result = morph_gdf(gdf, 'population', grid, n_iter=150)
    >>> print(f"Status: {result.status}")
    >>> print(f"Iterations: {result.iterations_completed}")
    >>>
    >>> # Access results
    >>> cartogram = result.gdf_result
    >>> history = result.history
    >>>
    >>> # Backward compatibility - still works as tuple
    >>> cartogram, history = morph_gdf(gdf, 'population', grid, n_iter=150)
    >>> print(f"Converged in {len(history.snapshots)} iterations")

    >>> # Compute displacement field with GeoDataFrame interface
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run morphing with displacement field computation
    >>> result = morph_gdf(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,
    ...     options=MorphOptions(n_iter=100, dt=0.2)
    ... )
    >>>
    >>> # Access displacement field results
    >>> displacement_field = result.displacement_field  # Format matches displacement_coords
    >>> displaced_coords = result.displaced_coords      # For refinement
    >>>
    >>> # Refine with displacement field
    >>> refined_result = morph_gdf(
    ...     result.geometries, 'population',
    ...     displacement_coords=displacement_coords,
    ...     previous_displaced_coords=result.displaced_coords,
    ...     options=MorphOptions(n_iter=50, dt=0.1)
    ... )
    """

    # --- Handle options ---
    if options is None:
        # Create default options
        options = MorphOptions()

    # --- Input validation and dataframe setup ---
    is_morphed, _refinement_id = _validate_morphed_dataframe(gdf, column)

    if is_morphed:
        # Refinement mode - use existing _original_areas
        gdf_work = gdf.copy()
        original_areas = gdf["_original_areas"].values
    else:
        # Initial mode - establish baseline
        gdf_work = gdf.copy()
        original_areas = gdf_work.area.values
        gdf_work["_original_areas"] = original_areas

        # Store metadata for future refinements
        gdf_work.attrs["cartoflow"] = {"morphed": True, "column_used": column, "refinement_id": str(uuid.uuid4())[:8]}

    # --- Prepare geometries and landmarks for morph_polygons ---
    geometries = gdf_work.geometry.values
    column_values = np.array(gdf_work[column])

    # Handle landmarks
    landmarks_geoms = None
    if landmarks is not None:
        landmarks_geoms = landmarks.geometry.values

    # Call the core morphing algorithm
    geometry_result = morph_geometries(
        geometries,
        column_values,
        original_areas,
        landmarks_geoms,
        options=options,
        displacement_coords=displacement_coords,
        previous_displaced_coords=previous_displaced_coords,
    )

    # --- Process results and snapshots ---
    if options.save_history and geometry_result.history:
        for snapshot in geometry_result.history.snapshots:
            snapshot_geoms = snapshot.geometry
            snapshot.geometry = gdf_work.copy()
            snapshot.geometry.geometry = snapshot_geoms

    # --- Create final GeoDataFrame ---
    gdf_work.geometry = geometry_result.geometries
    geometry_result.geometries = gdf_work

    # Handle landmarks if provided
    if geometry_result.landmarks is not None:
        landmarks_result = landmarks.copy()
        landmarks_result.geometry = geometry_result.landmarks
        geometry_result.landmarks = landmarks_result

    return geometry_result


def _validate_morphed_dataframe(gdf: Any, column: str) -> tuple[bool, str]:
    """
    Validate if dataframe is properly morphed and return refinement_id.

    This helper function checks if a GeoDataFrame has been previously morphed
    by looking for cartoflow metadata and _original_areas column. Used to
    determine if subsequent operations should be refinement or initial morphing.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object to validate
    column : str
        Name of the column being used for morphing

    Returns
    -------
    tuple[bool, str]
        (is_morphed, refinement_id) where:
        - is_morphed: True if dataframe was previously morphed
        - refinement_id: UUID string of the original morphing session, or None

    Raises
    ------
    ValueError
        If dataframe was morphed with different column than currently specified
    """
    """Validate if dataframe is properly morphed and return refinement_id."""
    # Check for metadata
    if not hasattr(gdf, "attrs") or "cartoflow" not in gdf.attrs:
        return False, None

    meta = gdf.attrs["cartoflow"]
    if not meta.get("morphed", False):
        return False, None

    # Check for _original_areas column
    if "_original_areas" not in gdf.columns:
        return False, None

    # Validate column consistency
    stored_column = meta.get("column_used")
    if stored_column and stored_column != column:
        raise ValueError(
            f"Dataframe was created using column '{stored_column}' "
            f"but refinement is using column '{column}'. "
            f"Use column '{stored_column}' for refinement."
        )

    return True, meta.get("refinement_id")


# ============================================================================
# Object-Oriented Interface
# ============================================================================

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RefinementRun:
    """Record of a single refinement run."""

    run_id: int
    options: MorphOptions
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Complete morphing result object
    morph_result: Optional[MorphResult] = None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Duration info
        if self.duration is not None:
            duration_str = f", duration={self.duration:.2f}s"
        elif self.end_time is not None:
            duration_str = f", duration={self.end_time - self.start_time:.2f}s"
        else:
            duration_str = ""

        # Status info
        if self.morph_result is not None:
            status_str = f", status={self.morph_result.status}"
            if self.morph_result.iterations_completed is not None:
                status_str += f", iter={self.morph_result.iterations_completed}"
        else:
            status_str = ", status=running"

        return f"RefinementRun(id={self.run_id}{duration_str}{status_str})"

    # Convenience properties for backward compatibility
    @property
    def gdf_result(self) -> Optional[Any]:
        """Get the morphed geometries (GeoDataFrame from morph_gdf, List from morph_geometries)."""
        return self.morph_result.geometries if self.morph_result else None

    @property
    def history(self) -> Optional[History]:
        """Get the computation history."""
        return self.morph_result.history if self.morph_result else None

    @property
    def landmarks_result(self) -> Optional[Any]:
        """Get the morphed landmark geometries."""
        return self.morph_result.landmarks if self.morph_result else None

    @property
    def status(self) -> Optional[str]:
        """Get the computation status."""
        return self.morph_result.status if self.morph_result else None

    @property
    def history_internals(self) -> Optional[History]:
        """Get internal computation data."""
        return self.morph_result.history_internals if self.morph_result else None

    @property
    def iterations_completed(self) -> Optional[int]:
        """Get the number of iterations completed."""
        return self.morph_result.iterations_completed if self.morph_result else None

    @property
    def final_mean_error(self) -> Optional[float]:
        """Get the final mean error."""
        return self.morph_result.final_mean_error if self.morph_result else None

    @property
    def final_max_error(self) -> Optional[float]:
        """Get the final maximum error."""
        return self.morph_result.final_max_error if self.morph_result else None


class MorphComputer:
    """
    Object-oriented interface for flow-based cartogram generation with refinement support.

    This class provides a stateful alternative to the flow_morph function, allowing for
    iterative refinement of cartograms with different options while maintaining history
    and the ability to rollback to previous states. Supports displacement field computation
    and refinement workflows.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values
    column : str
        Name of the column containing data values for cartogram generation
    landmarks : List[Geometry], optional
        Optional landmark geometries for tracking reference points
    options : MorphOptions, optional
        Optional configuration options. If None, default options are used.
    displacement_coords : array-like, optional
        Coordinates for displacement field computation. Stored immutably for refinement.

    Example
    -------
    >>> # Basic usage
    >>> computer = MorphComputer(gdf, 'population', options=MorphOptions(grid=grid))
    >>> result, history = computer.morph()
    >>>
    >>> # With displacement field support
    >>> computer = MorphComputer(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,  # Auto-detects format
    ...     options=MorphOptions(grid=grid)
    ... )
    >>> result, history = computer.morph()
    >>> displacement_field = computer.get_displacement_field_result()
    >>>
    >>> # Refine with displacement field - coordinates automatically propagated
    >>> computer.set_computation(mean_tol=0.02)
    >>> refined_result, refined_history = computer.morph()
    >>>
    >>> # More refinements are easy
    >>> computer.morph()  # Automatically refines from current state
    >>>
    >>> # Rollback if needed
    >>> computer.rollback()
    >>>
    >>> # Start over when ready
    >>> computer.reset()
    >>> computer.morph()  # Fresh start
    """

    def __init__(
        self,
        gdf: Any,
        column: str,
        landmarks: Any = None,
        options: Optional[MorphOptions] = None,
        displacement_coords: Any = None,
    ):
        """
        Initialize the MorphComputer.

        Parameters
        ----------
        gdf : Any
            GeoDataFrame-like object containing polygon geometries and data values
        column : str
            Name of the column containing data values for cartogram generation
        landmarks : List[Geometry], optional
            Optional landmark geometries for tracking reference points
        options : MorphOptions, optional
            Optional configuration options. If None, default options are used.
        displacement_coords : array-like, optional
            Coordinates for displacement field computation. Stored immutably for refinement.
        """
        # Immutable state
        self._gdf_original = gdf.copy()
        self._column = column
        self._landmarks_original = landmarks.copy() if landmarks is not None else None

        # Store original displacement coordinates for format conversion
        # self._displacement_coords_input = displacement_coords
        self._displacement_coords_format = (
            _detect_coordinate_format(displacement_coords) if displacement_coords is not None else None
        )
        self._displacement_coords_original = copy.deepcopy(displacement_coords)

        # Current working state (can be modified during refinement)
        self._gdf_current = gdf.copy()
        self._landmarks_current = landmarks.copy() if landmarks is not None else None
        self._displacement_coords_current = _normalize_coordinates(displacement_coords)

        # Refinement history
        self._refinement_runs: list[RefinementRun] = []

        # Initialize options with grid
        if options is None:
            self._current_options = MorphOptions()
        else:
            self._current_options = options

    def get_grid(self) -> Grid:
        """Get the current grid."""
        return self._get_grid()

        # Target areas (computed once from original data)
        self._target_areas = self._compute_target_areas()

    def _compute_target_areas(self) -> np.ndarray:
        """
        Compute target areas from original geometries and data.

        This method calculates the target areas that each geometry should achieve
        based on the data values in the specified column, maintaining the overall
        mass conservation principle of cartogram generation.

        Returns
        -------
        np.ndarray
            Array of target areas for each geometry
        """
        values = np.array(self._gdf_original[self._column])
        initial_areas = np.array(self._gdf_original.area)
        return np.sum(initial_areas) * values / np.sum(values)

    def _get_grid(self) -> Grid:
        """
        Get the current grid from options.

        This helper method retrieves the Grid object from the current options,
        ensuring that a valid grid is available for computation.

        Returns
        -------
        Grid
            Current grid object for spatial discretization

        Raises
        ------
        ValueError
            If no grid is set in the current options
        """
        if self._current_options.grid is None:
            raise ValueError("Grid is not set in options")
        return self._current_options.grid

    # ============================================================================
    # Fluent Options Interface
    # ============================================================================

    def set_computation(
        self,
        density_smooth: Optional[float] = None,
        dt: Optional[float] = None,
        n_iter: Optional[int] = None,
        recompute_every: Optional[int] = None,
        snapshot_every: Optional[int] = None,
        mean_tol: Optional[float] = None,
        max_tol: Optional[float] = None,
    ) -> "MorphComputer":
        """Set computation-related options."""
        if density_smooth is not None:
            self._current_options.density_smooth = density_smooth
        if dt is not None:
            self._current_options.dt = dt
        if n_iter is not None:
            self._current_options.n_iter = n_iter
        if recompute_every is not None:
            self._current_options.recompute_every = recompute_every
        if snapshot_every is not None:
            self._current_options.snapshot_every = snapshot_every
        if mean_tol is not None:
            self._current_options.mean_tol = mean_tol
        if max_tol is not None:
            self._current_options.max_tol = max_tol
        return self

    def set_anisotropy(
        self,
        Dx: Optional[float] = None,
        Dy: Optional[float] = None,
        anisotropy: Optional[Callable[[Grid], tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "MorphComputer":
        """Set anisotropy-related options."""
        if Dx is not None:
            self._current_options.Dx = Dx
        if Dy is not None:
            self._current_options.Dy = Dy
        if anisotropy is not None:
            self._current_options.anisotropy = anisotropy
        return self

    def set_smoothing(self, vsmooth: Optional[float] = None) -> "MorphComputer":
        """Set velocity field smoothing options."""
        if vsmooth is not None:
            self._current_options.vsmooth = vsmooth
        return self

    def set_grid(self, grid: Grid) -> "MorphComputer":
        """Set the grid for spatial discretization.

        Parameters
        ----------
        grid : Grid
            Grid defining spatial discretization for computation
        """
        self._current_options.grid = grid
        return self

    def set_output(
        self,
        save_history: Optional[bool] = None,
        save_internals: Optional[bool] = None,
        show_progress: Optional[bool] = None,
    ) -> "MorphComputer":
        """Set output and progress options."""
        if save_history is not None:
            self._current_options.save_history = save_history
        if save_internals is not None:
            self._current_options.save_internals = save_internals
        if show_progress is not None:
            self._current_options.show_progress = show_progress
        return self

    def set_options(self, options: MorphOptions) -> "MorphComputer":
        """Set complete options configuration.

        Parameters
        ----------
        options : MorphOptions
            Complete options object to use for morphing

        Returns
        -------
        MorphComputer
            Self for method chaining
        """
        self._current_options = options
        return self

    # ============================================================================
    # Core Computation Methods
    # ============================================================================

    def morph(self) -> tuple[Any, History]:
        """
        Morph geometries into cartogram.

        First call: Performs initial morphing from original data
        Subsequent calls: Refines the current morphed cartogram

        Returns
        -------
        gdf_result : Any
            GeoDataFrame with morphed cartogram geometries
        history : History
            History object containing computation snapshots
        """
        # Simple logic: if no morphed state, we're starting fresh from original
        # (no need to copy since _gdf_current is already a copy of original)

        return self._run_morph()

    def _run_morph(self) -> tuple[Any, History]:
        """
        Internal method to run morphing computation.

        This method handles the core morphing computation, including:
        - Setting up run records and timing
        - Creating appropriate options for the current refinement level
        - Executing the morphing algorithm
        - Recording results and updating internal state

        Returns
        -------
        Tuple[Any, History]
            (morphed_geometries, computation_history)
        """
        start_time = time.time()

        # Create run record
        run_id = len(self._refinement_runs)

        # Set dynamic progress message based on refinement stage
        num_geometries = len(self._gdf_current)
        if run_id == 0:
            # First run - morphing initial geometries
            default_progress_msg = f"Morphing {num_geometries} geometries"
        else:
            # Subsequent runs - refinement
            default_progress_msg = f"Morph refinement {run_id}"

        # Create options with progress message (respect custom message if provided)
        morph_options = copy.deepcopy(self._current_options)
        if morph_options.progress_message is None:
            morph_options.progress_message = default_progress_msg

        run_record = RefinementRun(
            run_id=run_id,
            options=morph_options,
            start_time=start_time,
        )

        try:
            # Run the morphing computation - always use current state
            # morph_gdf will automatically detect if this is initial or refinement
            result = morph_gdf(
                gdf=self._gdf_current,
                column=self._column,
                landmarks=self._landmarks_current,
                options=morph_options,
                displacement_coords=self._displacement_coords_original,
                previous_displaced_coords=self._displacement_coords_current,
            )

            # Store complete MorphResult object
            run_record.morph_result = result

            # Update current state
            self._gdf_current = result.geometries
            if self._landmarks_current is not None:
                self._landmarks_current = result.landmarks
            if self._displacement_coords_current is not None:
                self._displacement_coords_current = result.displaced_coords

            # Complete run record
            end_time = time.time()
            run_record.end_time = end_time
            run_record.duration = end_time - start_time

            # Add to refinement history
            self._refinement_runs.append(run_record)

            return result.geometries, result.history

        except Exception as e:
            # Mark run as failed
            end_time = time.time()
            run_record.end_time = end_time
            run_record.duration = end_time - start_time

            # Create a failed MorphResult for the run record
            run_record.morph_result = MorphResult(geometries=None, history=None, status=f"failed: {e!s}")

            self._refinement_runs.append(run_record)
            raise

    # ============================================================================
    # State Management and Utilities
    # ============================================================================

    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous refinement state.

        Parameters
        ----------
        steps : int, default=1
            Number of refinement runs to rollback

        Returns
        -------
        success : bool
            True if rollback was successful, False if not enough history
        """
        if steps <= 0 or len(self._refinement_runs) < steps:
            return False

        # Remove the last 'steps' refinement runs
        self._refinement_runs = self._refinement_runs[:-steps]

        # Reset to state before the rolled-back runs
        if len(self._refinement_runs) == 0:
            # No refinement runs left, reset to original
            self._gdf_current = self._gdf_original.copy()
            self._landmarks_current = self._landmarks_original.copy() if self._landmarks_original is not None else None
            self._displacement_coords_current = copy.deepcopy(self._displacement_coords_original)
        else:
            # Reset to the last completed refinement run
            last_run = self._refinement_runs[-1]
            self._gdf_current = last_run.geometries.copy()
            self._landmarks_current = last_run.landmarks.copy() if last_run.landmarks is not None else None
            # For displacement coordinates, we need to get them from the run's result
            if last_run.morph_result is not None:
                self._displacement_coords_current = last_run.morph_result.displaced_coords

        return True

    def reset(self) -> None:
        """Reset to initial state, clearing all refinement history."""
        self._gdf_current = self._gdf_original.copy()
        self._landmarks_current = self._landmarks_original.copy() if self._landmarks_original is not None else None
        self._displacement_coords_current = copy.deepcopy(self._displacement_coords_original)
        self._refinement_runs.clear()
        self._current_options = MorphOptions(grid=self._current_options.grid)

    def get_result(self) -> Any:
        """Get the current morphed geometries."""
        return self._gdf_current

    def get_refinement_history(self) -> list[RefinementRun]:
        """Get the complete refinement history."""
        return self._refinement_runs.copy()

    def get_current_options(self) -> MorphOptions:
        """Get the current options configuration."""
        return copy.deepcopy(self._current_options)

    def get_run_info(self, run_id: Optional[int] = None) -> RefinementRun:
        """
        Get information about a specific refinement run.

        Parameters
        ----------
        run_id : int, optional
            ID of the run to retrieve. If None, returns the last run.

        Returns
        -------
        run_info : RefinementRun
            Information about the requested run
        """
        if run_id is None:
            return self._refinement_runs[-1] if self._refinement_runs else None
        return self._refinement_runs[run_id] if 0 <= run_id < len(self._refinement_runs) else None

    def get_landmarks_result(self) -> Optional[Any]:
        """Get the current morphed landmark geometries.

        Returns
        -------
        Optional[Any]
            Current landmark geometries if landmarks were provided, None otherwise
        """
        return self._landmarks_current

    def get_displacement_coords_result(self) -> Optional[Any]:
        """Get the current displacement field coordinates.

        Returns
        -------
        Optional[Any]
            Current displacement coordinates if displacement_coords were provided, None otherwise
        """
        return self._displacement_coords_current

    def get_displacement_field_result(self) -> Optional[Any]:
        """Get the latest displacement field result.

        Returns
        -------
        Optional[Any]
            Latest displacement field in the same format as originally provided, None otherwise
        """
        if self._refinement_runs and self._refinement_runs[-1].morph_result:
            displacement_field = self._refinement_runs[-1].morph_result.displacement_field
            return displacement_field
        return None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Basic info
        geom_count = len(self._gdf_current) if self._gdf_current is not None else 0
        run_count = len(self._refinement_runs)

        # Landmark and displacement info
        landmark_str = ", landmarks" if self._landmarks_current is not None else ""
        displacement_str = ", displacement" if self._displacement_coords_current is not None else ""

        # Current status
        if run_count > 0:
            last_run = self._refinement_runs[-1]
            if last_run.morph_result is not None:
                status_str = f", status={last_run.morph_result.status}"
            else:
                status_str = ", status=running"
        else:
            status_str = ", status=initialized"

        return f"MorphComputer(geoms={geom_count}, runs={run_count}{status_str}{landmark_str}{displacement_str})"


# ============================================================================
# Development and Testing Utilities
# ============================================================================


def _test_morph_options_validation():
    """Test function to verify MorphOptions validation works correctly."""
    print("Testing MorphOptions validation...")

    # Test 1: Valid options should work
    try:
        opts = MorphOptions(dt=0.2, n_iter=100, mean_tol=0.05, max_tol=0.1)
        print("✓ Valid options accepted")
    except Exception as e:
        print(f"✗ Valid options rejected: {e}")
        return False

    # Test 2: Invalid dt should be caught
    try:
        opts = MorphOptions(dt=-0.1)
        print("✗ Invalid dt accepted")
        return False
    except MorphOptionsValidationError as e:
        print(f"✓ Invalid dt correctly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type for invalid dt: {e}")
        return False

    # Test 3: Invalid tolerance ordering should be caught
    try:
        opts = MorphOptions(mean_tol=0.2, max_tol=0.1)
        print("✗ Invalid tolerance ordering accepted")
        return False
    except MorphOptionsValidationError as e:
        print(f"✓ Invalid tolerance ordering correctly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type for tolerance ordering: {e}")
        return False

    # Test 4: Direct assignment validation
    try:
        opts = MorphOptions(dt=0.2, n_iter=100)
        opts.dt = -0.1  # Should raise validation error
        print("✗ Invalid direct assignment accepted")
        return False
    except MorphOptionsValidationError as e:
        print(f"✓ Invalid direct assignment correctly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type for direct assignment: {e}")
        return False

    # Test 5: with_options validation
    try:
        opts = MorphOptions(dt=0.2, n_iter=100)
        opts.with_options(dt=-0.1)  # Should raise validation error
        print("✗ Invalid with_options accepted")
        return False
    except MorphOptionsValidationError as e:
        print(f"✓ Invalid with_options correctly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type for with_options: {e}")
        return False

    print("All validation tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_morph_options_validation()
