"""Configuration options for symbol cartogram generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SymbolShape(str, Enum):
    """Shape of the symbols."""

    CIRCLE = "circle"
    SQUARE = "square"
    HEXAGON = "hexagon"


class AdjacencyMode(str, Enum):
    """How adjacency is computed."""

    BINARY = "binary"  # Adjacent or not (0/1)
    WEIGHTED = "weighted"  # Fraction of perimeter shared (asymmetric)
    AREA_WEIGHTED = "area_weighted"  # Neighbor area fraction (rows sum to 1)


class SymbolOrientation(str, Enum):
    """How symbols are oriented relative to their tile."""

    UPRIGHT = "upright"  # Symbol stays axis-aligned regardless of tile
    WITH_TILE = "with_tile"  # Symbol rotates/flips with its tile


class ForceMode(str, Enum):
    """How attraction force magnitude is computed.

    Applies to both origin attraction force and global centroid attraction force.
    """

    DIRECTION = "direction"  # Constant magnitude with drop-off near target (default)
    LINEAR = "linear"  # Force proportional to distance (spring)
    NORMALIZED = "normalized"  # Force proportional to distance / radius


@dataclass
class GridBasedLayoutOptions:
    """Options for grid-based placement.

    The assignment cost function combines four terms, harmonized with the
    ``TopologyPreservingSimulator`` force terminology:

    - **origin_weight**: Preference for assigning regions near their
      original centroid positions (analogous to origin attraction force).
    - **neighbor_weight**: Keep adjacent regions close on the grid
      (analogous to neighbor tangency force).
    - **topology_weight**: Preserve the relative direction between
      neighbors (analogous to angular topology force).
    - **compactness**: Prefer central/compact placement on the grid
      (analogous to global centroid attraction).

    Parameters
    ----------
    tiling : Tiling or str
        Tiling to use. Can be a ``Tiling`` instance for full control, or a
        string shorthand: ``"square"``, ``"hexagon"``, ``"triangle"``,
        ``"quadrilateral"``. Default is ``"hexagon"``.
    grid_size : int, tuple, or "auto"
        Number of grid cells. "auto" creates ~2x regions cells.
    origin_weight : float
        Weight for assigning regions near their original centroids.
        Default: 1.0.
    neighbor_weight : float
        Weight for keeping adjacent regions close on the grid.
        Default: 0.3.
    topology_weight : float
        Weight for preserving relative neighbor orientations.
        Default: 0.0.
    compactness : float
        Weight for compact/central placement. Default: 0.0.
    spacing : float
        Gap between symbols as fraction of cell size (0-1). Used when
        computing the maximum symbol size that fits within grid cells.
    fill_holes : bool
        If True, post-process the assignment to fill internal holes —
        unoccupied tiles surrounded by occupied tiles that don't correspond
        to geographic gaps in the original geometries. Islands and genuine
        geographic gaps (e.g. internal lakes) are preserved. Default: False.
    min_hole_fraction : float
        Minimum area of a geographic interior ring, as a fraction of one
        tile's area, for it to count as a genuine geographic gap. Rings
        smaller than this are treated as boundary artifacts and ignored.
        Default: 0.5.
    fix_islands : bool
        If True, post-process the assignment to reassign regions that are
        disconnected from the main assignment cluster to tiles adjacent to
        the cluster. True geographic islands (regions not adjacent to any
        other region) are preserved. Default: False.
    verbose : bool
        If True, print diagnostic information during fill_holes and
        fix_islands processing. Default: False.
    symbol_shape : SymbolShape or None
        Target symbol shape. When set, the grid algorithm computes
        grid-appropriate sizes to prevent overlap. None skips adjustment.
    symbol_orientation : SymbolOrientation
        How symbols are oriented relative to their tile. ``UPRIGHT`` keeps
        symbols axis-aligned; ``WITH_TILE`` rotates/flips with the tile.
    rotation : float
        Rotation angle for the entire tiling grid in degrees (counter-clockwise).
        Default: 0.0 (no rotation).

    """

    tiling: object = "hexagon"  # Tiling | str, but avoid circular import
    grid_size: int | tuple[int, int] | Literal["auto"] = "auto"
    origin_weight: float = 0.5
    neighbor_weight: float = 0.5
    topology_weight: float = 0.5
    compactness: float = 0.1
    spacing: float = 0.05
    fill_holes: bool = True
    min_hole_fraction: float = 0.5
    fix_islands: bool = True
    verbose: bool = False
    symbol_shape: SymbolShape | None = None
    symbol_orientation: SymbolOrientation = SymbolOrientation.UPRIGHT
    rotation: float = 0.0

    def validate(self) -> None:
        """Validate options."""
        if self.origin_weight < 0:
            raise ValueError("origin_weight must be >= 0")
        if self.neighbor_weight < 0:
            raise ValueError("neighbor_weight must be >= 0")
        if self.topology_weight < 0:
            raise ValueError("topology_weight must be >= 0")
        if self.compactness < 0:
            raise ValueError("compactness must be >= 0")
        if not 0 <= self.spacing <= 1:
            raise ValueError("spacing must be between 0 and 1")
        if self.min_hole_fraction < 0:
            raise ValueError("min_hole_fraction must be >= 0")


@dataclass
class CentroidLayoutOptions:
    """Options for centroid-based placement.

    Places symbols at geometry centroids with optional overlap removal.

    Parameters
    ----------
    spacing : float
        Gap between symbols as fraction of average symbol size. Default: 0.05
    remove_overlap : bool
        Whether to run overlap removal after placing at centroids. Default: True
    max_iterations : int
        Maximum iterations for overlap resolution. Default: 20
    overlap_tolerance : float
        Tolerance for overlap resolution convergence. Default: 1e-4

    """

    spacing: float = 0.05
    remove_overlap: bool = True
    max_iterations: int = 20
    overlap_tolerance: float = 1e-4

    def validate(self) -> None:
        """Validate options."""
        if not 0 <= self.spacing <= 1:
            raise ValueError("spacing must be between 0 and 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.overlap_tolerance <= 0:
            raise ValueError("overlap_tolerance must be positive")


@dataclass
class CirclePhysicsLayoutOptions:
    """Options for CirclePhysicsLayout (velocity-based physics).

    This simulator uses a two-phase approach:
    1. Separation phase: Strong repulsion until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Parameters
    ----------
    max_iterations : int
        Maximum iterations for simulation. Default: 500
    convergence_tolerance : float
        Threshold for declaring convergence. Default: 1e-4
    damping : float
        Velocity damping factor (0-1, exclusive). Higher values cause
        faster energy dissipation. Default: 0.85
    dt : float
        Integration timestep. Larger values give faster but less stable
        simulation. Default: 0.15
    max_velocity : float
        Maximum velocity magnitude (clamped). Default: 3.0
    k_repel : float
        Repulsion force coefficient. Higher values push overlapping
        circles apart more aggressively. Default: 15.0
    k_attract : float
        Base attraction force coefficient. Default: 2.0

    """

    # Simulation parameters
    max_iterations: int = 500
    convergence_tolerance: float = 1e-4

    # High-level placement parameters (passed to simulator constructor)
    spacing: float = 0.05  # Gap as fraction of avg symbol size
    compactness: float = 0.5  # 0 = centroid only, 1 = neighbor only
    topology_weight: float = 0.3  # 0 = ignore topology, 1 = strong

    # Physics-specific parameters
    damping: float = 0.85
    dt: float = 0.15
    max_velocity: float = 3.0
    k_repel: float = 15.0
    k_attract: float = 2.0

    def validate(self) -> None:
        """Validate options."""
        errors = []
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if not 0 <= self.spacing <= 1:
            errors.append("spacing must be between 0 and 1")
        if not 0 <= self.compactness <= 1:
            errors.append("compactness must be between 0 and 1")
        if not 0 <= self.topology_weight <= 1:
            errors.append("topology_weight must be between 0 and 1")
        if not 0 < self.damping < 1:
            errors.append("damping must be between 0 and 1 (exclusive)")
        if self.dt <= 0:
            errors.append("dt must be positive")
        if self.max_velocity <= 0:
            errors.append("max_velocity must be positive")
        if self.k_repel < 0:
            errors.append("k_repel must be non-negative")
        if self.k_attract < 0:
            errors.append("k_attract must be non-negative")
        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class CirclePackingLayoutOptions:
    """Options for CirclePackingLayout (two-stage with contact reaction).

    This simulator uses a fundamentally different approach:
    - Stage 1 (Feasibility): Global expansion + overlap projection
    - Stage 2 (Refinement): Force-based with contact reaction constraint

    The contact reaction allows circles to slide along each other without
    penetrating, enabling tighter packing while preserving topology.

    Parameters
    ----------
    max_iterations : int
        Maximum iterations for simulation. Default: 500
    convergence_tolerance : float
        Threshold for declaring convergence. Default: 1e-4
    overlap_tolerance : float
        Overlap tolerance for overlap resolution convergence, as fraction of
        average radius. Resolution completes when max overlap is below this.
        Default: 1e-4
    global_step_fraction : float
        Fraction of exact global expansion to apply per iteration (0-1].
        Higher = faster convergence but may overshoot. Default: 0.5
    local_step_fraction : float
        Fraction of local pairwise separation to apply per iteration (0-1].
        At 1.0, each circle moves 0.5 * overlap (its half of the full
        correction). At 0.5 (default), each moves 0.25 * overlap. Default: 0.5
    topology_gate_distance : float
        Topology force distance gate. Forces only act when circles are
        within topology_gate_distance * (r_i + r_j) distance. Default: 2.5
    neighbor_weight : float
        Neighbor tangency force strength. Pulls separated neighbors
        together. Default: 0.5
    origin_weight : float
        Origin attraction force strength. Pulls each circle toward its
        original position. Set to 0 to disable. Default: 0.0
        - 0: No origin attraction (current behavior)
        - 0.1-0.5: Gentle pull toward original positions
        - > 1.0: Strong pull, may interfere with topology preservation
    force_mode : ForceMode
        How attraction force magnitude is computed. Applies to both the
        global centroid attraction force and the origin attraction force.
        Default: ForceMode.DIRECTION
        - DIRECTION: Constant magnitude with drop-off near target
        - LINEAR: Force proportional to distance (spring)
        - NORMALIZED: Force proportional to distance / radius
    contact_tolerance : float
        Contact detection tolerance as fraction of radii sum. Default: 0.02
    max_step : float
        Maximum step size as fraction of average radius. Default: 0.3
    contact_transfer_ratio : float
        Balance between cancel (0) and transfer (1) of compressive forces
        at contact points. Default: 0.5
        - 0: Cancel (dissipate) compressive forces
        - 1: Transfer compressive forces between circles
        - 0.5: Equal mix
    contact_elasticity : float
        Controls net compression vs bounce behavior. Default: 0.0
        - 0: Neutral (momentum conserved)
        - < 0: Compression remains (forces tend to pass through)
        - > 0: Bounce behavior (forces tend to reflect)
        Note: Only has effect when 0 < contact_transfer_ratio < 1.
    size_sensitivity : float
        Controls how step size scales with circle radius in packing phase.
        Default: 0.0
        - 0: All circles use avg_radius (current behavior, backward compatible)
        - 1: Larger circles move faster (proportional to radius)
        - -1: Larger circles move slower (inverse relationship)
        - Other values: Interpolate between these behaviors
    expansion_max_iterations : int
        Maximum outer iterations for overlap resolution. Default: 20
    max_expansion_factor : float
        Maximum expansion factor clamp per iteration. Must be > 1.0.
        Default: 2.0
    contact_iterations : int
        Number of contact reaction passes per packing step. Default: 3
    overlap_projection_iters : int
        Overlap projection iterations per packing step. Default: 5
    step_smoothing_window : int
        EMA effective window size for step smoothing. Default: 20
    convergence_window : int
        EMA effective window size for displacement convergence tracking.
        Default: 50
    adaptive_ema : bool
        Whether EMA uses adaptive warmup (alpha starts at 1/k). Default: True

    """

    # Simulation parameters
    max_iterations: int = 500
    convergence_tolerance: float = 0.025

    # High-level placement parameters (passed to simulator constructor)
    spacing: float = 0.05  # Gap as fraction of avg symbol size
    compactness: float = 0.1  # 0 = centroid only, 1 = neighbor only
    topology_weight: float = 1.0  # 0 = ignore topology, 1 = strong

    # Overlap resolution phase
    overlap_tolerance: float = 1e-4
    global_step_fraction: float = 0.5
    local_step_fraction: float = 0.5

    # Packing phase: Force coefficients
    topology_gate_distance: float = 2.5
    neighbor_weight: float = 1.0
    origin_weight: float = 0.1  # Attraction to original position (0 = disabled)
    force_mode: ForceMode = ForceMode.DIRECTION  # Applies to both origin and global forces
    contact_tolerance: float = 0.02

    # Packing phase: Step control
    max_step: float = 0.3

    # Packing phase: Contact reaction parameters
    contact_transfer_ratio: float = 0.5
    contact_elasticity: float = 0.0

    # Packing phase: Size-based step scaling
    size_sensitivity: float = 0.0

    # Overlap resolution phase: Iteration limits
    expansion_max_iterations: int = 20
    max_expansion_factor: float = 2.0

    # Packing phase: Per-step iteration counts
    contact_iterations: int = 3
    overlap_projection_iters: int = 5

    # EMA configuration
    step_smoothing_window: int = 20
    convergence_window: int = 50
    adaptive_ema: bool = True

    def validate(self) -> None:
        """Validate options."""
        errors = []
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if not 0 <= self.spacing <= 1:
            errors.append("spacing must be between 0 and 1")
        if not 0 <= self.compactness <= 1:
            errors.append("compactness must be between 0 and 1")
        if not 0 <= self.topology_weight <= 1:
            errors.append("topology_weight must be between 0 and 1")
        if self.overlap_tolerance <= 0:
            errors.append("overlap_tolerance must be positive")
        if not 0 < self.global_step_fraction <= 1:
            errors.append("global_step_fraction must be > 0 and <= 1")
        if not 0 < self.local_step_fraction <= 1:
            errors.append("local_step_fraction must be > 0 and <= 1")
        if self.topology_gate_distance <= 0:
            errors.append("topology_gate_distance must be positive")
        if self.neighbor_weight < 0:
            errors.append("neighbor_weight must be non-negative")
        if self.origin_weight < 0:
            errors.append("origin_weight must be non-negative")
        if not 0 < self.contact_tolerance < 1:
            errors.append("contact_tolerance must be > 0 and < 1")
        if self.max_step <= 0:
            errors.append("max_step must be positive")
        if not 0 <= self.contact_transfer_ratio <= 1:
            errors.append("contact_transfer_ratio must be between 0 and 1")
        if not -1 <= self.contact_elasticity <= 1:
            errors.append("contact_elasticity must be between -1 and 1")
        if self.expansion_max_iterations < 1:
            errors.append("expansion_max_iterations must be >= 1")
        if self.max_expansion_factor <= 1.0:
            errors.append("max_expansion_factor must be > 1.0")
        if self.contact_iterations < 1:
            errors.append("contact_iterations must be >= 1")
        if self.overlap_projection_iters < 1:
            errors.append("overlap_projection_iters must be >= 1")
        if self.step_smoothing_window < 1:
            errors.append("step_smoothing_window must be >= 1")
        if self.convergence_window < 1:
            errors.append("convergence_window must be >= 1")
        if errors:
            raise ValueError("; ".join(errors))
