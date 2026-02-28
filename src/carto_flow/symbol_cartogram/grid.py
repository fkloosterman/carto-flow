"""Grid generation for symbol cartograms."""

from __future__ import annotations

from typing import Literal

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import box

from .symbols import create_hexagon


def generate_grid(
    bounds: tuple[float, float, float, float],
    shape: Literal["square", "hexagon"] = "hexagon",
    n_cells: int | None = None,
    cell_size: float | None = None,
    margin: float = 0.1,
    symbol_shape: str | None = None,
) -> gpd.GeoDataFrame:
    """Generate a regular grid of cells covering a bounding box.

    Parameters
    ----------
    bounds : tuple
        Bounding box as (minx, miny, maxx, maxy).
    shape : {"square", "hexagon"}
        Shape of grid cells.
    n_cells : int, optional
        Approximate number of cells. Mutually exclusive with cell_size.
    cell_size : float, optional
        Size of each cell. Mutually exclusive with n_cells.
    margin : float
        Fractional margin around bounding box.
    symbol_shape : str, optional
        Target symbol shape ("circle", "square", "hexagon"). When specified,
        grid layout may be adjusted to optimize symbol tiling.

    Returns
    -------
    gpd.GeoDataFrame
        Grid with columns:
        - geometry: Cell polygon
        - cell_id: Unique cell identifier
        - center_x, center_y: Cell center coordinates

    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    # Add margin
    minx -= width * margin
    maxx += width * margin
    miny -= height * margin
    maxy += height * margin
    width = maxx - minx
    height = maxy - miny

    # Determine cell size
    if cell_size is None and n_cells is None:
        raise ValueError("Must specify either n_cells or cell_size")

    if cell_size is None and n_cells is not None:
        total_area = width * height
        cell_area = total_area / n_cells
        if shape == "square":  # noqa: SIM108
            cell_size = np.sqrt(cell_area)
        else:
            # Hexagon area ~= 2.598 * s^2 where s is "size" (center to vertex)
            cell_size = np.sqrt(cell_area / 2.598)

    # At this point cell_size is guaranteed to be set
    if cell_size is None:
        raise RuntimeError("cell_size was not set; unexpected shape value")  # pragma: no cover

    if shape == "square":
        grid = _generate_square_grid(minx, miny, maxx, maxy, cell_size, symbol_shape=symbol_shape)
    else:
        grid = _generate_hexagon_grid(minx, miny, maxx, maxy, cell_size, symbol_shape=symbol_shape)

    # Store cell size info for symbol sizing
    grid.attrs["cell_size"] = cell_size
    grid.attrs["grid_shape"] = shape
    grid.attrs["symbol_shape"] = symbol_shape

    return grid


def _generate_square_grid(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    cell_size: float,
    symbol_shape: str | None = None,
) -> gpd.GeoDataFrame:
    """Generate a grid of square cells.

    When symbol_shape is "hexagon", adjusts the layout for proper hexagon tiling:
    - Vertical spacing is reduced to cell_size * sqrt(3) / 2
    - Alternating rows are offset by half the horizontal spacing
    """
    cells = []
    cell_id = 0

    # For hexagon symbols, use rectangular grid with hexagon proportions
    # (no row offset - true "square" arrangement)
    if symbol_shape == "hexagon":
        # Hexagons in a rectangular grid:
        # - Horizontal: touch at flat sides (width = size * sqrt(3))
        # - Vertical: touch at pointy vertices (height = size * 2)
        # Use cell_size as horizontal spacing, adjust vertical for hexagon ratio
        dx = cell_size
        dy = cell_size * 2 / np.sqrt(3)  # Maintains hexagon width:height ratio

        y = miny
        while y < maxy + dy:
            x = minx
            while x < maxx + dx:
                cell = box(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2)
                cells.append({
                    "geometry": cell,
                    "cell_id": cell_id,
                    "center_x": x,
                    "center_y": y,
                })
                cell_id += 1
                x += dx
            y += dy
    else:
        # Standard square grid
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell = box(x, y, x + cell_size, y + cell_size)
                center_x = x + cell_size / 2
                center_y = y + cell_size / 2
                cells.append({
                    "geometry": cell,
                    "cell_id": cell_id,
                    "center_x": center_x,
                    "center_y": center_y,
                })
                cell_id += 1
                y += cell_size
            x += cell_size

    return gpd.GeoDataFrame(cells, crs=None)


def _generate_hexagon_grid(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    size: float,
    symbol_shape: str | None = None,
) -> gpd.GeoDataFrame:
    """Generate a grid of hexagonal cells (pointy-top orientation).

    Parameters
    ----------
    size : float
        Distance from center to vertex.
    symbol_shape : str, optional
        Target symbol shape. When "square", uses brick pattern spacing
        for perfect square tiling.

    """
    cells = []
    cell_id = 0

    if symbol_shape == "square":
        # Brick pattern for square symbols:
        # Squares tile perfectly when staggered by half their width
        # Use size as the square's half-side
        square_size = size
        dx = square_size * 2  # Full square width
        dy = square_size * 2  # Full square height

        row = 0
        y = miny
        while y < maxy + dy:
            # Offset every other row by half square width
            x_offset = square_size if row % 2 == 1 else 0
            x = minx + x_offset

            while x < maxx + dx:
                # Use box for cell geometry
                cell = box(x - square_size, y - square_size, x + square_size, y + square_size)
                cells.append({
                    "geometry": cell,
                    "cell_id": cell_id,
                    "center_x": x,
                    "center_y": y,
                })
                cell_id += 1
                x += dx

            y += dy
            row += 1
    else:
        # Standard hexagon grid
        # Hexagon dimensions (pointy-top)
        hex_height = size * 2  # Vertex to vertex (vertical)
        hex_width = size * np.sqrt(3)  # Edge to edge (horizontal)

        # Spacing between centers
        dx = hex_width
        dy = hex_height * 0.75

        # Create base hexagon centered at origin
        base_hex = create_hexagon((0, 0), size, pointy_top=True)

        row = 0
        y = miny
        while y < maxy + dy:
            # Offset every other row
            x_offset = hex_width / 2 if row % 2 == 1 else 0
            x = minx + x_offset

            while x < maxx + dx:
                cell = translate(base_hex, x, y)
                cells.append({
                    "geometry": cell,
                    "cell_id": cell_id,
                    "center_x": x,
                    "center_y": y,
                })
                cell_id += 1
                x += dx

            y += dy
            row += 1

    return gpd.GeoDataFrame(cells, crs=None)


def compute_grid_symbol_size(
    grid: gpd.GeoDataFrame,
    symbol_shape: str,
    spacing: float = 0.05,
) -> float:
    """Compute symbol size that fits within grid cells without overlap.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid generated by generate_grid().
    symbol_shape : str
        Shape of symbols ("circle", "square", "hexagon").
    spacing : float
        Spacing between symbols as fraction of cell size.

    Returns
    -------
    float
        Maximum symbol size (radius for circles, half-side for squares,
        center-to-vertex for hexagons) that fits in the grid.

    """
    cell_size = grid.attrs.get("cell_size", 1.0)
    grid_shape = grid.attrs.get("grid_shape", "hexagon")

    # Compute maximum symbol size based on grid and symbol geometry
    # The key constraint is that symbols in adjacent cells must not overlap
    #
    # Note: Some symbol/grid combinations have inherent gaps even at spacing=0
    # because their geometries don't tile perfectly together.

    # Check if grid was generated with symbol-aware layout
    grid_symbol_shape = grid.attrs.get("symbol_shape")

    if grid_shape == "square":
        # Square grid: cell centers are cell_size apart (or adjusted for symbol)
        if symbol_shape == "circle":
            # Circles: 2*radius <= cell_size → radius = cell_size / 2
            max_size = cell_size / 2

        elif symbol_shape == "square":
            # Squares: 2*half_side <= cell_size → half_side = cell_size / 2
            max_size = cell_size / 2

        elif grid_symbol_shape == "hexagon":
            # Grid was adjusted for hexagon tiling (rectangular grid):
            # dx = cell_size, dy = cell_size * 2 / sqrt(3)
            # Hexagon width = size * sqrt(3) should equal dx = cell_size
            # → size = cell_size / sqrt(3)
            # Hexagon height = size * 2 = cell_size * 2 / sqrt(3) = dy ✓
            max_size = cell_size / np.sqrt(3)
        else:
            # Standard square grid - hexagons can't tile perfectly
            # Vertical: 2*size <= cell_size → size = cell_size / 2
            max_size = cell_size / 2

    # Hex grid neighbor distances:
    # - Horizontal: cell_size * sqrt(3)
    # - Diagonal: also cell_size * sqrt(3) (hex grid property)

    elif symbol_shape == "circle":
        # Circles: 2*radius <= cell_size * sqrt(3)
        max_size = cell_size * np.sqrt(3) / 2

    elif symbol_shape == "square":
        # Square tiling: half_side = cell_size; hex grid: diagonal constraint → 0.75 * cell_size
        max_size = cell_size if grid_symbol_shape == "square" else cell_size * 0.75

    else:  # hexagon (pointy-top)
        # Hexagons tile perfectly in hex grid
        # Width = size * sqrt(3), neighbors are size * sqrt(3) apart
        # So symbol size should equal cell_size for perfect tiling
        max_size = cell_size

    # Apply spacing
    return max_size * (1 - spacing)
