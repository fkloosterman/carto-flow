"""Visualization functions for symbol cartograms."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from numpy.typing import NDArray

    from .result import SymbolCartogram

# Default hatch patterns cycled when auto-assigning hatches to categories.
_HATCH_DEFAULTS: list[str] = ["/", "\\", ".", "x", "o", "+", "-", "|", "*", "O"]


def plot_comparison(
    original_gdf: gpd.GeoDataFrame,
    result: SymbolCartogram,
    column: str | None = None,
    figsize: tuple[float, float] = (14, 6),
    **kwargs: Any,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Side-by-side comparison of original geometries and symbols.

    Parameters
    ----------
    original_gdf : gpd.GeoDataFrame
        Original GeoDataFrame.
    result : SymbolCartogram
        Symbol cartogram result.
    column : str, optional
        Column for coloring.
    figsize : tuple
        Figure size.
    **kwargs
        Passed to GeoDataFrame.plot()

    Returns
    -------
    fig, (ax1, ax2)
        Figure and axes.

    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot original
    if column and column in original_gdf.columns:
        original_gdf.plot(ax=ax1, column=column, legend=True, **kwargs)
    else:
        original_gdf.plot(ax=ax1, **kwargs)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("Original")

    # Plot symbols
    result.plot(ax=ax2, column=column, **kwargs)
    ax2.set_title(f"Symbol Cartogram ({result.status.value})")

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_displacement(
    result: SymbolCartogram,
    ax: plt.Axes | None = None,
    arrow_scale: float = 1.0,
    arrow_color: str = "red",
    arrow_alpha: float = 0.7,
    show_symbols: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Plot displacement arrows from original centroids to symbol centers.

    Parameters
    ----------
    result : SymbolCartogram
        Symbol cartogram result.
    ax : plt.Axes, optional
        Axes to plot on.
    arrow_scale : float
        Scale factor for arrow width.
    arrow_color : str
        Arrow color.
    arrow_alpha : float
        Arrow transparency.
    show_symbols : bool
        Whether to show symbols underneath arrows.
    **kwargs
        Passed to result.plot() if show_symbols=True.

    Returns
    -------
    plt.Axes
        The axes with the plot.

    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if show_symbols:
        result.plot(ax=ax, **kwargs)

    # Get displacement vectors
    displacements = result.get_displacement_vectors()

    # Get original positions
    if result.layout_result is not None:
        origins = result.layout_result.positions
    elif result._source_gdf is not None:
        if result._valid_mask is not None:
            source_geoms = result._source_gdf.loc[result._valid_mask].geometry
        else:
            source_geoms = result._source_gdf.geometry
        origins = np.array([[g.centroid.x, g.centroid.y] for g in source_geoms])
    else:
        raise ValueError("No original positions available for displacement plot")

    # Plot arrows
    ax.quiver(
        origins[:, 0],
        origins[:, 1],
        displacements[:, 0],
        displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=arrow_color,
        alpha=arrow_alpha,
        width=0.003 * arrow_scale,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Displacement Vectors")

    return ax


def plot_adjacency(
    result: SymbolCartogram,
    original_gdf: gpd.GeoDataFrame | None = None,
    adjacency: NDArray | None = None,
    ax: plt.Axes | None = None,
    edge_color: str | None = None,
    edge_cmap: str = "viridis",
    edge_alpha: float = 0.6,
    edge_width: float = 1.5,
    node_size: float = 20,
    show_symbols: bool = True,
    show_original: bool = False,
    **kwargs: Any,
) -> plt.Axes:
    """Visualize the adjacency graph overlaid on the cartogram.

    Draws edges between adjacent symbol centers. Edge color can be fixed
    or mapped to adjacency weight via a colormap.

    Parameters
    ----------
    result : SymbolCartogram
        Symbol cartogram result.
    original_gdf : gpd.GeoDataFrame, optional
        Original GeoDataFrame for underlay when ``show_original=True``.
    adjacency : np.ndarray, optional
        Adjacency matrix of shape ``(n, n)``. If None, uses the matrix
        stored in the result (``result.layout_result.adjacency``).
    ax : plt.Axes, optional
        Axes to plot on. Created if not provided.
    edge_color : str, optional
        Fixed color for all edges. If None, edges are colored by weight
        using ``edge_cmap``.
    edge_cmap : str
        Colormap name for edge weights (used when ``edge_color`` is None).
    edge_alpha : float
        Edge transparency.
    edge_width : float
        Base edge line width.
    node_size : float
        Size of node markers at symbol centers.
    show_symbols : bool
        Whether to show symbol polygons underneath.
    show_original : bool
        Whether to show original geometry boundaries.
    **kwargs
        Passed to ``result.plot()`` if ``show_symbols=True``.

    Returns
    -------
    plt.Axes
        The axes with the plot.

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Resolve adjacency matrix
    adj = (
        adjacency
        if adjacency is not None
        else (result.layout_result.adjacency if result.layout_result is not None else None)
    )
    if adj is None:
        raise ValueError(
            "No adjacency matrix available. Pass one explicitly or ensure "
            "the result was created with adjacency computation.",
        )

    # Show underlay
    if show_original and original_gdf is not None:
        original_gdf.plot(ax=ax, facecolor="none", edgecolor="lightgray", linewidth=0.5)

    if show_symbols:
        result.plot(ax=ax, **kwargs)

    # Get symbol centers
    centers = np.column_stack([
        result.symbols["_symbol_x"].values,
        result.symbols["_symbol_y"].values,
    ])
    n = len(centers)

    # Collect edges
    segments = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            w = max(adj[i, j], adj[j, i])  # handle asymmetric matrices
            if w > 0:
                segments.append([centers[i], centers[j]])
                weights.append(w)

    if segments:
        segments_arr = np.array(segments)
        weights_arr = np.array(weights)

        if edge_color is not None:
            lc = LineCollection(
                segments_arr,
                colors=edge_color,
                alpha=edge_alpha,
                linewidths=edge_width,
            )
        else:
            norm = Normalize(vmin=weights_arr.min(), vmax=weights_arr.max())
            cmap = plt.get_cmap(edge_cmap)
            colors = cmap(norm(weights_arr))
            lc = LineCollection(
                segments_arr,
                colors=colors,
                alpha=edge_alpha,
                linewidths=edge_width,
            )
        ax.add_collection(lc)

    # Plot nodes
    if node_size > 0:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=node_size,
            c="black",
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Adjacency Graph")

    return ax


def plot_tiling(
    result: SymbolCartogram,
    ax: plt.Axes | None = None,
    show_symbols: bool = True,
    show_assigned: bool = True,
    show_unassigned: bool = True,
    assigned_color: str = "#d4e6f1",
    unassigned_color: str = "#f5f5f5",
    tile_edgecolor: str = "#999999",
    tile_linewidth: float = 0.5,
    tile_alpha: float = 0.5,
    **kwargs: Any,
) -> plt.Axes:
    """Visualize the tiling grid underlying a grid-placement cartogram.

    Plots all tile polygons, distinguishing assigned tiles (with a region)
    from unassigned tiles (empty). Optionally overlays the symbol geometries.

    Parameters
    ----------
    result : SymbolCartogram
        Symbol cartogram result from grid-based placement.
    ax : plt.Axes, optional
        Axes to plot on. Created if not provided.
    show_symbols : bool
        Whether to overlay symbol geometries. Default True.
    show_assigned : bool
        Whether to show tiles that have a region assigned. Default True.
    show_unassigned : bool
        Whether to show unassigned (empty) tiles. Default True.
    assigned_color : str
        Fill color for tiles that have a region assigned.
    unassigned_color : str
        Fill color for unassigned (empty) tiles.
    tile_edgecolor : str
        Edge color for tile outlines.
    tile_linewidth : float
        Line width for tile outlines.
    tile_alpha : float
        Fill transparency for tiles.
    **kwargs
        Passed to ``result.plot()`` if ``show_symbols=True``.

    Returns
    -------
    plt.Axes
        The axes with the plot.

    Raises
    ------
    ValueError
        If the result is not from grid-based placement.

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    tiling_result = getattr(result, "_tiling_result", None)
    assignments = getattr(result, "_assignments", None)

    if tiling_result is None:
        raise ValueError(
            "No tiling data available. plot_tiling() requires a result from grid-based layout (GridBasedLayout).",
        )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    assigned_set = set(assignments.tolist()) if assignments is not None else set()

    # Build two collections: assigned and unassigned tiles
    assigned_patches = []
    unassigned_patches = []
    for i, poly in enumerate(tiling_result.polygons):
        coords = np.array(poly.exterior.coords)
        patch = MplPolygon(coords, closed=True)
        if i in assigned_set:
            assigned_patches.append(patch)
        else:
            unassigned_patches.append(patch)

    if show_unassigned and unassigned_patches:
        pc_unassigned = PatchCollection(
            unassigned_patches,
            facecolor=unassigned_color,
            edgecolor=tile_edgecolor,
            linewidth=tile_linewidth,
            alpha=tile_alpha,
        )
        ax.add_collection(pc_unassigned)

    if show_assigned and assigned_patches:
        pc_assigned = PatchCollection(
            assigned_patches,
            facecolor=assigned_color,
            edgecolor=tile_edgecolor,
            linewidth=tile_linewidth,
            alpha=tile_alpha,
        )
        ax.add_collection(pc_assigned)

    if show_symbols:
        result.plot(ax=ax, **kwargs)

    # Auto-scale to tile bounds
    all_coords = np.vstack([np.array(poly.exterior.coords) for poly in tiling_result.polygons])
    ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Tiling Grid")

    return ax


# ---------------------------------------------------------------------------
# Private helpers for plot_symbols()
# ---------------------------------------------------------------------------


def _is_mpl_color(value: Any) -> bool:
    """Return True if *value* is a valid matplotlib color specification."""
    import matplotlib.colors as mc

    try:
        return mc.is_color_like(value)
    except Exception:
        return False


def _is_column(name: str, result: SymbolCartogram, source_gdf: gpd.GeoDataFrame | None) -> bool:
    """Return True if *name* exists in any available column source."""
    if source_gdf is not None and name in source_gdf.columns:
        return True
    if name in result.symbols.columns:
        return True
    internal = getattr(result, "_source_gdf", None)
    return bool(internal is not None and name in internal.columns)


def _find_column(
    name: str,
    result: SymbolCartogram,
    source_gdf: gpd.GeoDataFrame | None,
) -> NDArray:
    """Return column values as a 1-D array (length == len(result.symbols)).

    Lookup priority (highest first):
    1. Explicit *source_gdf* argument.
    2. ``result.symbols``.
    3. ``result._source_gdf`` (respecting ``_valid_mask``).

    Raises ``ValueError`` listing available columns when the column is absent.
    """
    n = len(result.symbols)

    # 1. Explicit source_gdf override
    if source_gdf is not None and name in source_gdf.columns:
        arr = source_gdf[name].values
        if len(arr) == n:
            return arr
        # Try applying valid_mask to match lengths
        mask = getattr(result, "_valid_mask", None)
        if mask is not None:
            arr = source_gdf.loc[mask, name].values
            if len(arr) == n:
                return arr

    # 2. result.symbols
    if name in result.symbols.columns:
        return result.symbols[name].values

    # 3. result._source_gdf
    internal = getattr(result, "_source_gdf", None)
    if internal is not None and name in internal.columns:
        mask = getattr(result, "_valid_mask", None)
        arr = internal.loc[mask, name].values if mask is not None else internal[name].values
        if len(arr) == n:
            return arr

    # Collect available columns for a helpful error message
    available: set[str] = set(result.symbols.columns)
    if source_gdf is not None:
        available |= set(source_gdf.columns)
    if internal is not None:
        available |= set(internal.columns)
    # Remove private internal columns from the suggestion list
    public = sorted(c for c in available if not c.startswith("_"))
    raise ValueError(f"Column '{name}' not found in symbols or source data. Available columns: {public}")


def _apply_cmap(
    values: NDArray,
    cmap: str,
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
) -> NDArray:
    """Apply *cmap* to a float array, returning an ``(n, 4)`` RGBA array."""
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(cmap)
    _vmin = vmin if vmin is not None else float(np.nanmin(values))
    _vmax = vmax if vmax is not None else float(np.nanmax(values))
    _norm = norm if norm is not None else mc.Normalize(vmin=_vmin, vmax=_vmax)
    return cm(_norm(values.astype(float)))


def _apply_color_mapping(
    col_vals: NDArray,
    cmap: str,
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
    color_map: dict[str, Any] | None,
) -> NDArray:
    """Map column values to an ``(n, 4)`` RGBA array.

    Numeric columns use *cmap* / *norm*.
    Categorical columns cycle through the ``"tab10"`` qualitative palette,
    with optional per-category overrides from *color_map*.
    """
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt

    is_numeric = np.issubdtype(col_vals.dtype, np.number)

    if is_numeric:
        return _apply_cmap(col_vals.astype(float), cmap, norm, vmin, vmax)

    # Categorical: assign colors from tab10 palette
    col_strs = col_vals.astype(str)
    unique_vals = list(dict.fromkeys(col_strs))  # preserve insertion order
    qual_cmap = plt.get_cmap("tab10")
    cat_colors: dict[str, Any] = {v: qual_cmap(i % 10) for i, v in enumerate(unique_vals)}

    # Apply per-category overrides
    if color_map:
        for k, v in color_map.items():
            cat_colors[str(k)] = mc.to_rgba(v)

    return np.array([cat_colors[str(v)] for v in col_strs])


def _to_rgba_array(colors: Any, n: int) -> NDArray:
    """Convert any color specification to an ``(n, 4)`` RGBA float32 array."""
    import matplotlib.colors as mc

    arr = np.asarray(colors)

    if arr.ndim == 2 and arr.shape == (n, 4):
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape == (n, 3):
        return np.concatenate([arr.astype(float), np.ones((n, 1))], axis=1)
    if arr.ndim == 1 and len(arr) == n:
        # Array of color strings / tuples
        return np.array([mc.to_rgba(c) for c in arr])

    # Scalar color → broadcast
    return np.tile(mc.to_rgba(colors), (n, 1))


def _resolve_color(
    param: Any,
    result: SymbolCartogram,
    source_gdf: gpd.GeoDataFrame | None,
    cmap: str,
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
    n: int,
    default: Any = "steelblue",
    color_map: dict[str, Any] | None = None,
) -> tuple[Any, bool, NDArray | None, str | None]:
    """Resolve a color parameter to a matplotlib-compatible color specification.

    Returns
    -------
    colors : resolved color(s) - scalar string, or ``(n, 4)`` RGBA array.
    is_mapped : True when driven by a data column (triggers legend).
    col_values : raw column values, or None.
    col_name : column name used, or None.

    """
    import matplotlib.colors as mc

    if param is None:
        return default, False, None, None

    # Pass through matplotlib special edge-color tokens unchanged
    if isinstance(param, str) and param in ("none", "face"):
        return param, False, None, None

    if isinstance(param, str):
        is_color = _is_mpl_color(param)
        is_col = _is_column(param, result, source_gdf)

        if is_color and not is_col:
            return param, False, None, None

        if is_col:
            if is_color:
                warnings.warn(
                    f"'{param}' is both a valid matplotlib color and a column name. "
                    "Treating it as a column name. Pass an explicit array to suppress "
                    "this warning.",
                    UserWarning,
                    stacklevel=5,
                )
            col_vals = _find_column(param, result, source_gdf)
            colors = _apply_color_mapping(col_vals, cmap, norm, vmin, vmax, color_map)
            return colors, True, col_vals, param

        # Not a color and not a column → try column lookup for a better error
        _find_column(param, result, source_gdf)  # raises ValueError

    # Array-like input
    arr = np.asarray(param)

    # (n, 4) or (n, 3) float arrays → direct RGBA
    if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] in (3, 4):
        if arr.shape[1] == 3:
            arr = np.concatenate([arr.astype(float), np.ones((n, 1))], axis=1)
        return arr.astype(float), False, None, None

    # (n,) numeric → apply colormap
    if arr.ndim == 1 and len(arr) == n and arr.dtype.kind in ("f", "i", "u"):
        colors = _apply_cmap(arr.astype(float), cmap, norm, vmin, vmax)
        return colors, False, arr, None

    # (n,) object array → list of colors
    if arr.ndim == 1 and len(arr) == n:
        return np.array([mc.to_rgba(str(c)) for c in arr]), False, None, None

    raise ValueError(
        f"Cannot interpret {type(param).__name__!r} value as a color specification. "
        f"Expected a color string, column name, or array of length {n}.",
    )


def _resolve_scalar(
    param: Any,
    result: SymbolCartogram,
    source_gdf: gpd.GeoDataFrame | None,
    value_range: tuple[float, float],
    n: int,
    default: float,
) -> tuple[float | NDArray, bool]:
    """Resolve *alpha* or *linewidth* to a scalar or per-symbol ``(n,)`` array.

    Returns ``(value, is_array)``.
    """
    if param is None:
        return default, False

    if isinstance(param, (int, float)):
        return float(param), False

    if isinstance(param, str):
        col_vals = _find_column(param, result, source_gdf).astype(float)
        lo, hi = value_range
        col_min, col_max = float(np.nanmin(col_vals)), float(np.nanmax(col_vals))
        t = (col_vals - col_min) / (col_max - col_min) if col_max > col_min else np.full(n, 0.5)
        return lo + t * (hi - lo), True

    arr = np.asarray(param, dtype=float)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"Expected scalar or 1-D array of length {n}, got shape {arr.shape}.")
    return arr, True


def _resolve_hatch(
    param: Any,
    result: SymbolCartogram,
    source_gdf: gpd.GeoDataFrame | None,
    hatch_map: dict[str, str] | None,
    n: int,
) -> tuple[list[str] | None, bool, dict[str, str] | None, str | None]:
    """Resolve hatch parameter to a per-symbol list of hatch strings.

    Returns
    -------
    hatch_list : per-symbol list of hatch strings, or None.
    is_mapped : True when driven by a data column (enables hatch legend).
    cat_hatch_map : ordered category → hatch mapping (only when is_mapped).
    col_name : column name used, or None.

    """
    if param is None:
        return None, False, None, None

    if isinstance(param, str):
        if _is_column(param, result, source_gdf):
            col_vals = _find_column(param, result, source_gdf).astype(str)
            unique_vals = list(dict.fromkeys(col_vals))
            _map: dict[str, str] = dict(hatch_map or {})
            for i, v in enumerate(unique_vals):
                if v not in _map:
                    _map[v] = _HATCH_DEFAULTS[i % len(_HATCH_DEFAULTS)]
            # Build ordered cat→hatch map (only unique values, insertion-order preserved)
            cat_hatch_map = {v: _map[v] for v in unique_vals}
            return [_map.get(v, "") for v in col_vals], True, cat_hatch_map, param
        # Treat as a literal global hatch pattern — not column-driven
        return [param] * n, False, None, None

    lst = list(param)
    if len(lst) != n:
        raise ValueError(f"hatch list length {len(lst)} != number of symbols {n}.")
    return lst, False, None, None


def _add_legend(
    ax: plt.Axes,
    col_values: NDArray,
    col_name: str | None,
    cmap: str,
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
    legend_kwds: dict,
    color_map: dict[str, Any] | None,
    role: str = "face",
) -> None:
    """Attach a colorbar (numeric) or patch legend (categorical) to *ax*.

    Parameters
    ----------
    role : {"face", "edge"}
        ``"face"`` renders filled patches; ``"edge"`` renders hollow patches
        (transparent fill, coloured border) so the legend represents edge
        styling.  Only affects the categorical path.

    """
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt

    is_numeric = np.issubdtype(np.asarray(col_values).dtype, np.number)

    if is_numeric:
        cm = plt.get_cmap(cmap)
        _vmin = vmin if vmin is not None else float(np.nanmin(col_values))
        _vmax = vmax if vmax is not None else float(np.nanmax(col_values))
        _norm = norm if norm is not None else mc.Normalize(vmin=_vmin, vmax=_vmax)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=_norm)
        sm.set_array([])
        cbar_kwds = {k: v for k, v in legend_kwds.items() if k not in ("title", "loc")}
        cbar = ax.get_figure().colorbar(sm, ax=ax, **cbar_kwds)
        label = legend_kwds.get("title", col_name or "")
        cbar.set_label(label)
    else:
        import matplotlib.patches as mpatches

        col_strs = np.asarray(col_values).astype(str)
        unique_vals = list(dict.fromkeys(col_strs))
        qual_cmap = plt.get_cmap("tab10")
        cat_colors: dict[str, Any] = {v: qual_cmap(i % 10) for i, v in enumerate(unique_vals)}
        if color_map:
            for k, v in color_map.items():
                cat_colors[str(k)] = mc.to_rgba(v)

        if role == "edge":
            handles = [
                mpatches.Patch(facecolor="none", edgecolor=cat_colors[v], linewidth=2, label=v) for v in unique_vals
            ]
        else:
            handles = [mpatches.Patch(facecolor=cat_colors[v], label=v) for v in unique_vals]

        legend_patch_kwds = {k: v for k, v in legend_kwds.items() if k not in ("shrink", "orientation")}
        if col_name and "title" not in legend_patch_kwds:
            legend_patch_kwds["title"] = col_name

        # If there is already a legend on the axes, preserve it with add_artist
        # before drawing the new one, so both are visible simultaneously.
        existing = ax.get_legend()
        if existing is not None:
            ax.add_artist(existing)
        ax.legend(handles=handles, **legend_patch_kwds)


def _add_hatch_legend(
    ax: plt.Axes,
    cat_hatch_map: dict[str, str],
    col_name: str | None,
    legend_kwds: dict,
) -> None:
    """Attach a patch legend showing hatch-pattern ↔ category mappings.

    Parameters
    ----------
    cat_hatch_map : dict
        Ordered mapping of category label → matplotlib hatch pattern string.
    col_name : str or None
        Column name used as the legend title (when not overridden via
        ``legend_kwds["title"]``).
    legend_kwds : dict
        Standard ``Axes.legend()`` keyword arguments.  An optional nested
        ``"patch_kw"`` key controls the visual appearance of the legend
        patches themselves (``facecolor``, ``edgecolor``, ``linewidth``):

        .. code-block:: python

            hatch_legend_kwds={
                "title": "Land use",
                "loc": "upper left",
                "patch_kw": {"facecolor": "lightyellow", "edgecolor": "darkgreen"},
            }

        Defaults: ``facecolor="none"``, ``edgecolor="black"``, ``linewidth=1.5``.

    """
    import matplotlib.patches as mpatches

    # Extract patch appearance overrides from the nested "patch_kw" key
    patch_kw_user: dict[str, Any] = legend_kwds.get("patch_kw", {})
    patch_defaults = {"facecolor": "none", "edgecolor": "black", "linewidth": 1.5}
    patch_kw = {**patch_defaults, **patch_kw_user}

    handles = [mpatches.Patch(hatch=h if h else None, label=cat, **patch_kw) for cat, h in cat_hatch_map.items()]

    legend_kw = {k: v for k, v in legend_kwds.items() if k not in ("shrink", "orientation", "patch_kw")}
    if col_name and "title" not in legend_kw:
        legend_kw["title"] = col_name

    existing = ax.get_legend()
    if existing is not None:
        ax.add_artist(existing)
    ax.legend(handles=handles, **legend_kw)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def plot_symbols(
    result: SymbolCartogram,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    *,
    source_gdf: gpd.GeoDataFrame | None = None,
    # Fill
    facecolor: Any = None,
    cmap: str = "viridis",
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    color_map: dict[str, Any] | None = None,
    # Transparency
    alpha: float | str | Sequence | None = 0.9,
    alpha_range: tuple[float, float] = (0.2, 1.0),
    # Edge
    edgecolor: Any = "none",
    edge_cmap: str | None = None,
    linewidth: float | str | Sequence | None = 0.5,
    linewidth_range: tuple[float, float] = (0.5, 3.0),
    # Hatch
    hatch: str | Sequence | None = None,
    hatch_map: dict[str, str] | None = None,
    # Legend
    legend: bool = True,
    legend_kwds: dict | None = None,
    edge_legend: bool = True,
    edge_legend_kwds: dict | None = None,
    hatch_legend: bool = True,
    hatch_legend_kwds: dict | None = None,
    # Labels
    label: str | Sequence | None = None,
    label_color: Any = "black",
    label_fontsize: float | str | Sequence | None = 8,
    label_fontsize_range: tuple[float, float] = (6.0, 14.0),
    label_kwargs: dict | None = None,
    # Other
    zorder: int = 1,
    title: str | None = None,
) -> plt.Axes:
    """Plot a symbol cartogram with rich, per-symbol visual styling.

    Every visual property can be specified in three ways:

    * **Globally** - a scalar value or color string applied to all symbols.
    * **Data-driven** - a column name (``str``) looked up in the cartogram or
      source data and mapped to the visual channel.
    * **Explicitly per-symbol** - a list or NumPy array with one value per symbol.

    Parameters
    ----------
    result : SymbolCartogram
        Cartogram to visualise.
    ax : plt.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    figsize : tuple
        Figure size when creating a new figure.
    source_gdf : gpd.GeoDataFrame, optional
        External GeoDataFrame used for column lookups (highest priority).
        Useful when you want to colour by a column not stored on the
        cartogram itself, e.g. ``result.plot(source_gdf=my_gdf, facecolor="gdp")``.
    facecolor : color, column name, or array-like, optional
        Symbol fill colour.  Accepts:

        * A matplotlib colour string (``"steelblue"``, ``"#2ca02c"``).
        * A column name → numeric columns are mapped through *cmap* / *norm*;
          categorical columns are auto-assigned colours from a qualitative
          palette (``"tab10"``), overridable via *color_map*.
        * A 1-D numeric array of length *n* → mapped through *cmap* / *norm*.
        * An ``(n, 3)`` or ``(n, 4)`` float array of RGB / RGBA values.
        * A list of colour strings or RGBA tuples, one per symbol.

        Defaults to ``"steelblue"`` when *None*.
    cmap : str
        Colormap name for numeric *facecolor* / *edgecolor* column mappings.
        Default ``"viridis"``.
    norm : matplotlib Normalize, optional
        Custom normalisation for the colourmap.
    vmin, vmax : float, optional
        Explicit data range for the colourmap normalisation.
    color_map : dict, optional
        Per-category colour overrides used when *facecolor* is a categorical
        column.  Unspecified categories receive auto-assigned palette colours.
        Example: ``{"Europe": "#2ca02c", "Africa": "#d62728"}``.
    alpha : float, column name, or array-like, optional
        Symbol opacity (0 = transparent, 1 = opaque).

        * Scalar float → global opacity.
        * Column name → linearly interpolated into *alpha_range*.
        * 1-D array of floats → per-symbol opacity.

        Default ``0.9``.
    alpha_range : (float, float)
        ``(min_alpha, max_alpha)`` used when *alpha* is a column name.
        Default ``(0.2, 1.0)``.
    edgecolor : color, column name, or array-like, optional
        Symbol edge colour.  Accepts the same forms as *facecolor*.
        Use ``"none"`` (default) for no visible border.
    edge_cmap : str, optional
        Separate colormap for edge colour mapping.  Falls back to *cmap*.
    linewidth : float, column name, or array-like, optional
        Edge line width.

        * Scalar float → global width.
        * Column name → linearly interpolated into *linewidth_range*.
        * 1-D array → per-symbol widths.

        Default ``0.5``.
    linewidth_range : (float, float)
        ``(min_lw, max_lw)`` used when *linewidth* is a column name.
        Default ``(0.5, 3.0)``.
    hatch : str, column name, or sequence, optional
        Fill hatching.

        * A matplotlib hatch pattern string (``"///"``, ``"..."``, etc.) →
          applied globally.
        * A column name → each unique category is assigned a hatch pattern
          from *_HATCH_DEFAULTS*, overridable via *hatch_map*.
        * A list / array of pattern strings, one per symbol.

        .. note::
            Hatching is only visible when *edgecolor* is not ``"none"``.
            A ``UserWarning`` is raised when hatching is requested without a
            visible edge colour.
    hatch_map : dict, optional
        Per-category hatch overrides used when *hatch* is a column name.
        Example: ``{"urban": "///", "rural": "...", "forest": "ooo"}``.
    legend : bool
        Whether to display a colourbar (numeric column) or patch legend
        (categorical column) when *facecolor* is data-driven.  Default ``True``.
    legend_kwds : dict, optional
        Extra keyword arguments forwarded to ``Figure.colorbar()`` (numeric)
        or ``Axes.legend()`` (categorical).  Use ``"title"`` to set the
        legend label.
    edge_legend : bool
        Whether to display a separate legend for *edgecolor* when it is
        data-driven from a **different** column than *facecolor*.
        When both map the same column, only one legend is shown.  Default ``True``.
    edge_legend_kwds : dict, optional
        Same as *legend_kwds* but for the edge-colour legend.
    hatch_legend : bool
        Whether to display a patch legend showing the hatch-pattern ↔ category
        mapping when *hatch* is a data-driven column name.  Default ``True``.
        Has no effect when *hatch* is a global pattern string or a list.
    hatch_legend_kwds : dict, optional
        Standard ``Axes.legend()`` keyword arguments for the hatch legend.
        Additionally accepts a nested ``"patch_kw"`` key (dict) for patch
        appearance (``facecolor``, ``edgecolor``, ``linewidth``):

        .. code-block:: python

            hatch_legend_kwds={
                "title": "Land use",
                "loc": "lower left",
                "patch_kw": {"facecolor": "lightyellow", "edgecolor": "darkgreen"},
            }
    label : str, column name, or sequence, optional
        Per-symbol text labels.

        * A column name → string representation of each value is used.
        * A list / array of strings, one per symbol.
        * ``None`` → no labels (default).
    label_color : color, column name, or array-like, optional
        Text colour for the labels.  Accepts the same forms as *facecolor*.
        Default ``"black"``.
    label_fontsize : float, column name, or array-like, optional
        Font size for labels.

        * Scalar float → global size.
        * Column name → linearly interpolated into *label_fontsize_range*.
        * 1-D array → per-symbol sizes.

        Default ``8``.
    label_fontsize_range : (float, float)
        ``(min, max)`` font-size range when *label_fontsize* is a column name.
        Default ``(6.0, 14.0)``.
    label_kwargs : dict, optional
        Additional keyword arguments forwarded to ``Axes.text()`` for every
        label (e.g. ``{"fontweight": "bold", "ha": "left"}``).
    zorder : int
        Matplotlib drawing order for symbol patches.  Labels are placed at
        ``zorder + 1``.  Default ``1``.
    title : str, optional
        Axes title.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Examples
    --------
    >>> # Global style
    >>> result.plot(facecolor="steelblue", edgecolor="white", linewidth=0.8)

    >>> # Data-driven fill → automatic colorbar
    >>> result.plot(facecolor="gdp_per_capita", cmap="YlOrRd")

    >>> # Data-driven edge colour → second legend auto-added
    >>> result.plot(facecolor="steelblue", edgecolor="region")

    >>> # Both fill and edge from different columns → two legends
    >>> result.plot(facecolor="pop_est", cmap="YlOrRd",
    ...             edgecolor="region", edge_cmap="Set1")

    >>> # Categorical fill with qualitative palette
    >>> result.plot(facecolor="continent")

    >>> # Categorical fill with partial colour overrides
    >>> result.plot(facecolor="continent",
    ...             color_map={"Europe": "#2ca02c", "Africa": "#d62728"})

    >>> # Per-symbol alpha driven by a column
    >>> result.plot(facecolor="steelblue",
    ...             alpha="population", alpha_range=(0.3, 1.0))

    >>> # Hatching by category — hatch legend added automatically
    >>> result.plot(facecolor="lightgray", edgecolor="black",
    ...             hatch="land_use",
    ...             hatch_map={"urban": "///", "rural": "...", "forest": "ooo"})

    >>> # Hatch legend with custom patch appearance
    >>> result.plot(facecolor="whitesmoke", edgecolor="black",
    ...             hatch="region",
    ...             hatch_legend_kwds={"title": "Region",
    ...                                "patch_kw": {"facecolor": "lightyellow"}})

    >>> # Explicit per-symbol array (e.g. random colours)
    >>> import numpy as np
    >>> result.plot(facecolor=np.random.rand(len(result.symbols), 4), legend=False)

    >>> # Labels from a column
    >>> result.plot(facecolor="pop_est", label="name")

    >>> # Labels with per-symbol colour
    >>> result.plot(facecolor="steelblue", label="iso_a3",
    ...             label_color="region", label_fontsize=7)

    >>> # Look up columns from an external GeoDataFrame
    >>> result.plot(source_gdf=world_gdf, facecolor="gdp_md_est",
    ...             label="name", cmap="plasma")

    """
    from collections import defaultdict

    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    symbols = result.symbols
    n = len(symbols)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Warn when hatching would be invisible
    if hatch is not None and edgecolor in (None, "none"):
        warnings.warn(
            "hatch is set but edgecolor='none' (the default). "
            "Hatching requires a visible edge colour. "
            "Set edgecolor to a colour such as 'black'.",
            UserWarning,
            stacklevel=2,
        )

    # --- Resolve visual properties ---
    face_colors, is_mapped, col_values, col_name = _resolve_color(
        facecolor,
        result,
        source_gdf,
        cmap,
        norm,
        vmin,
        vmax,
        n,
        default="steelblue",
        color_map=color_map,
    )

    alpha_val, alpha_is_arr = _resolve_scalar(
        alpha,
        result,
        source_gdf,
        alpha_range,
        n,
        default=0.9,
    )

    _edge_cmap = edge_cmap if edge_cmap is not None else cmap
    edge_colors, edge_is_mapped, edge_col_values, edge_col_name = _resolve_color(
        edgecolor,
        result,
        source_gdf,
        _edge_cmap,
        None,
        None,
        None,
        n,
        default="none",
    )

    lw_val, lw_is_arr = _resolve_scalar(
        linewidth,
        result,
        source_gdf,
        linewidth_range,
        n,
        default=0.5,
    )

    hatch_list, hatch_is_mapped, cat_hatch_map, hatch_col_name = _resolve_hatch(hatch, result, source_gdf, hatch_map, n)

    # --- Bake per-symbol alpha into RGBA facecolor ---
    if alpha_is_arr:
        face_rgba = _to_rgba_array(face_colors, n)
        face_rgba = face_rgba.copy()
        face_rgba[:, 3] = np.clip(np.asarray(alpha_val), 0.0, 1.0)
        face_colors = face_rgba
        collection_alpha = None
    else:
        collection_alpha = float(alpha_val) if alpha_val is not None else None

    # --- Build Shapely exterior → matplotlib Polygon patches ---
    patches: list[MplPolygon] = []
    for geom in symbols.geometry:
        coords = np.array(geom.exterior.coords)
        patches.append(MplPolygon(coords, closed=True))

    # --- Render ---
    if hatch_list is None:
        # Single PatchCollection - most efficient path
        pc = PatchCollection(
            patches,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=lw_val,
            alpha=collection_alpha,
            zorder=zorder,
        )
        ax.add_collection(pc)
    else:
        # Per-patch hatching requires one PatchCollection per unique hatch
        # because matplotlib applies a single hatch per collection.
        face_rgba = _to_rgba_array(face_colors, n)
        if collection_alpha is not None:
            face_rgba = face_rgba.copy()
            face_rgba[:, 3] = np.clip(collection_alpha, 0.0, 1.0)

        lw_arr = np.asarray(lw_val) if lw_is_arr else None
        groups: dict[str, list[int]] = defaultdict(list)
        for i, h in enumerate(hatch_list):
            groups[h].append(i)

        for h, indices in groups.items():
            group_patches = [patches[i] for i in indices]
            fc = face_rgba[indices]
            lw = lw_arr[indices] if lw_arr is not None else lw_val
            # Per-group edge colours
            ec = edge_colors[indices] if isinstance(edge_colors, np.ndarray) and edge_colors.ndim == 2 else edge_colors

            pc = PatchCollection(
                group_patches,
                facecolors=fc,
                edgecolors=ec,
                linewidths=lw,
                hatch=h if h else None,
                zorder=zorder,
            )
            ax.add_collection(pc)

    # --- Legend / colorbar for facecolor ---
    if legend and is_mapped and col_values is not None:
        _add_legend(
            ax,
            col_values,
            col_name,
            cmap,
            norm,
            vmin,
            vmax,
            legend_kwds or {},
            color_map,
            role="face",
        )

    # --- Legend for hatch patterns (only when column-driven) ---
    if hatch_legend and hatch_is_mapped and cat_hatch_map:
        _add_hatch_legend(ax, cat_hatch_map, hatch_col_name, hatch_legend_kwds or {})

    # --- Legend for edgecolor (only when mapped from a different column) ---
    if (
        edge_legend
        and edge_is_mapped
        and edge_col_values is not None
        and edge_col_name != col_name  # skip if same column already has a legend
    ):
        _add_legend(
            ax,
            edge_col_values,
            edge_col_name,
            _edge_cmap,
            None,
            None,
            None,
            edge_legend_kwds or {},
            None,
            role="edge",
        )

    # --- Per-symbol labels ---
    if label is not None:
        # Resolve label texts
        if isinstance(label, str) and _is_column(label, result, source_gdf):
            label_texts = [str(v) for v in _find_column(label, result, source_gdf)]
        else:
            label_list = list(label) if not isinstance(label, str) else [label] * n
            if len(label_list) != n:
                raise ValueError(f"label list length {len(label_list)} != number of symbols {n}.")
            label_texts = [str(v) for v in label_list]

        # Resolve label colours to a per-symbol list
        lc_colors, _, _, _ = _resolve_color(
            label_color,
            result,
            source_gdf,
            cmap,
            None,
            None,
            None,
            n,
            default="black",
        )
        if isinstance(lc_colors, np.ndarray) and lc_colors.ndim == 2:
            # RGBA array → list of tuples
            label_colors_list: list[Any] = [tuple(lc_colors[i]) for i in range(n)]
        elif isinstance(lc_colors, list):
            label_colors_list = lc_colors
        else:
            # Scalar colour → broadcast
            label_colors_list = [lc_colors] * n

        # Resolve label fontsize
        fs_val, fs_is_arr = _resolve_scalar(
            label_fontsize,
            result,
            source_gdf,
            label_fontsize_range,
            n,
            default=8.0,
        )
        fs_arr = np.asarray(fs_val) if fs_is_arr else None

        xs = result.symbols["_symbol_x"].values
        ys = result.symbols["_symbol_y"].values
        _lkw = {"ha": "center", "va": "center", **(label_kwargs or {})}

        for i, (x, y, text) in enumerate(zip(xs, ys, label_texts)):
            fs_i = float(fs_arr[i]) if fs_arr is not None else float(fs_val)  # type: ignore[arg-type]
            ax.text(
                x,
                y,
                text,
                color=label_colors_list[i],
                fontsize=fs_i,
                zorder=zorder + 1,
                **_lkw,
            )

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    return ax
