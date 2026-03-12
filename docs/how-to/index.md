# How-to Guides

Practical notebooks showing how to accomplish specific tasks with carto-flow. Each guide answers one focused question.

## Flow Cartogram

### [Diagnose and Improve Convergence](diagnose-convergence.ipynb)
Understand what `CONVERGED`, `COMPLETED`, and `STALLED` status mean, read error metrics, and apply the right fix — from iterative refinement with `CartogramWorkflow` to multi-resolution morphing.

### [Save and Export Results](save-and-export-results.ipynb)
Save a cartogram to disk and reload it, or export the morphed geometries to a GeoDataFrame / GeoPackage for use in GIS tools.

### [Reduce Boundary Distortion](reduce-boundary-distortion.ipynb)
Use `prescale_components`, `DensityBorderExtension`, anisotropy fields, and velocity modifiers like `BoundaryDecay` to control distortion at the edges of your cartogram.

### [Animate the Morphing Process](create-animation.ipynb)
Capture intermediate frames with `snapshot_every`, build an animation with `animate_morph_history()`, and save to GIF or MP4.

### [Co-Morph Landmarks and Coordinates](co-morph-landmarks-and-coordinates.ipynb)
Displace cities, annotations, or point coordinates consistently with the cartogram using `landmarks` (GeoDataFrames) or `displacement_coords` (NumPy arrays).

## Proportional Cartogram

### [Apply Partitions to a Flow Cartogram](partitions-on-flow-cartogram.ipynb)
Combine the two modules: morph geometries with the flow cartogram, then apply proportional splits or shrinks to show a second variable within each region.

## Symbol Cartogram

### [Style Symbols by Category](style-symbols-by-category.ipynb)
Separate layout computation from styling: map data columns to colors, shapes, opacity, hatching, and labels using `plot()` keyword arguments.
