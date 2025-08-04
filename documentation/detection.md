# Detection Module: `detection.py`

The `detection.py` module provides utilities for post-processing feature detection, specifically for merging elongated or clustered bacterial detections (rods) based on shape characteristics.

This is particularly useful when working with rod-shaped bacteria, as closely spaced detections may represent parts of the same object.

---

## Function: `merge_rods_by_shape()`

```python
def merge_rods_by_shape(df, merge_dist=10, max_rod_length=30, max_aspect_ratio=5):
```

### Arguments

* `df`: DataFrame containing detected coordinates (`x`, `y`). Output of `trackpy.locate()`.
* `merge_dist` (int): Maximum distance in pixels between detections to be considered for merging.
* `max_rod_length` (float): Maximum allowable rod length (in pixels) to be considered a single merged object.
* `max_aspect_ratio` (float): Minimum elongation (length/width) for an object to be merged.

### Behavior

1. Performs DBSCAN clustering on the `(x, y)` positions of detected features.
2. For each cluster:

   * If it contains only 1 detection → keep as-is.
   * If multiple detections:

     * Apply PCA to compute principal components of the cluster shape.
     * Measure elongation and rod length.
     * If aspect ratio is high and rod length is within limit → merge to single point (centroid).
     * Otherwise → keep each detection separately.

### Output

* Returns a new DataFrame with merged or original `(x, y)` coordinates, depending on clustering outcome and geometry.

---

## Usage

This function is typically called after locating features in an image or video frame, especially when `rod_shaped=True` is passed to the `run_tracking_pipeline()` function:

```python
from bactracker.detection import merge_rods_by_shape
import trackpy as tp

frame = preprocess_image(...)
df = tp.locate(frame, diameter=7, minmass=15)
merged_df = merge_rods_by_shape(df, merge_dist=7, max_rod_length=30, max_aspect_ratio=4)
```

---

## Integration in Pipeline

Within `tracking.py`, this function is automatically invoked during feature detection if the `rod_shaped=True` flag is set:

```python
if rod_shaped:
    f = merge_rods_by_shape(f, merge_dist=7, max_rod_length=30, max_aspect_ratio=4)
```

This step improves tracking robustness for elongated bacterial morphologies by reducing noise from fragmented detections.

---

## Dependencies

* `sklearn.cluster.DBSCAN` (for spatial clustering)
* `sklearn.decomposition.PCA` (for shape analysis)
* `numpy`, `pandas`

No external model training is required.
