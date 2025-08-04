# tracking.py Documentation

The `tracking.py` module runs the full particle detection and linking pipeline. It supports background subtraction, rod merging, and tracking parameter customization for multiple bacterial motility types.

## Function: `run_tracking_pipeline`

```python
def run_tracking_pipeline(
    input_media,
    output_tiff,
    param_file='tracking_params.pkl',
    show_plot=True,
    param_overrides=None,
    mode='swimming',
    rolling_bg=False,
    rod_shaped=False
):
```

### Parameters

| Parameter         | Type | Default                 | Description                                                                                                       |
| ----------------- | ---- | ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `input_media`     | str  | required                | Path to the input video file (e.g. `.avi`, `.mp4`).                                                               |
| `output_tiff`     | str  | required                | Path to the TIFF stack used for caching or saving frames.                                                         |
| `param_file`      | str  | `'tracking_params.pkl'` | File where tracking parameters are loaded from or saved to.                                                       |
| `show_plot`       | bool | `True`                  | Whether to display the final linked trajectory plot.                                                              |
| `param_overrides` | dict | `None`                  | Dictionary of manual parameter overrides.                                                                         |
| `mode`            | str  | `'swimming'`            | Tracking mode. Options: `'swimming'`, `'twitching'`, `'biofilm'`, `'swarming'`, `'gliding'`, `'pili_retraction'`. |
| `rolling_bg`      | bool | `False`                 | If True, use a rolling background instead of a static average background.                                         |
| `rod_shaped`      | bool | `False`                 | If True, merge rod-shaped features using PCA and DBSCAN.                                                          |

### Tracking Modes and Defaults

Each mode defines a preset group of detection parameters:

| Mode              | `chunk_size` | `diameter` | `minmass` | `search_range` | `memory` | `stub_threshold` | `invert` |
| ----------------- | ------------ | ---------- | --------- | -------------- | -------- | ---------------- | -------- |
| `swimming`        | 500          | 7          | 15        | 11             | 7        | 5                | False    |
| `twitching`       | 300          | 5          | 5         | 4              | 1        | 2                | False    |
| `biofilm`         | 300          | 4          | 3         | 2              | 0        | 1                | False    |
| `swarming`        | 500          | 9          | 10        | 20             | 10       | 10               | False    |
| `gliding`         | 500          | 6          | 5         | 3              | 5        | 4                | False    |
| `pili_retraction` | 400          | 5          | 4         | 6              | 2        | 2                | False    |

Parameter values from the selected mode can be overridden using `param_overrides`.

### Output

* Saves a TIFF stack to `output_tiff` if it doesn't already exist.
* Returns a `pandas.DataFrame` of linked trajectories (output of `trackpy.link_df`).
* If no features are detected, returns `None`.

### Output DataFrame Columns

| Column     | Description                              |
| ---------- | ---------------------------------------- |
| `x`        | X-coordinate of detection                |
| `y`        | Y-coordinate of detection                |
| `frame`    | Frame index                              |
| `particle` | Unique trajectory ID assigned by TrackPy |

### Behavior

1. If `output_tiff` exists, load it directly.
2. Otherwise, load the video from `input_media`, convert to grayscale, and build a frame stack.
3. Compute background (rolling or static average).
4. Perform feature detection on each frame using `trackpy.locate`.
5. Optionally merge elongated detections (`rod_shaped=True`).
6. Link features into trajectories using `trackpy.link_df`.
7. Filter out short trajectories using `trackpy.filter_stubs`.
8. Optionally display results using `trackpy.plot_traj()`.

### Example Usage

```python
from bactracker import run_tracking_pipeline

# Run tracking on a swimming video
tracks = run_tracking_pipeline('movie.avi', 'out.tiff', mode='swimming')
```

### Notes

* TIFF output is cached to avoid reprocessing on subsequent runs.
* Parameters are saved automatically to `param_file`.
* Merging of rods is done using PCA + DBSCAN via `merge_rods_by_shape()` from `detection.py`.
* Background subtraction uses either the mean of the first 100 frames (static) or a sliding window around each frame (rolling).
* This function is not interactive and is meant for automated or batch runs.
