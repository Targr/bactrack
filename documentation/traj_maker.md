# traj\_maker.py Documentation

This module contains the function to convert a filtered DataFrame of tracked particles (from `trackpy`) into a trajectory list format used throughout the `bactracker` analysis pipeline.

## Function: `build_trajectories`

```python
def build_trajectories(t_filtered, frame_interval=0.05):
```

### Parameters

| Parameter        | Type      | Default  | Description                                                                                                                                    |
| ---------------- | --------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `t_filtered`     | DataFrame | required | A DataFrame from `trackpy.link_df` and optionally filtered via `trackpy.filter_stubs`. Must contain `x`, `y`, `frame`, and `particle` columns. |
| `frame_interval` | float     | 0.05     | Time between frames in seconds. Used to compute time vectors for each trajectory.                                                              |

### Returns

* A `list` of `dicts`, where each dictionary represents a single trajectory with keys:

| Key        | Type    | Description                                                |
| ---------- | ------- | ---------------------------------------------------------- |
| `particle` | int     | Unique identifier for the tracked object.                  |
| `x`        | ndarray | X-coordinates of the particle over time.                   |
| `y`        | ndarray | Y-coordinates of the particle over time.                   |
| `time`     | ndarray | Relative time points, starting at 0 for each trajectory.   |
| `dt`       | ndarray | A single-element array indicating the frame interval used. |

### Example

```python
from bactracker.traj_maker import build_trajectories

# Assuming t_filtered is the output from run_tracking_pipeline
traj = build_trajectories(t_filtered, frame_interval=0.1)
```

### Notes

* `frame_interval` is used to compute the time vector for each trajectory based on the difference in frame number.
* The time vector always starts at 0 for each trajectory.
* The output format is designed to be compatible with `detect_motility()` and `summarize_trajectories()` functions.
* `dt` is included as an array for consistency with downstream expectations.
