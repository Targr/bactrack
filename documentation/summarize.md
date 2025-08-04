# summarize.py Documentation

The `summarize.py` module computes quantitative summaries of bacterial motility trajectories. It outputs both per-frame and per-trajectory metrics depending on which motility modes are enabled.

## Function: `summarize_trajectories`

```python
def summarize_trajectories(
    traj,
    mono=False,
    peri=False,
    twitch=False,
    biofilm=False,
    gliding=False,
    swarming=False,
    pili_retracting=False,
    microns_per_pixel=1.616
):
```

### Parameters

| Parameter           | Type  | Default  | Description                                                         |
| ------------------- | ----- | -------- | ------------------------------------------------------------------- |
| `traj`              | list  | required | List of trajectory dictionaries returned by `build_trajectories()`. |
| `mono`              | bool  | False    | Enable monotrichous (reversal-based) motility metrics.              |
| `peri`              | bool  | False    | Enable peritrichous (circling/tumble-based) motility metrics.       |
| `twitch`            | bool  | False    | Enable twitching motility metrics.                                  |
| `biofilm`           | bool  | False    | Enable biofilm-associated stationary behavior metrics.              |
| `gliding`           | bool  | False    | Enable gliding motility metrics.                                    |
| `swarming`          | bool  | False    | Enable swarming motility metrics.                                   |
| `pili_retracting`   | bool  | False    | Enable pili retraction burst metrics.                               |
| `microns_per_pixel` | float | 1.616    | Conversion factor from pixels to microns.                           |

### Returns

```python
df_per_frame, df_summary = summarize_trajectories(...)
```

* `df_per_frame`: `pandas.DataFrame` containing one row per timepoint, per trajectory.
* `df_summary`: `pandas.DataFrame` containing one row per trajectory.

### Metrics Included (depending on enabled flags)

#### Per-Frame (`df_per_frame`)

| Column        | Description                                   |
| ------------- | --------------------------------------------- |
| `traj_id`     | ID of the trajectory                          |
| `x`           | X-coordinate (in pixels)                      |
| `y`           | Y-coordinate (in pixels)                      |
| `time (sec)`  | Time value, computed using `frame_interval`   |
| `reversal`    | Binary flag per frame (only if `mono=True`)   |
| `twitch_jump` | Binary flag per frame (only if `twitch=True`) |

#### Per-Trajectory (`df_summary`)

| Column                          | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| `traj_id`                       | Unique trajectory ID                                           |
| `duration of trajectory (sec)`  | Total trajectory time span                                     |
| `speed (micron/sec)`            | Mean frame-to-frame speed, adjusted by pixel scaling           |
| `reversal_count`                | Number of directional reversals (if `mono=True`)               |
| `reversal_bias`                 | Fraction of frames with reversals (if `mono=True`)             |
| `tumblebias`                    | Fraction of frames identified as tumbles (if `peri=True`)      |
| `twitch_jump_count`             | Number of twitching displacement events (if `twitch=True`)     |
| `twitch_jump_fraction`          | Fraction of frames with twitching behavior                     |
| `twitch_jump_rate (Hz)`         | Twitch frequency, defined as count per second                  |
| `biofilm_net_displacement (µm)` | Net displacement from start to end (if `biofilm=True`)         |
| `biofilm_stationary_flag`       | Binary flag for stationarity (based on threshold)              |
| `gliding_consistency`           | Fraction of smooth directional motion (if `gliding=True`)      |
| `swarm_density`                 | Mean local density within 10-pixel radius (if `swarming=True`) |
| `is_swarming`                   | Binary flag if density exceeds `swarm_density_threshold`       |
| `pili_burst_count`              | Number of fast displacements (if `pili_retracting=True`)       |
| `pili_burst_fraction`           | Fraction of frames showing pili bursts                         |

### Notes

* The speed metric uses `microns_per_pixel` and the `dt` stored in each trajectory.
* Columns are only added to the output DataFrames if the relevant motility type is enabled.
* Trajectories that do not contain valid measurements (e.g., too short) may yield NaN in some fields.

### Example

```python
from bactracker.summarize import summarize_trajectories

# Assume 'traj' has been built and annotated with detect_motility
frame_df, summary_df = summarize_trajectories(traj, mono=True, twitch=True, microns_per_pixel=1.6)
```

This will return two DataFrames, one with per-frame measurements and one with summary statistics, both specific to monotrichous and twitching motility behaviors.
