# Parameter System Documentation

The `bactracker` module uses a configurable parameter system for detecting and classifying bacterial motility behaviors. These parameters are loaded from a `params.pkl` file using the function `load_or_create_params()` in `params.py`. If the file does not exist, it is created with a default set of values.

## Default Parameters

```python
default_params = {
    'frame_interval': 0.05,
    'min_circle_dt': 0.5,
    'max_circle_dt': 20.0,
    'circle_cost_threshold': 0.6,
    'fracCircling_threshold': 0.1,
    'twitch_jump_threshold': 5.0,
    'biofilm_disp_threshold': 3.0,
    'glide_angle_threshold': 0.1,
    'swarm_density_threshold': 3,
    'pili_retract_burst_threshold': 8.0
}
```

## Description of Parameters

| Parameter                      | Used In                         | Description                                                                                      |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `frame_interval`               | `traj_maker.py`, `summarize.py` | Time between consecutive frames in seconds. Used to compute real-time trajectories and speeds.   |
| `min_circle_dt`                | `classify.py` (peritrichous)    | Minimum duration of a loop (in seconds) to be classified as circling.                            |
| `max_circle_dt`                | `classify.py` (peritrichous)    | Maximum duration of a loop to be considered valid circling behavior.                             |
| `circle_cost_threshold`        | `classify.py` (peritrichous)    | Maximum allowed asymmetry (inverse radial uniformity) to accept a loop as circular.              |
| `fracCircling_threshold`       | `classify.py` (peritrichous)    | Minimum fraction of time spent circling to classify a bacterium as exhibiting circling behavior. |
| `twitch_jump_threshold`        | `classify.py` (twitching)       | Minimum frame-to-frame displacement (in pixels) to register as a twitch jump.                    |
| `biofilm_disp_threshold`       | `classify.py` (biofilm)         | Maximum net displacement (in pixels) over the trajectory to classify as stationary (biofilm).    |
| `glide_angle_threshold`        | `classify.py` (gliding)         | Maximum change in movement angle (radians) allowed to count as consistent gliding motion.        |
| `swarm_density_threshold`      | `classify.py` (swarming)        | Minimum average local neighbor count (within a radius of 10 pixels) to classify as swarming.     |
| `pili_retract_burst_threshold` | `classify.py` (pili retraction) | Minimum frame-to-frame displacement (in pixels) to register as a pili retraction burst.          |

## Loading Parameters

To load parameters for use:

```python
from bactracker.params import load_or_create_params
params = load_or_create_params('params.pkl')
```

If `params.pkl` does not exist, it will be created automatically with default values.

## Modifying Parameters

Parameters can be adjusted directly before use:

```python
params['twitch_jump_threshold'] = 4.0
```

To save the modified dictionary:

```python
import pickle
with open('params.pkl', 'wb') as f:
    pickle.dump(params, f)
```

## Parameters Not Included in the Dictionary

Some thresholds are passed directly to functions instead of being stored in the `params.pkl` file. For example:

* `reversal_angle_threshold` (used for monotrichous reversal detection in `detect_motility`) must be passed explicitly:

```python
detect_motility(traj, params, mono=True, reversal_angle_threshold=2.5)
```

## Usage in Pipeline

Example:

```python
from bactracker import run_tracking_pipeline, build_trajectories
from bactracker.classify import detect_motility
from bactracker.summarize import summarize_trajectories
from bactracker.params import load_or_create_params

params = load_or_create_params()
params['glide_angle_threshold'] = 0.05

t_filtered = run_tracking_pipeline("movie.avi", "out.tiff", mode="gliding")
traj = build_trajectories(t_filtered)
traj = detect_motility(traj, params, gliding=True)
df1, df2 = summarize_trajectories(traj, gliding=True)
```

This system ensures reproducibility and flexibility for users analyzing different types of motility behaviors.
