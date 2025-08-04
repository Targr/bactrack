# Motility Classification: `classify.py`

The `detect_motility()` function in `classify.py` is responsible for annotating bacterial trajectories with motility behavior features. This includes detection of circling, reversals, twitching, biofilm behavior, gliding consistency, swarming density, and pili retraction bursts.

Each motility mode is activated via its corresponding keyword argument (`peri`, `mono`, `twitch`, etc.).

---

## Function Signature

```python
def detect_motility(
    traj, params,
    peri=False, mono=False,
    twitch=False, biofilm=False,
    gliding=False, swarming=False, pili_retracting=False,
    reversal_angle_threshold=np.pi * 5/6,
    twitch_jump_threshold=5.0,
    biofilm_disp_threshold=3.0,
    glide_angle_threshold=0.1,
    swarm_density_threshold=3,
    pili_retract_burst_threshold=8.0
):
```

## Inputs

* `traj`: A list of trajectory dictionaries, each containing `x`, `y`, `time`, `dt`, and optionally other per-frame fields.
* `params`: Dictionary of parameter values (usually from `params.pkl`).
* Flags like `peri`, `mono`, `twitch`, etc.: Activate detection for a specific type of motility.
* Additional thresholds can be passed directly (if not using `params`).

---

## Motility Modes

### Peritrichous (`peri=True`)

* Detects loop-like (circling) behavior.
* Uses Shapely to compute self-intersections.
* Checks loop symmetry via a radial cost function.
* Computes:

  * `circling` (boolean array)
  * `fracCircling` (float)
  * `timeNotCircling` (float)
  * `tumble` (binary array)
  * `tumblebias` (float)

**Params used:**

* `min_circle_dt`
* `max_circle_dt`
* `circle_cost_threshold`
* `fracCircling_threshold`

### Monotrichous (`mono=True`)

* Detects reversals based on direction change.
* Compares consecutive movement vectors.
* Computes:

  * `reversal` (binary array)
  * `reversal_count` (int)
  * `reversal_bias` (float)

**Threshold:**

* `reversal_angle_threshold` (passed explicitly)

### Twitching (`twitch=True`)

* Detects sudden long displacements.
* Computes:

  * `twitch` (binary array)
  * `twitch_count` (int)
  * `twitch_fraction` (float)

**Params used:**

* `twitch_jump_threshold`

### Biofilm (`biofilm=True`)

* Labels low-net-displacement trajectories as stationary.
* Computes:

  * `biofilm_net_disp` (float)
  * `biofilm_stationary` (binary flag)

**Params used:**

* `biofilm_disp_threshold`

### Gliding (`gliding=True`)

* Assesses consistency of motion direction.
* Computes:

  * `gliding_consistency` (float)

**Params used:**

* `glide_angle_threshold`

### Swarming (`swarming=True`)

* Estimates local neighborhood density.
* Computes:

  * `swarm_density` (float)
  * `is_swarming` (binary flag)

**Params used:**

* `swarm_density_threshold`

### Pili Retraction (`pili_retracting=True`)

* Detects large frame-wise displacements.
* Computes:

  * `pili_bursts` (int)
  * `pili_fraction` (float)

**Params used:**

* `pili_retract_burst_threshold`

---

## Output

* Returns the input `traj` list, updated with per-cell motility annotations depending on which modes were enabled.

---

## Usage Example

```python
from bactracker.classify import detect_motility
params = load_or_create_params()
traj = detect_motility(traj, params, peri=True, mono=True, twitch=True)
```

This example annotates each trajectory in `traj` with circling behavior, reversal events, and twitch jumps based on thresholds provided in `params`.
