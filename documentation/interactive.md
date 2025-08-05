# Interactive Detection Tuning Documentation

The `interactive.py` module provides functionality for interactively tuning detection parameters using a Jupyter-based graphical interface. This is intended for parameter calibration prior to running the full tracking pipeline.

## Function: `interactive_detection_tuner`

```python
import bactracker as bt

bt.interactive_detection_tuner(
    video_path,
    max_frames=100,
    minmass_range=(1, 1, 15),
    diameter_range=(3, 1, 11),
    merge_dist_range=(1, 20, 7)
)
```

### Parameters

| Parameter          | Type  | Default     | Description                                                                 |
| ------------------ | ----- | ----------- | --------------------------------------------------------------------------- |
| `video_path`       | str   | required    | Path to the input video file or image stack (e.g. `.mp4`, `.avi`, `tif`).   |
| `max_frames`       | int   | 100         | Maximum number of frames to read from the video for analysis.               |
| `minmass_range`    | tuple | (1, 50, 15) | Range and default value for `minmass` slider. Format: (default, min, max).  |
| `diameter_range`   | tuple | (3, 15, 7)  | Range and default value for `diameter` slider. Format: (default, min, max). |
| `merge_dist_range` | tuple | (1, 20, 7)  | Range and default for merging rod-shaped detections.                        |

### Behavior

* Loads up to `max_frames` from the video file.
* Computes the average background from the loaded frames.
* Subtracts the average background from the first frame.
* Displays the background-subtracted frame with detected features overlaid.
* Provides interactive controls for:

  * `minmass`: Brightness threshold for `trackpy.locate`
  * `diameter`: Estimated size of features to detect (in pixels)
  * `consolidate_rods`: Boolean toggle to enable rod merging
  * `merge_dist`: Distance threshold for merging elongated detections

### Dependencies

* OpenCV (`cv2`)
* NumPy
* Matplotlib
* TrackPy
* ipywidgets

### Example Usage (in Jupyter)

```python
from bactracker.interactive import interactive_detection_tuner
interactive_detection_tuner('example_video.mp4')
```

This will display the first background-subtracted frame from the video with adjustable detection parameters.

## Notes

* This function is designed for interactive use and will not function in a non-interactive script or command-line interface.
* No outputs are returned or saved. This is strictly for visual inspection and tuning of detection settings prior to full processing.
* For rod-shaped bacteria, enabling `consolidate_rods` applies a shape-based merging algorithm to group elongated features using PCA and DBSCAN.
