
#bactracker

Bactracker is a python package for detecting, tracting, and classifying bacterial motility types from microscopy videos. It supports multiple forms of motility like swimming, twitching, gliding, and interprets interpretable summaries of behavior.

##features
-video loading and preprocessing with automatic background subtraction
-TrackPy-based feature detection and tracking with user-friendly, interactively tunable parameters (Jupyter/IPython widgets)
-support for rod-shape detection and consolidation and settings for peritrichous vs monotrichous bacterial types
-motility classification: swimming, twitching, gliding, biofilm motility, swarming, pili retraction
-summarization into tidy per-frame and per-trajectory tables with variables individualized to your footage (if your bacteria are peritrichous and swimming, it’ll give run/tumble data along with general coordinates, times, distances, and trajectory duration stats)

##installation
```bash
pip install git+https://github.com/Targr/bactrack.git
or clone and install locally:
git clone https://github.com/Targr/bactrack.git
cd bactracker
pip install .
