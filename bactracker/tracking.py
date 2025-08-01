import os
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import tifffile as tiff
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

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
    default_modes = {
        'swimming': {'chunk_size': 500, 'diameter': 7, 'minmass': 15, 'search_range': 11, 'memory': 7, 'stub_threshold': 5, 'invert': False},
        'twitching': {'chunk_size': 300, 'diameter': 5, 'minmass': 5, 'search_range': 4, 'memory': 1, 'stub_threshold': 2, 'invert': False},
        'biofilm': {'chunk_size': 300, 'diameter': 4, 'minmass': 3, 'search_range': 2, 'memory': 0, 'stub_threshold': 1, 'invert': False},
        'swarming': {'chunk_size': 500, 'diameter': 9, 'minmass': 10, 'search_range': 20, 'memory': 10, 'stub_threshold': 10, 'invert': False},
        'gliding': {'chunk_size': 500, 'diameter': 6, 'minmass': 5, 'search_range': 3, 'memory': 5, 'stub_threshold': 4, 'invert': False},
        'pili_retraction': {'chunk_size': 400, 'diameter': 5, 'minmass': 4, 'search_range': 6, 'memory': 2, 'stub_threshold': 2, 'invert': False},
    }

    if mode not in default_modes:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(default_modes.keys())}.")

    if os.path.exists(param_file):
        with open(param_file, 'rb') as f:
            params = pickle.load(f)
        print(f"Loaded parameters from {param_file}.")
    else:
        params = default_modes[mode].copy()
        print(f"Using default parameters for mode '{mode}'.")

    if param_overrides:
        for key, val in param_overrides.items():
            if key in params:
                params[key] = val
                print(f"Override: {key} = {val}")
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")

    with open(param_file, 'wb') as f:
        pickle.dump(params, f)
        print(f"Parameters saved to {param_file}.")

    print("Final tracking parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    if os.path.exists(output_tiff):
        print(f"TIFF found at {output_tiff}. Loading...")
        stack_reader = lambda s, e: tiff.imread(output_tiff, key=range(s, e)).astype(np.float32)
        with tiff.TiffFile(output_tiff) as tif:
            n_frames = len(tif.pages)
    else:
        print(f"Reading video {input_media} into memory...")
        cap = cv2.VideoCapture(input_media)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
        cap.release()

        full_stack = np.stack(frames)
        n_frames = full_stack.shape[0]
        print(f"Loaded {n_frames} frames.")
        stack_reader = lambda s, e: full_stack[s:e]

    print("Preparing background subtraction...")
    if rolling_bg:
        print("Using rolling background subtraction.")
    else:
        bg_sample = stack_reader(0, min(100, n_frames))
        avg_frame = np.mean(bg_sample, axis=0)

    features = []
    for start in range(0, n_frames, params['chunk_size']):
        end = min(start + params['chunk_size'], n_frames)
        print(f"Processing frames {start} to {end - 1}...")

        chunk = stack_reader(start, end)
        for i, frame in enumerate(chunk, start=start):
            if rolling_bg:
                win_start = max(0, i - 5)
                win_end = min(n_frames, i + 5)
                local_bg = np.mean(stack_reader(win_start, win_end), axis=0)
                subtracted = frame - local_bg
            else:
                subtracted = frame - avg_frame

            if np.max(subtracted) == 0:
                continue

            f = tp.locate(subtracted, diameter=params['diameter'],
                          minmass=params['minmass'], invert=params['invert'])
            if f is not None and len(f) > 0:
                if rod_shaped:
                    f = merge_rods_by_shape(f, merge_dist=7, max_rod_length=30, max_aspect_ratio=4)
                f['frame'] = i
                features.append(f)

    if features:
        f_all = pd.concat(features, ignore_index=True)
        print(f"Detected {len(f_all)} features.")

        t = tp.link_df(f_all, search_range=params['search_range'], memory=params['memory'])
        t_filtered = tp.filter_stubs(t, threshold=params['stub_threshold'])

        print(f"Final trajectory count: {t_filtered['particle'].nunique()}")

        if show_plot:
            tp.plot_traj(t_filtered)
            plt.show()

        return t_filtered
    else:
        print("No features detected.")
        return None
