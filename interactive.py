import cv2
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
from ipywidgets import interact, IntSlider, Checkbox
from .detection import merge_rods_by_shape

def interactive_detection_tuner(
    video_path,
    max_frames=100,
    minmass_range=(1, 50, 15),
    diameter_range=(3, 15, 7),
    merge_dist_range=(1, 20, 7)
):
    cap = cv2.VideoCapture(video_path)
    frames, frame_count = [], 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
        frame_count += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read from the video.")
    
    stack = np.stack(frames)
    avg_frame = np.mean(stack, axis=0)
    first_subtracted = stack[0] - avg_frame

    def show_detection(minmass, diameter, consolidate_rods, merge_dist):
        plt.figure(figsize=(8, 8))
        plt.imshow(first_subtracted, cmap='gray')
        detected = tp.locate(first_subtracted, diameter=diameter, minmass=minmass,
                             separation=5, invert=True)
        if detected is not None and len(detected) > 0:
            if consolidate_rods:
                detected = merge_rods_by_shape(detected, merge_dist)
            plt.scatter(detected['x'], detected['y'], s=50, c='none', edgecolors='lime', label='Detected')
            plt.legend()
        plt.axis('off')
        plt.title(f"minmass={minmass}, diameter={diameter}, consolidate_rods={consolidate_rods}")
        plt.show()

    interact(
        show_detection,
        minmass=IntSlider(*minmass_range),
        diameter=IntSlider(*diameter_range, step=2),
        consolidate_rods=Checkbox(value=True),
        merge_dist=IntSlider(*merge_dist_range)
    )
