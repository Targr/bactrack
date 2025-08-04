import numpy as np


def build_trajectories(t_filtered, frame_interval=0.05):
    traj = []
    for particle, group in t_filtered.groupby('particle'):
        traj.append({
            'particle': particle,
            'x': group['x'].values,
            'y': group['y'].values,
            'time': (group['frame'].values - group['frame'].values[0]) * frame_interval,
            'dt': np.array([frame_interval])
        })
    return traj
