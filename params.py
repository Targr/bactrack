import pickle

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

def load_or_create_params(param_file='params.pkl'):
    try:
        with open(param_file, 'rb') as f:
            params = pickle.load(f)
        print(f"Loaded parameters from {param_file}")
    except FileNotFoundError:
        with open(param_file, 'wb') as f:
            pickle.dump(default_params, f)
        params = default_params
        print(f"Created default parameter file: {param_file}")
    return params
