import pandas as pd
import numpy as np

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
    df1_rows = []
    df2_rows = []

    for traj_id, traj_i in enumerate(traj):
        x = np.array(traj_i['x'])
        y = np.array(traj_i['y'])
        time = np.array(traj_i['time'])
        dt = traj_i['dt'][0]

        row_summary = {
            'traj_id': traj_id,
            'duration of trajectory (sec)': time[-1] - time[0] if len(time) > 1 else 0
        }

        row_data = {
            'traj_id': [],
            'x': [],
            'y': [],
            'time (sec)': []
        }

        for xi, yi, ti in zip(x, y, time):
            row_data['traj_id'].append(traj_id)
            row_data['x'].append(xi)
            row_data['y'].append(yi)
            row_data['time (sec)'].append(ti)

        # --- Peritrichous ---
        if peri:
            if 'circling' in traj_i and isinstance(traj_i['circling'], np.ndarray):
                mask = ~traj_i['circling']
            else:
                mask = np.ones(len(x), dtype=bool)

            x_masked = x[mask]
            y_masked = y[mask]
            dx = np.diff(x_masked, prepend=np.nan)
            dy = np.diff(y_masked, prepend=np.nan)
            displacement = np.sqrt(dx**2 + dy**2)
            speed = (displacement * microns_per_pixel) * 20
            row_summary['speed (micron/sec)'] = np.nanmean(speed)
            row_summary['tumblebias'] = traj_i.get('tumblebias', np.nan)

        # --- Monotrichous ---
        if mono:
            dx = np.diff(x, prepend=np.nan)
            dy = np.diff(y, prepend=np.nan)
            displacement = np.sqrt(dx**2 + dy**2)
            speed = (displacement * microns_per_pixel) / dt
            row_summary['speed (micron/sec)'] = np.nanmean(speed)
            row_summary['reversal_count'] = traj_i.get('reversal_count', np.nan)
            row_summary['reversal_bias'] = traj_i.get('reversal_bias', np.nan)
            row_data['reversal'] = traj_i.get('reversal', np.zeros(len(x), dtype=int))

        # --- Twitch ---
        if twitch:
            twitch_mask = traj_i.get('twitch', np.zeros(len(x), dtype=int))
            count = traj_i.get('twitch_count', np.nan)
            frac = traj_i.get('twitch_fraction', np.nan)
            rate = count / row_summary['duration of trajectory (sec)'] if row_summary['duration of trajectory (sec)'] > 0 else np.nan
            row_summary.update({
                'twitch_jump_count': count,
                'twitch_jump_fraction': frac,
                'twitch_jump_rate (Hz)': rate
            })
            row_data['twitch_jump'] = twitch_mask

        # --- Biofilm ---
        if biofilm:
            net_disp = traj_i.get('biofilm_net_disp', np.nan)
            stationary = traj_i.get('biofilm_stationary', np.nan)
            row_summary['biofilm_net_displacement (µm)'] = net_disp * microns_per_pixel
            row_summary['biofilm_stationary_flag'] = stationary

        # --- Gliding ---
        if gliding:
            consistency = traj_i.get('gliding_consistency', np.nan)
            row_summary['gliding_consistency'] = consistency

        # --- Swarming ---
        if swarming:
            density = traj_i.get('swarm_density', np.nan)
            flag = traj_i.get('is_swarming', np.nan)
            row_summary['swarm_density'] = density
            row_summary['is_swarming'] = flag

        # --- Pili Retraction ---
        if pili_retracting:
            bursts = traj_i.get('pili_bursts', np.nan)
            fraction = traj_i.get('pili_fraction', np.nan)
            row_summary['pili_burst_count'] = bursts
            row_summary['pili_burst_fraction'] = fraction

        # Compile per-frame data
        df1_rows.extend([dict(zip(row_data, t)) for t in zip(*row_data.values())])
        df2_rows.append(row_summary)

    df1 = pd.DataFrame(df1_rows)
    df2 = pd.DataFrame(df2_rows)

    return df1, df2
