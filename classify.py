from shapely.geometry import LineString
import pickle
import numpy as np

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
    """
    Annotates trajectory data with motility features.
    Supports multiple bacterial motility types.
    """

    if peri:
        for traj_i in traj:
            t = traj_i['time']
            dt = traj_i['dt']

            if len(t) < params['min_circle_dt'] / dt:
                continue

            x = np.array(traj_i['x']) - np.nanmean(traj_i['x'])
            y = np.array(traj_i['y']) - np.nanmean(traj_i['y'])
            coords = list(zip(x, y))

            intersections = []
            for i in range(len(coords) - 1):
                seg1 = LineString([coords[i], coords[i+1]])
                for j in range(i+2, len(coords) - 1):
                    seg2 = LineString([coords[j], coords[j+1]])
                    if seg1.crosses(seg2):
                        pt = seg1.intersection(seg2)
                        intersections.append((pt.x, pt.y))

            t1, t2 = [], []
            for px, py in intersections:
                distances = np.abs(x - px) + np.abs(y - py)
                idx1 = np.argmin(distances)
                t1.append(idx1)

                mask = np.ones_like(x, dtype=bool)
                r1, r2 = max(0, idx1 - 3), min(len(x), idx1 + 4)
                mask[r1:r2] = False

                x2, y2 = np.copy(x), np.copy(y)
                x2[~mask], y2[~mask] = np.inf, np.inf

                distances2 = np.abs(x2 - px) + np.abs(y2 - py)
                idx2 = np.argmin(distances2)
                t2.append(idx2)

            t1, t2 = np.array(t1), np.array(t2)
            loop_durations = (t2 - t1 + 1) * dt
            valid_loops = (loop_durations >= params['min_circle_dt']) & (loop_durations <= params['max_circle_dt'])
            t1, t2 = t1[valid_loops], t2[valid_loops]

            costs = np.empty(len(t1))
            for j in range(len(t1)):
                seg_x = x[t1[j]:t2[j]+1]
                seg_y = y[t1[j]:t2[j]+1]
                c = [np.nanmean(seg_x), np.nanmean(seg_y)]
                rs = np.sqrt((seg_x - c[0])**2 + (seg_y - c[1])**2)
                r = np.nanmean(rs)
                costs[j] = (1 / r) * np.max(np.abs(r - rs))

            circles = costs < params['circle_cost_threshold']
            circling_mask = np.zeros(len(x), dtype=bool)
            for j in range(len(t1)):
                if circles[j]:
                    circling_mask[t1[j]:t2[j]+1] = True

            frac_circling = np.sum(circling_mask) / len(x)
            traj_i['circling'] = circling_mask if frac_circling >= params['fracCircling_threshold'] else np.zeros(len(x), dtype=bool)
            traj_i['fracCircling'] = frac_circling if frac_circling >= params['fracCircling_threshold'] else 0.0
            traj_i['timeNotCircling'] = np.sum(~circling_mask) * dt

            dx, dy = np.gradient(x), np.gradient(y)
            angle = np.arctan2(dy, dx)
            dangle = np.abs(np.diff(angle))
            dangle = np.concatenate([[0], dangle])
            tumble_mask = (dangle > np.pi / 3) & ~circling_mask
            tumble_labels = np.zeros(len(x), dtype=int)
            tumble_labels[tumble_mask] = 1
            traj_i['tumble'] = tumble_labels

        for traj_i in traj:
            tumble = traj_i.get('tumble')
            if isinstance(tumble, np.ndarray):
                is_tumble = (tumble == 1)
                denom = len(tumble)
                traj_i['tumblebias'] = np.sum(is_tumble) / denom if denom > 0 else np.nan

    if mono:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])

            if len(x) < 3:
                traj_i['reversal'] = np.zeros(len(x), dtype=int)
                traj_i['reversal_count'] = 0
                traj_i['reversal_bias'] = np.nan
                continue

            v1 = np.stack([np.diff(x[:-1]), np.diff(y[:-1])], axis=1)
            v2 = np.stack([np.diff(x[1:]), np.diff(y[1:])], axis=1)

            dot_products = np.einsum('ij,ij->i', v1, v2)
            norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cosine_angles = np.clip(dot_products / (norms + 1e-9), -1.0, 1.0)
            angle_changes = np.arccos(cosine_angles)

            reversal_mask = np.zeros(len(x), dtype=int)
            reversal_mask[2:] = angle_changes > reversal_angle_threshold

            traj_i['reversal'] = reversal_mask
            traj_i['reversal_count'] = np.sum(reversal_mask)
            traj_i['reversal_bias'] = traj_i['reversal_count'] / len(reversal_mask) if len(reversal_mask) > 0 else np.nan

    if twitch:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])

            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            displacement = np.sqrt(dx**2 + dy**2)

            twitch_mask = displacement > twitch_jump_threshold
            traj_i['twitch'] = twitch_mask.astype(int)
            traj_i['twitch_count'] = int(np.sum(twitch_mask))
            traj_i['twitch_fraction'] = float(np.mean(twitch_mask)) if len(twitch_mask) > 0 else np.nan

    if biofilm:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])

            net_disp = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            traj_i['biofilm_net_disp'] = net_disp
            traj_i['biofilm_stationary'] = int(net_disp < biofilm_disp_threshold)

    if gliding:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])
            dx = np.diff(x)
            dy = np.diff(y)
            angle = np.arctan2(dy, dx)
            dangle = np.abs(np.diff(angle))
            smooth_motion = dangle < glide_angle_threshold
            traj_i['gliding_consistency'] = float(np.mean(smooth_motion)) if len(smooth_motion) > 0 else np.nan

    if swarming:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])
            
            radius = 10  # pixels
            coords = np.stack([x, y], axis=1)
            density = []
            for i in range(len(coords)):
                dists = np.linalg.norm(coords - coords[i], axis=1)
                count = np.sum((dists < radius) & (dists > 0))
                density.append(count)
            traj_i['swarm_density'] = float(np.mean(density))
            traj_i['is_swarming'] = int(np.mean(density) > swarm_density_threshold)

    if pili_retracting:
        for traj_i in traj:
            x = np.array(traj_i['x'])
            y = np.array(traj_i['y'])
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            displacement = np.sqrt(dx**2 + dy**2)
            bursts = displacement > pili_retract_burst_threshold
            traj_i['pili_bursts'] = int(np.sum(bursts))
            traj_i['pili_fraction'] = float(np.mean(bursts)) if len(bursts) > 0 else np.nan

    return traj
