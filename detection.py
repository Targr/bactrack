import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def merge_rods_by_shape(df, merge_dist=10, max_rod_length=30, max_aspect_ratio=5):
    coords = df[['x', 'y']].values
    if len(coords) == 0:
        return df

    db = DBSCAN(eps=merge_dist, min_samples=1).fit(coords)
    df['cluster'] = db.labels_
    merged_rows = []

    for label in np.unique(db.labels_):
        cluster_pts = coords[db.labels_ == label]
        if len(cluster_pts) == 1:
            merged_rows.append({'x': cluster_pts[0][0], 'y': cluster_pts[0][1]})
            continue

        pca = PCA(n_components=2)
        pca.fit(cluster_pts)
        length = np.sqrt(pca.explained_variance_[0])
        width = np.sqrt(pca.explained_variance_[1])
        aspect = length / (width + 1e-6)

        if aspect > max_aspect_ratio and length * 2 < max_rod_length:
            merged_x, merged_y = np.mean(cluster_pts, axis=0)
        else:
            for pt in cluster_pts:
                merged_rows.append({'x': pt[0], 'y': pt[1]})
            continue

        merged_rows.append({'x': merged_x, 'y': merged_y})

    return pd.DataFrame(merged_rows)
