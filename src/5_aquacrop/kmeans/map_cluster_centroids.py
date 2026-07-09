import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_geographic_centroids(kmeans_final, df_coords_clean, final_labels, X_final_preprocessed):
    # ==============================================================================
    # CALCULATE K-MEANS CENTROIDS MAPPED TO GEOGRAPHY
    # ==============================================================================
    print("Status: Mapping KMeans cluster centers back to geographic coordinates...")

    # 1. Get the raw cluster centers from the trained KMeans model
    # (These centers live in the PCA compressed space, matching X_final_preprocessed)
    pca_centers = kmeans_final.cluster_centers_

    # 2. Sync coordinates and labels
    df_map = df_coords_clean.copy()
    df_map['Soil_Cluster'] = final_labels

    x_coord = df_map.columns[0]
    y_coord = df_map.columns[1]

    true_centroids_list = []

    # 3. For each cluster, find the real point closest to the mathematical PCA center
    for c_id in sorted(np.unique(final_labels)):
        # Isolate row indexes belonging to this specific cluster
        cluster_indices = np.where(final_labels == c_id)[0]

        # Extract the PCA preprocessed features for this cluster
        cluster_pca_features = X_final_preprocessed[cluster_indices]

        # Get the specific PCA center vector for this cluster
        center_vector = pca_centers[c_id].reshape(1, -1)

        # Compute distances in the PCA space to find the truest mathematical match
        distances = cdist(cluster_pca_features, center_vector, metric='euclidean').flatten()
        closest_idx = cluster_indices[distances.argmin()]

        # Extract the real geographic coordinates of that closest point
        real_sample = df_map.iloc[closest_idx]

        true_centroids_list.append({
            'Soil_Cluster': c_id,
            x_coord: real_sample[x_coord],
            y_coord: real_sample[y_coord]
        })

    # 4. Save into the final dataframe expected by your visualization/downstream cells
    cluster_centroids = pd.DataFrame(true_centroids_list)

    print("\n--- TRUE MODEL CENTROIDS FOR THE 4 CLUSTERS ---")
    print(cluster_centroids.to_string(index=False))
    
    return cluster_centroids