import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_feature_space_medoids(X_final_preprocessed, final_labels, df_map, x_coord, y_coord):
    # ==============================================================================
    # COMPUTE TRUE SOIL MEDOIDS USING FEATURE SPACE (NOT GEOGRAPHY)
    # ==============================================================================
    print("Status: Calculating mathematically rigorous soil medoids from feature space...")

    real_soil_medoids = []

    # Loop through each of the 4 clusters based on final_labels
    for c_id in sorted(np.unique(final_labels)):
        # 1. Isolate row indexes belonging to the current cluster
        cluster_indices = np.where(final_labels == c_id)[0]

        # 2. Extract the preprocessed features for this cluster
        cluster_features = X_final_preprocessed[cluster_indices]

        # 3. Calculate the theoretical center in the FEATURE space
        feature_centroid = cluster_features.mean(axis=0).reshape(1, -1)

        # 4. Find which real sample is closest to this feature center
        distances = cdist(cluster_features, feature_centroid, metric='euclidean').flatten()
        closest_relative_idx = distances.argmin()

        # 5. Map back to the absolute index of df_map
        absolute_idx = cluster_indices[closest_relative_idx]
        real_sample = df_map.iloc[absolute_idx]

        real_soil_medoids.append({
            'Soil_Cluster': c_id,
            'lon': real_sample[x_coord],
            'lat': real_sample[y_coord]
        })

    # Convert to DataFrame
    df_medoids = pd.DataFrame(real_soil_medoids)

    print("\n--- SOIL MEDOIDS FOR AQUACROP ---")
    print(df_medoids.to_string(index=False))
    
    return df_medoids