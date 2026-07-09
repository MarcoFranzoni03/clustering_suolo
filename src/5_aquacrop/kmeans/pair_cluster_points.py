import numpy as np
import pandas as pd

def pair_medoids_with_random_points(df_map, df_aquacrop_mappa, final_labels, medoids_coords, x_coord, y_coord):
    # ==============================================================================
    # DYNAMIC RECONSTRUCTION OF UNIQUE_POINTS FROM DF_MAP AND FINAL_LABELS
    # ==============================================================================
    print("Status: Reconstructing unique spatial grid points with their cluster labels...")
    
    # Set a deterministic seed for execution reproducibility across identical groups
    np.random.seed(42)

    # Create the unique points dataframe directly from your map dataset
    unique_points = pd.DataFrame({
        'lon': df_map[x_coord],
        'lat': df_map[y_coord],
        'Cluster_ID': final_labels
    }).drop_duplicates(subset=['lon', 'lat']).copy()

    # Initialize the distance column to store spatial offsets from the cluster center
    unique_points["dist_from_medoid"] = np.nan

    # Compute the Euclidean spatial distance for each point relative to its feature medoid
    for medoid in medoids_coords:
        c_id = medoid["cluster"]
        m_lon = medoid["lon"]
        m_lat = medoid["lat"]

        mask = unique_points["Cluster_ID"] == c_id
        if mask.any():
            unique_points.loc[mask, "dist_from_medoid"] = np.sqrt(
                (unique_points.loc[mask, "lon"] - m_lon) ** 2 +
                (unique_points.loc[mask, "lat"] - m_lat) ** 2
            )

    print(f"Status: Successfully mapped {len(unique_points)} unique coordinates using feature-space assignments.")

    # ==============================================================================
    # CORE PAIRING AND TARGETED PROFILE EXTRACTION LOOP
    # ==============================================================================
    cluster_pairs = {}

    print("Filtering and pairing medoids with representative cluster candidates...")

    for medoid in medoids_coords:
        c_id = medoid["cluster"]
        m_lon = medoid["lon"]
        m_lat = medoid["lat"]

        # Isolate all grid points belonging to the same management zone, excluding the medoid
        cluster_points = unique_points[
            (unique_points["Cluster_ID"] == c_id) &
            (unique_points["dist_from_medoid"] > 1e-5)
        ].copy()

        # Fallback to the full group if the distance filter strips all entries
        if cluster_points.empty:
            cluster_points = unique_points[
                unique_points["Cluster_ID"] == c_id
            ].copy()

        # Define a boundary window to extract a point that is representative but not adjacent
        d_min = cluster_points["dist_from_medoid"].quantile(0.25)
        d_max = cluster_points["dist_from_medoid"].quantile(0.50)

        # Filter candidate points located within the interquartile spatial envelope
        candidate_points = cluster_points[
            (cluster_points["dist_from_medoid"] >= d_min) &
            (cluster_points["dist_from_medoid"] <= d_max)
        ]

        # Fallback to the broad cluster population if the boundary slice is empty
        if candidate_points.empty:
            candidate_points = cluster_points

        # Draw exactly one random spatial point using a fixed state for consistency
        random_point = candidate_points.sample(
            n=1,
            random_state=7
        ).iloc[0]

        r_lon = random_point["lon"]
        r_lat = random_point["lat"]
        r_dist = random_point["dist_from_medoid"]

        # Extract the stacked, multi-layer soil profile for the central medoid location
        medoid_profile = (
            df_aquacrop_mappa[
                (np.isclose(df_aquacrop_mappa["lon"], m_lon, atol=1e-5)) &
                (np.isclose(df_aquacrop_mappa["lat"], m_lat, atol=1e-5))
            ]
            .sort_values("Layer")
        )

        # Extract the identical structural multi-layer soil profile for the paired random point
        random_profile = (
            df_aquacrop_mappa[
                (np.isclose(df_aquacrop_mappa["lon"], r_lon, atol=1e-5)) &
                (np.isclose(df_aquacrop_mappa["lat"], r_lat, atol=1e-5))
            ]
            .sort_values("Layer")
        )

        # Package paired coordinates and hydraulic layers into the comparative matrix
        cluster_pairs[c_id] = {
            "medoid": {
                "coords": (m_lon, m_lat),
                "profile": medoid_profile
            },
            "random_point": {
                "coords": (r_lon, r_lat),
                "profile": random_profile,
                "distance_from_medoid": r_dist,
                "cluster": c_id
            }
        }

        print(
            f"Cluster {c_id}: "
            f"Medoid=({m_lon:.6f}, {m_lat:.6f}) | "
            f"Random=({r_lon:.6f}, {r_lat:.6f}) | "
            f"Distance={r_dist:.5f}"
        )

    print("\n-> Structural pairing successfully completed for all management zones!")
    
    return cluster_pairs