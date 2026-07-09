import numpy as np
import pandas as pd

def extract_bisect_cluster2_control_data(unique_points, df_aquacrop_mappa):
    # ==============================================================================
    # BLOCCO 1: REGIONAL DISTANCE CONTROL POINTS (CLUSTER 2)
    # ==============================================================================
    # Isolate and sort points by distance from the medoid for Cluster 2
    c2_points_sorted = unique_points[
        (unique_points["Cluster_ID"] == 2) &
        (unique_points["dist_from_medoid"] > 1e-5)
    ].sort_values(by="dist_from_medoid")

    # Extract exactly 2 random points from the intermediate tier (index 1000 to 2000) using the updated seed
    df_c2_samples = c2_points_sorted.iloc[1000:2000].sample(n=2, random_state=1)

    print("--- 2 RANDOM POINTS FOR CLUSTER 2 (BISECT) ---")
    for idx, row in df_c2_samples.iterrows():
        print(f"Point {idx} -> Lon: {row['lon']:.6f} | Lat: {row['lat']:.6f} | Spatial Distance: {row['dist_from_medoid']:.5f}")
    
    # ==============================================================================
    # BLOCCO 2: PRINT HYDRAULIC PARAMETERS FOR THE 2 CLUSTER 2 CONTROL POINTS
    # ==============================================================================
    print("\nExtracting and displaying hydraulic soil profiles for the 2 Cluster 2 control points...\n")

    display_cols = ['Layer', 'Horizon', 'PWP_vol_pct', 'FC_vol_pct', 'SAT_vol_pct', 'Ksat_mm_day']

    # Iterating through the 2 sampled points of Cluster 2
    for idx, row in df_c2_samples.iterrows():
        print(f"--- CLUSTER 2 CONTROL POINT (Index: {idx}) ---")
        print(f"Location: Longitude = {row['lon']:.6f}, Latitude = {row['lat']:.6f}")
        print(f"Spatial Distance from Medoid: {row['dist_from_medoid']:.5f}")

        # Filter the primary database using exact coordinate lookup matching float precision
        sampled_profile = df_aquacrop_mappa[
            (np.isclose(df_aquacrop_mappa["lon"], row["lon"], atol=1e-5)) &
            (np.isclose(df_aquacrop_mappa["lat"], row["lat"], atol=1e-5))
        ].sort_values(by="Layer")

        # Display the specific hydraulic columns required by AquaCrop
        if not sampled_profile.empty:
            print(sampled_profile[display_cols].to_string(index=False))
        else:
            print(f"Warning: Profile for control point index {idx} is empty!")
            
        print("-" * 80)
        
    return df_c2_samples