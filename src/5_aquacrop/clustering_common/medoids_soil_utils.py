import numpy as np
import pandas as pd

def extract_medoids_soil_profiles(df_aquacrop_mappa, df_medoids):
    # ==============================================================================
    # EXTRACT FUNCTIONAL SOIL PROFILES FOR THE 4 BISECTING K-MEANS MEDOIDS
    # ==============================================================================
    print("Status: Building dynamic coordinates matrix from Bisecting K-Means medoids...")

    # Generates the coordinates dictionary automatically from df_medoids
    medoids_coords = [
        {"cluster": int(row['Soil_Cluster']), "lon": row['lon'], "lat": row['lat']}
        for _, row in df_medoids.iterrows()
    ]

    # Dictionary to store the multi-layer profile for each cluster
    soil_profiles = {}

    print("Extracting multi-layer soil profiles for AquaCrop simulation...")

    for medoid in medoids_coords:
        c_id = medoid["cluster"]

        # Filter the compiled map using a small spatial tolerance for float precision lookup
        profile = df_aquacrop_mappa[
            (np.isclose(df_aquacrop_mappa['lon'], medoid["lon"], atol=1e-5)) &
            (np.isclose(df_aquacrop_mappa['lat'], medoid["lat"], atol=1e-5))
        ].sort_values(by='Layer')

        # Store the 3-layer profile in the dictionary
        soil_profiles[c_id] = profile

        print(f"-> Cluster {c_id} profile extracted successfully ({len(profile)} layers found).")

    # Quick preview verification for all extracted profiles
    print("\n--- EXTRACTED HYDROLOGICAL PROFILES CHECK ---")
    display_cols = ['Layer', 'Horizon', 'PWP_vol_pct', 'FC_vol_pct', 'SAT_vol_pct', 'Ksat_mm_day']
    
    for c_id in sorted(soil_profiles.keys()):
        print(f"\n[Cluster {c_id} Medoid Soil Profile]")
        if not soil_profiles[c_id].empty:
            print(soil_profiles[c_id][display_cols].to_string(index=False))
        else:
            print(f"Warning: Profile for Cluster {c_id} is empty! Verify coordinate alignment in df_aquacrop_mappa.")
            
    return soil_profiles