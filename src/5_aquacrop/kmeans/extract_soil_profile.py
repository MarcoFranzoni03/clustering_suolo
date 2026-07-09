import numpy as np
import pandas as pd

def extract_medoids_soil_profiles(df_aquacrop_mappa):
    # ==============================================================================
    # 1. EXTRACT FUNCTIONAL SOIL PROFILES FOR THE 4 MEDOIDS
    # ==============================================================================
    # Define the coordinates of your mathematically computed medoids
    medoids_coords = [
        {"cluster": 0, "lon": 15.040663, "lat": 41.798625},
        {"cluster": 1, "lon": 15.627913, "lat": 41.791875},
        {"cluster": 2, "lon": 14.703163, "lat": 41.254125},
        {"cluster": 3, "lon": 15.090163, "lat": 41.287875}
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

    # ==============================================================================
    # 2. PRINT PROFILES SUMMARY DETAILED BY LAYER
    # ==============================================================================
    print("\n--- DETAILED HYDRAULIC PROFILES FOR ALL MEDOIDS ---")
    display_cols = ['Layer', 'Horizon', 'PWP_vol_pct', 'FC_vol_pct', 'SAT_vol_pct', 'Ksat_mm_day']
    
    for c_id in sorted(soil_profiles.keys()):
        print(f"\n==================== PROFILE CLUSTER {c_id} ====================")
        if not soil_profiles[c_id].empty:
            print(soil_profiles[c_id][display_cols].to_string(index=False))
        else:
            print(f"Warning: Profile for Cluster {c_id} is empty! Check coordinate alignment.")
            
    return soil_profiles