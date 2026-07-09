import pandas as pd

def audit_bisect_random_points(cluster_pairs):
    # ==============================================================================
    # BLOCCO 1: PRINT AND VERIFY ASSIGNED CLUSTER IDS FOR EACH RANDOM POINT
    # ==============================================================================
    print("Verifying the assigned Cluster ID for each sampled random point:\n")

    for c_id in sorted(cluster_pairs.keys()):
        # Extract coordinates and the verified cluster attribute from the dictionary
        coords = cluster_pairs[c_id]["random_point"]["coords"]
        assigned_cluster = cluster_pairs[c_id]["random_point"]["cluster"]

        print(f"Random Point at Longitude: {coords[0]:.6f}, Latitude: {coords[1]:.6f} -> Assigned Cluster ID: {assigned_cluster}")

    # ==============================================================================
    # BLOCCO 2: PRINT HYDRAULIC PARAMETERS FOR THE 4 RANDOM PAIRED POINTS
    # ==============================================================================
    print("\nExtracting and displaying hydraulic soil profiles for the 4 random paired points...")

    display_cols = ['Layer', 'Horizon', 'PWP_vol_pct', 'FC_vol_pct', 'SAT_vol_pct', 'Ksat_mm_day']

    # Iterate through the generated cluster pairs to print the dynamic target variables
    for c_id in sorted(cluster_pairs.keys()):
        print(f"\n--- RANDOM POINT HYDRAULIC PROFILE: CLUSTER {c_id} ---")

        # Retrieve the pre-extracted dynamic multi-layer profile from the database matrix
        random_profile = cluster_pairs[c_id]["random_point"]["profile"]

        # Isolate and print the specific AquaCrop input columns
        if not random_profile.empty:
            print(random_profile[display_cols].to_string(index=False))
        else:
            print(f"Warning: Profile for random paired point in Cluster {c_id} is empty!")