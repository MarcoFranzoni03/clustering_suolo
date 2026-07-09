def map_spatial_nan_distribution(ds_hybrid):
    # ==============================================================================
    # VISUAL DIAGNOSTIC: MAP THE SPATIAL LOCATION OF NANs
    # ==============================================================================
    print("Status: Mapping the spatial distribution of permanent NaNs...")

    # Count NaNs along the time dimension for each pixel (on MinTemp)
    spatial_nan_map = ds_hybrid['MinTemp'].isnull().sum(dim='time').values

    print("--- GEOGRAPHIC GRID OF NAN COUNT PER PIXEL (11 rows x 17 cols) ---")
    # Print the matrix as text to see the shape of the missing data
    for row in spatial_nan_map:
        # Convert numbers to strings for clean printing (2192 means permanently empty)
        print(" ".join(f"[{str(val).zfill(4)}]" for val in row))