def audit_hybrid_nans(ds_hybrid):
    # ==============================================================================
    # MULTI-VARIABLE NAN AUDIT ON HYBRID XARRAY DATASET
    # ==============================================================================
    print("Status: Scanning ds_hybrid variables for missing values (NaN)...")

    # Variables to check
    variables_to_check = ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET']

    print("--- XARRAY VARIABLES INTEGRITY CHECK ---")
    for var in variables_to_check:
        if var in ds_hybrid:
            # .sum() without arguments collapses all dimensions (time, lat, lon) into a single total
            nan_count = int(ds_hybrid[var].isnull().sum())
            print(f"- Variable '{var}': {nan_count} total NaN values")
        else:
            print(f"- Warning: Variable '{var}' not found in the dataset!")

    # ==============================================================================
    # DIAGNOSTIC: ARE NANs SPATIAL (LAND-SEA MASK) OR TEMPORAL?
    # ==============================================================================
    print("Status: Diagnosing the root cause of the 76907 NaNs...")

    # Take a single day slice (the first day) and see how many NaNs it has spatially
    single_day_nans = ds_hybrid['MinTemp'].isel(time=0).isnull().sum().values

    # Take a single pixel slice (0,0) and see how many NaNs it has temporally
    single_pixel_nans = ds_hybrid['MinTemp'].isel(latitude=0, longitude=0).isnull().sum().values

    print(f"Log: NaNs in a single day across the spatial grid: {single_day_nans} / 187 pixels")
    print(f"Log: NaNs in a single pixel across the 2193 days:     {single_pixel_nans} / 2193 days")