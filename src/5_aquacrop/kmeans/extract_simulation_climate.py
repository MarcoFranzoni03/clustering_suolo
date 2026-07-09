import pandas as pd

def isolate_simulation_climate(ds_hybrid, medoid_lon=15.040663, medoid_lat=41.798625):
    # ==============================================================================
    # 1. CROP DATASET WITH WARMUP PERIOD
    # ==============================================================================
    # Crop the dataset giving a 15-day warmup period before planting
    ds_hybrid = ds_hybrid.sel(time=slice('2024-11-01', '2025-06-30'))

    # Log the new dataset temporal boundaries to verify the operation
    print(f"Dataset successfully cropped!")
    print(f"New Start Date: {ds_hybrid['time'].min().values}")
    print(f"New End Date: {ds_hybrid['time'].max().values}")
    print(f"Total simulated days: {len(ds_hybrid['time'])}")

    # ==============================================================================
    # 2. CLIMATE EXTRACTION FOR UNIFORM BENCHMARK SIMULATION
    # ==============================================================================
    print("\n--- CLIMATE EXTRACTION FOR UNIFORM BENCHMARK SIMULATION ---")
    print(f"Log: Isolating meteorological time-series for Medoid ({medoid_lat}, {medoid_lon})...")

    # Isolate the daily meteorological time-series from the xarray Dataset using nearest-neighbor lookup
    climate_ts = ds_hybrid.sel(
        longitude=medoid_lon,
        latitude=medoid_lat,
        method='nearest'
    )

    # Convert the geogrid slice into a flat pandas DataFrame for seamless text formatting and file generation
    df_climate = climate_ts.to_dataframe().reset_index()

    print("Status: Core benchmark climate data successfully isolated and converted.")
    
    return df_climate