import numpy as np
import pandas as pd

def verify_precipitation_injection(active_coords, df_piogge, ds_hybrid):
    # ==============================================================================
    # SANITY CHECK: VERIFYING PRECIPITATION INJECTION (STATION VS HYBRID GRID)
    # ==============================================================================
    print("Initiating cross-verification for precipitation data...")

    # 1. Pick a reference station from your available rain data columns
    sample_station = active_coords['CLEAN_NAME'].iloc[0]
    print(f"Log: Selected '{sample_station}' as the validation sample.")

    # 2. Extract the coordinates of this station to find the nearest grid pixel
    station_info = active_coords[active_coords['CLEAN_NAME'] == sample_station].iloc[0]
    st_lat, st_lon = station_info['LAT'], station_info['LONG']

    # 3. Extract the 1D rainfall time series from both sources
    original_station_rain = df_piogge[sample_station].values
    injected_grid_rain = ds_hybrid['Precipitation'].sel(latitude=st_lat, longitude=st_lon, method='nearest').values

    # 4. Mathematical validation: check if the two arrays are identical
    arrays_match = np.allclose(original_station_rain, injected_grid_rain, atol=1e-5)

    print("\n--- VALIDATION RESULTS ---")
    if arrays_match:
        print(f"SUCCESS: The rainfall series in ds_hybrid perfectly matches df_piogge for station {sample_station}!")
    else:
        print(f"WARNING: Mismatch detected! Max difference: {np.max(np.abs(original_station_rain - injected_grid_rain))} mm")

    # 5. Visual sample comparison (First 10 days with data)
    print(f"\n--- DATA COMPARISON SAMPLE (First 10 days for {sample_station}) ---")
    comparison_df = pd.DataFrame({
        'Original (df_piogge)': original_station_rain,
        'Injected (ds_hybrid)': injected_grid_rain
    })
    print(comparison_df.head(10))