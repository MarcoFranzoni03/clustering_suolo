import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth
    using the Haversine formula (returns distance in kilometers).
    """
    R = 6371.0 # Earth's radius in km

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def inject_nearest_station_rain(ds_era5, df_coordinate, df_piogge):
    # ==============================================================================
    # GEOSPATIAL NEAREST-STATION INJECTION INTO XARRAY DATASET
    # ==============================================================================
    print("Status: Initiating Nearest-Station spatial injection into ds_era5...")

    # 1. Create a clean copy of ds_era5 to avoid modifying the original dataset in-place
    ds_hybrid = ds_era5.copy()

    # 2. Extract dimensions from xarray
    lat_array = ds_hybrid.latitude.values
    lon_array = ds_hybrid.longitude.values
    time_len = len(ds_hybrid.time)

    # 3. Create an empty numpy array with the correct shape (time, lat, lon) to host the new rain data
    new_precip_matrix = np.zeros((time_len, len(lat_array), len(lon_array)))

    # 4. Filter df_coordinate to keep only the stations currently present in df_piogge columns
    active_coords = df_coordinate[df_coordinate['CLEAN_NAME'].isin(df_piogge.columns)].copy()

    print(f"Log: Mapping grid pixels against {len(active_coords)} available rain stations...")

    # 5. Nested loops to iterate through every single pixel of the 5x5 grid
    for i, lat_nc in enumerate(lat_array):
        for j, lon_nc in enumerate(lon_array):

            min_dist = float('inf')
            closest_station_name = None

            # Calculate distance from this NC pixel to every real weather station
            for _, station_row in active_coords.iterrows():
                dist = haversine_distance(lat_nc, lon_nc, station_row['LAT'], station_row['LONG'])
                if dist < min_dist:
                    min_dist = dist
                    closest_station_name = station_row['CLEAN_NAME']

            # Extract the entire 2193-day rainfall series for the closest station
            # We skip the 'Data' column using .values to get the raw numpy array
            station_rainfall_series = df_piogge[closest_station_name].values

            # Inject the series into the temporary matrix at the specific grid intersection
            new_precip_matrix[:, i, j] = station_rainfall_series

    # 6. Overwrite the 'Precipitation' variable in our hybrid xarray dataset
    # We explicitly declare the dimensions to match xarray standards
    ds_hybrid['Precipitation'] = (('time', 'latitude', 'longitude'), new_precip_matrix)

    print("\nSuccess: Real-world precipitation data successfully injected into the xarray grid!")
    print(f"Log: Hybrid Dataset variable dimensions for Precipitation: {ds_hybrid['Precipitation'].shape}")
    
    return ds_hybrid