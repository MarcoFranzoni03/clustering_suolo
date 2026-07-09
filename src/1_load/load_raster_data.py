import os
import numpy as np
import pandas as pd
import rioxarray

VARIABLES = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc", "ocd", "wv0010", "wv0033", "wv1500"]
DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]

def load_raw_soil_layers(country="italy", base_data_dir="../data"):
    """
    Loads spatial raster maps (.tif) downloaded from SoilGrids for a target country,
    flattens the 2D grids into 1D arrays, extracts spatial coordinates, and assembles
    the final raw Pandas DataFrames. (Safe closing version to prevent excepthook loops)
    """
    target_dir = os.path.join(base_data_dir, country.lower(), "soil_data/soilgrids/data")
    
    raw_features_dict = {}
    spatial_coords = None

    print(f"\n--- Loading All Raw Layers for {country.upper()} ---")

    for var in VARIABLES:
        for depth in DEPTHS:
            file_name = f"{var}_{depth}cm_mean.tif"
            file_path = os.path.join(target_dir, file_name)

            if os.path.exists(file_path):
                try:
                    # Usiamo il context manager 'with' per garantire la chiusura del file raster
                    with rioxarray.open_rasterio(file_path) as src:
                        da = src.squeeze()

                        if spatial_coords is None:
                            lon_2d, lat_2d = np.meshgrid(da.x.values, da.y.values)
                            spatial_coords = {
                                "lon": lon_2d.flatten().astype(np.float32),  # Ottimizziamo la memoria a float32
                                "lat": lat_2d.flatten().astype(np.float32)
                            }

                        feature_name = f"{var}_{depth}cm"
                        raw_features_dict[feature_name] = da.values.flatten()
                        
                    print(f"Successfully loaded: {file_name}")

                except Exception as e:
                    print(f"Warning: Failed to read {file_name}. Error: {e}")
            else:
                print(f"Skipping: {file_name} (File not found)")

    ocs_path = os.path.join(target_dir, "ocs_0-30cm_mean.tif")
    if os.path.exists(ocs_path):
        try:
            with rioxarray.open_rasterio(ocs_path) as src:
                da = src.squeeze()
                raw_features_dict["ocs_0-30cm"] = da.values.flatten()
            print("Successfully loaded: ocs_0-30cm_mean.tif")
        except Exception:
            pass

    print("\n--- Completed: All existing raw layers are in memory ---")
    
    master_raw_dict = {**spatial_coords, **raw_features_dict}
    df_raw = pd.DataFrame(master_raw_dict)

    print(f"\n==========================================")
    print(f"RAW DATAFRAME ASSEMBLED SUCCESSFULLY!")
    print(f"Total initial rows (including NaNs): {df_raw.shape[0]}")
    print(f"Total initial columns (coords + features): {df_raw.shape[1]}")
    print(f"==========================================")

    df_coords = df_raw[["lon", "lat"]].copy()
    df_features = df_raw.drop(columns=["lon", "lat"]).copy()

    print(f"Coordinates shape: {df_coords.shape} | Features shape: {df_features.shape}")
    
    return df_raw, df_coords, df_features
