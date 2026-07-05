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
    the final raw Pandas DataFrames.
    """
    # Build relative local path instead of drive mount
    target_dir = os.path.join(base_data_dir, country.lower(), "soil_data/soilgrids/data")
    
    raw_features_dict = {}
    spatial_coords = None

    print(f"\n--- Loading All Raw Layers for {country.upper()} ---")

    # Loop through all combinations of variables and depths
    for var in VARIABLES:
        for depth in DEPTHS:
            file_name = f"{var}_{depth}cm_mean.tif"
            file_path = os.path.join(target_dir, file_name)

            if os.path.exists(file_path):
                try:
                    # Open the raster map using rioxarray and squeeze the band dimension
                    da = rioxarray.open_rasterio(file_path).squeeze()

                    # Dynamically extract spatial coordinates once from the first available file
                    if spatial_coords is None:
                        lon_2d, lat_2d = np.meshgrid(da.x.values, da.y.values)
                        spatial_coords = {
                            "lon": lon_2d.flatten(),
                            "lat": lat_2d.flatten()
                        }

                    # Flatten the 2D grid into a 1D array and save it under a unique feature name
                    feature_name = f"{var}_{depth}cm"
                    raw_features_dict[feature_name] = da.values.flatten()
                    da.close()
                    print(f"Successfully loaded: {file_name}")

                except Exception as e:
                    print(f"Warning: Failed to read {file_name}. Error: {e}")
            else:
                print(f"Skipping: {file_name} (File not found)")

    # Optional: Try to load the extra OCS variable (0-30cm only)
    ocs_path = os.path.join(target_dir, "ocs_0-30cm_mean.tif")
    if os.path.exists(ocs_path):
        try:
            da = rioxarray.open_rasterio(ocs_path).squeeze()
            raw_features_dict["ocs_0-30cm"] = da.values.flatten()
            da.close()
            print("Successfully loaded: ocs_0-30cm_mean.tif")
        except Exception:
            pass

    print("\n--- Completed: All existing raw layers are in memory ---")
    
    # --- ASSEMBLE DATAFRAME (Your two Colab cells combined) ---
    # 1. Combine spatial data and all individual raw depth layers into one dictionary
    master_raw_dict = {**spatial_coords, **raw_features_dict}

    # 2. Build the final Pandas DataFrame containing everything
    df_raw = pd.DataFrame(master_raw_dict)

    print(f"\n==========================================")
    print(f"RAW DATAFRAME ASSEMBLED SUCCESSFULLY!")
    print(f"Total initial rows (including NaNs): {df_raw.shape[0]}")
    print(f"Total initial columns (coords + features): {df_raw.shape[1]}")
    print(f"==========================================")

    # Separate spatial coordinates from chemical and physical soil features
    df_coords = df_raw[["lon", "lat"]]
    df_features = df_raw.drop(columns=["lon", "lat"])

    print(f"Coordinates shape: {df_coords.shape} | Features shape: {df_features.shape}")
    
    return df_raw, df_coords, df_features