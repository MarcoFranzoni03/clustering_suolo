import os
import pandas as pd
import xarray as xr

def load_and_prepare_weather():
    # ==============================================================================
    # 1. LOADING LAND-BASED WEATHER STATION DATA (RAW GITHUB URLS)
    # ==============================================================================
    # Verified raw links for station locations and rainfall timeseries
    URL_COORDINATE = "https://raw.githubusercontent.com/MarcoFranzoni03/clustering_suolo/refs/heads/main/meteo_data/coordinate_stazioni.csv"
    URL_PIOGGE = "https://raw.githubusercontent.com/MarcoFranzoni03/clustering_suolo/refs/heads/main/meteo_data/Stazioni_P_2007-01-01%2000_00_2026-02-13%2023_59.csv"

    print("Fetching CSV datasets from GitHub...")
    df_coordinate = pd.read_csv(URL_COORDINATE)
    df_piogge = pd.read_csv(URL_PIOGGE)
    print("CSV files successfully loaded into memory!")

    # ==============================================================================
    # 2. DOWNLOADING AND OPENING THE GRIDDED NETCDF (.NC) FILE
    # ==============================================================================
    # Media stream URL to download the binary NetCDF dataset directly into Colab
    URL_NETCDF = "https://github.com/MarcoFranzoni03/clustering_suolo/raw/main/meteo_data/era5_daily_aquacrop_inputs.nc"
    PATH_LOCAL_NETCDF = "era5_daily_aquacrop_inputs.nc"

    if not os.path.exists(PATH_LOCAL_NETCDF):
        print("\nDownloading gridded NetCDF dataset from GitHub...")
        # Using system wget to stream the binary file into the local workspace
        os.system(f"wget -q '{URL_NETCDF}' -O {PATH_LOCAL_NETCDF}")

    print("Parsing NetCDF matrix via xarray...")
    ds_era5 = xr.open_dataset(PATH_LOCAL_NETCDF)
    print("NetCDF dataset successfully initialized!")

    # ==============================================================================
    # 3. METADATA AND STRUCTURE INSPECTION
    # ==============================================================================
    print("\n--- METEOROLOGICAL INPUT AUDIT ---")
    print(f"Station Coordinates Matrix Shape: {df_coordinate.shape}")
    print(f"Precipitation Timeseries Shape:    {df_piogge.shape}")
    print("Extracted ERA5 NetCDF Variables:  ", list(ds_era5.data_vars))

    # ==============================================================================
    # RENAME AND FILTER VARIABLES INSIDE THE NETCDF DATASET
    # ==============================================================================
    print("Status: Renaming NetCDF variables to match AquaCrop strict naming conventions...")

    # Drop 't_mean' immediately as it is not utilized by the AquaCrop simulation engine
    if 't_mean' in ds_era5.data_vars:
        ds_era5 = ds_era5.drop_vars('t_mean')
        print("Log: Dropped unnecessary variable 't_mean'.")

    # Define the explicit translation dictionary for the remaining variables
    rename_dict = {
        't_min': 'MinTemp',
        't_max': 'MaxTemp',
        'et0': 'ReferenceET',
        'precip': 'Precipitation'
    }

    # Apply the renaming transformation across the xarray Dataset dimensions
    ds_era5 = ds_era5.rename(rename_dict)

    print("Success: NetCDF variables successfully renamed!")
    print("Current active variables in ds_era5:", list(ds_era5.data_vars))
    
    return df_coordinate, df_piogge, ds_era5