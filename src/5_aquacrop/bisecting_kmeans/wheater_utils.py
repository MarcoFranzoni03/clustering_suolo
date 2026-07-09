import pandas as pd
import xarray as xr

def export_bisect_medoid_climate(ds_hybrid, medoid_lat=41.798625, medoid_lon=15.040663, output_prefix="Bisect_Medoide_C0_24_25"):
    # ==============================================================================
    # BLOCCO 1: CLIMATE EXTRACTION AND BENCHMARK CONVERSION
    # ==============================================================================
    print(f"Status: Isolating meteorological time-series for Medoid ({medoid_lat}, {medoid_lon})...")

    # Isolate the daily meteorological time-series from the xarray Dataset using nearest-neighbor lookup
    climate_ts = ds_hybrid.sel(
        longitude=medoid_lon,
        latitude=medoid_lat,
        method='nearest'
    )

    # Convert the geogrid slice into a flat pandas DataFrame for seamless text formatting
    df_climate = climate_ts.to_dataframe().reset_index()
    print("Status: Core benchmark climate data successfully isolated and converted.")

    # ==============================================================================
    # BLOCCO 2: SPATIAL EXTRACTION, SEASONAL TIME-SLICING AND FILE GENERATION
    # ==============================================================================
    print("Extracting weather data grid slice for Medoid 0 and trimming the cropping season...")

    ds_slice = ds_hybrid.sel(
        latitude=medoid_lat,
        longitude=medoid_lon,
        method='nearest'
    ).sel(
        time=slice('2024-11-15', '2025-06-30')
    )

    # Conversion to DataFrame and missing value cleaning
    df_slice_m0 = ds_slice.to_dataframe().reset_index()
    df_slice_m0 = df_slice_m0.dropna(subset=['Precipitation', 'MinTemp', 'MaxTemp', 'ReferenceET']).reset_index(drop=True)

    # Extraction of start date metadata for standard FAO climate headers
    df_slice_m0['time'] = pd.to_datetime(df_slice_m0['time'])
    start_day = int(df_slice_m0['time'].min().day)
    start_month = int(df_slice_m0['time'].min().month)
    start_year = int(df_slice_m0['time'].min().year)

    print(f"Extraction completed! Total daily records to export: {len(df_slice_m0)}")

    # --- GENERATING STANDARD AQUACROP CLIMATE INPUT FILES ---

    # 1. Rainfall File Generation (.PLU)
    plu_filename = f"{output_prefix}.PLU"
    with open(plu_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Cluster 0 Medoid Rainfall - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, p in enumerate(df_slice_m0['Precipitation']):
            end_char = "" if i == len(df_slice_m0) - 1 else "\n"
            f.write(f"{p:8.1f}{end_char}")

    # 2. Temperature File Generation (.TMP)
    tmp_filename = f"{output_prefix}.TMP"
    with open(tmp_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Cluster 0 Medoid Temperatures - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, (tmin, tmax) in enumerate(zip(df_slice_m0['MinTemp'], df_slice_m0['MaxTemp'])):
            end_char = "" if i == len(df_slice_m0) - 1 else "\n"
            f.write(f"{tmin:8.1f}{tmax:8.1f}{end_char}")

    # 3. Reference Evapotranspiration File Generation (.ETo)
    eto_filename = f"{output_prefix}.ETo"
    with open(eto_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Cluster 0 Medoid Reference ET - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, et in enumerate(df_slice_m0['ReferenceET']):
            end_char = "" if i == len(df_slice_m0) - 1 else "\n"
            f.write(f"{et:8.1f}{end_char}")

    print(f"-> The {plu_filename}, {tmp_filename}, and {eto_filename} files have been successfully generated!")
    
    return df_climate, df_slice_m0