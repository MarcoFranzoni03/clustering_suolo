import pandas as pd
import xarray as xr

def export_medoid_climate_to_aquacrop(ds_hybrid, medoid_lat=41.798625, medoid_lon=15.040663, output_prefix="KMeans_Medoide_C0_24_25"):
    # ==============================================================================
    # SPATIAL EXTRACTION AND SEASONAL TIME-SLICING FOR MEDOID
    # ==============================================================================
    print(f"Status: Extracting weather data grid slice for Medoid ({medoid_lat}, {medoid_lon})...")

    ds_slice = ds_hybrid.sel(
        latitude=medoid_lat,
        longitude=medoid_lon,
        method='nearest'
    ).sel(
        time=slice('2024-11-15', '2025-06-30')
    )

    # ==============================================================================
    # CONVERSION TO DATAFRAME AND MISSING VALUE CLEANING
    # ==============================================================================
    df_slice = ds_slice.to_dataframe().reset_index()
    df_slice = df_slice.dropna(subset=['Precipitation', 'MinTemp', 'MaxTemp', 'ReferenceET']).reset_index(drop=True)

    # ==============================================================================
    # EXTRACTION OF START DATE METADATA FOR STANDARD FAO CLIMATE HEADERS
    # ==============================================================================
    df_slice['time'] = pd.to_datetime(df_slice['time'])
    start_day = int(df_slice['time'].min().day)
    start_month = int(df_slice['time'].min().month)
    start_year = int(df_slice['time'].min().year)

    print(f"Extraction completed! Total daily records to export: {len(df_slice)}")

    # ==============================================================================
    # GENERATING STANDARD AQUACROP CLIMATE INPUT FILES (CRASH-RESISTANT FORMAT)
    # ==============================================================================

    # 1. Rainfall File Generation (.PLU)
    plu_filename = f"{output_prefix}.PLU"
    with open(plu_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Medoid Rainfall - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, p in enumerate(df_slice['Precipitation']):
            # Suppress trailing newline only on the very last row to avoid engine parsing errors
            end_char = "" if i == len(df_slice) - 1 else "\n"
            f.write(f"{p:8.1f}{end_char}")

    # 2. Temperature File Generation (.TMP)
    tmp_filename = f"{output_prefix}.TMP"
    with open(tmp_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Medoid Temperatures - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, (tmin, tmax) in enumerate(zip(df_slice['MinTemp'], df_slice['MaxTemp'])):
            end_char = "" if i == len(df_slice) - 1 else "\n"
            f.write(f"{tmin:8.1f}{tmax:8.1f}{end_char}")

    # 3. Reference Evapotranspiration File Generation (.ETo)
    eto_filename = f"{output_prefix}.ETo"
    with open(eto_filename, "w", newline='\r\n') as f:
        f.write(f"Capitanata Medoid Reference ET - Season 2024-2025\n")
        f.write("1  : Daily records\n")
        f.write(f"{start_day}  : First day of record\n")
        f.write(f"{start_month}  : First month of record\n")
        f.write(f"{start_year}  : First year of record\n")
        f.write("========================\n")
        for i, et in enumerate(df_slice['ReferenceET']):
            end_char = "" if i == len(df_slice) - 1 else "\n"
            # ReferenceET values are written directly using the proper daily scale format
            f.write(f"{et:8.1f}{end_char}")

    print(f"Success: The {plu_filename}, {tmp_filename}, and {eto_filename} files have been successfully generated!")
    
    return df_slice