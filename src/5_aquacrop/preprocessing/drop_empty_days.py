import numpy as np
import pandas as pd

def remove_empty_initial_date(ds_hybrid, df_piogge):
    # ==============================================================================
    # AUDIT: WHICH DAYS ARE COMPLETELY NAN IN ERA5?
    # ==============================================================================
    print("Status: Locating the exact dates of the empty ERA5 layers...")

    # Calculate the number of NaNs for each day (collapsing latitude and longitude)
    nans_per_day = ds_hybrid['MinTemp'].isnull().sum(dim=['latitude', 'longitude']).values
    times = ds_hybrid.time.values

    # Find indexes where the whole grid is NaN (187 pixels missing)
    empty_day_indexes = np.where(nans_per_day == 187)[0]

    print(f"Log: Total completely empty days found in ERA5: {len(empty_day_indexes)}")

    if len(empty_day_indexes) > 0:
        print(f"- First empty date: {times[empty_day_indexes[0]]}")
        print(f"- Last empty date:  {times[empty_day_indexes[-1]]}")

        # Check a valid day to see if it's clean
        valid_day_indexes = np.where(nans_per_day == 0)[0]
        if len(valid_day_indexes) > 0:
            print(f"- First VALID date found: {times[valid_day_indexes[0]]}")

    # ==============================================================================
    # DROP THE EMPTY FIRST DAY (2019-12-31) FROM BOTH DATASETS
    # ==============================================================================
    print("Status: Dropping 2019-12-31 to ensure perfect data integrity...")

    # 1. Drop from the xarray Dataset (ds_hybrid)
    # We select only times that are strictly greater than 2019-12-31
    ds_hybrid = ds_hybrid.sel(time=ds_hybrid.time > np.datetime64('2019-12-31'))

    # 2. Drop from the pandas DataFrame (df_piogge)
    df_piogge = df_piogge[df_piogge['Data'] > '2019-12-31'].reset_index(drop=True)

    print(f"Log: New ds_hybrid time steps (days): {len(ds_hybrid.time)}")
    print(f"Log: New df_piogge rows (days):       {len(df_piogge)}")

    # 3. Quick re-check of total NaNs on ds_hybrid after dropping the day
    print("\n--- UPDATED RE-CHECK OF TOTAL NaNs ---")
    for var in ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET']:
        nan_count = int(ds_hybrid[var].isnull().sum())
        print(f"- Variable '{var}': {nan_count} remaining NaN values")
        
    return ds_hybrid, df_piogge