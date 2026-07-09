import pandas as pd

def audit_and_align_timelines(ds_era5, df_piogge):
    # ==============================================================================
    # TIME DIMENSION INTEGRITY AUDIT (MISSING DAYS DETECTION)
    # ==============================================================================
    print("Status: Checking NetCDF time dimension for missing consecutive days...")

    # Extract the unique time points from the xarray dataset and convert to Pandas DatetimeIndex
    nc_dates = pd.to_datetime(ds_era5['time'].values)

    # Define the theoretical perfect timeline (from the absolute min date to max date, step=1 day)
    perfect_timeline = pd.date_range(start=nc_dates.min(), end=nc_dates.max(), freq='D')

    # Identify missing dates by calculating the set difference
    missing_dates = perfect_timeline.difference(nc_dates)

    # Print structural diagnostics
    print(f"Log: Dataset timeline spans from {nc_dates.min().strftime('%Y-%m-%d')} to {nc_dates.max().strftime('%Y-%m-%d')}.")
    print(f"Log: Expected total days: {len(perfect_timeline)} | Actual recorded days: {len(nc_dates)}.")

    if len(missing_dates) == 0:
        print("Success: No missing days detected. The NetCDF time dimension is perfectly continuous.")
    else:
        print(f"Warning: Detected {len(missing_dates)} missing days within the time series matrix!")
        print("Missing dates index:", missing_dates.strftime('%Y-%m-%d').tolist())

    # ==============================================================================
    # IN-PLACE TEMPORAL ALIGNMENT OF PRECIPITATION DATASET
    # ==============================================================================
    print("Status: Aligning rainfall timeseries with NetCDF timeline limits...")

    # Ensure the 'Data' column is explicitly cast to datetime objects for filtering
    df_piogge['Data'] = pd.to_datetime(df_piogge['Data'])

    # Define target chronological boundaries based on NetCDF audit results
    start_boundary = '2019-12-31'
    end_boundary = '2025-12-31'

    # Filter the dataframe in-place by overwriting the variable
    df_piogge = df_piogge[
        (df_piogge['Data'] >= start_boundary) &
        (df_piogge['Data'] <= end_boundary)
    ].copy()

    # Sort chronologically to guarantee sequential order matching the NetCDF array
    df_piogge = df_piogge.sort_values(by='Data').reset_index(drop=True)

    print(f"Log: Target window set from {start_boundary} to {end_boundary}.")
    print(f"Log: Overwritten df_piogge dataframe shape: {df_piogge.shape}")

    # ==============================================================================
    # PRECIPITATION DATETIME CONTINUITY AUDIT
    # ==============================================================================
    print("Status: Auditing df_piogge chronological continuity...")

    # Extract the unique dates from the filtered rainfall dataframe
    csv_dates = df_piogge['Data']

    # Generate the ideal perfect timeline based on the min and max dates found
    perfect_csv_timeline = pd.date_range(start=csv_dates.min(), end=csv_dates.max(), freq='D')

    # Calculate the difference to spot any missing dates
    missing_csv_dates = perfect_csv_timeline.difference(csv_dates)

    print(f"Log: Rainfall dataframe timeline spans from {csv_dates.min().strftime('%Y-%m-%d')} to {csv_dates.max().strftime('%Y-%m-%d')}.")
    print(f"Log: Expected days: {len(perfect_csv_timeline)} | Actual rows in dataframe: {len(df_piogge)}.")

    if len(missing_csv_dates) == 0:
        print("Success: No days are missing. The chronological index of df_piogge is perfectly continuous.")
    else:
        print(f"Warning: Found {len(missing_csv_dates)} missing dates in the dataframe sequence!")
        print("Missing dates list:", missing_csv_dates.strftime('%Y-%m-%d').tolist())
        
    return df_piogge