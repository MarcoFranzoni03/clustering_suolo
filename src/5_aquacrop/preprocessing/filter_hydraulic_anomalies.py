import pandas as pd

def filter_negative_hydraulic_values(df_aquacrop_mappa):
    # ==============================================================================
    # HYDRAULIC QUALITY CONTROL AND FILTERING
    # ==============================================================================
    # Define the core hydraulic columns to check for negative values
    hydraulic_cols = ['PWP_vol_pct', 'FC_vol_pct', 'SAT_vol_pct', 'Ksat_mm_day']

    # Count rows before filtering to track data loss
    rows_before = len(df_aquacrop_mappa)

    # Overwrite the DataFrame keeping only rows where all specified columns are >= 0
    df_aquacrop_mappa = df_aquacrop_mappa[(df_aquacrop_mappa[hydraulic_cols] >= 0).all(axis=1)].reset_index(drop=True)

    # Calculate and print the quality control summary
    rows_after = len(df_aquacrop_mappa)
    dropped_rows = rows_before - rows_after

    print(f"Quality Control Check Completed:")
    print(f"-> Rows before filtering: {rows_before}")
    print(f"-> Rows after removing negative anomalies: {rows_after}")
    print(f"-> Total anomalous rows dropped: {dropped_rows}")
    
    return df_aquacrop_mappa