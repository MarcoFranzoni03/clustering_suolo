import pandas as pd

def audit_negative_weather_values(ds_hybrid):
    # ==============================================================================
    # HYBRID DATASET AUDIT: ANOMALOUS NEGATIVE VALUE CHECK
    # ==============================================================================
    # Convert the boolean mask to a single array and count where at least one weather variable is negative
    neg_weather_rows = (ds_hybrid[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET']] < 0).to_array().any(dim='variable').sum().values

    print(f"Total grid-points/days with negative values: {neg_weather_rows}")

    # ==============================================================================
    # INSPECTING ROWS WITH NEGATIVE WEATHER VALUES IN DS_HYBRID
    # ==============================================================================
    print("Filtering and extracting rows containing negative weather values...")

    # 1. Convert the xarray dataset to a pandas DataFrame and drop structural NaNs
    df_hybrid_flat = ds_hybrid[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET']].to_dataframe().dropna()

    # 2. Create a boolean mask identifying rows where at least one variable is below zero
    rows_with_negatives = (df_hybrid_flat < 0).any(axis=1)

    # 3. Filter the DataFrame to isolate only the anomalous rows
    df_negatives_only = df_hybrid_flat[rows_with_negatives]

    print(f"Log: Found {len(df_negatives_only)} rows with negative entries out of {len(df_hybrid_flat)} total rows.")
    print("\n--- FIRST ROWS WITH NEGATIVE VALUES ---")

    # 4. Display the first 15 rows to inspect which variables are driving the negatives
    print(df_negatives_only.head(15))