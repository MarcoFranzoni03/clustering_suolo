import pandas as pd
from sklearn.impute import KNNImputer

def analyze_and_impute_rainfall(df_piogge):
    # ==============================================================================
    # INVESTIGATION: COUNT TOTAL AND PER-COLUMN MISSING VALUES (NaN)
    # ==============================================================================
    print("Status: Analyzing df_piogge for missing values (NaN)...")

    # 1. Calculate the total number of NaN values in the entire dataframe
    total_nans = df_piogge.isna().sum().sum()

    # 2. Calculate NaNs column by colum (excluding the 'Data' column)
    nans_per_station = df_piogge.drop(columns=['Data']).isna().sum()

    # 3. Filter only stations that actually have at least one NaN
    stations_with_nans = nans_per_station[nans_per_station > 0].sort_values(ascending=False)

    print(f"Log: Total NaN data points found across the entire dataset: {total_nans}\n")

    if total_nans == 0:
        print("Result: There are absolutely no NaN values in the dataset.")
    else:
        print("--- STATIONS WITH MISSING VALUES ---")
        for station, count in stations_with_nans.items():
            percentage = (count / len(df_piogge)) * 100
            print(f"- {station}: {count} missing days ({percentage:.2f}% of the series)")

    # ==============================================================================
    # DROP HIGH-VACANCY STATIONS (> 40% NaN) FROM RAINFALL DATAFRAME
    # ==============================================================================
    print("Status: Filtering out unreliable weather stations...")

    # Set the strict threshold (40% of the total 2193 days)
    threshold_percent = 40.0
    max_allowed_nans = len(df_piogge) * (threshold_percent / 100.0)

    # Calculate NaNs per column (excluding the 'Data' index)
    nans_per_station = df_piogge.drop(columns=['Data']).isna().sum()

    # Identify columns that violate the threshold
    columns_to_drop = nans_per_station[nans_per_station > max_allowed_nans].index.tolist()

    print(f"Log: Stations exceeding {threshold_percent}% missing data: {columns_to_drop}")

    # Drop the identified columns from df_piogge
    df_piogge = df_piogge.drop(columns=columns_to_drop)

    print(f"Success: Dropped {len(columns_to_drop)} stations.")
    print(f"Log: Remaining active columns in df_piogge: {len(df_piogge.columns) - 1} (plus 'Data')")

    # ==============================================================================
    # FINAL MISSING DATA IMPUTATION VIA KNN (DIRECTLY ON DF_PIOGGE)
    # ==============================================================================
    print("Status: Initializing KNN Imputer for remaining precipitation gaps...")

    # 1. Isolate the numeric precipitation matrix (excluding the 'Data' column)
    rain_features = df_piogge.drop(columns=['Data'])

    # 2. Configure the KNN Imputer (using 3 neighbors, weighted by similarity)
    imputer = KNNImputer(n_neighbors=3, weights="uniform")

    # 3. Fit and transform directly on the matrix
    print("Log: Computing mathematical replacements for remaining NaNs...")
    imputed_matrix = imputer.fit_transform(rain_features)

    # 4. Overwrite the original df_piogge columns with the clean, imputed data
    df_piogge.iloc[:, 1:] = imputed_matrix

    # 5. Double-check that total NaNs are now exactly 0
    final_nans = df_piogge.drop(columns=['Data']).isna().sum().sum()
    print(f"Success: Imputation complete. Total NaN values left in df_piogge: {final_nans}")

    return df_piogge