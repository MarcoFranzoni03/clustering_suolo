import re
import pandas as pd

def clean_and_harmonize_stations(df_coordinate, df_piogge):
    # ==============================================================================
    # 1. STRIP WHITE SPACES FROM DF_COORDINATE COLUMNS
    # ==============================================================================
    print("Status: Stripping leading and trailing whitespaces from column headers...")

    # Rewrite column names by applying strip() to each header
    df_coordinate.columns = df_coordinate.columns.str.strip()

    print("Success: Columns successfully cleaned!")
    print("Updated column headers:", df_coordinate.columns.tolist())

    # ==============================================================================
    # 2. INSPECT STATION NAMES FOR ANOMALIES AND SPECIAL CHARACTERS
    # ==============================================================================
    print("Status: Listing all unique station names from df_coordinate...")

    # Sort the names alphabetically to make inspection easier
    unique_names = sorted(df_coordinate['NOME STAZIONE'].unique())

    print(f"Log: Found {len(unique_names)} unique stations in the coordinate dataset.\n")
    print("--- STATION NAMES LIST ---")
    for idx, name in enumerate(unique_names):
        # repr() is used here to expose hidden characters like \t, \n or trailing spaces
        print(f"{idx+1:02d}. {repr(name)}")

    # ==============================================================================
    # 3. STRING REWRITING AND HARMONIZATION FOR BOTH DATASETS
    # ==============================================================================
    print("Status: Executing synchronized name harmonization across both dataframes...")

    # Step 1: Clean and overwrite the coordinate names in df_coordinate
    df_coordinate['CLEAN_NAME'] = df_coordinate['NOME STAZIONE'].astype(str)\
        .str.upper()\
        .str.replace(r'\s+', ' ', regex=True)\
        .str.strip()\
        .str.replace("'", "", regex=False)\
        .str.replace("`", "", regex=False)

    # Step 2: Clean and overwrite the column names in df_piogge
    piogge_rename_map = {}
    for col in df_piogge.columns:
        if col == 'Data':
            piogge_rename_map[col] = 'Data'
        else:
            # Extract name before the pipeline symbol
            clean_col = col.split('|')[0].upper()
            # Replace multiple spaces with a single space
            clean_col = re.sub(r'\s+', ' ', clean_col).strip()
            # Remove punctuation
            clean_col = clean_col.replace("'", "").replace("`", "")
            piogge_rename_map[col] = clean_col

    df_piogge = df_piogge.rename(columns=piogge_rename_map)

    # Step 3: Verify the intersection between the two clean sets
    set_coords = set(df_coordinate['CLEAN_NAME'])
    set_piogge = set(df_piogge.columns) - {'Data'}
    matched_stations = set_coords.intersection(set_piogge)

    print(f"Log: Unique station names in coordinates file: {len(set_coords)}")
    print(f"Log: Unique weather columns in rainfall file:     {len(set_piogge)}")
    print(f"Success: Perfectly matched stations ready to use: {len(matched_stations)}")

    return df_coordinate, df_piogge