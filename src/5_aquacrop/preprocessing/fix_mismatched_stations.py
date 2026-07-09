import pandas as pd

def resolve_station_mismatches(df_coordinate, df_piogge):
    # ==============================================================================
    # IDENTIFY UNMATCHED RAINFALL STATIONS
    # ==============================================================================
    print("Status: Identifying the 7 unmatched stations from df_piogge...")

    # Find columns in df_piogge that are NOT present in df_coordinate['CLEAN_NAME']
    unmatched_rain_stations = [col for col in df_piogge.columns if col != 'Data' and col not in df_coordinate['CLEAN_NAME'].values]

    print(f"Log: Found {len(unmatched_rain_stations)} stations without coordinate matches.\n")
    print("--- UNMATCHED RAINFALL STATIONS ---")
    for idx, name in enumerate(sorted(unmatched_rain_stations)):
        print(f"{idx+1}. {name}")

    # ==============================================================================
    # IDENTIFY UNMATCHED COORDINATE STATIONS
    # ==============================================================================
    print("Status: Identifying stations in df_coordinate without rainfall data...")

    # Get all clean station names from coordinates
    coord_names = df_coordinate['CLEAN_NAME'].values

    # Find which of these names are NOT present in df_piogge columns
    unmatched_coord_stations = [name for name in coord_names if name not in df_piogge.columns]

    print(f"Log: Found {len(unmatched_coord_stations)} coordinate stations without a matching rainfall column.\n")
    print("--- UNMATCHED COORDINATE STATIONS ---")
    for idx, name in enumerate(sorted(unmatched_coord_stations)):
        print(f"{idx+1:02d}. {name}")

    # ==============================================================================
    # FIX PUNCTUATION SPACING AND RE-RUN HARMONIZATION MATCH
    # ==============================================================================
    print("Status: Fixing dot-spacing typos in df_piogge column names...")

    # Create a dictionary to map the 7 mismatched names to their official coordinate equivalents
    typo_fix_map = {
        'CAN. S.MARIA - P.TE SP12': 'CAN. S. MARIA - P.TE SP12',
        'FOGGIA IST.AGR.': 'FOGGIA IST. AGR.',
        'MASSERIA S.CHIARA': 'MASSERIA S. CHIARA',
        'ROCCHETTA S.ANTONIO': 'ROCCHETTA S. ANTONIO',
        'S.AGATA DI PUGLIA': 'S. AGATA DI PUGLIA',
        'S.PAOLO DI CIVITATE': 'S. PAOLO DI CIVITATE',
        'S.SEVERO': 'S. SEVERO'
    }

    # Apply the manual mapping to df_piogge columns
    df_piogge = df_piogge.rename(columns=typo_fix_map)

    # Re-calculate the intersection
    set_coords = set(df_coordinate['CLEAN_NAME'])
    set_piogge = set(df_piogge.columns) - {'Data'}
    matched_stations = set_coords.intersection(set_piogge)

    print(f"Log: Unique weather columns in rainfall file:     {len(set_piogge)}")
    print(f"Success: Perfectly matched stations ready to use: {len(matched_stations)} / {len(set_piogge)}")

    return df_piogge