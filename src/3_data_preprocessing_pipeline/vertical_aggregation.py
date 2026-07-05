import pandas as pd

def compute_depth_weighted_average(df_features):
    """
    Compresses the 6 standardized SoilGrids depth intervals down to unified depth-integrated 
    properties using a vertical thickness-weighted average based on trapezoidal integration.
    """
    print("\n--- Initiating Vertical Profile Aggregation (Thickness Weighting) ---")
    
    # 1. Define the standard SoilGrids depth intervals (thickness in cm)
    depth_weights = {
        "0-5cm": 5,
        "5-15cm": 10,
        "15-30cm": 15,
        "30-60cm": 30,
        "60-100cm": 40,
        "100-200cm": 100
    }
    total_depth = 200

    # 2. Extract base names of variables and sort them to guarantee a stable column order
    base_features = sorted(list(set([
        col.split('_')[0] for col in df_features.columns 
        if '_' in col and not col.startswith('ocs')
    ])))

    # 3. Compute the weighted average using float64 to prevent dynamic integer overflow
    weighted_features_dict = {}
    for feature in base_features:
        weighted_sum = 0
        for suffix, weight in depth_weights.items():
            col_name = f"{feature}_{suffix}"
            if col_name in df_features.columns:
                # Cast to float64 to dynamically prevent int16 memory overflow
                weighted_sum += df_features[col_name].astype(float) * weight

        weighted_features_dict[feature] = weighted_sum / total_depth

    # 4. Rebuild the compact features DataFrame
    df_features_weighted = pd.DataFrame(weighted_features_dict)

    # 5. Bring back 'ocs_0-30cm' directly as it is already an integrated property
    if 'ocs_0-30cm' in df_features.columns:
        df_features_weighted['ocs'] = df_features['ocs_0-30cm'].astype(float)

    print(f"Aggregation complete! Compressed to {df_features_weighted.shape[1]} total properties.")
    print("====================================================================")
    
    return df_features_weighted