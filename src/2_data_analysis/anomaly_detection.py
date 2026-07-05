import pandas as pd

def audit_negative_anomalies(df_features_unique):
    """
    Audits and profiles negative value anomalies (such as -32768 NoData placeholders)
    across all unique physical-chemical soil layers, logging their frequencies and column locations.
    """
    print("\n--- Phase 1: Auditing Negative Value Anomalies ---")
    
    # 1. Count negative values per column
    negative_counts = (df_features_unique < 0).sum()
    columns_with_negatives = negative_counts[negative_counts > 0]
    
    print("Columns with negative values (sorted by frequency):")
    print(columns_with_negatives.sort_values(ascending=False))
    
    negative_cols = df_features_unique.columns[(df_features_unique < 0).any()]
    print("\n--- Detailed Negative Values Breakdown (Unique Records) ---")
    
    # 2. Iterate and break down distinct negative frequencies
    for col in negative_cols:
        negative_values = df_features_unique.loc[df_features_unique[col] < 0, col]
        distinct_counts = negative_values.value_counts()
        print(f"\nColumn: {col}")
        for value, count in distinct_counts.items():
            print(f"  -> Value: {value} | Found in {count} pixels")
            
    # 3. Use 'wv0010_5-15cm' as a diagnostic tracer to study systematic corruption spreading
    if "wv0010_5-15cm" in df_features_unique.columns:
        corrupted_subset = df_features_unique[df_features_unique["wv0010_5-15cm"] < 0]
        print(f"\nAnalyzing {len(corrupted_subset)} corrupted tracer pixels across all features...\n")
        print(f"{'Variable Name':<25} | {'Pixels at EXACTLY 0':<20} | {'Percentage (%)':<10}")
        print("-" * 65)
        
        for col in df_features_unique.columns:
            zero_or_neg_count = (corrupted_subset[col] <= 0).sum()
            percentage = ((zero_or_neg_count / len(corrupted_subset)) * 100) if len(corrupted_subset) > 0 else 0
            print(f"{col:<25} | {zero_or_neg_count:<20} | {percentage:.2f}%")


def sanitize_and_remove_coastal_noise(df_features_unique, df_coords_unique):
    """
    Purges identified NoData dropouts using the volumetric water tracer, aligns 
    spatial coordinates, applies a non-negative machine-learning safe clip, 
    and surgically removes edge coastal boundary noise via multi-depth bulk density masking.
    """
    print("\n--- Phase 2: Executing Spatial Anomaly Purging & Coastal Filtering ---")
    
    # 1. Isolate the valid land rows using the diagnostic water content tracer
    if "wv0010_5-15cm" in df_features_unique.columns:
        clean_rows_mask = df_features_unique["wv0010_5-15cm"] >= 0
        valid_rows = df_features_unique[clean_rows_mask].index
        
        df_features = df_features_unique.loc[valid_rows]
        df_coords = df_coords_unique.loc[valid_rows]
    else:
        df_features = df_features_unique.copy()
        df_coords = df_coords_unique.copy()
        
    # 2. Apply a safety clip to truncate minor micro-interpolation artifacts below zero
    df_features = df_features.clip(lower=0)
    print(f"Purged NoData dropouts. Intermediate valid records: {df_features.shape[0]}")
    
    # 3. Systematic Coastal Noise Removal across all 6 vertical profile layers
    bdod_layers = [f"bdod_{d}" for d in ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]]
    
    # Check if all specified bdod layers exist in df before masking
    available_bdod = [layer for layer in bdod_layers if layer in df_features.columns]
    
    if available_bdod:
        # Keep only rows where ALL available bulk density layers are strictly greater than zero
        pristine_soil_mask = (df_features[available_bdod] > 0).all(axis=1)
        
        df_features_clean = df_features[pristine_soil_mask]
        df_coords_clean = df_coords.loc[pristine_soil_mask]
        
        print(f"\n=== PHYSICAL BOUNDARY CLEANING COMPLETED ===")
        print(f"Surgically dropped all edge pixels containing 0 in any bulk density layer.")
        print(f"Pristine records remaining in df_features: {df_features_clean.shape[0]}")
        print(f"Features remaining before vertical aggregation: {df_features_clean.shape[1]}")
        print(f"=======================================")
        
        return df_features_clean, df_coords_clean
    else:
        print("Warning: Bulk density layers not found. Skipping coastal noise removal.")
        return df_features, df_coords