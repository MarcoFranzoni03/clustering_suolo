import pandas as pd

def analyze_and_remove_duplicates(df_features, df_coords):
    """
    Analyzes the structural impact of identical soil profiles (clones),
    logs the spatial bias percentage, and sanitizes the datasets by retaining
    only unique land records to ensure unbiased downstream clustering.
    """
    print("\n--- Initiating Structural Profile Deduplication ---")
    
    # 1. Count and log the total number of identical chemical-physical rows
    duplicate_rows_count = df_features.duplicated().sum()
    duplicate_percentage = (duplicate_rows_count / df_features.shape[0]) * 100
    
    print(f"Total identical rows found: {duplicate_rows_count}")
    print(f"Percentage of duplicated data: {duplicate_percentage:.2f}%")
    
    # 2. Isolate duplicated rows for internal auditing alignment
    df_features_duplicates = df_features[df_features.duplicated(keep=False)]
    df_coords_duplicates = df_coords.loc[df_features_duplicates.index]
    
    # Assemble a sorted dataframe preview for localized spatial validation
    df_visual_check = pd.concat([df_coords_duplicates, df_features_duplicates], axis=1)
    df_visual_check_sorted = df_visual_check.sort_values(by=['lon', 'lat'])
    
    # 3. Remove rows with identical physical-chemical profiles to purge clustering bias
    df_features_unique = df_features.drop_duplicates()
    df_coords_unique = df_coords.loc[df_features_unique.index]
    
    print(f"\n=== DATABASE SANITIZATION COMPLETED ===")
    print(f"Original records: {df_features.shape[0]}")
    print(f"Unique land records: {df_features_unique.shape[0]}")
    print(f"Total bias-inducing clones dropped: {df_features.shape[0] - df_features_unique.shape[0]}")
    print(f"=======================================")
    
    # We return the unique dataframes along with the sorted preview 
    # so you can still view it in the notebook if needed via .head()
    return df_features_unique, df_coords_unique, df_visual_check_sorted