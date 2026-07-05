import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def run_statistical_preprocessing_pipeline(df_features_weighted, df_coords, nominal_threshold=10, contamination_rate=0.02, pca_variance=0.90):
    """
    Executes a complete, end-to-end statistical data preprocessing pipeline:
    1. Inspects missing value distribution and cardinalities.
    2. Dynamically isolates numerical attributes based on unique value criteria.
    3. Stabilizes and scales data through an Imputer + StandardScaler pipeline.
    4. Evaluates and drops multidimensional outliers using an Isolation Forest, synchronizing coordinates.
    5. Compresses structural collinearity extracting orthogonal Principal Components via PCA.
    """
    print("\n--- Phase 1: Auditing Missing Values and Cardinality ---")
    
    # 1. Audit missing values profile
    missing_in_cols = df_features_weighted.isna().sum()
    cols_with_missing = missing_in_cols[missing_in_cols > 0]
    if not cols_with_missing.empty:
        print("Columns with missing values detected:")
        print(cols_with_missing)
    else:
        print("Pass: No missing values found across weighted attributes.")

    # 2. Count unique values to separate potential nominal features
    cols_unique_vals_count = df_features_weighted.nunique().sort_values()
    
    # 3. Dynamic feature selection based on configuration threshold
    numerical_cols = [
        col for col, val in cols_unique_vals_count.items()
        if val > nominal_threshold
    ]
    df_features_pipeline_input = df_features_weighted[numerical_cols]
    print(f"Features selected for pipeline based on uniqueness: {len(numerical_cols)} columns.")

    print("\n--- Phase 2: Feature Stabilization & Standardization (Phase A) ---")
    
    # 4. Standardizing features column-wise using median imputer and standard Z-score scaling
    prep_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_scaled = prep_pipeline.fit_transform(df_features_pipeline_input)
    print(f"Phase A Complete: Features stabilized. Shape: {X_scaled.shape}")

    print("\n--- Phase 3: Multidimensional Outlier Filtering (Isolation Forest) ---")
    
    # 5. Isolation Forest deployment to drop the worst anomalous matrix rows
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_scaled)
    
    clean_mask = (outlier_preds == 1)
    X_scaled_clean = X_scaled[clean_mask]
    
    # Synchronize geospatial coordinates with outlier filtration output
    df_coords_clean = df_coords.loc[df_features_pipeline_input.index].loc[clean_mask]
    
    print(f"Intermediary Phase Complete: Removed {(~clean_mask).sum()} anomalous pixels.")
    print(f"Remaining clean pixels for downstream modeling: {X_scaled_clean.shape[0]}")

    print("\n--- Phase 4: Feature Compression & Dimensionality Reduction (Phase B) ---")
    
    # 6. Apply PCA to project features into an uncorrelated orthogonal space
    reduction_pipeline = Pipeline([
        ('pca', PCA(n_components=pca_variance))
    ])
    
    X_final_preprocessed = reduction_pipeline.fit_transform(X_scaled_clean)
    
    n_comps = reduction_pipeline['pca'].n_components_
    total_variance_explained = np.sum(reduction_pipeline['pca'].explained_variance_ratio_) * 100
    
    print(f"Phase B Complete: PCA automatically extracted {n_comps} Principal Components.")
    print(f"Total Cumulative Variance Retained: {total_variance_explained:.2f}%")
    
    # 7. Structure reconstruction for subsequent visual audit tracking
    pca_cols = [f"PC{i+1}" for i in range(n_comps)]
    df_pca_visualization = pd.DataFrame(
        data=X_final_preprocessed,
        columns=pca_cols,
        index=df_coords_clean.index
    )
    
    print(f"\nPipeline Execution Finished! 'X_final_preprocessed' matrix is ready for clustering.")
    print("====================================================================================")
    
    return X_final_preprocessed, df_coords_clean, df_pca_visualization