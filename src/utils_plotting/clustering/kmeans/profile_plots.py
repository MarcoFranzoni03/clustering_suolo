import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_feature_profiles(X_scaled_clean, numerical_cols, final_labels):
    """
    Reconstructs the standardized DataFrame, computes the mean Z-score profile (centroids)
    for each of the 4 definitive clusters, and renders an evaluation heatmap.
    """
    print("\n--- Generating Cluster Feature Profile Heatmap ---")
    
    # 1. Reconstruct the clean standardized dataframe with original feature names
    df_scaled_clean = pd.DataFrame(X_scaled_clean, columns=numerical_cols)

    # 2. Append the definitive K=4 cluster labels
    df_scaled_clean['Soil_Cluster'] = final_labels

    # 3. Compute the centroids in the original feature space (mean Z-score per cluster)
    original_space_centroids = df_scaled_clean.groupby('Soil_Cluster').mean()

    # 4. Plot the feature importance/profile heatmap for each cluster
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        original_space_centroids,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".3f",
        cbar_kws={'label': 'Mean Z-Score Value'}
    )

    plt.title("Original Feature Profiles Across the 4 Definitive Clusters", fontsize=12, fontweight='bold')
    plt.ylabel("Soil Management Zone (Cluster)")
    plt.xlabel("Original Numerical Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return original_space_centroids