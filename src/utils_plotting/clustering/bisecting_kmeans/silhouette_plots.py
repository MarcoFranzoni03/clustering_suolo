import numpy as np
import matplotlib.pyplot as plt

def plot_definitive_bisecting_silhouette(X_final_preprocessed, sample_size=10000):
    """
    Extracts a randomized, memory-safe subsample from the PCA matrix and renders 
    the definitive Yellowbrick Silhouette profile for the BisectingKMeans architecture.
    """
    print(f"\n--- Generating Bisecting K-Means Silhouette Profile (N={sample_size}) ---")
    from sklearn.cluster import BisectingKMeans
    from yellowbrick.cluster import silhouette_visualizer

    # 1. Enforce statistical replicability for the sampling process
    np.random.seed(42) 

    # 2. Extract a randomized, independent subset directly from the orthogonal PCA space
    indices_bisect = np.random.choice(X_final_preprocessed.shape[0], sample_size, replace=False)
    X_subsample_bisect_pca = X_final_preprocessed[indices_bisect]

    # 3. Generate the definitive Silhouette profile plot aligned with the Bisecting architecture geometry
    plt.figure(figsize=(9, 6))
    silhouette_visualizer(
        BisectingKMeans(n_clusters=4, init='k-means++', random_state=42),
        X_subsample_bisect_pca,
        colors='yellowbrick'
    )
    
    return X_subsample_bisect_pca