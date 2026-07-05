import numpy as np
import matplotlib.pyplot as plt

def plot_dbscan_k_distance(X_final_preprocessed, min_pts=14):
    """
    Computes NN distances across the entire PCA dataset using memory-safe KD-Trees,
    sorts the global gradient, and renders the high-definition K-Distance plot 
    to empirically determine the optimal Epsilon (Eps) radius threshold.
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"\n--- Computing {min_pts}-NN Distances across the Entire Dataset ({X_final_preprocessed.shape[0]} samples) ---")
    
    # 1. Process the COMPLETE orthogonal PCA dataset (no subsampling)
    X_full_pca = X_final_preprocessed.astype(np.float32)

    # 2. Instantiate the estimator using the 'auto' tree algorithm for maximum memory safety
    neighbors = NearestNeighbors(n_neighbors=min_pts, algorithm='auto', metric='euclidean', n_jobs=-1)
    neighbors_fit = neighbors.fit(X_full_pca)
    distances, _ = neighbors_fit.kneighbors(X_full_pca)

    # 3. Sort the distance matrix along the sample axis to outline the true global density profile
    distances = np.sort(distances, axis=0)

    # 4. Isolate the target nearest neighbor column (index min_pts - 1)
    global_target_distances = distances[:, min_pts - 1]

    # 5. Render the high-definition global K-Distance plot
    plt.figure(figsize=(11, 6))
    plt.plot(global_target_distances, color='darkorange', linewidth=2.5, label=f'{min_pts}-NN Global Sorted Distances')

    plt.title(f"DBSCAN Structural Selection: Global {min_pts}-Distance Plot (Full Spatial Matrix)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Total Spatial Samples Sorted by Real Density Distance Gradient")
    plt.ylabel("Epsilon Radius Threshold (Eps)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    
    return global_target_distances