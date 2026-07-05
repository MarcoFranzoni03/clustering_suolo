import numpy as np
import matplotlib.pyplot as plt

def plot_gmm_silhouette_profile(X_final_preprocessed, sample_size=10000):
    """
    Reconstructs a custom high-definition Silhouette plot (Yellowbrick style) 
    for the Gaussian Mixture Model using a memory-safe spatial matrix slice.
    """
    print(f"\n--- Generating Custom GMM Silhouette Profile Plot (N={sample_size}) ---")
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.cm as cm

    # 1. Memory-safe subsampling for perfect synchronization with previous plots
    np.random.seed(42)
    indices_gmm = np.random.choice(X_final_preprocessed.shape[0], sample_size, replace=False)
    X_subsample_gmm_pca = X_final_preprocessed[indices_gmm]

    # 2. Fit a localized GMM on the sample to extract the hard labels for Silhouette
    gmm_eval = GaussianMixture(n_components=4, covariance_type='full', n_init=10, random_state=42)
    sample_labels = gmm_eval.fit_predict(X_subsample_gmm_pca)

    # Compute global silhouette average and individual sample scores
    gmm_silhouette_avg = silhouette_score(X_subsample_gmm_pca, sample_labels)
    sample_silhouette_values = silhouette_samples(X_subsample_gmm_pca, sample_labels)

    # 3. Custom High-Definition Silhouette Profile Plot
    fig, ax1 = plt.subplots(figsize=(9, 6))

    y_lower = 10
    for i in range(4): # Loop over the 4 clusters
        # Aggregate and sort silhouette scores for the current cluster
        ith_cluster_silhouette_values = sample_silhouette_values[sample_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Select color from a high-contrast palette
        color = cm.get_cmap("Set1")(float(i) / 4)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontweight='bold')
        y_lower = y_upper + 10  # 10 for the gaps between segments

    ax1.set_title("Silhouette Plot of GaussianMixture Clustering for 10000 Samples in 4 Centers", fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("silhouette coefficient values")
    ax1.set_ylabel("cluster label")

    # The vertical line for average silhouette score
    ax1.axvline(x=gmm_silhouette_avg, color="red", linestyle="--", linewidth=2, label=f"Average Silhouette Score: {gmm_silhouette_avg:.4f}")
    ax1.set_yticks([])  # Clear the y-axis labels / ticks
    ax1.set_xlim([-0.4, 0.6])
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
    
    return X_subsample_gmm_pca