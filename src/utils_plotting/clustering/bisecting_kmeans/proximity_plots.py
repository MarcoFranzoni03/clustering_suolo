import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def plot_bisecting_ordered_proximity_matrix(X_final_preprocessed, final_labels, sample_size=5000):
    """
    Computes Proximity and Incidence matrices for Bisecting K-Means, calculates their
    Pearson correlation coefficient, and renders the high-definition Ordered Proximity Matrix Heatmap.
    """
    print(f"\n--- Computing Bisecting Ordered Proximity Matrix & Correlation (N={sample_size}) ---")
    
    # 1. Set a memory-safe sample size extracted directly from the orthogonal PCA space
    np.random.seed(42) # Enforce statistical replicability

    if X_final_preprocessed.shape[0] > sample_size:
        indices = np.random.choice(X_final_preprocessed.shape[0], sample_size, replace=False)
        X_sample = X_final_preprocessed[indices].astype(np.float32)
        labels_sample = np.asarray(final_labels[indices], dtype=int)
    else:
        X_sample = X_final_preprocessed.astype(np.float32)
        labels_sample = np.asarray(final_labels, dtype=int)
        sample_size = X_final_preprocessed.shape[0]

    # 2. Compute the Proximity Matrix (Pairwise Euclidean Distances in PCA space)
    proximity_matrix = squareform(pdist(X_sample, metric='euclidean')).astype(np.float32)

    # 3. Compute the Ideal Cluster Incidence Matrix (1 if same cluster, 0 otherwise)
    incidence_matrix = (labels_sample[:, None] == labels_sample[None, :]).astype(np.uint8)

    # 4. Extract upper triangular indices to compute the Pearson correlation matrix
    tri_indices = np.triu_indices(sample_size, k=1)
    flat_proximity = proximity_matrix[tri_indices]
    flat_incidence = incidence_matrix[tri_indices]

    # Compute Pearson correlation coefficient between geometric space and partition logic
    inc_prox_correlation = np.corrcoef(flat_proximity, flat_incidence)[0, 1]
    print(f"Matrix Correlation (Proximity vs Incidence) for Bisecting K-Means (N={sample_size}): {inc_prox_correlation:.4f}")

    # Free up flat arrays from memory immediately before sorting and rendering the heatmap
    del tri_indices, flat_proximity, flat_incidence

    # 5. Sort indices based on cluster labels to align the blocks along the main diagonal
    sorting_indices = np.argsort(labels_sample)
    ordered_proximity = proximity_matrix[sorting_indices][:, sorting_indices]

    # Free up un-sorted proximity matrix to recover RAM overhead
    del proximity_matrix

    # 6. Plot the high-definition Ordered Proximity Matrix
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        ordered_proximity,
        cmap="Blues",
        cbar_kws={'label': 'Euclidean Distance (PCA Space)'},
        xticklabels=False, yticklabels=False
    )

    plt.title(f"Ordered Proximity Matrix | Bisecting K-Means (K=4) | N={sample_size} | Correlation: {inc_prox_correlation:.4f}", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    
    return inc_prox_correlation