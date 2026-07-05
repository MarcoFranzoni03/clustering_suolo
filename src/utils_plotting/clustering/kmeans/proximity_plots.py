import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def plot_ordered_proximity_matrix(X_scaled_clean, final_labels, sample_size=10000):
    """
    Computes the Proximity and Incidence matrices for a memory-safe sample,
    calculates their Pearson correlation coefficient, sorts the results by cluster,
    and renders a high-definition Ordered Proximity Matrix Heatmap.
    """
    print(f"\n--- Computing Ordered Proximity Matrix & Correlation (N={sample_size}) ---")
    
    # 1. Set a memory-safe sample size (N=10,000 ensures maximum performance within 12GB RAM)
    if X_scaled_clean.shape[0] > sample_size:
        indices = np.random.choice(X_scaled_clean.shape[0], sample_size, replace=False)
        X_sample = X_scaled_clean[indices].astype(np.float32)
        labels_sample = final_labels[indices]
    else:
        X_sample = X_scaled_clean.astype(np.float32)
        labels_sample = final_labels
        sample_size = X_scaled_clean.shape[0]

    # 2. Compute the Proximity Matrix (Pairwise Euclidean Distances)
    proximity_matrix = squareform(pdist(X_sample, metric='euclidean')).astype(np.float32)

    # 3. Compute the Incidence Matrix
    incidence_matrix = (labels_sample[:, None] == labels_sample[None, :]).astype(np.uint8)

    # 4. Extract upper triangular indices to compute the correlation
    tri_indices = np.triu_indices(sample_size, k=1)
    flat_proximity = proximity_matrix[tri_indices]
    flat_incidence = incidence_matrix[tri_indices]

    # Compute Pearson correlation coefficient
    inc_prox_correlation = np.corrcoef(flat_proximity, flat_incidence)[0, 1]
    print(f"Matrix Correlation (Proximity vs Incidence) for N={sample_size}: {inc_prox_correlation:.4f}")

    # Free up flat arrays from memory immediately before sorting and plotting
    del tri_indices, flat_proximity, flat_incidence

    # 5. Sort indices based on cluster labels to align the blocks on the diagonal
    sorting_indices = np.argsort(labels_sample)
    ordered_proximity = proximity_matrix[sorting_indices][:, sorting_indices]

    # Free up un-sorted proximity matrix to recover RAM
    del proximity_matrix

    # 6. Plot the high-definition Ordered Proximity Matrix
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        ordered_proximity,
        cmap="Blues",
        cbar_kws={'label': 'Euclidean Distance'},
        xticklabels=False, yticklabels=False
    )

    plt.title(f"Ordered Proximity Matrix (K=4) | N={sample_size} | Correlation: {inc_prox_correlation:.4f}", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    
    return inc_prox_correlation