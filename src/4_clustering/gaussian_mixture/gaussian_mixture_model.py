import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

def fit_definitive_gmm(X_final_preprocessed, n_components=4, covariance_type='full'):
    """
    Trains the definitive Gaussian Mixture Model on the full PCA preprocessed matrix,
    extracts both hard assignments and soft probabilities, and computes equivalent
    geometric validation metrics (SSE, DB, CH) for cross-model auditing.
    """
    print(f"Training the definitive Gaussian Mixture Model (K={n_components}, {covariance_type} covariance) on the PCA space...")
    start_time = time.time()

    # Initialize the optimized GMM structure based on the operational trade-off
    gmm_final = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=10,
        random_state=42
    )

    # Fit on the full preprocessed matrix
    gmm_final.fit(X_final_preprocessed)

    # Extract hard cluster assignments and soft-clustering probabilities
    gmm_labels = gmm_final.predict(X_final_preprocessed)
    gmm_probabilities = gmm_final.predict_proba(X_final_preprocessed)

    elapsed = time.time() - start_time
    print(f"GMM training completed successfully in {elapsed:.1f} seconds.")
    
    # Ensure cluster labels from GMM are structured as a standard NumPy integer array
    gmm_labels_array = np.asarray(gmm_labels, dtype=int)

    # ==============================================================================
    # 1. MATHEMATICAL METRICS EVALUATION FOR GMM (FULL PCA FEATURE SPACE)
    # ==============================================================================

    # Extract the definitive centroids (means of the Gaussian components)
    gmm_centroids = gmm_final.means_

    # Compute the comprehensive pairwise Euclidean distance matrix between all samples and all 4 GMM means
    gmm_distances_to_all = cdist(X_final_preprocessed, gmm_centroids, metric='euclidean')

    # Filter out only the distance connecting each specific pixel to its formally assigned Gaussian component
    gmm_distances_to_assigned = gmm_distances_to_all[np.arange(len(gmm_labels_array)), gmm_labels_array]

    # Compute the final equivalent Sum of Squared Errors (SSE / Inertia) for the GMM partition
    gmm_sse = np.sum(gmm_distances_to_assigned ** 2)

    # Calculate standard academic clustering validation indices across the complete preprocessed matrix
    gmm_db = davies_bouldin_score(X_final_preprocessed, gmm_labels_array)
    gmm_ch = calinski_harabasz_score(X_final_preprocessed, gmm_labels_array)

    # Print out the structural evaluation dashboard for the Gaussian Mixture Model
    print("\n======= GAUSSIAN MIXTURE MODEL METRICS (K=4) =======")
    print(f"Equivalent SSE (Inertia): {gmm_sse:.2f}")
    print(f"Davies-Bouldin Index: {gmm_db:.4f}")
    print(f"Calinski-Harabasz Index: {gmm_ch:.2f}")
    print("====================================================\n")
    
    return gmm_final, gmm_labels_array, gmm_probabilities