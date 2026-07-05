import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def fit_definitive_kmeans(X_final_preprocessed, n_clusters=4):
    """
    Trains the definitive KMeans model on the entire preprocessed dataset using 
    optimized convergence constraints and prints out final architectural validation metrics.
    """
    print(f"Training the definitive K-Means model on the entire dataset (K={n_clusters})...")
    start_time = time.time()

    # Initialize the final model with max_iter=1000 for absolute convergence safety
    kmeans_final = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=1000,
        random_state=42
    )

    # Fit and predict on the full preprocessed matrix
    final_labels = kmeans_final.fit_predict(X_final_preprocessed)

    elapsed = time.time() - start_time
    print(f"Training completed successfully in {elapsed:.1f} seconds.")
    
    # Calculate and print the official metrics for the definitive model
    final_sse = kmeans_final.inertia_
    final_db = davies_bouldin_score(X_final_preprocessed, final_labels)
    final_ch = calinski_harabasz_score(X_final_preprocessed, final_labels)

    print("\n======= DEFINITIVE MODEL METRICS (K=4) =======")
    print(f"SSE (Inertia): {final_sse:.2f}")
    print(f"Davies-Bouldin Index: {final_db:.4f}")
    print(f"Calinski-Harabasz Index: {final_ch:.2f}")
    print("==============================================\n")
    
    return kmeans_final, final_labels