import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def fit_definitive_dbscan(X_final_preprocessed, eps=1.0, min_samples=14):
    """
    Trains the definitive density-based DBSCAN model on the complete preprocessed matrix,
    parallelizes execution across all CPU cores, and computes the structural allocation distribution.
    """
    print(f"Training the definitive DBSCAN model (eps={eps}, min_samples={min_samples})...")
    start_time = time.time()

    # Instantiate the density-based estimator with the optimized parameters
    dbscan_final = DBSCAN(
        eps=eps,
        min_samples=min_samples, # Aligned with the 14-NN rule used for the elbow plot
        n_jobs=-1       # Parallelizes execution across all CPU cores to guarantee maximum efficiency
    )

    # Fit on the complete preprocessed matrix used for all previous algorithms
    dbscan_labels = dbscan_final.fit_predict(X_final_preprocessed)

    elapsed = time.time() - start_time
    print(f"DBSCAN training completed successfully in {elapsed:.1f} seconds.")

    # Quick integrity check on the extracted density structures
    unique_labels = np.unique(dbscan_labels)
    print(f"Unique clusters identified: {list(unique_labels)} (Note: -1 represents the unclassified noise)")
    
    # Calculate the absolute and percentage distribution of DBSCAN assignments
    labels_series = pd.Series(dbscan_labels)
    distribution_abs = labels_series.value_counts().sort_index()
    distribution_pct = labels_series.value_counts(normalize=True).sort_index() * 100

    print("\n======= DBSCAN STRUCTURAL DISTRIBUTION =======")
    for label in distribution_abs.index:
        name = f"Cluster {label}" if label != -1 else "Unclassified Noise (-1)"
        print(f"{name:<24} | Samples: {distribution_abs[label]:<8} | Percentage: {distribution_pct[label]:.2f}%")
    print("===============================================\n")
    
    return dbscan_final, dbscan_labels