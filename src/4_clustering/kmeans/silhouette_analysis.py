import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_silhouette_analysis(X_final_preprocessed, sample_size=30000, k_candidates=[2, 3, 4, 5]):
    """
    Executes a memory-safe Silhouette analysis on a random subsample of the preprocessed 
    spatial matrix to determine cluster cohesion and separation limits.
    """
    print("======= EXPANDED SILHOUETTE VERDICT =======")
    
    # 1. Deterministic subsampling to preserve system RAM memory bounds
    np.random.seed(42)
    subsample_idx = np.random.choice(len(X_final_preprocessed), size=sample_size, replace=False)
    X_subsample = X_final_preprocessed[subsample_idx]

    # Dictionary to keep the subsample matrix for the plot function downstream
    analysis_results = {
        'X_subsample': X_subsample,
        'scores': {}
    }

    # 2. Iterate and evaluate each K candidate configuration
    for k_candidate in k_candidates:
        start_time = time.time()

        km = KMeans(n_clusters=k_candidate, init='k-means++', n_init=5, max_iter=150, random_state=42)
        labels_sub = km.fit_predict(X_subsample)

        sil_avg = silhouette_score(X_subsample, labels_sub)

        elapsed = time.time() - start_time
        print(f"Average Silhouette Score for K={k_candidate}: {sil_avg:.4f} (Took {elapsed:.1f}s)")
        
        analysis_results['scores'][k_candidate] = sil_avg
        
    print("==================================================")
    
    return X_subsample, analysis_results