import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import optuna

# Disable optuna logs to avoid cluttering the terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_kmeans_tuning(X_final_preprocessed):
    """
    Executes a deterministic grid search tuning using Optuna to evaluate 
    KMeans clustering performance for K values from 2 to 10.
    """
    # Dictionary to store all metrics for later comparison
    all_metrics_history = {}

    def objective(trial):
        # 1. FIXED RANGE: Changed lower bound to 2 to accommodate the new target
        suggested_k = trial.suggest_int('n_clusters', 2, 10)

        # Keeping max_iter=150 to ensure optimal speed during grid exploration
        kmeans = KMeans(n_clusters=suggested_k, init='k-means++', random_state=42, max_iter=150, n_init=10)
        labels = kmeans.fit_predict(X_final_preprocessed)

        # 1. Internal objective (Inertia / SSE)
        sse = kmeans.inertia_

        # 2. External validation metrics (Davies-Bouldin & Calinski-Harabasz)
        db_score = davies_bouldin_score(X_final_preprocessed, labels)
        ch_score = calinski_harabasz_score(X_final_preprocessed, labels)

        # Save all of them tracked by the current K
        all_metrics_history[suggested_k] = {
            'SSE (Inertia)': sse,
            'Davies-Bouldin': db_score,
            'Calinski-Harabasz': ch_score
        }

        return db_score

    # This forces Optuna to test the exact mathematical sequence from 2 to 10
    search_space = {"n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    grid_sampler = optuna.samplers.GridSampler(search_space)

    # We pass the grid_sampler to the study creation
    study = optuna.create_study(sampler=grid_sampler, direction='minimize')

    # 3. EXPANDED TRIALS: Set to 9 to cleanly evaluate all 9 options in the grid
    study.optimize(objective, n_trials=9)

    # Reconstruct a comprehensive and SORTED evaluation table
    df_metrics = pd.DataFrame(all_metrics_history).T
    df_metrics.index.name = 'Number of Clusters (K)'
    df_metrics = df_metrics.sort_index()

    print("\nCOMPLETE MULTI-METRIC VALIDATION TABLE (WITH K=2):")
    
    return df_metrics, all_metrics_history