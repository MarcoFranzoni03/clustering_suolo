import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
import optuna

# Disable optuna logs to avoid cluttering the output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_hdbscan_tuning(X_final_preprocessed, sample_size=5000, n_trials=30):
    """
    Executes a multi-trial Optuna study on a memory-safe representative sample to optimize 
    HDBSCAN hyperparameters by maximizing the native global cluster persistence (stability) scores.
    """
    print(f"\n--- Launching HDBSCAN Hyperparameter Optimization via Optuna (Sample N={sample_size}) ---")
    
    # 1. Extract a memory-safe representative sample for the objective function
    np.random.seed(42)
    sample_idx = np.random.choice(X_final_preprocessed.shape[0], sample_size, replace=False)
    X_objective_sample = X_final_preprocessed[sample_idx].astype(np.float32)

    def objective(trial):
        # 2. Define the hyperparameter search space for the hierarchical density framework
        min_cluster_size = trial.suggest_int('min_cluster_size', 50, 2000, step=50)
        min_samples = trial.suggest_int('min_samples', 5, 100, step=5)
        cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0.0, 1.5, step=0.1)
        cluster_selection_method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])

        # 3. Instantiate the HDBSCAN estimator for the current optimization trial
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            n_jobs=-1
        )

        labels = hdb.fit_predict(X_objective_sample)
        unique_labels = np.unique(labels)
        unique_clusters = unique_labels[unique_labels != -1]

        # 4. Strategic penalties to guide Optuna away from degenerate structural solutions
        if len(unique_clusters) <= 1:
            return -1.0  # Return the worst possible structural evaluation score

        # Calculate the unclassified spatial noise ratio
        noise_pct = np.sum(labels == -1) / len(labels)
        if noise_pct > 0.30: # Penalize the trial if it rejects more than 30% of the dataset as noise
            return -1.0

        # 5. Extract the native cluster persistence (stability) scores
        try:
            mean_persistence = np.mean(hdb.cluster_persistence_)
            score = float(mean_persistence)
        except:
            return -1.0

        return score

    # 6. Execute the Optuna study maximizing the multivariate density-based stability index
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\n======= OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETED =======")
    print(f"Best Global Cluster Persistence Score: {study.best_value:.4f}")
    print("Optimized Hyperparameters Selected:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("============================================================")
    
    return study.best_params, study.best_value