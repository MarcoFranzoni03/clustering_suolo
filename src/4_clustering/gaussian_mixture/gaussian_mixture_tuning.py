import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import optuna

# Disable optuna logs to avoid terminal cluttering
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_gmm_tuning(X_final_preprocessed, sample_size=30000):
    """
    Executes a systematic grid-search via Optuna across K values (2 to 8) to optimize 
    the covariance matrix structures of a Gaussian Mixture Model based on BIC minimization.
    """
    print(f"\n--- Starting Systematic GMM Tuning & Architecture Search (Sample N={sample_size}) ---")
    
    # Memory-safe subsampling to guarantee stability within 12GB RAM bounds
    np.random.seed(42)
    indices_gmm = np.random.choice(X_final_preprocessed.shape[0], sample_size, replace=False)
    X_gmm_sample = X_final_preprocessed[indices_gmm].astype(np.float32)

    k_values = list(range(2, 9))
    best_bic_scores = []
    best_aic_scores = []
    best_covariances = []

    # Iterate over each K value to ensure explicit exploration
    for k in k_values:
        print(f"Optimizing GMM architecture for K={k}...")

        def objective(trial):
            covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                n_init=5,
                random_state=42
            )
            gmm.fit(X_gmm_sample)
            return gmm.bic(X_gmm_sample)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=8, show_progress_bar=False)

        best_cov = study.best_params['covariance_type']
        best_bic = study.best_value

        # Re-fit the optimal setup to capture the corresponding AIC score
        best_gmm = GaussianMixture(n_components=k, covariance_type=best_cov, n_init=5, random_state=42)
        best_gmm.fit(X_gmm_sample)
        best_aic = best_gmm.aic(X_gmm_sample)

        best_bic_scores.append(best_bic)
        best_aic_scores.append(best_aic)
        best_covariances.append(best_cov)

    # Reconstruct a structured summary dashboard
    df_gmm_results = pd.DataFrame({
        'K': k_values,
        'Best_Covariance': best_covariances,
        'BIC': best_bic_scores,
        'AIC': best_aic_scores
    })

    print("\n======= SYSTEMATIC GMM TUNING SUMMARY =======")
    print(df_gmm_results.to_string(index=False))
    print("=============================================\n")
    
    # Package results into a dictionary for the downstream plotter module
    tuning_history = {
        'k_values': k_values,
        'bic_scores': best_bic_scores,
        'aic_scores': best_aic_scores,
        'covariances': best_covariances
    }

    return df_gmm_results, tuning_history