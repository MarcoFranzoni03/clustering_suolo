import numpy as np
import matplotlib.pyplot as plt

def plot_three_metric_elbow_profile(all_metrics_history):
    """
    Generates a 1x3 horizontal subplot layout visualizing the mathematical behavior 
    of SSE (Inertia), Davies-Bouldin Index, and Calinski-Harabasz Index across K values.
    """
    # Extract and sort the K values and their relative metrics from your history
    ks_sorted = sorted(all_metrics_history.keys())
    sse_points = [all_metrics_history[k]['SSE (Inertia)'] for k in ks_sorted]
    db_points = [all_metrics_history[k]['Davies-Bouldin'] for k in ks_sorted]
    ch_points = [all_metrics_history[k]['Calinski-Harabasz'] for k in ks_sorted]

    # Setup a 1x3 horizontal subplot layout to isolate the scales
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Capitanata Soil Clustering: 3-Metric Elbow Profile', fontsize=15, fontweight='bold', y=1.02)

    # Panel 1: SSE / Inertia (The classic Elbow plot)
    axs[0].plot(ks_sorted, sse_points, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    axs[0].set_title('1. SSE / Inertia (Look for the Elbow)', fontsize=11, fontweight='bold')
    axs[0].set_xlabel('Number of Clusters (K)')
    axs[0].set_ylabel('Sum of Squared Errors')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Davies-Bouldin (Minimization)
    axs[1].plot(ks_sorted, db_points, marker='s', linestyle='-', color='#ff7f0e', linewidth=2)
    axs[1].set_title('2. Davies-Bouldin Index (Lower is Better)', fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Number of Clusters (K)')
    axs[1].set_ylabel('Score')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Calinski-Harabasz (Maximization)
    axs[2].plot(ks_sorted, ch_points, marker='^', linestyle='-', color='#2ca02c', linewidth=2)
    axs[2].set_title('3. Calinski-Harabasz Index (Higher is Better)', fontsize=11, fontweight='bold')
    axs[2].set_xlabel('Number of Clusters (K)')
    axs[2].set_ylabel('Score')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()