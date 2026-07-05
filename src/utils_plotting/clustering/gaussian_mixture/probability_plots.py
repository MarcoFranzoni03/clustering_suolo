import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gmm_boundary_certainty(gmm_probabilities):
    """
    Computes and renders the Density Profile of GMM soft-clustering responsibilities
    to evaluate boundary transition certainty and assignment confidence distributions.
    """
    print("\n--- Generating GMM Boundary Certainty & Confidence Distribution Plot ---")
    
    # 1. Extract the maximum probability (assignment confidence) for each sample
    max_probabilities = np.max(gmm_probabilities, axis=1)

    # 2. Render the Density Profile of Soft-Clustering Responsibilities
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        max_probabilities,
        fill=True,
        color="purple",
        linewidth=2.5,
        alpha=0.4,
        label="Assignment Confidence Distribution"
    )

    # Statistical references
    plt.axvline(x=0.25, color="red", linestyle=":", linewidth=2, label="Theoretical Random Guess Limit (1/K)")
    plt.axvline(x=np.mean(max_probabilities), color="black", linestyle="--", linewidth=2,
                label=f"Mean Confidence: {np.mean(max_probabilities):.4f}")

    plt.title("GMM Boundary Certainty: Distribution of Maximum Posterior Probabilities (K=4)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Probability of Assigned Component (Confidence Score)")
    plt.ylabel("Relative Spatial Sample Density")
    plt.xlim([0.25, 1.0]) # Min theoretical confidence for K=4 is 0.25
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    
    return max_probabilities