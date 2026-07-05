import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_outlier_audit(df_features_pipeline_input, outlier_preds):
    """
    Generates a professional Seaborn scatter plot visualizing the multidimensional outliers
    purged by the Isolation Forest model against Bulk Density and Soil Organic Carbon.
    """
    # 1. Create a temporary DataFrame combining original features and outlier predictions
    df_audit = df_features_pipeline_input.copy()
    df_audit['Is_Outlier'] = outlier_preds
    df_audit['Is_Outlier'] = df_audit['Is_Outlier'].map({1: 'Sane Soil', -1: 'Outlier'})

    # 2. Plot the data using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_audit,
        x='bdod',
        y='soc',
        hue='Is_Outlier',
        palette={'Sane Soil': '#1f77b4', 'Outlier': '#d62728'},
        alpha=0.6,
        s=15
    )

    # 3. Add professional styling and titles
    plt.title('Outlier Auditing: Bulk Density vs Soil Organic Carbon', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Bulk Density (bdod)', fontsize=12)
    plt.ylabel('Soil Organic Carbon (soc)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Classification', loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_pca_scree_plot(reduction_pipeline):
    """
    Generates a professional Scree Plot displaying both individual and running cumulative
    explained variance percentages captured by the Principal Component Analysis (PCA).
    """
    # 1. Extract individual and cumulative variance from fitted PCA
    explained_variance = reduction_pipeline['pca'].explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)

    # 2. Setup the plot figure
    plt.figure(figsize=(9, 5))
    x_axis = range(1, len(explained_variance) + 1)

    # Plot bars for the variance of each individual component
    plt.bar(x_axis, explained_variance, alpha=0.7, color='#2ca02c', label='Individual Explained Variance')

    # Plot the running cumulative line
    plt.plot(x_axis, cumulative_variance, marker='o', linestyle='--', color='#d62728', label='Cumulative Explained Variance')

    # Draw the 90% threshold line that guided our pipeline
    plt.axhline(y=90, color='black', linestyle=':', linewidth=1.5, label='90% Variance Target')

    # Annotate the cumulative values on top of each marker
    for x, y in zip(x_axis, cumulative_variance):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Professional styling and labelling
    plt.title('PCA Auditing: Scree Plot & Cumulative Variance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Principal Components (PCs)', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.xticks(x_axis)
    plt.ylim(0, 110)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()