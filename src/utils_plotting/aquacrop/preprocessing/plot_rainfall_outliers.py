import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visual_outlier_audit(df_piogge):
    # ==============================================================================
    # VISUAL OUTLIER AUDIT VIA MULTI-STATION SCATTERPLOT
    # ==============================================================================
    print("Status: Generating scatterplot to visually audit for extreme precipitation outliers...")

    # Melt the dataframe to transform it from wide format to long format for easy plotting
    df_melted = df_piogge.melt(id_vars=['Data'], var_name='Stazione', value_name='Pioggia_mm')

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=df_melted, x='Data', y='Pioggia_mm', alpha=0.4, color='teal', edgecolor=None)

    # Formatting the plot
    plt.title('Precipitation Distribution across All Stations (2020-2025) - Outlier Check', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Rainfall (mm)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Tight layout and show
    plt.tight_layout()
    plt.show()

    # Print the top 5 highest rainfall events recorded to have the exact numbers
    print("\nLog: Top 5 maximum rainfall events in the dataset:")
    print(df_melted.sort_values(by='Pioggia_mm', ascending=False).head(5).to_string(index=False))