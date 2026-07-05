import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_definitive_dbscan_map(df_coords_clean, dbscan_labels, dbscan_final):
    """
    Renders the high-contrast geographical density map for DBSCAN assignments,
    isolating spatial noise (-1) from structured high-density macro agronomic zones.
    """
    print("\n--- Generating Definitive DBSCAN Density Structure Map ---")
    
    # 1. Create the synchronized spatial dataframe linking coordinates and DBSCAN labels
    df_map_dbscan = df_coords_clean.copy()
    df_map_dbscan['DBSCAN_Cluster'] = dbscan_labels

    # 2. Extract spatial coordinate column names from the clean coordinates dataframe
    x_coord = df_map_dbscan.columns[0]
    y_coord = df_map_dbscan.columns[1]

    # 3. Render the definitive geographical map for DBSCAN
    plt.figure(figsize=(13, 11))
    sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue='DBSCAN_Cluster',
        data=df_map_dbscan,
        palette='bright',       # High-contrast colors to isolate noise and micro-islands
        alpha=0.9,
        s=1,                   # Smallest marker size to prevent pixel bleeding and overlap
        edgecolor=None
    )

    plt.title(f"Capitanata Precision Agriculture: DBSCAN Density Structure Map\n(Eps={dbscan_final.eps}, MinPts={dbscan_final.min_samples})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Spatial Coordinate X / Longitude")
    plt.ylabel("Spatial Coordinate Y / Latitude")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(title="Density Class (Cluster)", loc="upper right", markerscale=8)
    plt.axis('equal')       # Strictly prevents structural or geographic distortion
    plt.tight_layout()
    plt.show()