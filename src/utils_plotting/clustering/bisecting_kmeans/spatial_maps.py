import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_definitive_bisecting_map(df_coords_clean, final_labels):
    """
    Renders the high-contrast definitive geographical map mapping the 4 
    hierarchical management zones across the Capitanata coordinate grid space.
    """
    print("\n--- Generating Definitive Bisecting Geographical Cluster Map ---")
    
    # 1. Create the synchronized spatial dataframe linking coordinates and K=4 hierarchical labels
    df_map = df_coords_clean.copy()
    df_map['Soil_Cluster'] = final_labels

    # 2. Extract spatial coordinate column names from the clean coordinates dataframe
    x_coord = df_map.columns[0]
    y_coord = df_map.columns[1]

    # 3. Render the definitive hierarchical geographical map
    plt.figure(figsize=(13, 11))
    sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue='Soil_Cluster',
        data=df_map,
        palette='Set1',       # High-contrast categorical palette for the 4 hierarchical zones
        alpha=0.9,
        s=1,                  # Minimal marker size to ensure ultra-high definition and prevent pixel overlap
        edgecolor=None
    )

    plt.title("Capitanata Precision Agriculture: Definitive 4-Zone Bisecting K-Means Map", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Spatial Coordinate X / Longitude")
    plt.ylabel("Spatial Coordinate Y / Latitude")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(title="Management Zone (Bisecting Cluster)", loc="upper right", markerscale=10)
    plt.axis('equal')       # Enforces a strict 1:1 aspect ratio to completely eliminate geographic distortion
    plt.tight_layout()
    plt.show()


def plot_bisecting_satellite_map(df_coords_clean, final_labels, output_filename='Capitanata_Satellite_Bisecting_KMeans_Map.png'):
    """
    Overlays translucent hierarchical clustering fields over real-time ESRI high-resolution 
    satellite imagery via contextily to evaluate morphological and geographical ground-truth.
    """
    print("\n--- Generating Translucent Bisecting Satellite Ground-Truth Validation Map ---")
    import contextily as cx

    # 1. Create the synchronized spatial dataframe linking coordinates and K=4 hierarchical labels
    df_map = df_coords_clean.copy()
    df_map['Soil_Cluster'] = final_labels

    x_coord = df_map.columns[0]
    y_coord = df_map.columns[1]

    # 2. Initialize the geographical plot figure matching the academic standard layout
    plt.figure(figsize=(14, 12))

    # Setting zorder=2 forces the points to sit strictly on top of the satellite baseline
    ax = sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue='Soil_Cluster',
        data=df_map,
        palette='Set1',        # High-contrast categorical palette for the 4 hierarchical zones
        alpha=0.35,            # Perfectly balanced transparency to see both clusters and terrain
        s=1.0,                 # Calibrated size to ensure visibility over high-res satellite textures
        edgecolor=None,
        zorder=2               # Ensures the hierarchical clustering layer is rendered on top
    )

    # 3. Inject the real-time satellite imagery baseline via contextily
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:4326",                           # Explicit coordinate reference system for standard Lat/Lon
            source=cx.providers.Esri.WorldImagery,    # Professional high-resolution satellite provider
            alpha=1.0,                                # Full opacity for the underlying satellite matrix
            zorder=1                                  # Forces the background layer underneath the scatter points
        )
    except Exception as e:
        # Fallback execution if the spatial coordinates are already converted into Web Mercator meters
        print("[Spatial CRS Notification] Standard EPSG:4326 mapping failed. Executing default projection...")
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, alpha=1.0, zorder=1)

    # 4. Refine academic plot styling and geomorphological mapping attributes
    plt.title("Capitanata Precision Agriculture: Translucent Satellite Ground-Truth Validation (4-Zone Bisecting K-Means)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Geographical Coordinate X / Longitude", fontsize=11)
    plt.ylabel("Geographical Coordinate Y / Latitude", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.3, zorder=3)

    plt.legend(title="Bisecting Management Zone", loc="upper right", markerscale=8, facecolor='white', framealpha=0.9)
    plt.axis('equal')
    plt.tight_layout()

    # Save the final high-resolution, readable chart for the thesis attachments
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Hierarchical satellite map exported successfully as: {output_filename}")
    plt.show()