import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx

def plot_soil_clusters_on_map(df_map, cluster_centroids, x_coord, y_coord):
    # ==============================================================================
    # GEOGRAPHIC VALIDATION WITH CONTEXTILY STATIC BACKGROUND MAP
    # ==============================================================================
    print("Status: Rendering static geographic map with contextily background...")

    # 1. Setup the figure and axis
    fig, ax = plt.subplots(figsize=(13, 11))

    # 2. Plot the soil dataset points (using your exact variables)
    sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue='Soil_Cluster',
        data=df_map,
        palette='Set1',
        alpha=0.6, # Slightly lowered alpha to let the background map show through
        s=2,       # Small points to prevent overlap
        ax=ax,
        edgecolor=None
    )

    # 3. Overlay the 4 centroids as large black stars
    ax.scatter(
        x=cluster_centroids[x_coord],
        y=cluster_centroids[y_coord],
        color='black',
        marker='*',
        s=300,
        edgecolor='white',
        linewidth=1.5,
        label='Cluster Centroid',
        zorder=5 # Ensures centroids stay on top of everything
    )

    # 4. Add the background map via contextily
    # WebMercator (EPSG:3857) vs WGS84 (EPSG:4326) check:
    # If your coordinates are standard Lat/Lon (like 41.4, 15.4), we specify the crs
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik)

    # 5. Map styling and boundaries
    ax.set_title("Capitanata Soil Clusters & Centroids on Regional Map", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(title="Management Zone (Cluster)", loc="upper right")

    plt.tight_layout()
    plt.show()