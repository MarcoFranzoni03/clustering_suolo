import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx

def plot_bisect_cluster2_focused_validation(df_map, df_medoids, df_c2_samples, x_coord, y_coord):
    # ==============================================================================
    # PLOT VALIDATION MAP: FOCUS ON CLUSTER 2 (2 POINTS) - BISECTING K-MEANS
    # ==============================================================================
    print("Status: Generating focused validation map for Cluster 2...")

    fig, ax = plt.subplots(figsize=(13, 11))

    # Background map: all the soil points mapped via cluster labels
    sns.scatterplot(
        data=df_map,
        x=x_coord,
        y=y_coord,
        hue="Soil_Cluster",
        palette="Set1",
        alpha=0.4,
        s=2,
        edgecolor=None,
        ax=ax
    )

    # Vectorize all Medoids (to keep anchors on the map)
    cluster_ids = sorted(list(df_medoids['Soil_Cluster'].unique()))
    med_lons = [df_medoids[df_medoids['Soil_Cluster'] == c]['lon'].values[0] for c in cluster_ids]
    med_lats = [df_medoids[df_medoids['Soil_Cluster'] == c]['lat'].values[0] for c in cluster_ids]

    ax.scatter(
        med_lons,
        med_lats,
        color="black",
        marker="*",
        s=180,
        edgecolor="white",
        linewidth=1.2,
        label="Cluster Medoid",
        zorder=10
    )

    # Overlay ONLY the 2 specific random control points for Cluster 2
    ax.scatter(
        df_c2_samples['lon'],
        df_c2_samples['lat'],
        color="yellow",
        marker="X",
        s=120,
        edgecolor="black",
        linewidth=1.0,
        label="Cluster 2 Control Points",
        zorder=11
    )

    # Link only the Cluster 2 Medoid to its 2 specific control points (dynamically extracted)
    m2_lon = df_medoids[df_medoids['Soil_Cluster'] == 2]['lon'].values[0]
    m2_lat = df_medoids[df_medoids['Soil_Cluster'] == 2]['lat'].values[0]

    for _, row in df_c2_samples.iterrows():
        ax.plot(
            [m2_lon, row['lon']],
            [m2_lat, row['lat']],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=9
        )

    # Basemap injection
    cx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=cx.providers.OpenStreetMap.Mapnik
    )

    # Styling and Legend setup
    ax.set_title(
        "Capitanata Management Zones: Cluster 2 Multi-Point Robustness Validation (Bisect)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Longitude (Degrees)")
    ax.set_ylabel("Latitude (Degrees)")
    ax.grid(True, linestyle=":", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([60])

    ax.legend(
        handles,
        labels,
        title="Zone & Cluster 2 Validation",
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

    print("Status: Focused validation plot rendered successfully.")