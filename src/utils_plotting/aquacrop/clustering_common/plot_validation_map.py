import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx

def plot_validation_map(df_map, cluster_pairs, x_coord, y_coord):
    # ==============================================================================
    # PLOT DEFINITIVE MAP (BISECTING K-MEANS VALIDATION)
    # ==============================================================================
    print("Status: Generating Bisecting K-Means validation map...")

    fig, ax = plt.subplots(figsize=(13, 11))

    # Background map: all the soil points
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

    # Medoids and random points coordinates
    cluster_ids = sorted(cluster_pairs.keys())

    med_lons = [cluster_pairs[c]["medoid"]["coords"][0] for c in cluster_ids]
    med_lats = [cluster_pairs[c]["medoid"]["coords"][1] for c in cluster_ids]

    rand_lons = [cluster_pairs[c]["random_point"]["coords"][0] for c in cluster_ids]
    rand_lats = [cluster_pairs[c]["random_point"]["coords"][1] for c in cluster_ids]

    # Medoids
    ax.scatter(
        med_lons,
        med_lats,
        color="black",
        marker="*",
        s=180,
        edgecolor="white",
        linewidth=1.2,
        label="Bisect Medoid",
        zorder=10
    )

    # Random points
    ax.scatter(
        rand_lons,
        rand_lats,
        color="yellow",
        marker="X",
        s=120,
        edgecolor="black",
        linewidth=1.0,
        label="Random Paired Point",
        zorder=11
    )

    # Links every medoid to his random point
    for c in cluster_ids:
        m_lon, m_lat = cluster_pairs[c]["medoid"]["coords"]
        r_lon, r_lat = cluster_pairs[c]["random_point"]["coords"]

        ax.plot(
            [m_lon, r_lon],
            [m_lat, r_lat],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            zorder=9
        )

    # Basemap injection
    cx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=cx.providers.OpenStreetMap.Mapnik
    )

    # Style & Axis configuration
    ax.set_title(
        "Capitanata Management Zones: Bisecting K-Means Spatial Robustness Validation",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.3)

    # Clean and resize legend handlers
    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([60])

    ax.legend(
        handles,
        labels,
        title="Bisect Zones & Pairing",
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()