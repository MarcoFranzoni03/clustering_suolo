import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx

def plot_final_management_map(df_map, df_medoids, x_coord, y_coord):
    # ==============================================================================
    # PLOT FINAL MANAGEMENT MAP WITH REAL MEDOIDS AND CONTEXTILY
    # ==============================================================================
    print("Status: Generating the definitive Capitanata map with real medoids...")

    # 1. Setup the figure and axis with a professional aspect ratio
    fig, ax = plt.subplots(figsize=(13, 11))

    # 2. Plot all soil grid points (using your dynamic x_coord and y_coord)
    sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue='Soil_Cluster',
        data=df_map,
        palette='Set1',
        alpha=0.5,    # Slightly transparent to read the town names underneath
        s=2,          # Fine resolution
        ax=ax,
        edgecolor=None
    )

    # 3. Plot the 4 Real Medoids as crisp, highly visible stars (No Legend Distortion)
    ax.scatter(
        x=df_medoids[x_coord],
        y=df_medoids[y_coord],
        color='black',
        marker='*',
        s=150,        # Sized correctly so it doesn't inflate the legend naturally
        edgecolor='white',
        linewidth=1.0,
        label='Cluster Medoid (Real Point)',
        zorder=10     # Forces them on top of everything
    )

    # 4. Inject the OpenStreetMap basemap underneath
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik)

    # 5. Fine styling and Standard Legend
    ax.set_title("Capitanata Precision Agriculture: Validated Medoids for AquaCrop", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=':', alpha=0.3)

    # 6. EXPLICIT FIX FOR LEGEND: Simple, robust render without property-checks
    ax.legend(title="Management Zone (Cluster)", loc="upper right")

    plt.tight_layout()
    plt.show()

    print("\nStatus: Map rendered successfully. Checking coordinates mapping...")