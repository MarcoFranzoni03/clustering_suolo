import matplotlib.pyplot as plt

def plot_regional_reference_et(ds_hybrid):
    # ==============================================================================
    # VISUALIZING REGIONAL MEAN REFERENCE ET (ALL LAND PIXELS OVER TIME)
    # ==============================================================================
    print("Calculating the spatial mean across all grid pixels for ReferenceET...")

    # 1. Compute the mean across the spatial dimensions (latitude and longitude)
    # This collapses the 3D grid into a 1D time-series of the regional average
    regional_et = ds_hybrid['ReferenceET'].mean(dim=['latitude', 'longitude'])

    # 2. Initialize the plot
    plt.figure(figsize=(14, 6))

    # 3. Plot the regional average timeline
    regional_et.plot(color='crimson', linewidth=1.5, label='Regional Mean ET0 (mm/day)')

    # 4. Style and labels
    plt.title('Regional Seasonal Dynamics of Reference Evapotranspiration (ET0) - Full Capitanata Grid', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline (Years)', fontsize=12)
    plt.ylabel('Mean ET0 (mm/day)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # 5. Render the chart
    plt.tight_layout()
    plt.show()