import matplotlib.pyplot as plt

def plot_regional_precipitation(ds_hybrid):
    # ==============================================================================
    # VISUALIZING REGIONAL MEAN PRECIPITATION (ALL LAND PIXELS OVER TIME)
    # ==============================================================================
    print("Calculating the spatial mean across all grid pixels for Precipitation...")

    # 1. Compute the mean precipitation across the spatial dimensions (latitude and longitude)
    regional_rain = ds_hybrid['Precipitation'].mean(dim=['latitude', 'longitude'])

    # 2. Initialize the plot
    plt.figure(figsize=(14, 5))

    # 3. Plot the regional rainfall timeline as a bar-like structure or thin lines
    # using alpha to handle overlapping events over the 5 years
    plt.plot(regional_rain.time, regional_rain.values, color='royalblue', linewidth=1, label='Regional Mean Precipitation (mm/day)')
    plt.fill_between(regional_rain.time, 0, regional_rain.values, color='royalblue', alpha=0.3)

    # 4. Style and labels
    plt.title('Regional Precipitation Timeline (5-Year Overview) - Full Capitanata Grid', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline (Years)', fontsize=12)
    plt.ylabel('Mean Precipitation (mm/day)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    # 5. Render the chart
    plt.tight_layout()
    plt.show()