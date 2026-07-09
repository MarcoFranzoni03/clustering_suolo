import matplotlib.pyplot as plt
import pandas as pd

def visual_temperature_audit(ds_hybrid):
    # ==============================================================================
    # PLOT ABSOLUTE TEMPERATURE EXTREMES TO DETECT WEATHER OUTLIERS
    # ==============================================================================
    print("Status: Calculating absolute grid-wide extremes for outlier detection...")

    # 1. Compute the absolute max and min across all spatial pixels for each timestep
    # This ensures that even a single corrupted pixel on a specific day will be fully visible
    daily_absolute_min = ds_hybrid['MinTemp'].min(dim=['latitude', 'longitude']).to_series()
    daily_absolute_max = ds_hybrid['MaxTemp'].max(dim=['latitude', 'longitude']).to_series()

    # 2. Setup a professional, wide timeline figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # 3. Plot the absolute highest and lowest boundaries found in the entire dataset
    ax.plot(daily_absolute_max.index, daily_absolute_max.values, color='#d62728', label='Absolute Maximum Grid Temperature', alpha=0.8, linewidth=1.2)
    ax.plot(daily_absolute_min.index, daily_absolute_min.values, color='#1f77b4', label='Absolute Minimum Grid Temperature', alpha=0.8, linewidth=1.2)

    # 4. Add reference lines for physical sanity checks (e.g., extreme historical thresholds)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.0)
    ax.axhline(y=45, color='purple', linestyle=':', alpha=0.5, linewidth=1.2, label='Extreme Heat Threshold (45°C)')
    ax.axhline(y=-10, color='cyan', linestyle=':', alpha=0.5, linewidth=1.2, label='Extreme Frost Threshold (-10°C)')

    # 5. Chart styling and layout tuning
    ax.set_title("Capitanata Dataset Quality Control: Absolute Spatial Grid Extremes Over Time", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Timeline Horizon / Date")
    ax.set_ylabel("Air Temperature (°C)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')

    # Adjust y-limits dynamically with a safety envelope to easily spot vertical spikes
    all_values = pd.concat([daily_absolute_max, daily_absolute_min])
    ax.set_ylim(all_values.min() - 5, all_values.max() + 5)

    plt.tight_layout()
    plt.show()

    print(f"Status: Quality control plot rendered. Dataset range verified between {all_values.min():.2f}°C and {all_values.max():.2f}°C.")