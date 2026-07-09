import numpy as np
import pandas as pd

def calculate_real_agronomic_ksat(clay_fraction, bdod):
    """
    Estimates Saturated Hydraulic Conductivity (Ksat) based on clay fraction
    and applies a cubic scaling factor to account for soil compaction (Bulk Density).
    """
    # Determine baseline Ksat from clay fraction thresholds
    if clay_fraction > 0.40:
        base_ksat = 25.0  # mm/day (Typical agricultural clay loam / silty clay)
    elif clay_fraction > 0.25:
        base_ksat = 55.0  # mm/day (Silty loam / clay loam)
    else:
        base_ksat = 120.0 # mm/day (Sandy loam)

    # If bulk density is missing, invalid, or zero, bypass compaction scaling
    if bdod <= 0.01 or np.isnan(bdod):
        return base_ksat

    # Apply compaction correction factor relative to a standard baseline (1.35 g/cm3)
    normal_bdod = 1.35
    compaction_factor = bdod / normal_bdod

    # Lower bound clamp at 5.0 mm/day to prevent numerical stalls in AquaCrop due to extreme compaction
    return max(5.0, base_ksat / (compaction_factor ** 3))

def build_aquacrop_soil_profile(df_raw):
    # ==============================================================================
    # --- LAYER 1: TOPSOIL (0 - 15 cm) -> Thickness-weighted: 5cm (0-5) and 10cm (5-15) ---
    # ==============================================================================
    w_l1_1, w_l1_2 = 5.0, 10.0
    w_l1_tot = w_l1_1 + w_l1_2

    # Aggregate hydraulic properties using thickness weights
    pwp_l1 = (df_raw['wv1500_0-5cm'] * w_l1_1 + df_raw['wv1500_5-15cm'] * w_l1_2) / w_l1_tot
    fc_l1  = (df_raw['wv0033_0-5cm'] * w_l1_1 + df_raw['wv0033_5-15cm'] * w_l1_2) / w_l1_tot
    sat_l1 = (df_raw['wv0010_0-5cm'] * w_l1_1 + df_raw['wv0010_5-15cm'] * w_l1_2) / w_l1_tot

    # Extract and scale clay fraction (from g/kg to true decimal fraction) and bulk density (to g/cm3)
    clay_l1_frac = (((df_raw['clay_0-5cm'] * w_l1_1 + df_raw['clay_5-15cm'] * w_l1_2) / w_l1_tot) / 10.0) / 100.0
    bdod_l1_real = ((df_raw['bdod_0-5cm'] * w_l1_1 + df_raw['bdod_5-15cm'] * w_l1_2) / w_l1_tot) / 100.0

    # Vectorized computation of Ksat via list comprehension
    ksat_l1 = [calculate_real_agronomic_ksat(c, b) for c, b in zip(clay_l1_frac, bdod_l1_real)]

    # ==============================================================================
    # --- LAYER 2: ACTIVE ROOTZONE (15 - 60 cm) -> Thickness-weighted: 15cm (15-30) and 30cm (30-60) ---
    # ==============================================================================
    w_l2_1, w_l2_2 = 15.0, 30.0
    w_l2_tot = w_l2_1 + w_l2_2

    # Aggregate hydraulic properties using thickness weights
    pwp_l2 = (df_raw['wv1500_15-30cm'] * w_l2_1 + df_raw['wv1500_30-60cm'] * w_l2_2) / w_l2_tot
    fc_l2  = (df_raw['wv0033_15-30cm'] * w_l2_1 + df_raw['wv0033_30-60cm'] * w_l2_2) / w_l2_tot
    sat_l2 = (df_raw['wv0010_15-30cm'] * w_l2_1 + df_raw['wv0010_30-60cm'] * w_l2_2) / w_l2_tot

    # Extract and scale clay fraction (from g/kg to true decimal fraction) and bulk density (to g/cm3)
    clay_l2_frac = (((df_raw['clay_15-30cm'] * w_l2_1 + df_raw['clay_30-60cm'] * w_l2_2) / w_l2_tot) / 10.0) / 100.0
    bdod_l2_real = ((df_raw['bdod_15-30cm'] * w_l2_1 + df_raw['bdod_30-60cm'] * w_l2_2) / w_l2_tot) / 100.0

    # Vectorized computation of Ksat via list comprehension
    ksat_l2 = [calculate_real_agronomic_ksat(c, b) for c, b in zip(clay_l2_frac, bdod_l2_real)]

    # ==============================================================================
    # --- LAYER 3: DEEP SUBSOIL (60 - 100 cm) -> Direct mapping (No aggregation required) ---
    # ==============================================================================
    pwp_l3 = df_raw['wv1500_60-100cm']
    fc_l3  = df_raw['wv0033_60-100cm']
    sat_l3 = df_raw['wv0010_60-100cm']

    # Convert raw units directly (No depth-averaging needed since it maps to a single raw interval)
    clay_l3_frac = (df_raw['clay_60-100cm'] / 10.0) / 100.0
    bdod_l3_real = df_raw['bdod_60-100cm'] / 100.0

    # Vectorized computation of Ksat via list comprehension
    ksat_l3 = [calculate_real_agronomic_ksat(c, b) for c, b in zip(clay_l3_frac, bdod_l3_real)]

    # ==============================================================================
    # --- DATA RECONSTRUCTION & FORMATTING FOR AQUACROP ---
    # ==============================================================================
    df_list = []
    for idx, layer_num, hor_name, thick, pwp, fc, sat, ksat in [
        (df_raw.index, 1, 'Topsoil', 0.15, pwp_l1, fc_l1, sat_l1, ksat_l1),
        (df_raw.index, 2, 'Active Rootzone', 0.45, pwp_l2, fc_l2, sat_l2, ksat_l2),
        (df_raw.index, 3, 'Deep Subsoil', 0.40, pwp_l3, fc_l3, sat_l3, ksat_l3)
    ]:
        # Construct individual layer DataFrame with explicit column mapping for safety
        df_layer = pd.DataFrame({
            'Original_Index': idx,
            'lon': df_raw['lon'],
            'lat': df_raw['lat'],
            'Layer': layer_num,
            'Horizon': hor_name,
            'Thickness_m': thick,
            'PWP_vol_pct': pwp / 10.0,  # Scale from cm3/dm3 (permille) to volumetric %
            'FC_vol_pct': fc / 10.0,    # Scale from cm3/dm3 (permille) to volumetric %
            'SAT_vol_pct': sat / 10.0,  # Scale from cm3/dm3 (permille) to volumetric %
            'Ksat_mm_day': ksat
        })
        df_list.append(df_layer)

    # Merge all functional layers back together, sort sequentially by location and depth profile
    df_aquacrop_mappa = pd.concat(df_list, ignore_index=True).sort_values(by=['Original_Index', 'Layer']).reset_index(drop=True)
    print("\n-> Map successfully compiled!")
    
    return df_aquacrop_mappa