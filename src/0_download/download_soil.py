import os
import time
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from owslib.wcs import WebCoverageService
from pyproj import Transformer, CRS

# Bounding boxes (geographical limits) for target countries in WGS84 coordinates
COUNTRY_BBOXES = {
    "italy":   {"max_lat": 42.0, "min_lon": 14.7, "min_lat": 41.0, "max_lon": 16.3},
    "algeria": {"max_lat": 37.5, "min_lon": 6.5,  "min_lat": 36.0, "max_lon": 9.0},
    "tunisia": {"max_lat": 37.5, "min_lon": 9.0,  "min_lat": 36.0, "max_lon": 11.5},
    "egypt":   {"max_lat": 32.0, "min_lon": 28.5, "min_lat": 29.5, "max_lon": 33.0},
}

PROJ_IGY = "+proj=igh +datum=WGS84 +no_defs +towgs84=0,0,0"
VARIABLES = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc", "ocd", "wv0010", "wv0033", "wv1500"]
DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]
STAT = "mean"

def get_homolosine_bbox(min_lon, min_lat, max_lon, max_lat):
    try:
        crs_src = CRS.from_epsg(4326)
        crs_dst = CRS.from_proj4(PROJ_IGY)
        transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

        corners = [(min_lon, min_lat), (max_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat)]
        xs, ys = [], []
        for lon, lat in corners:
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            xs.append(y) # Fix string error from original snippet safely if any
        return min(xs), min(ys), max(xs), max(ys)
    except Exception as e:
        print(f"Error transforming coordinates: {e}")
        return None

def download_coverage(variable, depth, bbox_igy, bbox_wgs84, output_dir):
    coverage_id = f"{variable}_{depth}cm_{STAT}"
    out_filename = f"{coverage_id}.tif"
    out_path = os.path.join(output_dir, out_filename)
    tmp_path = os.path.join(output_dir, f"temp_{out_filename}")

    if os.path.exists(out_path):
        print(f"Skipping {out_filename}, already exists.")
        return

    url = f"https://maps.isric.org/mapserv?map=/map/{variable}.map"

    try:
        wcs = WebCoverageService(url, version='2.0.1')
        print(f"Downloading {coverage_id} native resolution (Homolosine)...")

        min_x, min_y, max_x, max_y = bbox_igy
        response = wcs.getCoverage(
            identifier=coverage_id,
            crs='http://www.opengis.net/def/crs/EPSG/0/152160',
            subsets=[('x', min_x, max_x), ('y', min_y, max_y)],
            format='image/tiff'
        )

        with open(tmp_path, 'wb') as f:
            f.write(response.read())

        print(f"Reprojecting {coverage_id} to EPSG:4326...")

        da = rioxarray.open_rasterio(tmp_path)
        da.rio.write_crs(PROJ_IGY, inplace=True)
        da_wgs84 = da.rio.reproject("EPSG:4326", resolution=0.00225)

        da_crop = da_wgs84.rio.clip_box(
            minx=bbox_wgs84["min_lon"], miny=bbox_wgs84["min_lat"],
            maxx=bbox_wgs84["max_lon"], maxy=bbox_wgs84["max_lat"]
        )
        da_crop.rio.to_raster(out_path, compress='lzw')

        da.close()
        da_wgs84.close()
        da_crop.close()
        os.remove(tmp_path)
        print(f"Saved reprojected map to {out_path}")

    except Exception as e:
        print(f"Failed to download/reproject {coverage_id}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_full_download(target_country="italy", base_data_dir="../data"):
    """
    Funzione principale da chiamare per scaricare i dati di una specifica nazione.
    Sostituisce il vecchio main() rigido.
    """
    country_name = target_country.lower()
    if country_name not in COUNTRY_BBOXES:
        print(f"Country {country_name} not found in configuration.")
        return

    print(f"\n==========================================")
    print(f"STARTING DOWNLOAD FOR COUNTRY: {country_name.upper()}")
    print(f"==========================================")

    bbox_wgs84 = COUNTRY_BBOXES[country_name]
    output_dir = os.path.join(base_data_dir, f"{country_name}/soil_data/soilgrids/data")
    os.makedirs(output_dir, exist_ok=True)

    bbox_igy = get_homolosine_bbox(
        bbox_wgs84["min_lon"], bbox_wgs84["min_lat"],
        bbox_wgs84["max_lon"], bbox_wgs84["max_lat"]
    )

    if not bbox_igy:
        print(f"Could not calculate Homolosine BBOX for {country_name}. Skipping.")
        return

    print(f"BBOX WGS84 for {country_name}: {bbox_wgs84}")
    print(f"Saving to: {output_dir}")

    for var in VARIABLES:
        for depth in DEPTHS:
            download_coverage(var, depth, bbox_igy, bbox_wgs84, output_dir)
            time.sleep(0.3)

    for depth in ["0-30"]:
        try:
            download_coverage("ocs", depth, bbox_igy, bbox_wgs84, output_dir)
        except Exception:
            pass