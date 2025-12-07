#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate WorldPop 1 km population rasters (ppp_YYYY_1km_Aggregated.tif)
to mixed admin units using area-weighted zonal sums.

For each polygon and year:
    worldpop_sum = sum(value * fractional_pixel_coverage)

Inputs
- mixed-gpkg <path>   : GeoPackage with mixed units
- mixed-layer <name>  : layer name in the GeoPackage
- worldpop-dir <dir>  : directory with ppp_YYYY_1km_Aggregated.tif files

Output
- CSV with columns: [mixed_id, level, iso3, year, worldpop_sum]
"""

import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.errors import GEOSException
from exactextract import exact_extract

MIN_YEAR, MAX_YEAR = 2000, 2016

mixed_gpkg = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
mixed_layer = "mixed"
worldpop_dir = "WorldPop_1km_gridded"
worldpop_glob = "ppp_*_1km_Aggregated.tif"
outdir = "mixed_outputs"
outcsv = "worldpop_mixed_2000_2016_new.csv"
force_input_crs = None


# ---------------------- Helpers ----------------------
def parse_year(path: str) -> Optional[int]:
    """Extract a year from the filename (limited to 2000–2016)."""
    for s in re.findall(r"(\d{4})", Path(path).name):
        y = int(s)
        if MIN_YEAR <= y <= MAX_YEAR:
            return y
    return None


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries via buffer(0)."""
    fixed = gdf.copy()
    try:
        fixed["geometry"] = fixed.geometry.buffer(0)
    except GEOSException:
        pass
    return fixed


def safe_reproject_to_raster_crs(gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to the CRS of the raster, if needed."""
    with rasterio.open(raster_path) as src:
        r_crs = src.crs
    if r_crs is None:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs != r_crs:
        gdf = gdf.to_crs(r_crs)
    return gdf


def intersect_with_raster_bounds(
    gdf: gpd.GeoDataFrame, raster_path: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (subset that intersects raster, non-overlapping remainder)."""
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        rb = box(left, bottom, right, top)
    inter_mask = gdf.intersects(rb)
    return gdf[inter_mask].copy(), gdf[~inter_mask].copy()


def _stat_from_value(v, key: str) -> float:
    """
    Extract a statistic (e.g. 'sum') from exact_extract output, which may be either:
      - {'sum': ...}
      - {'type': 'Feature', 'properties': {'sum': ...}, ...}
    """
    if v is None:
        return np.nan
    if isinstance(v, dict):
        if key in v:
            return v.get(key, np.nan)
        props = v.get("properties")
        if isinstance(props, dict):
            return props.get(key, np.nan)
    return np.nan


def aggregate_one_area_weighted(raster_path: str, mixed_attr: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Area-weighted sum via exact_extract:
    For each polygon, sum of (raster value * fractional pixel coverage within polygon).
    """
    gdf = safe_reproject_to_raster_crs(mixed_attr, raster_path)
    gdf = ensure_valid_geometries(gdf)

    gdf_hit, _gdf_miss = intersect_with_raster_bounds(gdf, raster_path)

    out = gdf[["mixed_id", "level", "iso3"]].copy()
    out["worldpop_sum"] = np.nan

    if len(gdf_hit) > 0:
        with rasterio.open(raster_path, masked=False) as src:
            vals = exact_extract(src, gdf_hit, ["sum"])
        hit_sum = np.array([_stat_from_value(v, "sum") for v in vals], dtype=float)
        out.loc[gdf_hit.index, "worldpop_sum"] = hit_sum

    return out


# ---------------------- Main ----------------------
def main() -> None:
    outdir_p = Path(outdir)
    outcsv_p = outdir_p / outcsv
    outdir_p.mkdir(parents=True, exist_ok=True)

    mixed = gpd.read_file(mixed_gpkg, layer=mixed_layer)
    if force_input_crs:
        mixed = mixed.set_crs(force_input_crs, allow_override=True)

    if "mixed_id" not in mixed.columns:
        mixed = mixed.reset_index(drop=True)
        mixed["mixed_id"] = mixed.index.astype(str)

    mixed = mixed[["mixed_id", "level", "iso3", "geometry"]].copy()
    mixed["mixed_id"] = mixed["mixed_id"].astype(str)
    mixed["level"] = mixed["level"].astype(str).str.lower()
    mixed["iso3"] = mixed["iso3"].astype(str)
    mixed = ensure_valid_geometries(mixed)
    mixed = mixed[~mixed.geometry.is_empty]

    worldpop_glob_path = str(Path(worldpop_dir) / worldpop_glob)
    files = sorted(glob.glob(worldpop_glob_path))

    frames: List[pd.DataFrame] = []
    for fpath in files:
        year = parse_year(fpath)
        if year is None:
            continue
        df = aggregate_one_area_weighted(fpath, mixed)
        df["year"] = year
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result[["mixed_id", "level", "iso3", "year", "worldpop_sum"]].sort_values(["mixed_id", "year"])
    result.to_csv(outcsv_p, index=False)


if __name__ == "__main__":
    main()
