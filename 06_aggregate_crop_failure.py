#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate crop-failure population exposure to MIXED admin boundaries (fractional).

Inputs
- Mixed boundaries GPKG layer (default: layer "mixed", columns: mixed_id, level, geometry)
- NetCDF crop-failure rasters (variable: "exposure")

Outputs
- agg_mixed_fractional/cropfailed_population_mixed_fractional_<tag>.csv
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import rasterio
from shapely.geometry import box, MultiPolygon
from exactextract import exact_extract

# ---------------------- Config ----------------------
INPUT_FILES = {
    "LPJmL_gswp3-w5e5_historical":      "ISIMIP3_forKarinHeisen/LPJmL_gswp3-w5e5_historical_cropfailedarea_global_annual_population_1901_2016.nc",
    "EPIC-IIASA_gswp3-w5e5_historical": "ISIMIP3_forKarinHeisen/EPIC-IIASA_gswp3-w5e5_historical_cropfailedarea_global_annual_population_1901_2016.nc",
}
MIXED_GPKG_PATH = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
MIXED_LAYER     = "mixed"
OUTPUT_DIR      = "agg_mixed_fractional"
VAR_NAME        = "exposure"
NODATA          = -9999.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------- Helpers ----------------------
def to_multipolygon(geom):
    """Coerce geometry to MultiPolygon; drop non-area types."""
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "Polygon":
        return MultiPolygon([geom])
    if gt == "MultiPolygon":
        return geom
    if gt == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(list(g.geoms))
        return MultiPolygon(polys) if polys else None
    return None


def load_mixed(path: str, layer: str) -> gpd.GeoDataFrame:
    adm = gpd.read_file(path, layer=layer)
    adm = adm.to_crs("EPSG:4326") if adm.crs else adm.set_crs("EPSG:4326")

    adm = adm.dropna(subset=["geometry"]).copy()
    try:
        adm["geometry"] = adm.geometry.make_valid()
    except Exception:
        adm["geometry"] = adm.buffer(0)

    meta = adm[["mixed_id", "level"]].groupby("mixed_id", as_index=False).first()
    geom = adm.dissolve(by="mixed_id").reset_index()[["mixed_id", "geometry"]]
    adm = gpd.GeoDataFrame(
        meta.merge(geom, on="mixed_id"),
        geometry="geometry",
        crs="EPSG:4326",
    )

    adm["geometry"] = adm["geometry"].apply(to_multipolygon)
    return adm.dropna(subset=["geometry"])[["mixed_id", "level", "geometry"]]


def std_spatial(da: xr.DataArray) -> xr.DataArray:
    """
    Rename to x,y; wrap lon to [-180,180]; sort x; flip y to descending;
    strip common fill; set CRS; transpose to (time,y,x).
    """
    xdim = next((d for d in ("lon", "longitude", "x") if d in da.dims), None)
    ydim = next((d for d in ("lat", "latitude", "y") if d in da.dims), None)

    if (xdim, ydim) != ("x", "y"):
        da = da.rename({xdim: "x", ydim: "y"})

    x = da["x"]
    y = da["y"]
    if float(np.asarray(x.max())) > 180:
        da = da.assign_coords(x=((x + 180) % 360) - 180)
    da = da.sortby("x")
    if float(y[0]) < float(y[-1]):
        da = da.reindex(y=y[::-1])

    fills = []
    for src in (da.attrs, da.encoding):
        for key in ("_FillValue", "missing_value"):
            if key in src:
                try:
                    fills.append(float(src[key]))
                except Exception:
                    pass
    for f in set(fills + [-9999.0, -1e20, 1e20]):
        if np.isfinite(f):
            da = da.where(da != f)

    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False).rio.write_crs(
        "EPSG:4326", inplace=False
    )
    if "time" in da.dims and list(da.dims)[0] != "time":
        da = da.transpose("time", "y", "x")
    return da


def write_tif(da_yx: xr.DataArray, path: str):
    da_yx = da_yx.astype("float32")
    da_yx = xr.where(np.isfinite(da_yx), da_yx, np.float32(NODATA))
    da_yx = da_yx.rio.write_nodata(NODATA, inplace=False)
    da_yx.rio.to_raster(path, dtype="float32")


def build_year(time_coord: xr.DataArray, nc_path: str) -> list[int]:
    """
    Minimal time → year mapping.
    Priority:
      1) CF units with 'since' (years since YYYY-..)
      2) Filename base-year + integer offsets (e.g., *_1901_2016.nc)
      3) Absolute years as-is
    """
    vals = np.asarray(time_coord.values)
    units = (time_coord.attrs.get("units", "") or "").lower()

    if "since" in units:
        try:
            base = units.split("since", 1)[1].strip()
            base_year = int(base[:4])
            return [int(base_year + float(v)) for v in vals]
        except Exception:
            pass

    m = re.search(r"_(\d{4})_(\d{4})\.nc$", os.path.basename(nc_path))
    if m and np.isfinite(vals).all():
        base_year = int(m.group(1))
        return [int(base_year + float(v)) for v in vals]

    return [int(v) for v in vals]


def aggregate_one(nc_path: str, tag: str, admin: gpd.GeoDataFrame):
    ds = xr.open_dataset(nc_path, decode_times=True)
    da = std_spatial(ds[VAR_NAME])

    x = da.coords["x"].values
    y = da.coords["y"].values
    bbox = box(float(x.min()), float(y.min()), float(x.max()), float(y.max()))
    adm = gpd.clip(admin, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326"))
    adm = adm.dropna(subset=["geometry"]).copy()
    try:
        adm["geometry"] = adm.geometry.make_valid()
    except Exception:
        adm["geometry"] = adm.buffer(0)
    adm["geometry"] = adm["geometry"].apply(to_multipolygon)
    adm = adm.dropna(subset=["geometry"]).copy()

    years = build_year(da.time, nc_path)
    tids = list(range(len(years)))

    out = []
    for ti, year in zip(tids, years):
        arr = da.isel(time=ti)
        with tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".tif", delete=False) as tmp:
            tif = tmp.name
        write_tif(arr, tif)

        with rasterio.open(tif) as rds:
            stats = exact_extract(
                rds,
                adm,
                ["mean"],
                include_cols=["mixed_id", "level"],
                output="pandas",
            )
        os.remove(tif)

        stats = stats.rename(columns={"mean": "cropfailed_mean_population"})
        stats["year"] = int(year)
        out.append(stats[["mixed_id", "level", "year", "cropfailed_mean_population"]])

    res = pd.concat(out, ignore_index=True)
    res.to_csv(
        os.path.join(OUTPUT_DIR, f"cropfailed_population_mixed_fractional_{tag}.csv"),
        index=False,
    )

# ---------------------- Main ----------------------
def main():
    admin = load_mixed(MIXED_GPKG_PATH, MIXED_LAYER)
    for tag, path in INPUT_FILES.items():
        aggregate_one(path, tag, admin)


if __name__ == "__main__":
    main()
