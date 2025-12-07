#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate heatwave exposure to MIXED admin boundaries (fractional).

Inputs
- Mixed boundaries GPKG layer (default: layer "mixed", columns: mixed_id, level, geometry)
- NetCDF heatwave rasters (variable: "exposure")

Outputs
- agg_mixed_fractional/heatwave_population_mixed_fractional_<tag>.csv
"""

import os
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
    "gswp3-w5e5_historical": "ISIMIP3_forKarinHeisen/hwmid-humidex_gswp3-w5e5_historical_heatwavedarea_global_annual_population_1901_2016.nc",
    "gswp3-w5e5_picontrol":  "ISIMIP3_forKarinHeisen/hwmid-humidex_gswp3-w5e5_picontrol_heatwavedarea_global_annual_population_1901_2016.nc",
    "20crv3_historical":     "ISIMIP3_forKarinHeisen/hwmid-humidex_20crv3_historical_heatwavedarea_global_annual_population_1901_2016.nc",
    "20crv3_picontrol":      "ISIMIP3_forKarinHeisen/hwmid-humidex_20crv3_picontrol_heatwavedarea_global_annual_population_1901_2016.nc",
}
MIXED_GPKG_PATH = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
MIXED_LAYER     = "mixed"
OUTPUT_DIR      = "agg_mixed_fractional"
NODATA          = -9999.0
VAR_NAME        = "exposure"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------- Helpers ----------------------
def to_multipolygon(geom):
    """Return a MultiPolygon with only polygonal parts; None if no polygons."""
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


def load_mixed(gpkg_path: str, layer: str) -> gpd.GeoDataFrame:
    adm = gpd.read_file(gpkg_path, layer=layer)
    adm = adm.set_crs("EPSG:4326") if adm.crs is None else adm.to_crs("EPSG:4326")

    # assume mixed_id and level are present
    adm = adm[adm.geometry.notnull()].copy()
    try:
        adm["geometry"] = adm.geometry.make_valid()
    except Exception:
        adm["geometry"] = adm.buffer(0)

    meta = adm[["mixed_id", "level"]].groupby("mixed_id", as_index=False).first()
    diss = adm.dissolve(by="mixed_id").reset_index()[["mixed_id", "geometry"]]
    adm = gpd.GeoDataFrame(
        meta.merge(diss, on="mixed_id"),
        geometry="geometry",
        crs="EPSG:4326",
    )

    adm["geometry"] = adm["geometry"].apply(to_multipolygon)
    adm = adm[adm.geometry.notnull()].copy()
    return adm[["mixed_id", "level", "geometry"]]


def std_spatial(da: xr.DataArray) -> xr.DataArray:
    """Standardize to (time, y, x), lon[-180,180], lat descending, NaNs for fill, CRS=EPSG:4326."""
    xdim = next((d for d in ("lon", "longitude", "x") if d in da.dims), None)
    ydim = next((d for d in ("lat", "latitude", "y") if d in da.dims), None)

    if xdim != "x" or ydim != "y":
        da = da.rename({xdim: "x", ydim: "y"})

    x = da["x"]
    if float(np.asarray(x.max())) > 180:
        da = da.assign_coords(x=((x + 180) % 360) - 180)
    da = da.sortby("x")

    y = da["y"]
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
    for f in set(fills + [-9999.0, -1.0e20, 1.0e20]):
        if np.isfinite(f):
            da = da.where(da != f)

    da = (
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        .rio.write_crs("EPSG:4326", inplace=False)
    )
    if "time" in da.dims and list(da.dims)[0] != "time":
        da = da.transpose("time", "y", "x")
    return da


def write_year_tif(da_yx: xr.DataArray, path: str):
    da_yx = da_yx.astype("float32")
    da_yx = xr.where(np.isfinite(da_yx), da_yx, np.float32(NODATA))
    da_yx = da_yx.rio.write_nodata(NODATA, inplace=False)
    da_yx.rio.to_raster(path, dtype="float32")


def aggregate_one(nc_path: str, tag: str, admin: gpd.GeoDataFrame):
    ds = xr.open_dataset(nc_path)
    hw = std_spatial(ds[VAR_NAME])

    x = hw.coords["x"].values
    y = hw.coords["y"].values
    bbox = box(float(x.min()), float(y.min()), float(x.max()), float(y.max()))
    adm = gpd.clip(admin, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326"))
    adm = adm[adm.geometry.notnull()].copy()
    try:
        adm["geometry"] = adm.geometry.make_valid()
    except Exception:
        adm["geometry"] = adm.buffer(0)
    adm["geometry"] = adm["geometry"].apply(to_multipolygon)
    adm = adm[adm.geometry.notnull()].copy()

    out_rows = []
    n_t = hw.sizes["time"] if "time" in hw.dims else 1

    for ti in range(n_t):
        year = int(pd.to_datetime(hw.time.values[ti]).year) if "time" in hw.dims else None
        arr = hw.isel(time=ti) if "time" in hw.dims else hw

        with tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".tif", delete=False) as tmp:
            tif = tmp.name
        write_year_tif(arr, tif)

        with rasterio.open(tif) as rds:
            stats = exact_extract(
                rds,
                adm,
                ["mean"],
                include_cols=["mixed_id", "level"],
                output="pandas",
            )
        os.remove(tif)

        stats = stats.rename(columns={"mean": "heatwave_mean_population"})
        if year is not None:
            stats["year"] = year
        cols = ["mixed_id", "level"] + (["year"] if year is not None else []) + ["heatwave_mean_population"]
        out_rows.append(stats[cols])

    res = pd.concat(out_rows, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, f"heatwave_population_mixed_fractional_{tag}.csv")
    res.to_csv(out_path, index=False)

# ---------------------- Main ----------------------
def main():
    admin = load_mixed(MIXED_GPKG_PATH, MIXED_LAYER)
    for tag, path in INPUT_FILES.items():
        aggregate_one(path, tag, admin)


if __name__ == "__main__":
    main()
