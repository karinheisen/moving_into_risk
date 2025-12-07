#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maps: Population exposure (mean 2000–2015) on MIXED polygons
Hazards:
  - Heatwaves (GSWP3-W5E5)
  - Wildfires (LPJmL)
  - Crop failures (LPJmL)
Projection: Robinson
Color: linear white -> yellow -> red (0 = white, 99 % clip)
Layout: no titles, vertical colorbar “Population exposed” on the right, Antarctica cropped
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------- Config ----------------------
MIXED_GPKG  = "admin_cutoff_outputs_mean_only/mixed_boundaries_new_nov.gpkg"
MIXED_LAYER = "mixed"
CRS_PROJ    = "+proj=robin"
YEAR_MIN, YEAR_MAX = 2000, 2015

HAZARDS = [
    {
        "name": "heatwaves",
        "csv":  "agg_mixed_fractional/heatwave_population_mixed_fractional_gswp3-w5e5_historical_nov.csv",
        "value_col": "heatwave_mean_population",
        "out_png": "map_heatwave_population_robinson_white0_gswp3-w5e5_historical_nov.png",
    },
    {
        "name": "wildfires_lpjml",
        "csv":  "agg_mixed_fractional/wildfire_population_mixed_fractional_lpjml_gswp3-w5e5_historical_nov.csv",
        "value_col": "wildfire_mean_population",
        "out_png": "map_wildfire_population_robinson_white0_lpjml_gswp3-w5e5_historical_nov.png",
    },
    {
        "name": "cropfailures_lpjml",
        "csv":  "agg_mixed_fractional/cropfailed_population_mixed_fractional_LPJmL_gswp3-w5e5_historical_nov.csv",
        "value_col": "cropfailed_mean_population",
        "out_png": "map_cropfailed_population_robinson_white0_LPJmL_gswp3-w5e5_historical_nov.png",
    },
]

# white -> yellow -> orange -> red
COLORS = [
    (1.0, 1.0, 1.0),
    (1.0, 0.95, 0.6),
    (1.0, 0.75, 0.2),
    (1.0, 0.4, 0.1),
    (0.7, 0.0, 0.0),
]
CMAP = LinearSegmentedColormap.from_list("heat_white_yellow_red", COLORS, N=256)

# ---------------------- Helpers ----------------------
def load_mixed(path, layer):
    gdf = gpd.read_file(path, layer=layer)
    gdf["mixed_id"] = gdf["mixed_id"].astype(str)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.centroid.y > -60].copy()
    return gdf


def load_csv(csv_path, value_col):
    df = pd.read_csv(csv_path)
    df["mixed_id"] = df["mixed_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    return (
        df.groupby("mixed_id", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "val_mean"})
    )


def limits(series):
    vals = series.to_numpy()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0, 1
    return 0, float(np.nanpercentile(vals, 99))


def plot_map(gdf_mixed, val_df, out_png):
    merged = gdf_mixed.merge(val_df, on="mixed_id", how="left").to_crs(CRS_PROJ)
    outline = merged.copy()
    outline["geometry"] = outline.geometry.boundary

    vmin, vmax = limits(merged["val_mean"])
    fig, ax = plt.subplots(figsize=(13.5, 7.0))

    outline.plot(ax=ax, linewidth=0.4, color="#999999", alpha=0.9)

    merged.plot(
        column="val_mean",
        linewidth=0.08,
        edgecolor="white",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        missing_kwds={"color": "lightgrey", "hatch": "///"},
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP)
    sm.set_clim(vmin, vmax)

    fig.tight_layout()

    bbox = ax.get_position()
    cbar_height = bbox.height * 0.8
    cbar_y0 = bbox.y0 + (bbox.height - cbar_height) / 2
    cbar_width = 0.02
    cbar_x0 = bbox.x1 + 0.005

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_height])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Population exposed", labelpad=6)

    ax.set_axis_off()
    ax.set_aspect("equal")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------- Main ----------------------
def main():
    gdf = load_mixed(MIXED_GPKG, MIXED_LAYER)
    for hz in HAZARDS:
        val_df = load_csv(hz["csv"], hz["value_col"])
        plot_map(gdf, val_df, hz["out_png"])


if __name__ == "__main__":
    main()
