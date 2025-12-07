#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map: Average WorldPop (2000–2015) on MIXED polygons
- Linear scale, 0 = white -> yellow -> red
- Clip at 95th percentile with arrow on high end
- Vertical colorbar on the right (labeled in millions)
- Antarctica cropped, Robinson projection
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ---------------------- Config ----------------------
CSV_PATH     = "mixed_outputs/worldpop_mixed_2000_2016_new.csv"
MIXED_GPKG   = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
MIXED_LAYER  = "mixed"
YEAR_MIN, YEAR_MAX = 2000, 2015
CRS_PROJ     = "+proj=robin"
OUT_PNG      = "map_worldpop_mean_2000_2015.png"

CLIP_PERCENTILE = 98
UNIT_MODE = "millions"   # "auto" | "raw" | "thousands" | "millions" | "billions"

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
    # remove Antarctica (centroid below -60°)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.centroid.y > -60].copy()
    return gdf

def human_unit(vmax, mode="auto"):
    if mode == "raw":       return 1.0, "Population"
    if mode == "thousands": return 1e3, "Population (thousands)"
    if mode == "millions":  return 1e6, "Population (millions)"
    if mode == "billions":  return 1e9, "Population (billions)"
    if vmax >= 1e9: return 1e9, "Population (billions)"
    if vmax >= 1e6: return 1e6, "Population (millions)"
    if vmax >= 1e3: return 1e3, "Population (thousands)"
    return 1.0, "Population"

def load_worldpop(csv_path):
    df = pd.read_csv(csv_path)
    df["mixed_id"] = df["mixed_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    val_col = "worldpop_sum"
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    val_df = (
        df.groupby("mixed_id", as_index=False)[val_col]
          .mean()
          .rename(columns={val_col: "pop_mean"})
    )
    return val_df

def limits(series, pct=98):
    vals = series.to_numpy()
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = 0.0
    vmax = float(np.nanpercentile(vals, pct))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmin, vmax

def plot_map(gdf_mixed, val_df, out_png):
    merged = gdf_mixed.merge(val_df, on="mixed_id", how="left").to_crs(CRS_PROJ)
    outline = merged.copy()
    outline["geometry"] = outline.geometry.boundary

    vmin, vmax = limits(merged["pop_mean"], CLIP_PERCENTILE)
    unit_div, cbar_label = human_unit(vmax, UNIT_MODE)

    fig, ax = plt.subplots(figsize=(13.5, 7.0))

    # darker outlines
    outline.plot(ax=ax, linewidth=0.35, color="#888888", alpha=1.0)

    merged.plot(
        column="pop_mean",
        linewidth=0.08,
        edgecolor="white",
        cmap=CMAP,
        vmin=vmin, vmax=vmax,
        ax=ax,
        missing_kwds={"color": "lightgrey", "hatch": "///"},
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP)
    sm.set_clim(vmin, vmax)
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.035,
        pad=0.02,
        shrink=0.9,
        extend="max",
    )

    def fmt(x, _pos):
        return f"{x / unit_div:,.0f}"

    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    cbar.set_label(cbar_label, labelpad=6)

    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------- Main ----------------------
def main():
    gdf_mixed = load_mixed(MIXED_GPKG, MIXED_LAYER)
    val_df = load_worldpop(CSV_PATH)
    plot_map(gdf_mixed, val_df, OUT_PNG)

if __name__ == "__main__":
    main()
