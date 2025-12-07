#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four maps per model on mixed boundaries

A) Net IN-migration areas (mig > 0): color = exposure sign (red ↑, blue ↓), shade = |exposure trend|
B) Net OUT-migration areas (mig < 0): color = exposure sign (red ↑, blue ↓), shade = |exposure trend|
C) Exposure INCREASE areas (trend > 0): color = migration sign (purple in-mig, green out-mig), shade = |migration count|
D) Exposure DECREASE areas (trend < 0): color = migration sign (purple in-mig, green out-mig), shade = |migration count|
"""

import os
from typing import List, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------- Config ----------------------
GLOBAL_YEAR_MIN = 1901
GLOBAL_YEAR_MAX = 2015
EARLY_START, EARLY_END     = 1901, 1930
RECENT_START, RECENT_END   = 1986, 2015
PRESENT_START, PRESENT_END = 2000, 2015

MIXED_GPKG  = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
MIXED_LAYER = "mixed"
MIG_COUNTS_CSV = "mixed_outputs/migration_counts_mixed_2000_2016_new.csv"

DATASETS: List[Dict] = [
    {"file": "agg_mixed_fractional/heatwave_population_mixed_fractional_gswp3-w5e5_historical.csv",
     "column": "heatwave_mean_population", "slug": "heatwaves_gswp3-w5e5_historical"},
    {"file": "agg_mixed_fractional/heatwave_population_mixed_fractional_20crv3_historical.csv",
     "column": "heatwave_mean_population", "slug": "heatwaves_20crv3_historical"},
    {"file": "agg_mixed_fractional/cropfailed_population_mixed_fractional_LPJmL_gswp3-w5e5_historical.csv",
     "column": "cropfailed_mean_population", "slug": "cropfail_lpjml_gswp3-w5e5_historical"},
    {"file": "agg_mixed_fractional/cropfailed_population_mixed_fractional_EPIC-IIASA_gswp3-w5e5_historical.csv",
     "column": "cropfailed_mean_population", "slug": "cropfail_epic-iiasa_gswp3-w5e5_historical"},
    {"file": "agg_mixed_fractional/wildfire_population_mixed_fractional_lpjml_gswp3-w5e5_historical.csv",
     "column": "wildfire_mean_population", "slug": "wildfire_lpjml_gswp3-w5e5_historical"},
    {"file": "agg_mixed_fractional/wildfire_population_mixed_fractional_classic_gswp3-w5e5_historical.csv",
     "column": "wildfire_mean_population", "slug": "wildfire_classic_gswp3-w5e5_historical"},
]

COLOR_SCALE = 0.7
CRS_LATLON  = "EPSG:4326"
CRS_PROJ    = "+proj=robin"

OUT_DIR = "maps_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- Helpers ----------------------
def _ensure_str_id(df: pd.DataFrame, col: str = "mixed_id") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out

def _symmetric_limits(series: pd.Series) -> float:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    vmax = float(np.nanmax(np.abs(s))) if not s.empty else 1.0
    return vmax if vmax > 0 else 1.0

def _savefig(fig, fname: str):
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches="tight")
    print(f"[saved] {fname}")


def load_admin_mixed(path: str, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer).to_crs(CRS_LATLON)
    gdf = gdf[gdf.geometry.notnull()].copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    meta = gdf[["mixed_id"]].drop_duplicates()
    diss = gdf.dissolve(by="mixed_id", as_index=False)[["mixed_id", "geometry"]]
    out  = meta.merge(diss, on="mixed_id", how="left")
    out  = _ensure_str_id(out, "mixed_id")
    out  = gpd.GeoDataFrame(out, geometry="geometry", crs=CRS_LATLON)

    bbox = out.bounds
    out = out[bbox["miny"] > -60].copy() # removes regions south of 60°S (Antarctica)

    out = out.to_crs(CRS_PROJ) #projects to Robinson
    return out

def load_migration_mean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_str_id(df, "mixed_id")
    ccol = [c for c in ["netMgr_count", "mig_count", "net_migration_count"] if c in df.columns][0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[ccol] = pd.to_numeric(df[ccol], errors="coerce")
    m = (df[(df["year"] >= PRESENT_START) & (df["year"] <= PRESENT_END)]
         .groupby("mixed_id", as_index=False)[ccol].mean()
         .rename(columns={ccol: "mig_mean_2000_2015"}))
    return m

def load_exposure(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_str_id(df, "mixed_id")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["mixed_id", "year", col]]

def exposure_trend(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def _avg(sub):
        return sub.groupby("mixed_id", as_index=False)[col].mean().rename(columns={col: "val"})
    early  = _avg(df[(df["year"] >= EARLY_START) & (df["year"] <= EARLY_END)])
    recent = _avg(df[(df["year"] >= RECENT_START) & (df["year"] <= RECENT_END)])
    t = early.merge(recent, on="mixed_id", how="inner", suffixes=("_early", "_recent"))
    t["exposure_trend"] = t["val_recent"] - t["val_early"]
    return t[["mixed_id", "exposure_trend"]]


def _add_vertical_cbar(fig, ax, norm, label: str, tick_fmt: str | None = None):
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])

    cax = inset_axes(
        ax,
        width="2.2%",
        height="80%",
        loc="center right",
        borderpad=0.8,
        bbox_to_anchor=(0, 0.05, 1, 1),
        bbox_transform=ax.transAxes
    )

    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="both", extendfrac=0.05)
    cbar.set_label(label)

    if tick_fmt:
        cbar.ax.yaxis.set_major_formatter(StrMethodFormatter(f'{{x:,{tick_fmt}}}'))
    elif "Exposure" in label:
        cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    else:
        cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    cbar.outline.set_linewidth(0.8)
    return cbar

def plot_migration_split(admin: gpd.GeoDataFrame, merged: pd.DataFrame, slug: str):
    g = admin.merge(merged, on="mixed_id", how="left")
    vmax = _symmetric_limits(g["exposure_trend"]) * COLOR_SCALE
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    exposure_tick_fmt = ".2f" if slug.startswith("cropfail_") else None

    masks = [g["mig_mean_2000_2015"] > 0, g["mig_mean_2000_2015"] < 0]
    suffix = ["INmig_exposure_trend", "OUTmig_exposure_trend"]
    for m, sfx in zip(masks, suffix):
        fig, ax = plt.subplots(figsize=(10, 7))
        admin.boundary.plot(ax=ax, linewidth=0.2, color="#999")
        g[m & g["exposure_trend"].notna()].plot(
            ax=ax, column="exposure_trend", cmap="RdBu_r", norm=norm, linewidth=0
        )
        ax.set_axis_off()
        _add_vertical_cbar(fig, ax, norm, "Exposure change", tick_fmt=exposure_tick_fmt)
        plt.tight_layout()
        _savefig(fig, f"{slug}_{sfx}.png")
        plt.close(fig)

def plot_exposure_split(admin: gpd.GeoDataFrame, merged: pd.DataFrame, slug: str):
    g = admin.merge(merged, on="mixed_id", how="left")
    vmax = _symmetric_limits(g["mig_mean_2000_2015"]) * COLOR_SCALE
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    masks = [g["exposure_trend"] > 0, g["exposure_trend"] < 0]
    suffix = ["EXPincrease_migration", "EXPdecrease_migration"]
    for m, sfx in zip(masks, suffix):
        fig, ax = plt.subplots(figsize=(10, 7))
        admin.boundary.plot(ax=ax, linewidth=0.2, color="#999")
        g[m & g["mig_mean_2000_2015"].notna()].plot(
            ax=ax, column="mig_mean_2000_2015", cmap="RdBu_r", norm=norm, linewidth=0
        )
        ax.set_axis_off()
        _add_vertical_cbar(fig, ax, norm, "Net migration")
        plt.tight_layout()
        _savefig(fig, f"{slug}_{sfx}.png")
        plt.close(fig)

# ---------------------- Main ----------------------
def main():
    admin = load_admin_mixed(MIXED_GPKG, MIXED_LAYER)
    mig = load_migration_mean(MIG_COUNTS_CSV)
    for spec in DATASETS:
        if not os.path.exists(spec["file"]):
            continue
        exp = load_exposure(spec["file"], spec["column"])
        tr = exposure_trend(exp, spec["column"])
        merged = mig.merge(tr, on="mixed_id", how="inner")
        if not merged.empty:
            plot_migration_split(admin, merged, spec["slug"])
            plot_exposure_split(admin, merged, spec["slug"])

if __name__ == "__main__":
    main()
