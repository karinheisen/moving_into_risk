#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mean annual net migration rate (per 1,000 persons/year), 2000–2015
MIXED admin boundaries -- Robinson projection -- 99% symmetric clip
DISCRETE symmetric quasi-log color scale (“blocky” legend)
"""

from math import floor, log10

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------- Config ----------------------
CSV_PATH    = "mixed_outputs/migration_counts_mixed_2000_2016_new.csv"
MIXED_GPKG  = "admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg"
MIXED_LAYER = "mixed"
OUT_PNG     = "map_mean_net_migration_rate_mixed_blocks_99clip.png"

YEAR_MIN, YEAR_MAX = 2000, 2015

ID_COL    = "mixed_id"
RATE_COL  = "netMgr_rate_per_1000"
YEAR_COL  = "year"

CRS_PROJ = "+proj=robin"

# ---------------------- Helpers ----------------------
def make_sym_quasilog_boundaries(lim_val, linthr):
    emin = floor(log10(linthr)) if linthr > 0 else -6
    emax = floor(log10(lim_val))
    pos_edges = []
    for e in range(emin, emax + 1):
        for m in (1, 2, 5):
            v = m * (10 ** e)
            if linthr <= v <= lim_val:
                pos_edges.append(v)
    pos_edges = sorted(set(pos_edges + [lim_val]))
    neg_edges = [-v for v in reversed(pos_edges)]
    boundaries = neg_edges + [-linthr, 0.0, linthr] + pos_edges
    b = np.array(sorted(set(boundaries)))
    b[0] = -lim_val
    b[-1] = lim_val
    return b


def fmt_num(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def edge_labels_clean(bnds):
    labs = []
    for i, v in enumerate(bnds):
        if i == 0 or i == len(bnds) - 1:
            val = str(int(round(v)))
            labs.append(("≤" if i == 0 else "≥") + val)
        else:
            labs.append(fmt_num(v))
    return labs


# ---------------------- Main ----------------------
def main() -> None:
    gdf = gpd.read_file(MIXED_GPKG, layer=MIXED_LAYER)
    gdf[ID_COL] = gdf[ID_COL].astype(str)

    gdf = gdf[gdf["iso3"] != "ATA"].copy() #removes Antarctica

    df = pd.read_csv(CSV_PATH)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[ID_COL] = df[ID_COL].astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce")

    df = df[(df[YEAR_COL] >= YEAR_MIN) & (df[YEAR_COL] <= YEAR_MAX)].copy()

    mean_rate = (
        df.groupby(ID_COL, as_index=False)[RATE_COL]
          .mean()
          .rename(columns={RATE_COL: "mean_rate_per_1000"})
    )

    plot_df = gdf.merge(mean_rate, on=ID_COL, how="left").to_crs(CRS_PROJ)
    gdf_proj = gdf.to_crs(CRS_PROJ)

    vals = plot_df["mean_rate_per_1000"].to_numpy()
    vals = vals[np.isfinite(vals)]

    lim = float(np.nanquantile(np.abs(vals), 0.99)) if vals.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(np.abs(vals))) if vals.size else 1.0

    linthresh = max(lim / 50.0, 1e-6)

    boundaries = make_sym_quasilog_boundaries(lim, linthresh)

    cmap = plt.get_cmap("RdBu_r", len(boundaries) - 1)
    colors = cmap(np.arange(cmap.N))
    discrete_cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=discrete_cmap.N, clip=False)

    clipped_low  = np.any(vals < boundaries[0] - 1e-12)
    clipped_high = np.any(vals > boundaries[-1] + 1e-12)
    extend_mode = (
        "both"
        if (clipped_low and clipped_high)
        else ("min" if clipped_low else ("max" if clipped_high else "neither"))
    )

    tick_positions = boundaries
    tick_labels = edge_labels_clean(boundaries)

    fig, ax = plt.subplots(figsize=(13.5, 7))

    gdf_proj.boundary.plot(ax=ax, linewidth=0.3, color="#dddddd", alpha=0.9)

    plot_df.plot(
        column="mean_rate_per_1000",
        cmap=discrete_cmap,
        linewidth=0.08,
        edgecolor="white",
        norm=norm,
        ax=ax,
    )

    minx, miny, maxx, maxy = plot_df.total_bounds
    pad_x = 0.01 * (maxx - minx)
    pad_y = 0.01 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_axis_off()
    ax.set_aspect("equal")

    sm = ScalarMappable(norm=norm, cmap=discrete_cmap)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.6)
    cbar = fig.colorbar(sm, cax=cax, extend=extend_mode)
    cbar.set_label("Net migration rate (per 1,000 per year)")
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)


if __name__ == "__main__":
    main()
