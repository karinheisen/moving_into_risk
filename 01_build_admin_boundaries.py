#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build mixed admin boundaries

Rules:
Collapses a country to ADM-0 if ANY of these hold:
  1) no ADM-1 geometry exists
  2) mean ADM-1 area < CUT_MEAN
  3) insufficient ADM-1 UNIT coverage: fraction of ADM-1 units with any data
     falls below MIN_FRAC_UNITS_WITH_DATA.

Outputs
- admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg (layers: mixed, admin0_only)
- admin_cutoff_outputs_mean_only/countries_collapsed_mean_only_<CUT_MEAN>_new.csv
- admin_cutoff_outputs_mean_only/map_rule_mean_only_<CUT_MEAN>_new.png
"""

from pathlib import Path
import re
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Config ----------------------
ADM1_GPKG = "polyg_adm1_dataNetMgr.gpkg"  # Niva table (contains ADM1 rows; uses ADM0 when no ADM1 exists)
ISO_COL   = "iso3"      # country code
ADM1_ID   = "GID_1"     # admin-1 id (fallbacks below)
NAME_COL  = "Country"   # optional country name
EQ_EPSG   = 6933        # equal-area CRS for area calc
CUT_MEAN  = 15_000.0    # km² threshold for collapse

# Coverage thresholds
# Keep up to 20% of ADM-1 units missing; collapse if more than that are missing
MIN_FRAC_UNITS_WITH_DATA = 0.80   # require data for at least 80% of ADM-1 units

OUTDIR = Path("admin_cutoff_outputs_mean_only")
OUTDIR.mkdir(exist_ok=True)
OUT_GPKG = OUTDIR / "mixed_boundaries_new.gpkg"
OUT_PLOT = OUTDIR / f"map_rule_mean_only_{int(CUT_MEAN)}_new.png"
OUT_CSV  = OUTDIR / f"countries_collapsed_mean_only_{int(CUT_MEAN)}_new.csv"


# ---------------------- Helpers ----------------------
def coerce_adm1_id(df: pd.DataFrame, target: str, candidates=("ADM1_PCODE", "ADM1_CODE", "ID_1", "NAME_1")) -> str:
    if target in df.columns:
        df[target] = df[target].astype(str)
        return target
    for c in candidates:
        if c in df.columns:
            df[target] = df[c].astype(str)
            return target
    raise ValueError(f"Could not find an ADM1 id column; tried {target} and {candidates}")


def find_year_cols(cols, y0=2000, y1=2016):
    pat = re.compile(r"^netMgr_(\d{4})$")
    years = [c for c in map(str, cols) if (m := pat.match(c)) and y0 <= int(m.group(1)) <= y1]
    return sorted(years)


def adm1_coverage_table(adm1_gpkg: str, iso_col: str, gid1_col: str) -> pd.DataFrame:
    """
    Per-ISO3 coverage stats over ADM-1 migration data:
      - n_adm1_data: number of ADM-1 units with any row in ADM1 table
      - mean_frac_years_nonmissing: average fraction of years non-missing across its ADM-1 units
    """
    df = gpd.read_file(adm1_gpkg)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    df[iso_col]  = df[iso_col].astype(str).str.strip()
    df[gid1_col] = df[gid1_col].astype(str).str.strip()
    ycols = find_year_cols(df.columns)

    # keep rows with a valid adm1 id
    df = df[df[gid1_col].ne("")].copy()

    # per (iso3, gid1): fraction of non-missing years
    frac = 1.0 - df[ycols].isna().mean(axis=1)
    tmp = df[[iso_col, gid1_col]].copy()
    tmp["frac_years_nonmissing"] = frac

    # aggregate per ISO3
    cov = (
        tmp.groupby(iso_col, dropna=False)
        .agg(
            n_adm1_data=(gid1_col, "nunique"),
            mean_frac_years_nonmissing=("frac_years_nonmissing", "mean"),
        )
        .reset_index()
    )
    return cov


def reason_string(row: pd.Series) -> str:
    reasons = []
    if bool(row.get("rule_no_adm1_geom", False)):
        reasons.append("no ADM1 geometry")
    if bool(row.get("rule_small_mean_area", False)):
        reasons.append(f"mean ADM1 area < {int(CUT_MEAN):,} km²")
    if bool(row.get("rule_partial_adm1_coverage", False)):
        reasons.append("insufficient ADM1 unit coverage")
    return "; ".join(reasons)


# ---------------------- Main ----------------------
def main():
    g = gpd.read_file(ADM1_GPKG).to_crs(4326)

    coerce_adm1_id(g, ADM1_ID)

    g = g[g.geometry.notnull() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()

    # whitespace-safe ADM1-ID presence
    g[ADM1_ID] = g[ADM1_ID].astype(str).str.strip()
    g["has_gid1"] = g[ADM1_ID].ne("")

    # dissolve to unique ADM1 units
    g_adm1 = g.loc[g["has_gid1"], [ISO_COL, ADM1_ID, "geometry"]].copy()
    if not g_adm1.empty:
        g_adm1 = g_adm1.dissolve(by=[ISO_COL, ADM1_ID], as_index=False)
        adm1_eq = g_adm1.to_crs(EQ_EPSG)
        g_adm1["area_km2"] = adm1_eq.area / 1e6
        stats = (
            g_adm1.groupby(ISO_COL, dropna=False)["area_km2"]
            .agg(
                mean_admin1_km2="mean",
                median_admin1_km2="median",
                n_admin1="size",
            )
            .reset_index()
        )
    else:
        stats = pd.DataFrame(
            {
                ISO_COL: g[ISO_COL].unique(),
                "mean_admin1_km2": pd.NA,
                "median_admin1_km2": pd.NA,
                "n_admin1": 0,
            }
        )

    # mark countries with any ADM1 geometry present
    any_adm1 = (
        g.groupby(ISO_COL, dropna=False)["has_gid1"]
        .any()
        .reset_index()
        .rename(columns={"has_gid1": "has_any_adm1"})
    )
    stats = stats.merge(any_adm1, on=ISO_COL, how="outer")
    stats["has_any_adm1"] = stats["has_any_adm1"].fillna(False)

    # coverage from ADM1 migration table
    cov = adm1_coverage_table(ADM1_GPKG, iso_col=ISO_COL, gid1_col=ADM1_ID)
    stats = stats.merge(cov, on=ISO_COL, how="left")
    stats["n_adm1_data"] = stats["n_adm1_data"].fillna(0).astype(int)
    stats["mean_frac_years_nonmissing"] = stats["mean_frac_years_nonmissing"].fillna(0.0)

    # fraction of ADM-1 units (by geometry) that have any data rows
    def _frac_units_with_data(r):
        n_geom = r.get("n_admin1", 0)
        n_data = r.get("n_adm1_data", 0)
        if pd.isna(n_geom) or n_geom == 0:
            return 0.0
        return float(n_data) / float(n_geom)

    stats["frac_units_with_data"] = stats.apply(_frac_units_with_data, axis=1)
    stats["pct_units_missing"] = 100.0 * (1.0 - stats["frac_units_with_data"])

    # rule booleans
    stats["rule_no_adm1_geom"] = ~stats["has_any_adm1"]
    stats["rule_small_mean_area"] = (stats["mean_admin1_km2"] < CUT_MEAN).fillna(False)

    # Coverage rule only checks unit coverage. Year coverage is not used to collapse.
    stats["rule_partial_adm1_coverage"] = stats["frac_units_with_data"] < MIN_FRAC_UNITS_WITH_DATA

    # collapse decision
    stats["flag_collapse"] = (
        stats["rule_no_adm1_geom"]
        | stats["rule_small_mean_area"]
        | stats["rule_partial_adm1_coverage"]
    )

    # human-readable reason
    stats["collapse_reason"] = stats.apply(reason_string, axis=1)
    stats.loc[~stats["flag_collapse"], "collapse_reason"] = pd.NA

    # tag rows in full layer
    g = g.merge(
        stats[
            [
                ISO_COL,
                "flag_collapse", "collapse_reason", "rule_no_adm1_geom", "rule_small_mean_area", "rule_partial_adm1_coverage",
                "frac_units_with_data", "mean_frac_years_nonmissing", "pct_units_missing",
            ]
        ],
        on=ISO_COL,
        how="left",
    )
    g["flag_collapse"] = g["flag_collapse"].fillna(True)  # conservative to ADM0 if uncertain

    # ADM0: countries collapsed
    admin0 = g[g["flag_collapse"]].dissolve(by=ISO_COL, as_index=False)
    admin0 = admin0[[ISO_COL, "geometry"]].merge(
        stats[
            [
                ISO_COL,
                "collapse_reason", "rule_no_adm1_geom". "rule_small_mean_area", "rule_partial_adm1_coverage",
                "frac_units_with_data", "mean_frac_years_nonmissing", "pct_units_missing",
            ]
        ],
        on=ISO_COL,
        how="left",
    )
    admin0["level"] = "admin0"

    # ADM1: countries not collapsed (use dissolved unique units)
    g_adm1 = g_adm1.merge(stats[[ISO_COL, "flag_collapse"]], on=ISO_COL, how="left")
    admin1_keep = g_adm1[~g_adm1["flag_collapse"]][[ISO_COL, ADM1_ID, "geometry"]].copy()
    admin1_keep["level"] = "admin1"
    # fill diagnostics with neutral values for ADM1 units
    admin1_keep["collapse_reason"] = pd.NA
    for c in ["rule_no_adm1_geom", "rule_small_mean_area", "rule_partial_adm1_coverage"]:
        admin1_keep[c] = False
    admin1_keep["frac_units_with_data"] = pd.NA
    admin1_keep["mean_frac_years_nonmissing"] = pd.NA
    admin1_keep["pct_units_missing"] = pd.NA

    # combine
    mixed = pd.concat(
        [
            admin0[
                [
                    ISO_COL,
                    "level", "geometry", "collapse_reason", "rule_no_adm1_geom", "rule_small_mean_area",
                    "rule_partial_adm1_coverage", "frac_units_with_data", "mean_frac_years_nonmissing", "pct_units_missing",
                ]
            ],
            admin1_keep[
                [
                    ISO_COL,
                    ADM1_ID,
                    "level", "geometry", "collapse_reason", "rule_no_adm1_geom", "rule_small_mean_area",
                    "rule_partial_adm1_coverage", "frac_units_with_data", "mean_frac_years_nonmissing", "pct_units_missing",
                ]
            ],
        ],
        ignore_index=True,
    )
    mixed = gpd.GeoDataFrame(mixed, geometry="geometry", crs=g.crs).reset_index(drop=True)
    mixed["mixed_id"] = mixed.index.astype(str)

    # plot
    fig, ax = plt.subplots(figsize=(15, 8))
    mixed.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.25)
    admin0.boundary.plot(ax=ax, color="red", linewidth=0.7, label="Shown at admin-0")
    ax.set_title(
        f"Admin boundaries used in study\n"
        f"Rules: no ADM1; mean ADM1 area < {int(CUT_MEAN):,} km²; insufficient ADM1 unit coverage",
        fontsize=16,
    )
    ax.legend()
    ax.axis("off")
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # collapsed countries CSV (with diagnostics)
    collapsed = stats[stats["flag_collapse"]].copy()
    if NAME_COL in g.columns:
        collapsed = (
            collapsed.merge(
                g[[ISO_COL, NAME_COL]].drop_duplicates(),
                on=ISO_COL,
                how="left",
            )
            .drop_duplicates([ISO_COL])
        )

    cols_order = [
        ISO_COL,
        NAME_COL if NAME_COL in collapsed.columns else None,
        "n_admin1", "n_adm1_data", "mean_admin1_km2", "median_admin1_km2", "has_any_adm1", "frac_units_with_data", "pct_units_missing",
        "mean_frac_years_nonmissing", "rule_no_adm1_geom", "rule_small_mean_area", "rule_partial_adm1_coverage", "flag_collapse", "collapse_reason",
    ]
    cols_order = [c for c in cols_order if c is not None]
    collapsed = collapsed[cols_order].sort_values([ISO_COL]).reset_index(drop=True)
    collapsed.to_csv(OUT_CSV, index=False)

    # save
    mixed.to_file(OUT_GPKG, layer="mixed", driver="GPKG")
    admin0.to_file(OUT_GPKG, layer="admin0_only", driver="GPKG")

    print("Saved:")
    print(f" - {OUT_PLOT}")
    print(f" - {OUT_CSV}")
    print(f" - {OUT_GPKG} (layers: mixed, admin0_only)")

    print("\nCounts by level (mixed):")
    print(mixed["level"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
