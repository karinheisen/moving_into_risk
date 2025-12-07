#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate net-migration rates (per 1,000 people per year) to MIXED admin boundaries (2000–2016).

Rules
- If mixed.level == "admin1": join ADM1 rates by GID_1.
- If mixed.level == "admin0": use ADM0 rate by iso3.
- Duplicates at ADM1/ADM0 are allowed only when rows are identical across all year columns; otherwise abort.

Inputs
- <mixed_gpkg>: layer "mixed" (columns: mixed_id [str], level ["admin1"/"admin0"], iso3 [str], GID_1 [str])
- <adm1_gpkg>: ADM1 table with GID_1 + netMgr_YYYY columns
- <adm0_gpkg>: ADM0 table with iso3 + netMgr_YYYY columns

Output
- <out_csv>: columns [mixed_id, year, netMgr_rate_per_1000]
"""

import re
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd

# ---------------------- Helpers ----------------------
def find_year_cols(cols: List[str], y0: int, y1: int) -> List[str]:
    pat = re.compile(r"^netMgr_(\d{4})$")
    return sorted(
        c for c in map(str, cols)
        if (m := pat.match(c)) and y0 <= int(m.group(1)) <= y1
    )


def signature_series(df: pd.DataFrame, year_cols: List[str]) -> pd.Series:
    # Represent NaN as None to treat identical NaN patterns as equal
    return df[year_cols].apply(
        lambda r: tuple(None if pd.isna(v) else v for v in r.values.tolist()),
        axis=1,
    )


def dedupe_by_exact(df: pd.DataFrame, key: str, year_cols: List[str]) -> pd.DataFrame:
    """Keep one row per key ONLY if duplicates are exactly identical across all year columns; else raise."""
    df2 = df[[key] + year_cols].copy()
    if not df2.duplicated(subset=[key]).any():
        return df2.drop_duplicates(subset=[key])

    df2["_sig"] = signature_series(df2, year_cols)
    nunq = df2.groupby(key)["_sig"].nunique(dropna=False)
    conflicts = nunq[nunq > 1]
    if not conflicts.empty:
        examples = ", ".join(map(str, conflicts.index[:5]))
        raise ValueError(
            f"Conflicting duplicates for {key}: {len(conflicts)} keys (e.g., {examples}). "
            "Source tables must be uniquely keyed or exactly identical per key."
        )
    return df2.drop_duplicates(subset=[key]).drop(columns=["_sig"])


def melt_years(df: pd.DataFrame, id_col: str, year_cols: List[str], value_name: str) -> pd.DataFrame:
    long = df[[id_col] + year_cols].melt(
        id_vars=[id_col], value_vars=year_cols, var_name="var", value_name=value_name
    )
    long["year"] = long["var"].str.replace("netMgr_", "", regex=False).astype(int)
    return long.drop(columns=["var"])


def safe_concat(parts: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    parts = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

# ---------------------- Main ----------------------
def main() -> None:
    y0, y1 = 2000, 2016
    outdir = Path("mixed_outputs")
    outcsv = outdir / "migration_mixed_rate_2000_2016_new.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load mixed attributes
    mixed = gpd.read_file("admin_cutoff_outputs_mean_only/mixed_boundaries_new.gpkg", layer="mixed")
    if "geometry" in mixed.columns:
        mixed = mixed.drop(columns=["geometry"])
    if "mixed_id" not in mixed.columns:
        mixed = mixed.reset_index(drop=True)
        mixed["mixed_id"] = mixed.index.astype(str)

    required = ("mixed_id", "level", "iso3", "GID_1")
    mixed = mixed[list(required)].copy()
    mixed["mixed_id"] = mixed["mixed_id"].astype(str)
    mixed["level"] = mixed["level"].astype(str).str.lower()
    mixed["iso3"] = mixed["iso3"].astype(str)
    mixed["GID_1"] = mixed["GID_1"].astype(str)

    mix1 = mixed[mixed["level"] == "admin1"].copy()
    mix0 = mixed[mixed["level"] == "admin0"].copy()

    # Load ADM1 / ADM0 tables
    adm1 = gpd.read_file("polyg_adm1_dataNetMgr.gpkg")
    if "geometry" in adm1.columns:
        adm1 = adm1.drop(columns=["geometry"])
    ycols1 = find_year_cols(list(adm1.columns), y0, y1)
    adm1_uni = dedupe_by_exact(adm1[["GID_1"] + ycols1], "GID_1", ycols1)

    adm0 = gpd.read_file("polyg_adm0_dataNetMgr.gpkg")
    if "geometry" in adm0.columns:
        adm0 = adm0.drop(columns=["geometry"])
    ycols0 = find_year_cols(list(adm0.columns), y0, y1)
    adm0_uni = dedupe_by_exact(adm0[["iso3"] + ycols0], "iso3", ycols0)

    # A) admin1 rows: join by GID_1
    a = pd.DataFrame(columns=["mixed_id", "year", "netMgr_rate_per_1000"])
    if not mix1.empty:
        mix1_j = mix1.dropna(subset=["GID_1"]).drop_duplicates(subset=["GID_1"], keep="first")
        m1 = mix1_j.merge(adm1_uni, on="GID_1", how="left", validate="1:1")
        a = melt_years(m1, "mixed_id", ycols1, "netMgr_rate_per_1000")[
            ["mixed_id", "year", "netMgr_rate_per_1000"]
        ]

    # B) admin0 rows: map by iso3
    b = pd.DataFrame(columns=["mixed_id", "year", "netMgr_rate_per_1000"])
    if not mix0.empty:
        iso_map = mix0[["iso3", "mixed_id"]].drop_duplicates()
        m0 = iso_map.merge(adm0_uni, on="iso3", how="left", validate="m:1")
        b = melt_years(m0, "mixed_id", ycols0, "netMgr_rate_per_1000")[
            ["mixed_id", "year", "netMgr_rate_per_1000"]
        ]

    # Combine and finalize
    out = safe_concat([a, b], ["mixed_id", "year", "netMgr_rate_per_1000"])
    out = out.dropna(subset=["mixed_id", "year"]).copy()
    out["mixed_id"] = out["mixed_id"].astype(str)
    out["year"] = out["year"].astype(int)
    out = out[(out["year"] >= y0) & (out["year"] <= y1)]
    out = out.sort_values(["mixed_id", "year"]).reset_index(drop=True)

    expected_years = (y1 - y0 + 1)

    out.to_csv(outcsv, index=False)
    print(f"Saved: {outcsv}  ({out['mixed_id'].nunique():,} mixed units × {expected_years} years)")


if __name__ == "__main__":
    main()
