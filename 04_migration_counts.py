#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute migration counts for mixed units by multiplying
per-1000 migration rates with WorldPop population (per mixed_id, per year).

Inputs
- MIGRATION_CSV : long table with columns [mixed_id, year, netMgr_rate_per_1000]
- WORLDPOP_CSV  : long table with columns [mixed_id, year, worldpop_sum]

Output
- OUTPUT_CSV    : long table with rate, population_thousands, and netMgr_count

If WorldPop values are already in thousands, set POP_IS_THOUSANDS = True.
"""

import os
import sys
import pandas as pd

# ---------------------- Config ----------------------
MIGRATION_CSV = "mixed_outputs/migration_mixed_rate_2000_2016_new.csv"
WORLDPOP_CSV  = "mixed_outputs/worldpop_mixed_2000_2016_new.csv"
OUTPUT_CSV    = "mixed_outputs/migration_counts_mixed_2000_2016_new.csv"

ID_COL           = "mixed_id"
YEAR_COL         = "year"
RATE_COL         = "netMgr_rate_per_1000"
WPOP_VALUE_COL   = "worldpop_sum"           # persons per mixed_id-year
POP_IS_THOUSANDS = False                    # If WorldPop values are already in thousands, set POP_IS_THOUSANDS = True.

YEAR_MIN, YEAR_MAX = 2000, 2016

# ---------------------- Helpers ----------------------
def _to_thousands(series: pd.Series) -> pd.Series:
    """Convert population series to thousands if needed."""
    return series if POP_IS_THOUSANDS else series / 1000.0

# ---------------------- Main ----------------------
def main(mig_path=None, wpop_path=None, out_path=None):
    mig_path = mig_path or MIGRATION_CSV
    wpop_path = wpop_path or WORLDPOP_CSV
    out_path = out_path or OUTPUT_CSV

    # Load migration (long)
    mig = pd.read_csv(mig_path)
    mig[RATE_COL] = pd.to_numeric(mig[RATE_COL], errors="coerce")
    mig = mig[(mig[YEAR_COL] >= YEAR_MIN) & (mig[YEAR_COL] <= YEAR_MAX)].copy()

    # Load WorldPop (long)
    wpop = pd.read_csv(wpop_path)
    wpop[WPOP_VALUE_COL] = pd.to_numeric(wpop[WPOP_VALUE_COL], errors="coerce")
    wpop = wpop[(wpop[YEAR_COL] >= YEAR_MIN) & (wpop[YEAR_COL] <= YEAR_MAX)].copy()

    # Merge and compute counts
    df = pd.merge(
        mig,
        wpop[[ID_COL, YEAR_COL, WPOP_VALUE_COL]],
        on=[ID_COL, YEAR_COL],
        how="left",
    )
    df["population_thousands"] = _to_thousands(df[WPOP_VALUE_COL])
    df["netMgr_count"] = df[RATE_COL] * df["population_thousands"]

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    mig_arg  = sys.argv[1] if len(sys.argv) > 1 else None
    wpop_arg = sys.argv[2] if len(sys.argv) > 2 else None
    out_arg  = sys.argv[3] if len(sys.argv) > 3 else None
    main(mig_arg, wpop_arg, out_arg)
