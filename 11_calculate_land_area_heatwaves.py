#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heatwave exposure in + and - migration areas

For each model (20CRv3, GSWP3-W5E5):
- Present-day exposure (real climate, 2000–2015)
- 20th century trend (real climate, 1986–2015 minus 1901–1930)
- Present-day exposure (counterfactual, 2000–2015)

Dropout resampling test for land area:
- Compute relative difference in exposure between + and - migration cells:
      ((E_in - E_out) / |E_out|) * 100
- Drop 20% of grid cells at random
- Repeat 1000 times
- Use the distribution of relative differences to obtain a 95% interval
- Result is considered robust if the 95% interval does not cross zero
"""

import rioxarray
import xarray as xr
import numpy as np

# ---------------------- Config ----------------------
MIGRATION_RASTER = "raster_netMgr_2000_2019_annual.tif"

HIST_20CR = "hwmid-humidex_20crv3_historical_heatwavedarea_global_annual_landarea_1901_2016.nc"
CF_20CR   = "hwmid-humidex_20crv3_picontrol_heatwavedarea_global_annual_landarea_1901_2016.nc"

HIST_GSWP = "hwmid-humidex_gswp3-w5e5_historical_heatwavedarea_global_annual_landarea_1901_2016.nc"
CF_GSWP   = "hwmid-humidex_gswp3-w5e5_picontrol_heatwavedarea_global_annual_landarea_1901_2016.nc"

DROP_FRACTION = 0.2
N_ITERATIONS = 1000
RANDOM_SEED = 42

# ---------------------- Helpers ----------------------
def load_migration_masks(raster_path: str):
    print("Loading migration raster (2000–2015)...")

    raster = rioxarray.open_rasterio(raster_path, masked=True)
    mean_migration = raster.isel(band=slice(0, 16)).mean(dim="band", skipna=True)
    mean_migration_30min = mean_migration.coarsen(y=6, x=6, boundary="trim").mean()

    mig_vals = mean_migration_30min.values
    pos_mask = mig_vals > 0
    neg_mask = mig_vals < 0

    print("  Positive migration cells:", np.count_nonzero(pos_mask))
    print("  Negative migration cells:", np.count_nonzero(neg_mask))
    print()

    return mig_vals, pos_mask, neg_mask


def relative_diff_percent(mean_pos, mean_neg):
    if not np.isfinite(mean_neg) or mean_neg == 0:
        return np.nan
    return (mean_pos - mean_neg) / abs(mean_neg) * 100.0


def compute_stats(nc_hist, nc_cf, label, mig_vals, pos_mask, neg_mask):
    print(f"=== {label} ===")

    ds_hist = xr.open_dataset(nc_hist)
    ds_cf   = xr.open_dataset(nc_cf)

    heat_hist = ds_hist["exposure"]
    heat_cf   = ds_cf["exposure"]

    present_hist = heat_hist.sel(time=slice("2000-01-01", "2015-12-31")).mean(dim="time")
    present_cf   = heat_cf.sel(time=slice("2000-01-01", "2015-12-31")).mean(dim="time")

    modern_hist = heat_hist.sel(time=slice("1986-01-01", "2015-12-31")).mean(dim="time")
    early_hist  = heat_hist.sel(time=slice("1901-01-01", "1930-12-31")).mean(dim="time")
    trend_hist  = modern_hist - early_hist

    pres_hist_vals = present_hist.values
    pres_cf_vals   = present_cf.values
    trend_vals     = trend_hist.values

    def masked_means(arr):
        mean_pos = np.nanmean(arr[pos_mask])
        mean_neg = np.nanmean(arr[neg_mask])
        return mean_pos, mean_neg

    pres_real_pos, pres_real_neg = masked_means(pres_hist_vals)
    trend_real_pos, trend_real_neg = masked_means(trend_vals)
    pres_cf_pos, pres_cf_neg = masked_means(pres_cf_vals)

    pres_real_rel = relative_diff_percent(pres_real_pos, pres_real_neg)
    trend_real_rel = relative_diff_percent(trend_real_pos, trend_real_neg)
    pres_cf_rel = relative_diff_percent(pres_cf_pos, pres_cf_neg)

    print(
        f"  Present-day exposure (real):           "
        f"+ {pres_real_pos:.4f}, - {pres_real_neg:.4f}, rel diff = {pres_real_rel:.1f}%"
    )
    print(
        f"  20th century trend (real):             "
        f"+ {trend_real_pos:.4f}, - {trend_real_neg:.4f}, rel diff = {trend_real_rel:.1f}%"
    )
    print(
        f"  Present-day exposure (counterfactual): "
        f"+ {pres_cf_pos:.4f}, - {pres_cf_neg:.4f}, rel diff = {pres_cf_rel:.1f}%"
    )
    print()

    np.random.seed(RANDOM_SEED)

    def bootstrap_relative(values):
        diffs = []
        flat_size = mig_vals.size

        for _ in range(N_ITERATIONS):
            mask = np.ones(flat_size, dtype=bool)
            drop_n = int(DROP_FRACTION * flat_size)
            drop_idx = np.random.choice(flat_size, drop_n, replace=False)
            mask[drop_idx] = False

            mask_grid = mask.reshape(mig_vals.shape)

            pos_mask_d = pos_mask & mask_grid
            neg_mask_d = neg_mask & mask_grid

            mpos = np.nanmean(values[pos_mask_d])
            mneg = np.nanmean(values[neg_mask_d])

            diff_pct = relative_diff_percent(mpos, mneg)
            if np.isfinite(diff_pct):
                diffs.append(diff_pct)

        diffs = np.array(diffs)
        mean_diff = np.nanmean(diffs)
        ci = np.nanpercentile(diffs, [2.5, 97.5])
        robust = not (ci[0] <= 0 <= ci[1])

        return mean_diff, ci, robust

    print(f"  Running dropout test ({N_ITERATIONS} iterations, drop 20%) on relative differences...")

    b_pres_real_mean, b_pres_real_ci, b_pres_real_robust = bootstrap_relative(pres_hist_vals)
    b_trend_real_mean, b_trend_real_ci, b_trend_real_robust = bootstrap_relative(trend_vals)
    b_pres_cf_mean, b_pres_cf_ci, b_pres_cf_robust = bootstrap_relative(pres_cf_vals)

    print(
        f"  Dropout rel diff (present-day real): {b_pres_real_mean:.1f}% "
        f"[{b_pres_real_ci[0]:.1f}%, {b_pres_real_ci[1]:.1f}%], robust={b_pres_real_robust}"
    )
    print(
        f"  Dropout rel diff (20th c. trend real): {b_trend_real_mean:.1f}% "
        f"[{b_trend_real_ci[0]:.1f}%, {b_trend_real_ci[1]:.1f}%], robust={b_trend_real_robust}"
    )
    print(
        f"  Dropout rel diff (present-day CF): {b_pres_cf_mean:.1f}% "
        f"[{b_pres_cf_ci[0]:.1f}%, {b_pres_cf_ci[1]:.1f}%], robust={b_pres_cf_robust}"
    )
    print()

    return dict(
        pres_real_pos=pres_real_pos,
        pres_real_neg=pres_real_neg,
        trend_real_pos=trend_real_pos,
        trend_real_neg=trend_real_neg,
        pres_cf_pos=pres_cf_pos,
        pres_cf_neg=pres_cf_neg,
        pres_real_rel=pres_real_rel,
        trend_real_rel=trend_real_rel,
        pres_cf_rel=pres_cf_rel,
        boot_pres_real_mean=b_pres_real_mean,
        boot_pres_real_ci=b_pres_real_ci,
        boot_pres_real_robust=b_pres_real_robust,
        boot_trend_real_mean=b_trend_real_mean,
        boot_trend_real_ci=b_trend_real_ci,
        boot_trend_real_robust=b_trend_real_robust,
        boot_pres_cf_mean=b_pres_cf_mean,
        boot_pres_cf_ci=b_pres_cf_ci,
        boot_pres_cf_robust=b_pres_cf_robust,
    )

# ---------------------- Main ----------------------
def main() -> None:
    mig_vals, pos_mask, neg_mask = load_migration_masks(MIGRATION_RASTER)

    compute_stats(HIST_20CR, CF_20CR, "20CRv3", mig_vals, pos_mask, neg_mask)
    compute_stats(HIST_GSWP, CF_GSWP, "GSWP3-W5E5", mig_vals, pos_mask, neg_mask)

if __name__ == "__main__":
    main()
