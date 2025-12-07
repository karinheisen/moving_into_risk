#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-model comparison using NET MIGRATION COUNTS vs. POPULATION EXPOSED
(Admin-1 with fallback everywhere)

- Loads migration COUNTS (best_id, year, netMgr_count).
- Hard-crops all inputs to a global year window (by default 1901–2015).
- Classifies regions by sign of mean net migration COUNT over 2000–2015.
- For each exposure dataset (admin-1 with fallback; POPULATION EXPOSED columns), computes:
    - Present-day relative difference (2000–2015) between net-in vs net-out regions
    - Counterfactual present-day difference
    - Trend difference: (1986–2015 mean) – (1901–1930 mean), then compare net-in vs net-out
- Performs statistical tests:
    - Welch's t-test 
    - Bootstrap confidence intervals
    - Permutation p-values
- Plots summary bars & can write CSV summaries.
"""

from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # Welch's t-test

# ---------------------- Config ----------------------
GLOBAL_YEAR_MIN = 1901
GLOBAL_YEAR_MAX = 2015

PRESENT_START, PRESENT_END = 2000, 2015
EARLY_START, EARLY_END     = 1901, 1930
RECENT_START, RECENT_END   = 1986, 2015

MIG_COUNTS_CSV = "migration_weighted_corrected.csv"
MIG_ID_COL     = "best_id"
MIG_YEAR_COL   = "year"
MIG_COUNT_COL  = "netMgr_count"

EXPOSURE_DATASETS: List[Dict] = [
    {
        "file":    "agg_admin1_fractional_simple/heatwave_population_admin1_fractional_gswp3-w5e5_historical.csv",
        "column":  "heatwave_mean_population",
        "label":   "Heatwaves – GSWP3-W5E5",
        "cf_file": "agg_admin1_fractional_simple/heatwave_population_admin1_fractional_gswp3-w5e5_picontrol.csv",
        "present": True,
        "trend":   True,
    },
    {
        "file":    "agg_admin1_fractional_simple/heatwave_population_admin1_fractional_20crv3_historical.csv",
        "column":  "heatwave_mean_population",
        "label":   "Heatwaves – 20CRv3",
        "cf_file": "agg_admin1_fractional_simple/heatwave_population_admin1_fractional_20crv3_picontrol.csv",
        "present": True,
        "trend":   True,
    },
    {
        "file":    "agg_admin1_fractional_simple/cropfailed_population_admin1_fractional_LPJmL.csv",
        "column":  "cropfailed_mean_population",
        "label":   "Crop failures – LPJmL",
        "cf_file": None,
        "present": False,
        "trend":   True,
    },
    {
        "file":    "agg_admin1_fractional_simple/cropfailed_population_admin1_fractional_EPIC-IIASA.csv",
        "column":  "cropfailed_mean_population",
        "label":   "Crop failures – EPIC-IIASA",
        "cf_file": None,
        "present": False,
        "trend":   True,
    },
    {
        "file":    "agg_admin1_fractional_simple/wildfire_population_admin1_fractional_lpjml_gswp3-w5e5_historical.csv",
        "column":  "wildfire_mean_population",
        "label":   "Wildfire – LPJmL",
        "cf_file": "agg_admin1_fractional_simple/wildfire_population_admin1_fractional_lpjml_gswp3-w5e5_picontrol.csv",
        "present": True,
        "trend":   True,
    },
    {
        "file":    "agg_admin1_fractional_simple/wildfire_population_admin1_fractional_classic_gswp3-w5e5_historical.csv",
        "column":  "wildfire_mean_population",
        "label":   "Wildfire – Classic",
        "cf_file": "agg_admin1_fractional_simple/wildfire_population_admin1_fractional_classic_gswp3-w5e5_picontrol.csv",
        "present": True,
        "trend":   True,
    },
]

PLOT_WIDTH, PLOT_HEIGHT = 14, 6
Y_CAP = 100

BOOT_ITERS = 5000
PERM_ITERS = 10000
RNG_SEED   = 42

# ---------------------- Helpers ----------------------
def load_migration_counts(path: str,
                          id_col: str = MIG_ID_COL,
                          year_col: str = MIG_YEAR_COL,
                          count_col: str = MIG_COUNT_COL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    df[id_col]    = df[id_col].astype(str).str.strip()
    df[year_col]  = pd.to_numeric(df[year_col], errors="coerce")
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df.dropna(subset=[id_col, year_col])
    df[year_col] = df[year_col].astype(int)

    df = df[(df[year_col] >= GLOBAL_YEAR_MIN) & (df[year_col] <= GLOBAL_YEAR_MAX)].copy()

    df_class = df[(df[year_col] >= PRESENT_START) & (df[year_col] <= PRESENT_END)].copy()
    mig_mean = (
        df_class
        .groupby(id_col, as_index=False)[count_col]
        .mean()
        .rename(columns={count_col: "netMgr_count_mean_2000_2015"})
    )
    mig_mean["migration_class"] = np.where(
        mig_mean["netMgr_count_mean_2000_2015"] > 0, "positive",
        np.where(mig_mean["netMgr_count_mean_2000_2015"] < 0, "negative", "neutral")
    )
    mig_mean = mig_mean.rename(columns={id_col: "best_id"})
    return df, mig_mean

def load_exposure_csv(path: str, id_col: str = "best_id") -> pd.DataFrame:
    df = pd.read_csv(path)

    df[id_col] = df[id_col].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=[id_col, "year"])
    df["year"] = df["year"].astype(int)

    df = df[(df["year"] >= GLOBAL_YEAR_MIN) & (df["year"] <= GLOBAL_YEAR_MAX)].copy()
    return df


def calc_diff_and_se(merged: pd.DataFrame, value_col: str) -> Tuple[float, float]:
    """
    Relative difference:

        Diff (%) = (E_in - E_out) / |E_out| * 100

    i.e. exposure in net in-migration regions relative to net out-migration regions.
    """
    grouped = merged.groupby("migration_class")[value_col]
    means = grouped.mean()
    ses   = grouped.std() / np.sqrt(grouped.count())

    if ("positive" in means and "negative" in means
        and pd.notna(means["negative"]) and means["negative"] != 0):

        denom = abs(means["negative"])
        diff  = (means["positive"] - means["negative"]) / denom * 100

        diff_se = np.sqrt(
            (ses.get("positive", 0.0) ** 2) +
            (ses.get("negative", 0.0) ** 2)
        ) / denom * 100
        return float(diff), float(diff_se)

    return np.nan, np.nan

def welch_ttest(merged: pd.DataFrame, value_col: str) -> Tuple[float, float, int, int]:
    """
    Welch's two-sample t-test on exposure values between migration classes.
    Returns: t_stat, p_value, n_pos, n_neg
    """
    pos = merged.loc[merged["migration_class"] == "positive", value_col].to_numpy()
    neg = merged.loc[merged["migration_class"] == "negative", value_col].to_numpy()
    #val_col = an exposure variable -- divide b/t in and out
    # null hyp = in and out exposure have equal mean exposure
    # if p<0.05 -> reject null hypothesis
    # thereby tests whether the mean exposure values differ significantly between net in and net out regions.

    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) >= 2 and len(neg) >= 2:
        t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False, nan_policy="omit")
        return float(t_stat), float(p_val), len(pos), len(neg)
    else:
        return np.nan, np.nan, len(pos), len(neg)

def _relative_diff_percent(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """
    Percent metric used in plotting:
        ((mu_pos - mu_neg) / |mu_neg|) * 100
    i.e. difference relative to the mean in net OUT-migration regions.
    """
    mu_pos = np.mean(pos_vals) if len(pos_vals) else np.nan
    mu_neg = np.mean(neg_vals) if len(neg_vals) else np.nan
    if (not np.isfinite(mu_pos) or not np.isfinite(mu_neg)
            or mu_neg == 0):
        return np.nan
    return ((mu_pos - mu_neg) / abs(mu_neg)) * 100.0

def bootstrap_relative_diff_ci(pos_vals: np.ndarray,
                               neg_vals: np.ndarray,
                               n_iter: int = BOOT_ITERS,
                               seed: Optional[int] = RNG_SEED,
                               ci: float = 95.0) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the percent relative difference.
    Returns (est, ci_low, ci_high).
    """
    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)
    pos_vals = pos_vals[np.isfinite(pos_vals)]
    neg_vals = neg_vals[np.isfinite(neg_vals)]
    if len(pos_vals) < 2 or len(neg_vals) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_iter):
        pos_b = rng.choice(pos_vals, size=len(pos_vals), replace=True)
        neg_b = rng.choice(neg_vals, size=len(neg_vals), replace=True)
        stat = _relative_diff_percent(pos_b, neg_b)
        if np.isfinite(stat):
            boot_stats.append(stat)

    if len(boot_stats) == 0:
        return np.nan, np.nan, np.nan

    boot_stats = np.array(boot_stats)
    est = _relative_diff_percent(pos_vals, neg_vals)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot_stats, [alpha, 100.0 - alpha])
    return est, float(lo), float(hi)

def permutation_relative_diff_p(pos_vals: np.ndarray,
                                neg_vals: np.ndarray,
                                n_iter: int = PERM_ITERS,
                                seed: Optional[int] = RNG_SEED) -> float:
    """
    Permutation p-value (two-sided) for the percent relative difference
    by shuffling group labels. Returns np.nan if not computable.
    """
    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)
    pos_vals = pos_vals[np.isfinite(pos_vals)]
    neg_vals = neg_vals[np.isfinite(neg_vals)]
    n_pos, n_neg = len(pos_vals), len(neg_vals)
    if n_pos < 2 or n_neg < 2:
        return np.nan

    obs = _relative_diff_percent(pos_vals, neg_vals)
    if not np.isfinite(obs):
        return np.nan

    all_vals = np.concatenate([pos_vals, neg_vals])
    N = len(all_vals)
    rng = np.random.default_rng(seed)

    count = 0
    valid = 0
    for _ in range(n_iter):
        idx = rng.permutation(N)
        pos_new = all_vals[idx[:n_pos]]
        neg_new = all_vals[idx[n_pos:]]
        stat = _relative_diff_percent(pos_new, neg_new)
        if np.isfinite(stat):
            valid += 1
            if abs(stat) >= abs(obs):
                count += 1

    if valid == 0:
        return np.nan
    return (count + 1) / (valid + 1)



def analyze_dataset(exp_df: pd.DataFrame,
                    label: str,
                    exp_col: str,
                    mig_class_df: pd.DataFrame,
                    cf_df: Optional[pd.DataFrame] = None,
                    analyze_present: bool = True,
                    analyze_trend: bool = True) -> List[Tuple[str, str, float, float]]:
    results: List[Tuple[str, str, float, float]] = []
    print(f"\n=== {label} ===")

    if exp_col not in exp_df.columns:
        raise ValueError(f"Exposure column '{exp_col}' not found for {label}. Available: {', '.join(exp_df.columns)}")

    if analyze_present:
        per = exp_df[(exp_df["year"] >= PRESENT_START) & (exp_df["year"] <= PRESENT_END)]
        mean_exp = per.groupby("best_id", as_index=False)[exp_col].mean()
        merged = pd.merge(
            mig_class_df[["best_id", "migration_class", "netMgr_count_mean_2000_2015"]],
            mean_exp, on="best_id", how="inner"
        )
        diff, diff_se = calc_diff_and_se(merged, exp_col)

        t_stat, p_val, n_pos, n_neg = welch_ttest(merged, exp_col)

        pos_vals = merged.loc[merged["migration_class"] == "positive", exp_col].to_numpy()
        neg_vals = merged.loc[merged["migration_class"] == "negative", exp_col].to_numpy()
        est_b, ci_lo, ci_hi = bootstrap_relative_diff_ci(pos_vals, neg_vals)
        p_perm = permutation_relative_diff_p(pos_vals, neg_vals)

        print(
            f"{label} – Present: {diff:.1f}% ± {diff_se:.1f}% (SE) | "
            f"Welch t={t_stat:.2f}, p={p_val:.3e} (n+={n_pos}, n-={n_neg}) | "
            f"Bootstrap 95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%] | Perm p={p_perm:.3e}"
        )

        results.append((label, "Present", diff, diff_se))

    if analyze_present and cf_df is not None:
        if exp_col not in cf_df.columns:
            raise ValueError(f"CF file for {label} missing '{exp_col}'. Available: {', '.join(cf_df.columns)}")
        cf_per = cf_df[(cf_df["year"] >= PRESENT_START) & (cf_df["year"] <= PRESENT_END)]
        cf_mean_exp = cf_per.groupby("best_id", as_index=False)[exp_col].mean()
        merged_cf = pd.merge(
            mig_class_df[["best_id", "migration_class", "netMgr_count_mean_2000_2015"]],
            cf_mean_exp, on="best_id", how="inner"
        )
        diff_cf, diff_cf_se = calc_diff_and_se(merged_cf, exp_col)

        t_cf, p_cf, n_pos_cf, n_neg_cf = welch_ttest(merged_cf, exp_col)

        pos_vals_cf = merged_cf.loc[merged_cf["migration_class"] == "positive", exp_col].to_numpy()
        neg_vals_cf = merged_cf.loc[merged_cf["migration_class"] == "negative", exp_col].to_numpy()
        est_b_cf, ci_lo_cf, ci_hi_cf = bootstrap_relative_diff_ci(pos_vals_cf, neg_vals_cf)
        p_perm_cf = permutation_relative_diff_p(pos_vals_cf, neg_vals_cf)

        print(
            f"{label} – Present (cf): {diff_cf:.1f}% ± {diff_cf_se:.1f}% (SE) | "
            f"Welch t={t_cf:.2f}, p={p_cf:.3e} (n+={n_pos_cf}, n-={n_neg_cf}) | "
            f"Bootstrap 95% CI: [{ci_lo_cf:.1f}%, {ci_hi_cf:.1f}%] | Perm p={p_perm_cf:.3e}"
        )

        results.append((label, "Present (cf)", diff_cf, diff_cf_se))

    if analyze_trend:
        early  = exp_df[(exp_df["year"] >= EARLY_START)  & (exp_df["year"] <= EARLY_END)]
        recent = exp_df[(exp_df["year"] >= RECENT_START) & (exp_df["year"] <= RECENT_END)]
        early_mean  = early.groupby("best_id", as_index=False)[exp_col].mean().rename(columns={exp_col: "early_mean"})
        recent_mean = recent.groupby("best_id", as_index=False)[exp_col].mean().rename(columns={exp_col: "recent_mean"})
        trend_df = pd.merge(early_mean, recent_mean, on="best_id", how="inner")
        trend_df["trend"] = trend_df["recent_mean"] - trend_df["early_mean"]

        merged_trend = pd.merge(
            mig_class_df[["best_id", "migration_class", "netMgr_count_mean_2000_2015"]],
            trend_df[["best_id", "trend"]], on="best_id", how="inner"
        )
        diff_trend, diff_trend_se = calc_diff_and_se(merged_trend, "trend")

        t_tr, p_tr, n_pos_tr, n_neg_tr = welch_ttest(merged_trend, "trend")

        pos_vals_tr = merged_trend.loc[merged_trend["migration_class"] == "positive", "trend"].to_numpy()
        neg_vals_tr = merged_trend.loc[merged_trend["migration_class"] == "negative", "trend"].to_numpy()
        est_b_tr, ci_lo_tr, ci_hi_tr = bootstrap_relative_diff_ci(pos_vals_tr, neg_vals_tr)
        p_perm_tr = permutation_relative_diff_p(pos_vals_tr, neg_vals_tr)

        print(
            f"{label} – Trend: {diff_trend:.1f}% ± {diff_trend_se:.1f}% (SE) | "
            f"Welch t={t_tr:.2f}, p={p_tr:.3e} (n+={n_pos_tr}, n-={n_neg_tr}) | "
            f"Bootstrap 95% CI: [{ci_lo_tr:.1f}%, {ci_hi_tr:.1f}%] | Perm p={p_perm_tr:.3e}"
        )

        results.append((label, "Trend", diff_trend, diff_trend_se))

    return results

# ---------------------- Main ----------------------
def main():
    mig_counts_all, mig_class = load_migration_counts(
        MIG_COUNTS_CSV,
        id_col=MIG_ID_COL,
        year_col=MIG_YEAR_COL,
        count_col=MIG_COUNT_COL
    )

    results: List[Tuple[str, str, float, float]] = []

    for spec in EXPOSURE_DATASETS:
        label      = spec["label"]
        exp_col    = spec["column"]
        file       = spec["file"]
        cf_file    = spec.get("cf_file")
        do_present = spec.get("present", True)
        do_trend   = spec.get("trend", True)

        exp_df = load_exposure_csv(file)
        cf_df  = load_exposure_csv(cf_file) if (cf_file and do_present) else None

        res = analyze_dataset(
            exp_df, label, exp_col, mig_class,
            cf_df=cf_df, analyze_present=do_present, analyze_trend=do_trend
        )
        results.extend(res)

    df_plot = pd.DataFrame(results, columns=["Category", "Metric", "Diff", "SE"])

    blocks = [
        ("Heatwaves", ["Heatwaves – GSWP3-W5E5",
                       "Heatwaves – 20CRv3"]),
        ("Crop failures", ["Crop failures – LPJmL",
                           "Crop failures – EPIC-IIASA"]),
        ("Wildfire", ["Wildfire – LPJmL",
                      "Wildfire – Classic"]),
    ]

    categories = []
    for _, labels in blocks:
        for lbl in labels:
            if lbl in df_plot["Category"].values:
                categories.append(lbl)

    df_agg = (
        df_plot
        .groupby(["Category", "Metric"], as_index=False)
        .agg(Diff=("Diff", "mean"), SE=("SE", "mean"))
    )

    x = np.arange(len(categories)) * 1.5
    width = 0.18
    spacing = 0.25
    offsets = [-spacing, 0, spacing]
    colors = ["#4575b4", "#74add1", "#d73027"]
    metrics = ["Present", "Present (cf)", "Trend"]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    LABEL_PAD = 3.0
    MARGIN = 2.0
    y_min_cap, y_max_cap = -Y_CAP, Y_CAP

    for i, metric in enumerate(metrics):
        vals = []
        for cat in categories:
            row = df_agg[(df_agg["Category"] == cat) & (df_agg["Metric"] == metric)]
            if len(row) == 1:
                vals.append(row["Diff"].iloc[0])
            else:
                vals.append(np.nan)

        bars = ax.bar(x + offsets[i], vals, width, label=metric, color=colors[i])

        for bar, val in zip(bars, vals):
            if pd.isna(val):
                continue
            bx = bar.get_x() + bar.get_width() / 2.0

            if abs(val) <= Y_CAP:
                if val >= 0:
                    y_label = val + LABEL_PAD
                    y_label = min(y_label, y_max_cap - MARGIN)
                    va = "bottom"
                else:
                    y_label = val - LABEL_PAD
                    y_label = max(y_label, y_min_cap + MARGIN)
                    va = "top"
                ax.text(bx, y_label, f"{val:.1f}%", ha="center", va=va, fontsize=9)
            else:
                y_edge = 0.98 * Y_CAP * (1 if val > 0 else -1)
                y_text = (Y_CAP - 0.20 * Y_CAP) * (1 if val > 0 else -1)
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(bx, y_edge),
                    xytext=(bx, y_text),
                    ha="center",
                    va="bottom" if val > 0 else "top",
                    fontsize=9,
                    arrowprops=dict(arrowstyle='-|>', lw=1)
                )

    ax.set_ylim(-Y_CAP, Y_CAP)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative difference (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("– ", 1)[-1] for c in categories], rotation=45, ha="right")
    ax.legend()

    start = 0
    for block_name, labels in blocks:
        block_labels = [lbl for lbl in labels if lbl in categories]
        if block_labels:
            end = start + len(block_labels) - 1
            ax.axvspan(x[start]-0.75, x[end]+0.75, color="grey", alpha=0.08)
            ymax = ax.get_ylim()[1]
            ax.text(
                (x[start]+x[end]) / 2, ymax * 0.95, block_name,
                ha="center", va="top", fontsize=12, fontweight="bold"
            )
            start = end + 1

    plt.tight_layout()
    plt.savefig("exposure_vs_migration_counts.png", dpi=300)
    plt.show()

    # df_plot.to_csv("exposure_vs_migration_counts_summary.csv", index=False)

if __name__ == "__main__":
    main()
