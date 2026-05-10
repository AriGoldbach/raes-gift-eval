"""
Read every existing all_results.csv under gift-eval/results/, compute the
per-config oracle (best component per metric), and report the gap between the
current leaderboard top and what a perfect-selection ensemble over our pool
would score.

This is the cheapest possible decisive analysis — it tells us whether
selection-only is enough to top the leaderboard, or we have to do
full forecast-level mixing.

Usage:
  python oracle_analysis.py --results-root ~/raes/gift-eval/results \
      [--pool TimesFM-2.5 chronos_bolt_base ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

METRIC_COLS = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "eval_metrics/ND[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/CRPS",  # may not be present — handled by load
]
PRIMARY = "eval_metrics/MASE[0.5]"
SECONDARY = "eval_metrics/mean_weighted_sum_quantile_loss"


def load_all(results_root: Path, model_filter: Optional[set[str]] = None) -> pd.DataFrame:
    rows = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        f = d / "all_results.csv"
        if not f.exists():
            continue
        if model_filter is not None and d.name not in model_filter:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] {d.name}: {e}", file=sys.stderr)
            continue
        if "dataset" not in df.columns or "model" not in df.columns:
            continue
        df["_model_dir"] = d.name
        rows.append(df)
    if not rows:
        raise SystemExit("No valid result CSVs found.")
    return pd.concat(rows, ignore_index=True)


def per_config_rank(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a long DF (dataset, _model_dir, value, rank). Lower value = lower rank."""
    sub = df[["dataset", "_model_dir", metric]].dropna()
    sub = sub[~np.isinf(sub[metric])]
    sub["rank"] = sub.groupby("dataset")[metric].rank(method="min", ascending=True)
    return sub


def report_leaderboard(df: pd.DataFrame, metric: str, top: int = 15) -> pd.DataFrame:
    """Mean rank across configs per model. Lower mean rank = closer to #1."""
    ranked = per_config_rank(df, metric)
    n_configs_per_model = ranked.groupby("_model_dir").size()
    full_pool = n_configs_per_model[n_configs_per_model == n_configs_per_model.max()].index
    full = ranked[ranked["_model_dir"].isin(full_pool)]
    leaderboard = full.groupby("_model_dir")["rank"].mean().sort_values()
    return leaderboard.head(top), len(full_pool)


def oracle_per_config(df: pd.DataFrame, pool: list[str], metric: str) -> pd.DataFrame:
    """For the given pool, pick the best (lowest) value per dataset config."""
    sub = df[df["_model_dir"].isin(pool)][["dataset", "_model_dir", metric]].dropna()
    sub = sub[~np.isinf(sub[metric])]
    idx = sub.groupby("dataset")[metric].idxmin()
    picks = sub.loc[idx]
    return picks


def oracle_rank_among_full_pool(df: pd.DataFrame, pool: list[str], metric: str) -> float:
    """Compute what the oracle's mean rank would be against the full leaderboard."""
    picks = oracle_per_config(df, pool, metric)
    pick_value_by_ds = dict(zip(picks["dataset"], picks[metric]))
    sub = df[["dataset", "_model_dir", metric]].dropna()
    sub = sub[~np.isinf(sub[metric])]
    rows = []
    for ds, oracle_v in pick_value_by_ds.items():
        ds_models = sub[sub["dataset"] == ds][[metric]].values.flatten()
        rank = (ds_models < oracle_v).sum() + 1
        rows.append({"dataset": ds, "oracle_value": oracle_v, "rank_in_pool": rank})
    return pd.DataFrame(rows)["rank_in_pool"].mean()


def best_components_per_config(df: pd.DataFrame, pool: list[str], metric: str) -> pd.DataFrame:
    """Which pool member wins each dataset, used for visualizing diversity of picks."""
    picks = oracle_per_config(df, pool, metric)
    counts = picks["_model_dir"].value_counts()
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--pool", nargs="*", default=None,
                   help="Candidate models for our ensemble pool. Defaults to all leaderboard "
                        "members declared zero-shot/agentic with no leakage.")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    df = load_all(args.results_root)
    print(f"Loaded {df['_model_dir'].nunique()} models, {len(df)} (model, config) rows.")
    print(f"Configs in latest entry: {df.groupby('_model_dir').size().max()}")
    print()

    for metric in [PRIMARY, SECONDARY, "eval_metrics/ND[0.5]"]:
        print(f"=== Top {args.top} models by mean rank on {metric} (lower = better) ===")
        lb, full_pool_size = report_leaderboard(df, metric, top=args.top)
        for i, (m, r) in enumerate(lb.items(), 1):
            print(f"  #{i:2d} {m:35s}  mean rank = {r:6.3f}")
        print(f"  (computed over {full_pool_size} models with full coverage)")
        print()

    if args.pool is None:
        # default candidate pool — modern, public, declared no-leakage
        candidate_pool = [
            "TimesFM-2.5", "chronos-2", "chronos_bolt_base", "chronos_bolt_small",
            "Moirai2", "Moirai_base", "Moirai_large", "moirai_small",
            "FlowState-r1.1", "Granite-FlowState-r1.1",
            "FLAIR", "Toto", "tabpfn_ts", "TempoPFN",
            "tempo_ensemble", "Kairos_50m", "sundial_base_128m",
        ]
        # filter to those actually present
        present = set(df["_model_dir"].unique())
        candidate_pool = [m for m in candidate_pool if m in present]
        print(f"Using default candidate pool ({len(candidate_pool)} models present):")
        for m in candidate_pool:
            print(f"  - {m}")
        print()
    else:
        candidate_pool = args.pool

    print(f"=== Oracle of pool over {PRIMARY} ===")
    rank = oracle_rank_among_full_pool(df, candidate_pool, PRIMARY)
    print(f"Oracle mean rank against full leaderboard: {rank:.3f}")
    print()
    print("=== Per-config winner counts (which model wins which config) ===")
    counts = best_components_per_config(df, candidate_pool, PRIMARY)
    for m, n in counts.items():
        print(f"  {m:35s}  wins {n:3d} configs")
    print()

    # Extra: the same for wSQL.
    print(f"=== Oracle of pool over {SECONDARY} ===")
    rank2 = oracle_rank_among_full_pool(df, candidate_pool, SECONDARY)
    print(f"Oracle mean rank against full leaderboard: {rank2:.3f}")
    print()
    counts2 = best_components_per_config(df, candidate_pool, SECONDARY)
    print(f"=== Per-config winner counts on wSQL ===")
    for m, n in counts2.items():
        print(f"  {m:35s}  wins {n:3d} configs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
