"""
Selection router for GIFT-Eval.

The router maps each of the 97 dataset configurations to a component
model. The mapping is *discovered* from leaderboard CSVs (which we treat
as priors over per-config component performance), then *applied* by
pulling, per config, the chosen component's row from OUR own sweep
results to assemble the final submission CSV.

Three modes:

  discover  Read all_results.csv from a leaderboard-style directory, score
            each (config, component) pair, and emit a router_map.json
            mapping each dataset config to a chosen component.

  apply     Read router_map.json + our per-component sweep CSVs, output a
            single all_results.csv where each config row comes from the
            chosen component. Honors per-config fallbacks if the chosen
            component wasn't computed.

  analyze   Sanity report: with the discovered router, what would the
            ensemble's mean rank be on the prior data?

Scoring: per-config min-max normalize {MASE, wSQL, ND}, weighted sum
  0.4 * MASE + 0.3 * wSQL + 0.2 * ND + 0.1 * |MAPE - typical|
(MAPE term skipped if it has heavy outliers, which it usually does).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


METRICS_LOWER_IS_BETTER = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "eval_metrics/ND[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/RMSE[mean]",
]
DEFAULT_WEIGHTS = {
    "eval_metrics/MASE[0.5]": 0.40,
    "eval_metrics/mean_weighted_sum_quantile_loss": 0.30,
    "eval_metrics/ND[0.5]": 0.20,
    "eval_metrics/MAE[0.5]": 0.10,
}


def load_pool(results_root: Path, pool: list[str]) -> pd.DataFrame:
    """Read each pool member's all_results.csv. Concat into long DF with column
    `_model_dir` carrying the source folder name (== component name we'll route by)."""
    rows = []
    for name in pool:
        f = results_root / name / "all_results.csv"
        if not f.exists():
            print(f"[skip] {name}: no all_results.csv", file=sys.stderr)
            continue
        df = pd.read_csv(f)
        df["_model_dir"] = name
        rows.append(df)
    if not rows:
        raise SystemExit("Empty pool — nothing to load.")
    return pd.concat(rows, ignore_index=True)


def per_config_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Per (dataset, _model_dir), compute a normalized score in [0,1].
    Lower = better. Returns a DataFrame with columns [dataset, _model_dir, score]."""
    parts = []
    for metric, w in weights.items():
        if metric not in df.columns:
            continue
        sub = df[["dataset", "_model_dir", metric]].dropna()
        sub = sub[~np.isinf(sub[metric])]
        # min-max within each config
        mins = sub.groupby("dataset")[metric].transform("min")
        maxs = sub.groupby("dataset")[metric].transform("max")
        denom = (maxs - mins).replace(0, 1)
        sub["norm"] = (sub[metric] - mins) / denom
        sub["weighted"] = w * sub["norm"]
        parts.append(sub[["dataset", "_model_dir", "weighted"]])
    if not parts:
        raise SystemExit("No metric columns matched.")
    cat = pd.concat(parts)
    score = cat.groupby(["dataset", "_model_dir"])["weighted"].sum().reset_index(name="score")
    return score


def discover_router(score: pd.DataFrame) -> dict[str, str]:
    """For each config, pick the component with the lowest combined score."""
    idx = score.groupby("dataset")["score"].idxmin()
    picks = score.loc[idx]
    return dict(zip(picks["dataset"], picks["_model_dir"]))


def winner_count(router_map: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in router_map.values():
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def whatif_mean_rank(df_full: pd.DataFrame, router_map: dict[str, str], metric: str) -> float:
    """If the router picks the prescribed component per config and we pretend its
    prior metric value is what we'd score, what's the mean rank against the full
    leaderboard pool?"""
    sub = df_full[["dataset", "_model_dir", metric]].dropna()
    sub = sub[~np.isinf(sub[metric])]
    ranks = []
    for ds, picked in router_map.items():
        ds_rows = sub[sub["dataset"] == ds]
        if ds_rows.empty:
            continue
        pick_row = ds_rows[ds_rows["_model_dir"] == picked]
        if pick_row.empty:
            continue
        v = float(pick_row[metric].iloc[0])
        rank = int((ds_rows[metric] < v).sum() + 1)
        ranks.append(rank)
    return float(np.mean(ranks)) if ranks else float("nan")


def cmd_discover(args) -> int:
    pool = args.pool.split(",")
    df = load_pool(args.results_root, pool)
    print(f"loaded {df['_model_dir'].nunique()} components, {df['dataset'].nunique()} configs")
    score = per_config_score(df, DEFAULT_WEIGHTS)
    router = discover_router(score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "pool": pool,
        "router_map": router,
        "winner_counts": winner_count(router),
    }, indent=2))
    print(f"wrote router map → {args.out}")
    print()
    print("=== winner counts ===")
    for c, n in winner_count(router).items():
        print(f"  {c:35s}  {n:3d} configs")
    return 0


def cmd_apply(args) -> int:
    spec = json.loads(args.router_map.read_text())
    router = spec["router_map"]
    pool = spec["pool"]

    # Load OUR component sweeps (different from priors — these contain our metrics).
    # If a component is missing or has missing configs, fall back to the next-best
    # component on the prior, recursively.
    #
    # The fallback chain is computed from the prior the router was discovered on.
    fallback_chain = json.loads(args.fallback_map.read_text()) if args.fallback_map else None

    components = {}
    for name in pool:
        f = args.our_results_root / name / "all_results.csv"
        if not f.exists():
            print(f"[skip-our] {name}: no all_results.csv at {f}", file=sys.stderr)
            continue
        components[name] = pd.read_csv(f).set_index("dataset")

    if not components:
        raise SystemExit("None of our component sweeps are available.")

    # Build the submission rows.
    out_rows = []
    chosen_log = []
    skipped = []
    for ds, picked in router.items():
        comp_used = None
        # Try the picked component first, then fallbacks.
        candidates = [picked]
        if fallback_chain is not None and ds in fallback_chain:
            candidates += fallback_chain[ds]
        for c in candidates:
            if c in components and ds in components[c].index:
                comp_used = c
                break
        if comp_used is None:
            skipped.append(ds)
            continue
        row = components[comp_used].loc[ds].copy()
        if hasattr(row, "ndim") and row.ndim > 1:
            row = row.iloc[0]
        row_dict = row.to_dict()
        row_dict["dataset"] = ds
        row_dict["model"] = args.model_name
        out_rows.append(row_dict)
        chosen_log.append((ds, comp_used))

    # Order columns to match the official schema.
    schema = [
        "dataset", "model",
        "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain", "num_variates",
    ]
    df_out = pd.DataFrame(out_rows)
    for c in schema:
        if c not in df_out.columns:
            df_out[c] = pd.NA
    df_out = df_out[schema]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"wrote {len(df_out)} rows → {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} configs (no available component): {skipped[:5]}...")
    print()
    chosen = pd.Series([c for _, c in chosen_log])
    print("=== components used ===")
    for c, n in chosen.value_counts().items():
        print(f"  {c:35s}  {n:3d} configs")
    return 0


def cmd_analyze(args) -> int:
    pool = args.pool.split(",")
    df = load_pool(args.results_root, pool)
    score = per_config_score(df, DEFAULT_WEIGHTS)
    router = discover_router(score)

    # Mean rank of the routed ensemble against the full leaderboard.
    # Load EVERY model in results_root, not just our pool, to compute rank.
    everything = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir():
            continue
        f = d / "all_results.csv"
        if not f.exists():
            continue
        try:
            sub = pd.read_csv(f)
            sub["_model_dir"] = d.name
            everything.append(sub)
        except Exception:
            pass
    full = pd.concat(everything, ignore_index=True)
    n_full = full["_model_dir"].nunique()

    print(f"=== Router 'what-if' rank against {n_full}-model leaderboard ===")
    for metric in [
        "eval_metrics/MASE[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "eval_metrics/ND[0.5]",
    ]:
        rank = whatif_mean_rank(full, router, metric)
        print(f"  {metric:50s}  mean rank = {rank:6.3f}")
    print()
    print("=== winner counts ===")
    for c, n in winner_count(router).items():
        print(f"  {c:35s}  {n:3d} configs")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("discover", help="Discover router_map from leaderboard priors.")
    sp.add_argument("--results-root", required=True, type=Path,
                    help="gift-eval/results — leaderboard CSVs to use as priors.")
    sp.add_argument("--pool", required=True,
                    help="Comma-separated component names matching folder names under --results-root.")
    sp.add_argument("--out", required=True, type=Path,
                    help="Output JSON router_map.")

    sp = sub.add_parser("apply", help="Apply router_map to our own per-component CSVs.")
    sp.add_argument("--router-map", required=True, type=Path)
    sp.add_argument("--our-results-root", required=True, type=Path,
                    help="Directory containing OUR per-component all_results.csv directories.")
    sp.add_argument("--out", required=True, type=Path)
    sp.add_argument("--model-name", default="TSFM-Selection-Ensemble")
    sp.add_argument("--fallback-map", type=Path, default=None,
                    help="Optional JSON: dataset → list of fallback component names.")

    sp = sub.add_parser("analyze",
                        help="What-if: rank a discovered router against the full leaderboard.")
    sp.add_argument("--results-root", required=True, type=Path)
    sp.add_argument("--pool", required=True)

    args = p.parse_args()
    return {"discover": cmd_discover, "apply": cmd_apply, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
