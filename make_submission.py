"""
Package an all_results.csv + config.json into a leaderboard-PR-ready
directory under results/<MODEL_NAME>/.

Validates schema (98-row, 15-column expectation), normalizes column order,
and emits a minimal config.json matching the agentic-submission template
used by TSOrchestra, MoiraiAgent, etc.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
]


def validate(df: pd.DataFrame, expected_rows: int = 97) -> list[str]:
    errs = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        errs.append(f"Missing columns: {missing}")
    if extra:
        errs.append(f"Extra columns: {extra}")
    if len(df) != expected_rows:
        errs.append(f"Expected {expected_rows} rows, found {len(df)}")
    metric_cols = [c for c in EXPECTED_COLUMNS if c.startswith("eval_metrics/")]
    for c in metric_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_nan = s.isna().sum()
        if n_nan > 0:
            errs.append(f"{c} has {n_nan} NaN/non-numeric values")
        if (s < 0).any():
            errs.append(f"{c} has negative values: {df.loc[(s<0)].dataset.head(3).tolist()}")
    if "model" in df.columns:
        n_unique = df.model.nunique()
        if n_unique != 1:
            errs.append(f"`model` column has {n_unique} unique values; should be 1")
    return errs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path,
                   help="Input all_results.csv (e.g., from router.py apply).")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Target directory: e.g., gift-eval/results/<MODEL_NAME>/")
    p.add_argument("--model-name", default="TSFM-Selection-Ensemble")
    p.add_argument("--model-type", default="agentic",
                   choices=["statistical", "deep-learning", "agentic", "pretrained",
                            "fine-tuned", "zero-shot"])
    p.add_argument("--model-dtype", default="float32")
    p.add_argument("--model-link", default="",
                   help="HF model link, if applicable.")
    p.add_argument("--code-link", required=True,
                   help="Public GitHub URL to replication code.")
    p.add_argument("--org", default="California State University Northridge")
    p.add_argument("--testdata-leakage", default="No", choices=["Yes", "No"])
    p.add_argument("--replication-code-available", default="Yes",
                   choices=["Yes", "No"])
    p.add_argument("--expected-rows", type=int, default=97)
    p.add_argument("--strict", action="store_true",
                   help="Fail if validation errors are found.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["model"] = args.model_name  # force consistent value
    df = df[EXPECTED_COLUMNS]  # reorder

    errs = validate(df, expected_rows=args.expected_rows)
    if errs:
        print("=== VALIDATION ISSUES ===")
        for e in errs:
            print(f"  - {e}")
        if args.strict:
            return 2
    else:
        print("Validation passed.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "all_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows → {out_csv}")

    cfg = {
        "model": args.model_name,
        "model_type": args.model_type,
        "model_dtype": args.model_dtype,
        "model_link": args.model_link,
        "code_link": args.code_link,
        "org": args.org,
        "testdata_leakage": args.testdata_leakage,
        "replication_code_available": args.replication_code_available,
    }
    out_cfg = args.out_dir / "config.json"
    out_cfg.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"Wrote config.json → {out_cfg}")
    print()
    print(json.dumps(cfg, indent=4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
