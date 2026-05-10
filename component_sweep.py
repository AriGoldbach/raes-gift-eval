"""
Run one component model on all 98 GIFT-Eval configs (or a subset) and
emit the leaderboard-format all_results.csv.

Uses the official per-component Predictor classes from tsf_models.py
(vendored from gift-eval/notebooks/) directly with gluonts.evaluate_model.

Usage:
  python component_sweep.py \
      --component chronos-2 \
      --out ~/raes/results/chronos-2/all_results.csv \
      --data-root ~/raes/data/GiftEval \
      --gift-eval-root ~/raes/gift-eval \
      [--limit 3]
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MeanWeightedSumQuantileLoss, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset as GiftEvalDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_models import build_predictor, QUANTILE_LEVELS


SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T "
    "M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

ALL_DATASETS = sorted(set(SHORT_DATASETS) | set(MED_LONG_DATASETS))

CSV_HEADER = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
]


def make_metrics():
    return [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5),
        MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
        RMSE(), NRMSE(), ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]


_NAME_REMAP = {
    "saugeenday": "saugeen",
}

def _normalize_key(raw: str) -> str:
    """Map dirty dataset name to dataset_properties.json key."""
    k = raw.lower()
    if k.endswith("_with_missing"):
        k = k[: -len("_with_missing")]
    return _NAME_REMAP.get(k, k)


def parse_ds_name(ds_name: str, props: dict) -> tuple[str, str]:
    """Return (props_key, freq) — props_key is normalized for lookup; freq comes from
    name suffix or properties map."""
    if "/" in ds_name:
        raw_key, ds_freq = ds_name.split("/", 1)
    else:
        raw_key = ds_name
        ds_freq = None
    ds_key = _normalize_key(raw_key)
    if ds_freq is None:
        ds_freq = props[ds_key]["frequency"]
    return ds_key, ds_freq


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--component", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--gift-eval-root", default=Path.home() / "raes/gift-eval", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-component batch size (overrides default in build_predictor).")
    p.add_argument("--terms", default="short,medium,long")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--model-name-out", default=None)
    p.add_argument("--evaluator-batch-size", type=int, default=512,
                   help="GluonTS evaluate_model batch_size (separate from predictor's).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("sweep")

    os.environ["GIFT_EVAL"] = str(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    props_path = args.gift_eval_root / "notebooks/dataset_properties.json"
    with open(props_path) as f:
        dataset_properties_map = json.load(f)

    terms = args.terms.split(",")

    work = []
    for ds_name in ALL_DATASETS:
        for term in terms:
            if term in ("medium", "long") and ds_name not in MED_LONG_DATASETS:
                continue
            work.append((ds_name, term))
    if args.start_from:
        work = work[args.start_from:]
    if args.limit:
        work = work[: args.limit]

    log.info(f"Component: {args.component}, configs: {len(work)}, device: {args.device}")

    metrics = make_metrics()
    model_name_out = args.model_name_out or args.component

    if not args.out.exists() or args.start_from == 0:
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)

    rows_done = 0
    failures = []
    for i, (ds_name, term) in enumerate(work, 1):
        ds_key, ds_freq = parse_ds_name(ds_name, dataset_properties_map)
        ds_config = f"{ds_key}/{ds_freq}/{term}"
        t0 = time.time()
        try:
            probe = GiftEvalDataset(name=ds_name, term=term, to_univariate=False)
            to_uni = probe.target_dim != 1
            ds = GiftEvalDataset(name=ds_name, term=term, to_univariate=to_uni)
            season = get_seasonality(ds.freq)

            opts = {}
            if args.batch_size is not None:
                opts["batch_size"] = args.batch_size

            predictor = build_predictor(
                component=args.component,
                prediction_length=ds.prediction_length,
                device=args.device,
                **opts,
            )

            evaluate_kwargs = dict(
                test_data=ds.test_data,
                metrics=metrics,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season,
            )
            # gluonts 0.15.x adds a batch_size kwarg; 0.14.x does not.
            try:
                res = evaluate_model(predictor, batch_size=args.evaluator_batch_size, **evaluate_kwargs)
            except TypeError:
                res = evaluate_model(predictor, **evaluate_kwargs)

            row = [
                ds_config, model_name_out,
                float(res["MSE[mean]"].iloc[0]), float(res["MSE[0.5]"].iloc[0]),
                float(res["MAE[0.5]"].iloc[0]), float(res["MASE[0.5]"].iloc[0]),
                float(res["MAPE[0.5]"].iloc[0]), float(res["sMAPE[0.5]"].iloc[0]),
                float(res["MSIS"].iloc[0]),
                float(res["RMSE[mean]"].iloc[0]), float(res["NRMSE[mean]"].iloc[0]),
                float(res["ND[0.5]"].iloc[0]),
                float(res["mean_weighted_sum_quantile_loss"].iloc[0]),
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
            ]
            with open(args.out, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            rows_done += 1
            dt = time.time() - t0
            log.info(
                f"[{i}/{len(work)}] {ds_config} done in {dt:.1f}s "
                f"MASE={row[5]:.4f} wSQL={row[12]:.4f}"
            )

            # Free GPU memory between configs.
            del predictor
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            dt = time.time() - t0
            log.error(f"[{i}/{len(work)}] {ds_config} FAILED in {dt:.1f}s: "
                      f"{type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            failures.append(ds_config)

    log.info(f"Done. {rows_done}/{len(work)} rows written to {args.out}")
    if failures:
        log.warning(f"Failed configs ({len(failures)}): {failures}")
    return 0 if rows_done == len(work) else 1


if __name__ == "__main__":
    sys.exit(main())
