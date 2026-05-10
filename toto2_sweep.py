"""
Toto-2.0 sweep over GIFT-Eval. Self-contained for venv313 (Python 3.13).

Mirrors component_sweep.py but uses Toto2GluonTSModel directly (the
Datadog-provided GluonTS predictor wrapper).
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

import torch

from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MeanWeightedSumQuantileLoss, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset as GiftEvalDataset
from toto2 import Toto2Model, Toto2GluonTSModel, Toto2GluonTSModelConfig

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

NAME_REMAP = {"saugeenday": "saugeen"}

def _normalize_key(raw):
    k = raw.lower()
    if k.endswith("_with_missing"):
        k = k[: -len("_with_missing")]
    return NAME_REMAP.get(k, k)

def parse_ds_name(ds_name, props):
    if "/" in ds_name:
        raw, freq = ds_name.split("/", 1)
    else:
        raw, freq = ds_name, None
    key = _normalize_key(raw)
    if freq is None:
        freq = props[key]["frequency"]
    return key, freq

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="Datadog/Toto-2.0-313m")
    p.add_argument("--model-name-out", default="Toto-2.0-313m")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--gift-eval-root", default=Path.home() / "raes/gift-eval", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--terms", default="short,medium,long")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("toto2")

    os.environ["GIFT_EVAL"] = str(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    props = json.load(open(args.gift_eval_root / "notebooks/dataset_properties.json"))
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

    log.info(f"Toto-2 sweep: {len(work)} configs, model={args.checkpoint}, batch={args.batch_size}")

    log.info("Loading model checkpoint...")
    base_model = Toto2Model.from_pretrained(args.checkpoint, map_location=args.device)
    log.info("Model loaded.")

    metrics = [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5),
        MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
        RMSE(), NRMSE(), ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]

    if not args.out.exists() or args.start_from == 0:
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)

    rows_done = 0
    for i, (ds_name, term) in enumerate(work, 1):
        ds_key, ds_freq = parse_ds_name(ds_name, props)
        ds_config = f"{ds_key}/{ds_freq}/{term}"
        t0 = time.time()
        try:
            probe = GiftEvalDataset(name=ds_name, term=term, to_univariate=False)
            to_uni = probe.target_dim != 1
            ds = GiftEvalDataset(name=ds_name, term=term, to_univariate=to_uni)
            season = get_seasonality(ds.freq)

            cfg = Toto2GluonTSModelConfig(
                prediction_length=ds.prediction_length,
                context_length=args.context_length,
                target_dim=1,
                past_feat_dynamic_real_dim=0,
                feat_dynamic_real_dim=0,
                decode_block_size=None,
                has_missing_values=True,
                quantiles=QUANTILE_LEVELS,
            )
            gts_model = Toto2GluonTSModel(base_model, cfg)
            predictor = gts_model.create_predictor(batch_size=args.batch_size, device=args.device)

            try:
                res = evaluate_model(
                    predictor,
                    test_data=ds.test_data,
                    metrics=metrics,
                    batch_size=512,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season,
                )
            except TypeError:
                res = evaluate_model(
                    predictor,
                    test_data=ds.test_data,
                    metrics=metrics,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season,
                )

            row = [
                ds_config, args.model_name_out,
                float(res["MSE[mean]"].iloc[0]), float(res["MSE[0.5]"].iloc[0]),
                float(res["MAE[0.5]"].iloc[0]), float(res["MASE[0.5]"].iloc[0]),
                float(res["MAPE[0.5]"].iloc[0]), float(res["sMAPE[0.5]"].iloc[0]),
                float(res["MSIS"].iloc[0]),
                float(res["RMSE[mean]"].iloc[0]), float(res["NRMSE[mean]"].iloc[0]),
                float(res["ND[0.5]"].iloc[0]),
                float(res["mean_weighted_sum_quantile_loss"].iloc[0]),
                props[ds_key]["domain"], props[ds_key]["num_variates"],
            ]
            with open(args.out, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            rows_done += 1
            dt = time.time() - t0
            log.info(f"[{i}/{len(work)}] {ds_config} done in {dt:.1f}s "
                     f"MASE={row[5]:.4f} wSQL={row[12]:.4f}")
            del predictor, gts_model
            torch.cuda.empty_cache()
        except Exception as e:
            dt = time.time() - t0
            log.error(f"[{i}/{len(work)}] {ds_config} FAILED in {dt:.1f}s: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())

    log.info(f"Done. {rows_done}/{len(work)} rows.")
    return 0 if rows_done == len(work) else 1

if __name__ == "__main__":
    sys.exit(main())
