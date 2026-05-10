"""
Official per-component Predictor classes vendored from gift-eval/notebooks/.

Each Predictor exposes `.predict(test_data_input) -> List[Forecast]` and is
directly compatible with gluonts.evaluate_model. We use the official
leaderboard wrappers verbatim — no MoiraiAgent-style chunking — so our
results match the official entries on configs where the same model is run.

Components covered:
  - ChronosPredictor    (chronos-2 + chronos-bolt-{small,base,large})
  - TimesFmPredictor    (timesfm-2.5-200m-pytorch)
  - MoiraiQuantilePredictor  (Moirai-2.0-R-{small,base,large})
  - MoiraiSamplePredictor    (Moirai-1.1-R-{small,base,large})
  - TiRexGiftEvalWrapper     (TiRex-1.1-gifteval)
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Type

import numpy as np
import torch

from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm


QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seeds()

logger = logging.getLogger("tsf_models")
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Chronos-2 (and Chronos-Bolt) — Amazon. Vendored from chronos-2.ipynb.
# --------------------------------------------------------------------------- #

class ChronosPredictor:
    def __init__(
        self,
        model_name: str,
        prediction_length: int,
        batch_size: int = 100,
        quantile_levels: list[float] = QUANTILE_LEVELS,
        predict_batches_jointly: bool = True,
        **kwargs,
    ):
        from chronos import BaseChronosPipeline, Chronos2Pipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(model_name, **kwargs)
        self._is_chronos2 = isinstance(self.pipeline, Chronos2Pipeline)
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        # predict_batches_jointly is only for chronos-2.
        self.predict_batches_jointly = predict_batches_jointly and self._is_chronos2

    def _pack(self, items):
        for item in items:
            yield {"target": item["target"]}

    def predict(self, test_data_input) -> List[Forecast]:
        bs = self.batch_size
        input_data = list(self._pack(test_data_input))
        is_univariate = input_data[0]["target"].ndim == 1
        if self.predict_batches_jointly:
            logger.info("chronos-2 cross-learning mode active")

        while True:
            try:
                if self._is_chronos2:
                    # chronos-2: list of {"target": np.array} dicts; takes batch_size + cross-learning flag.
                    quantiles, _ = self.pipeline.predict_quantiles(
                        inputs=input_data,
                        prediction_length=self.prediction_length,
                        batch_size=bs,
                        quantile_levels=self.quantile_levels,
                        predict_batches_jointly=self.predict_batches_jointly,
                    )
                else:
                    # chronos-bolt: predict_quantiles stacks ALL inputs into one
                    # tensor before internal batching, which OOMs on big configs
                    # (6k+ series). Chunk externally and concat.
                    chunk_size = bs
                    out_chunks = []
                    n_total = len(input_data)
                    for i in range(0, n_total, chunk_size):
                        chunk_tensors = [
                            torch.as_tensor(d["target"], dtype=torch.float32)
                            for d in input_data[i : i + chunk_size]
                        ]
                        q_chunk, _ = self.pipeline.predict_quantiles(
                            inputs=chunk_tensors,
                            prediction_length=self.prediction_length,
                            quantile_levels=self.quantile_levels,
                        )
                        out_chunks.append(q_chunk)
                        torch.cuda.empty_cache()
                    # q_chunk for chronos-bolt is a tensor of shape (B, T, Q).
                    if isinstance(out_chunks[0], list):
                        quantiles = []
                        for chunk in out_chunks:
                            quantiles.extend(chunk)
                    else:
                        quantiles = torch.cat(out_chunks, dim=0)
                        # Transpose (B, T, Q) → (B, Q, T) so per-item array
                        # matches QuantileForecast schema (forecast_arrays must
                        # have shape[0] == len(forecast_keys)).
                        quantiles = quantiles.permute(0, 2, 1)
                # chronos-2: list of Tensors → stack and permute
                if isinstance(quantiles, list):
                    quantiles = torch.stack(quantiles)
                    quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                else:
                    quantiles = quantiles.cpu().numpy()
                if is_univariate and quantiles.ndim == 4:
                    quantiles = quantiles.squeeze(-1)
                break
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                logger.warning(f"OOM at chronos batch_size={bs}, halving to {bs//2}")
                bs //= 2
                torch.cuda.empty_cache()

        forecasts = []
        for arr, ts in zip(quantiles, test_data_input):
            forecast_start = ts["start"] + len(ts["target"])
            forecasts.append(QuantileForecast(
                item_id=ts.get("item_id", None),
                forecast_arrays=arr,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=forecast_start,
            ))
        return forecasts


# --------------------------------------------------------------------------- #
# TimesFM-2.5-200M — Google. Vendored from timesfm2p5.ipynb.
# --------------------------------------------------------------------------- #

class TimesFmPredictor:
    def __init__(
        self,
        prediction_length: int,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        per_core_batch_size: int = 128,
    ):
        from timesfm.timesfm_2p5 import timesfm_2p5_torch
        # HF's hub_mixin.from_pretrained injects 'proxies' (etc.) into the
        # constructor, which TimesFM_2p5_200M_torch.__init__ rejects.
        # Bypass it by calling _from_pretrained directly with only the kwargs
        # that the timesfm code expects.
        self.tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch._from_pretrained(
            model_id=model_name,
            revision=None,
            cache_dir=None,
            local_files_only=False,
            token=None,
        )
        self.prediction_length = prediction_length
        self.quantile_levels = QUANTILE_LEVELS
        self.per_core_batch_size = per_core_batch_size

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        from timesfm import configs
        forecast_outputs = []
        # Round prediction_length up to a multiple of patch size for safe horizon bound.
        max_horizon = (
            (self.prediction_length + self.tfm.model.o - 1) // self.tfm.model.o
        ) * self.tfm.model.o
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            max_context = 0
            for entry in batch:
                arr = np.array(entry["target"])
                if arr.shape[0] > max_context:
                    max_context = arr.shape[0]
                context.append(arr)
            max_context = ((max_context + self.tfm.model.p - 1) // self.tfm.model.p) * self.tfm.model.p
            max_context = min(15360, max_context)
            # Disable infer_is_positive and force_flip_invariance — they crash to
            # NaN on datasets with non-strictly-positive values (electricity, ETT,
            # bitbrains, weather, etc.). Tighter max_horizon = less compile cache.
            self.tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    infer_is_positive=False,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=False,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=self.per_core_batch_size,
                ),
            )
            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length,
                inputs=context,
            )
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
            torch.cuda.empty_cache()
        forecast_outputs = np.concatenate(forecast_outputs)

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecasts.append(QuantileForecast(
                item_id=ts.get("item_id", None),
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=ts["start"] + len(ts["target"]),
            ))
        return forecasts


# --------------------------------------------------------------------------- #
# Moirai-2.0-R-{small,base,large} — Salesforce. Vendored from moirai2.ipynb.
# --------------------------------------------------------------------------- #

def _get_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class MoiraiQuantilePredictor:
    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        context_length: int = 4000,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        device: str = "auto",
        batch_size: int = 512,
        quantile_levels: tuple = tuple(QUANTILE_LEVELS),
    ):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = _get_device(device)
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_path),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        ).to(self.device)

    def predict(self, test_data_input):
        while True:
            try:
                forecast_quantiles = []
                for batch in batcher(test_data_input, batch_size=self.batch_size):
                    past_target = [entry["target"] for entry in batch]
                    forecasts = self.model.predict(past_target)
                    forecast_quantiles.append(forecasts)
                forecast_quantiles = np.concatenate(forecast_quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                if self.batch_size <= 1:
                    raise
                logger.warning(f"OOM at moirai batch_size={self.batch_size}, halving")
                self.batch_size //= 2
                torch.cuda.empty_cache()

        out = []
        for item, ts in zip(forecast_quantiles, test_data_input):
            out.append(QuantileForecast(
                item_id=ts.get("item_id", None),
                forecast_arrays=item,
                start_date=ts["start"] + len(ts["target"]),
                forecast_keys=list(map(str, self.quantile_levels)),
            ))
        return out


# --------------------------------------------------------------------------- #
# Moirai-1.1-R-{small,base,large} — sample-based predictor.
# Adapted from moirai.ipynb pattern (Moirai-1.0 example).
# --------------------------------------------------------------------------- #

class MoiraiSamplePredictor:
    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        context_length: int = 4000,
        patch_size: int = 32,
        num_samples: int = 200,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        device: str = "auto",
        batch_size: int = 512,
        quantile_levels: tuple = tuple(QUANTILE_LEVELS),
    ):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.prediction_length = prediction_length
        self.device = _get_device(device)
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(model_path),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        ).to(self.device)

    def predict(self, test_data_input):
        while True:
            try:
                # Moirai-1.x: get a real GluonTS predictor via create_predictor
                # (the model itself doesn't expose .predict()). It produces
                # SampleForecast objects.
                gts_predictor = self.model.create_predictor(batch_size=self.batch_size)
                sample_forecasts = list(gts_predictor.predict(test_data_input))
                break
            except torch.cuda.OutOfMemoryError:
                if self.batch_size <= 1:
                    raise
                logger.warning(f"OOM at moirai-1.x batch_size={self.batch_size}, halving")
                self.batch_size //= 2
                torch.cuda.empty_cache()

        out = []
        for sf, ts in zip(sample_forecasts, test_data_input):
            samples = sf.samples  # (num_samples, T) or (num_samples, T, D)
            if samples.ndim == 3 and samples.shape[-1] == 1:
                samples = samples.squeeze(-1)
            qs = np.quantile(samples, q=list(self.quantile_levels), axis=0)  # (Q, T)
            out.append(QuantileForecast(
                item_id=ts.get("item_id", sf.item_id),
                forecast_arrays=qs.astype(np.float32),
                start_date=sf.start_date,
                forecast_keys=list(map(str, self.quantile_levels)),
            ))
        return out


# --------------------------------------------------------------------------- #
# TiRex-1.1-gifteval — NX-AI. Vendored from tirex.ipynb.
# --------------------------------------------------------------------------- #

@dataclass
class TiRexGiftEvalWrapper:
    model: Any
    pred_len: int = 32
    resample_strategy: Optional[str] = "frequency"
    item_id_attr: bool = True

    @classmethod
    def from_pretrained(cls, model_name: str, prediction_length: int, device: str = "cuda"):
        from tirex import load_model
        model = load_model(model_name, device=device)
        return cls(model=model, pred_len=prediction_length)

    @property
    def prediction_length(self) -> int:
        return self.pred_len

    def predict(self, test_data_input):
        return self.model.forecast_gluon(
            test_data_input,
            prediction_length=self.pred_len,
            output_type="gluonts",
            resample_strategy=self.resample_strategy,
        )


# --------------------------------------------------------------------------- #
# Component registry: name → factory(prediction_length, **opts) → Predictor
# --------------------------------------------------------------------------- #

def build_predictor(
    component: str,
    prediction_length: int,
    device: str = "cuda",
    **opts,
):
    if component == "chronos-2":
        return ChronosPredictor(
            model_name="amazon/chronos-2",
            prediction_length=prediction_length,
            batch_size=opts.get("batch_size", 100),
            predict_batches_jointly=True,
            device_map=device,
            torch_dtype="float32",
        )
    if component == "chronos-bolt-base":
        return ChronosPredictor(
            model_name="amazon/chronos-bolt-base",
            prediction_length=prediction_length,
            batch_size=opts.get("batch_size", 256),
            device_map=device,
            torch_dtype="float32",
        )
    if component == "chronos-bolt-small":
        return ChronosPredictor(
            model_name="amazon/chronos-bolt-small",
            prediction_length=prediction_length,
            batch_size=opts.get("batch_size", 512),
            device_map=device,
            torch_dtype="float32",
        )
    if component == "timesfm-2.5":
        return TimesFmPredictor(
            prediction_length=prediction_length,
            model_name="google/timesfm-2.5-200m-pytorch",
            per_core_batch_size=opts.get("per_core_batch_size", 128),
        )
    if component == "moirai-2.0-R-small":
        return MoiraiQuantilePredictor(
            model_path="Salesforce/moirai-2.0-R-small",
            prediction_length=prediction_length,
            context_length=opts.get("context_length", 4000),
            batch_size=opts.get("batch_size", 512),
            device=device,
        )
    if component == "moirai-1.1-R-large":
        return MoiraiSamplePredictor(
            model_path="Salesforce/moirai-1.1-R-large",
            prediction_length=prediction_length,
            context_length=opts.get("context_length", 4000),
            patch_size=opts.get("patch_size", 32),
            num_samples=opts.get("num_samples", 200),
            batch_size=opts.get("batch_size", 256),
            device=device,
        )
    if component == "moirai-1.1-R-base":
        return MoiraiSamplePredictor(
            model_path="Salesforce/moirai-1.1-R-base",
            prediction_length=prediction_length,
            context_length=opts.get("context_length", 4000),
            patch_size=opts.get("patch_size", 32),
            num_samples=opts.get("num_samples", 200),
            batch_size=opts.get("batch_size", 512),
            device=device,
        )
    if component == "tirex-1.1-gifteval":
        return TiRexGiftEvalWrapper.from_pretrained(
            "NX-AI/TiRex-1.1-gifteval",
            prediction_length=prediction_length,
            device=device,
        )
    raise KeyError(f"Unknown component: {component}")
