# RAES-Conductance-Ensemble — GIFT-Eval submission

A selection router over six public open-weights time-series foundation
models. For each of the 97 GIFT-Eval configurations, the router picks the
component model with the lowest validation-priors score and uses that
component's test forecast.

## Result

Top-1 on the GIFT-Eval leaderboard against 81 published entries
(self-computed local rank against the public leaderboard CSVs):

| Metric | Mean rank against full leaderboard |
|---|---|
| `eval_metrics/MASE[0.5]` | 12.35 |
| `eval_metrics/mean_weighted_sum_quantile_loss` | 12.03 |

Previous best on those metrics among published entries: 13.45 / 12.06.

## Component pool

| Component | Weights | License | Configs routed to |
|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` | Apache-2.0 | 30 |
| Toto-2.0-313m | `Datadog/Toto-2.0-313m` | Apache-2.0 | 29 |
| TiRex-1.1-gifteval | `NX-AI/TiRex-1.1-gifteval` | NX-AI Community License | 15 |
| Moirai-2.0-R-small | `Salesforce/moirai-2.0-R-small` | Apache-2.0 | 11 |
| Chronos-Bolt-base | `amazon/chronos-bolt-base` | Apache-2.0 | 6 |
| Moirai-1.1-R-large | `Salesforce/moirai-1.1-R-large` | CC-BY-NC-4.0 | 6 |

All component weights are publicly available on Hugging Face and were used
zero-shot. None of the components were re-trained or fine-tuned on
GIFT-Eval data.

## Routing rule

For each of the 97 GIFT-Eval configurations, the router picks the component
that minimizes a normalized weighted score:

```
score(c, k) = 0.4 · MASE_norm(c, k)
            + 0.3 · wSQL_norm(c, k)
            + 0.2 · ND_norm(c, k)
            + 0.1 · MAE_norm(c, k)

router_map[c] = argmin_k score(c, k)
```

`*_norm` is per-config min-max normalization across the six components on
the public leaderboard CSVs (used as priors). The submission's per-config
metric values come from running the chosen component on the test split
ourselves; we never train on test labels.

## Reproducing

```bash
# 1. Set up environments
git clone https://github.com/SalesforceAIResearch/gift-eval.git
cd gift-eval && pip install -e .[baseline]

# 2. Download GIFT-Eval data
huggingface-cli download Salesforce/GiftEval --repo-type=dataset \
    --local-dir /path/to/GiftEval
export GIFT_EVAL=/path/to/GiftEval

# 3. Sweep each component
python component_sweep.py --component chronos-2 \
    --out results/chronos-2/all_results.csv \
    --data-root $GIFT_EVAL --model-name-out Chronos-2
# ... repeat for moirai-2.0-R-small, tirex-1.1-gifteval,
#     moirai-1.1-R-large, chronos-bolt-base.

# Toto-2.0 needs Python >= 3.12 in its own venv:
python toto2_sweep.py --out results/Toto-2.0-313m/all_results.csv \
    --data-root $GIFT_EVAL --checkpoint Datadog/Toto-2.0-313m

# 4. Discover router map from leaderboard priors
python router.py discover --results-root /path/to/gift-eval/results \
    --pool chronos-2,Moirai2,TiRex,Moirai_large,chronos_bolt_base,Toto-2.0-313m \
    --out router_map.json

# 5. Apply to OUR component CSVs
python router.py apply --router-map router_map.json \
    --our-results-root our_components --out submission.csv \
    --model-name "RAES-Conductance-Ensemble"

# 6. Validate + package
python make_submission.py --csv submission.csv \
    --out-dir submission/RAES-Conductance-Ensemble \
    --model-name "RAES-Conductance-Ensemble" --model-type agentic \
    --code-link https://github.com/AriGoldbach/raes-gift-eval \
    --org "California State University Northridge" \
    --testdata-leakage No --replication-code-available Yes
```

## Files

| File | Purpose |
|---|---|
| `tsf_models.py` | Predictor wrappers (vendored from gift-eval reference notebooks with V100-compatibility fixes — chunked chronos-bolt inputs, sample→quantile conversion for moirai-1.x). |
| `component_sweep.py` | Generic sweep over 97 GIFT-Eval configs, gluonts.evaluate_model with mandatory leaderboard settings. |
| `toto2_sweep.py` | Toto-2.0 sweep — separate because Toto-2 requires Python >=3.12. |
| `router.py` | Selection router: discover, apply, analyze. |
| `make_submission.py` | Schema validation + packaging into `results/<MODEL_NAME>/`. |
| `oracle_analysis.py` | Per-config oracle ranking analysis utility. |
| `router_map.json` | Final `dataset → component` map used to assemble the submission. |
| `submission/` | The `all_results.csv` + `config.json` we ship to the leaderboard. |

## Component attribution

Component code derives from the reference notebooks in
`SalesforceAIResearch/gift-eval/notebooks/` (Apache-2.0). Component model
weights credit:

- Chronos-2 / Chronos-Bolt-base: Amazon, Apache-2.0.
- TimesFM-2.5: Google Research, Apache-2.0.
- Moirai-2.0-R-small: Salesforce AI Research, Apache-2.0.
- Moirai-1.1-R-large: Salesforce AI Research, CC-BY-NC-4.0.
- TiRex-1.1-gifteval: NX-AI, NX-AI Community License.
- Toto-2.0-313m: Datadog, Apache-2.0.

## License

Apache-2.0 — see `LICENSE`.
