# DGX Agent Prompt

You are inheriting a live competition repo for the Nedbank Transaction Volume Forecasting Challenge.

Operate as a rigorous Lead Data Engineer, ML Engineer, and competition researcher.

## Objective

Improve the public leaderboard score beyond the current best:

- current public-best score: `0.383097231`
- current public-best uploaded file: `02_isotonic_anchor_b0p08.csv`

The immediate goal is to generate 2 to 4 stronger upload-ready CSVs while preserving reproducibility.

## Non-negotiable constraints

- Use `polars` for loading, joining, filtering, and feature engineering.
- Do not drop rows from `Train.csv` or `Test.csv`.
- All enrichments must remain left joins onto the base population.
- Submission files must contain:
  - `UniqueID`
  - `next_3m_txn_count = np.log1p(raw_prediction)`
- Keep model training reproducible:
  - fixed seeds
  - deterministic folds where applicable
  - no manual spreadsheet post-processing
- Use only open-source tools and only the competition data.

## Current repo state

Important current branches:

- `rolling_panel_supervision_lab.py`
  - rolling monthly-panel supervised branch
- `rolling_panel_public_stability_lab.py`
  - stability-aware reranking around the rolling-panel anchor
- `rolling_panel_public_calibration_lab.py`
  - smooth global recalibration around the rolling-panel public-best file
- `rolling_panel_public_calibration_feedback_lab.py`
  - microblends around the new public-best isotonic candidate

Current practical anchor progression:

1. `03_anchor_global_xgb_a025.csv`
2. `02_isotonic_anchor_b0p08.csv`

The isotonic family has now proven itself publicly, so work should anchor on that file first.

## First actions

1. Install dependencies from `requirements-dgx.txt`.
2. Verify the dataset layout:
   - `Train.csv`
   - `Test.csv`
   - `SampleSubmission.csv`
   - `transactions_features/transactions_features.parquet`
   - `financials_features/financials_features.parquet`
   - `demographics_clean/demographics_clean.parquet`
3. Run:

```bash
python dgx_search_runner.py --data-dir . --pseudo-public-splits 960
```

4. Inspect:
   - `outputs/dgx_search_runner/<timestamp>/README.md`
   - latest `recommended_selection/` folders under:
     - `outputs/rolling_panel_public_stability_lab/`
     - `outputs/rolling_panel_public_calibration_lab/`
     - `outputs/rolling_panel_public_calibration_feedback_lab/`

## What to optimize next

Prioritize in this order:

1. descendants of `isotonic_anchor_b0p08`
   - finer isotonic microsteps
   - smooth global recalibration
   - tiny log-space blends toward stronger isotonic siblings
2. stricter pseudo-public stability
   - increase pseudo-public resamples if runtime allows
   - keep the ranking rule deterministic
3. broader recalibration only if the isotonic microblend family stalls
   - dual-feature positive recalibration
   - Huber or affine recalibration if they improve both OOF and pseudo-public stability

Do not spend the first iteration on:

- sparse mask corrections
- unrelated new model families
- broad prediction-surface changes detached from the current public-best file

## Required outputs

For every serious run, produce:

- `candidate_metrics.csv`
- `submission_evaluation.md`
- `recommended_selection/`
- a short `summary.md`

If a local public reference file exists, ensure the generated submissions are also scored through the repo’s `evaluate.py` logic via `submission_eval_utils.py`.

## Documentation discipline

When you improve or change the active search path, update:

- `docs/competition-playbook.md`
- `docs/experiments.md`
- `docs/WIKIFLOW.md`
- the lab-specific document under `docs/`

## Decision rule

Favor candidates that satisfy all three:

- better OOF RMSLE
- high pseudo-public win rate against the current public-best anchor
- low mean absolute log drift from the current public-best anchor

If there is a conflict, prefer the candidate that stays closer to the current public-best file unless the improvement is materially larger.
