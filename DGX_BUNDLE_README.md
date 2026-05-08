# DGX Bundle

This bundle is the portable handoff package for running the current Nedbank competition search on a stronger machine.

## What is inside

- core repo scripts needed for the active rolling-panel search path
- docs explaining the current best branch and run order
- `requirements-dgx.txt`
- `run_dgx_search.sh`
- `dgx_search_runner.py`
- `DGX_AGENT_PROMPT.md`
- current recommended upload packs

## What is not inside

- the raw competition data, unless you copy it separately

You still need to place these in the bundle root before running:

- `Train.csv`
- `Test.csv`
- `SampleSubmission.csv`
- `transactions_features/transactions_features.parquet`
- `financials_features/financials_features.parquet`
- `demographics_clean/demographics_clean.parquet`

## Fast start

```bash
python3 -m pip install -r requirements-dgx.txt
python3 dgx_search_runner.py --data-dir . --pseudo-public-splits 960
```

## Read first

1. `DGX_AGENT_PROMPT.md`
2. `docs/competition-playbook.md`
3. `docs/rolling-panel-public-calibration-lab.md`
4. `docs/rolling-panel-public-stability-lab.md`
5. `docs/rolling-panel-supervision-lab.md`
