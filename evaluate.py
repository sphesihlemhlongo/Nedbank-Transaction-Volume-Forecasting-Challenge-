"""
Nedbank Transaction Forecasting Challenge - Local Evaluation Script
==================================================================
Use this to score your predictions locally before submitting to Zindi.

Usage:
    python evaluate.py <submission.csv> <reference.csv> [raw|zindi_log|auto]

Examples:
    python evaluate.py my_submission.csv PublicReference.csv raw
    python evaluate.py my_upload_ready_submission.csv PublicReference.csv zindi_log
    python evaluate.py candidate.csv PublicReference.csv

The submission CSV must have columns: UniqueID, next_3m_txn_count.

Important:
    - `raw` expects raw transaction-count predictions.
    - `zindi_log` expects upload-ready values equal to np.log1p(raw_prediction).
    - `auto` prints both interpretations because the official local bundle and the
      platform instructions use different submission interfaces.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(diff))))


def rmsle_from_raw(y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float64)
    if np.any(y_pred_raw < 0):
        raise ValueError("Predictions must be non-negative for RMSLE.")
    return float(np.sqrt(np.mean(np.square(np.log1p(y_pred_raw) - np.log1p(y_true)))))


def zindi_log_rmse(y_true: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    if np.any(y_pred_log < 0):
        raise ValueError("Submitted log predictions must be non-negative.")
    return rmse(np.log1p(y_true), y_pred_log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a submission against a reference file. Use `raw` for count predictions, "
            "`zindi_log` for upload-ready np.log1p(prediction) files, or `auto` to print both."
        )
    )
    parser.add_argument("submission_path")
    parser.add_argument("reference_path")
    parser.add_argument(
        "submission_mode",
        nargs="?",
        default="auto",
        choices=("raw", "zindi_log", "auto"),
        help="Interpretation of the submission values. Defaults to auto.",
    )
    return parser.parse_args()


def validate_and_merge(submission_path: str, reference_path: str) -> pd.DataFrame:
    sub = pd.read_csv(submission_path)
    ref = pd.read_csv(reference_path)

    required_cols = {"UniqueID", "next_3m_txn_count"}
    if not required_cols.issubset(sub.columns):
        print(f"ERROR: Submission must have columns: {required_cols}")
        print(f"Found: {set(sub.columns)}")
        sys.exit(1)

    if not required_cols.issubset(ref.columns):
        print(f"ERROR: Reference must have columns: {required_cols}")
        print(f"Found: {set(ref.columns)}")
        sys.exit(1)

    merged = ref.merge(sub, on="UniqueID", suffixes=("_true", "_pred"))
    if len(merged) != len(ref):
        missing = set(ref["UniqueID"]) - set(sub["UniqueID"])
        print(f"ERROR: {len(missing)} UniqueIDs in reference not found in submission.")
        print(f"Expected {len(ref)} rows, matched {len(merged)}.")
        sys.exit(1)

    if merged["next_3m_txn_count_pred"].isna().any():
        n_nan = int(merged["next_3m_txn_count_pred"].isna().sum())
        print(f"ERROR: {n_nan} NaN values in predictions.")
        sys.exit(1)

    if merged["next_3m_txn_count_pred"].min() < 0:
        print("ERROR: Predictions must be non-negative.")
        sys.exit(1)

    return merged


def print_raw_score(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    score = rmsle_from_raw(y_true, y_pred)
    print(f"Interpretation: raw counts")
    print(f"Metric: RMSLE(raw counts)")
    print(f"Score: {score:.6f}")


def print_zindi_log_score(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    score = zindi_log_rmse(y_true, y_pred)
    print("Interpretation: upload-ready zindi_log values")
    print("Metric: RMSE(log1p(true), submitted)")
    print(f"Score: {score:.6f}")
    equivalent_raw = np.expm1(np.asarray(y_pred, dtype=np.float64))
    equivalent_rmsle = rmsle_from_raw(y_true, equivalent_raw)
    print(f"Equivalent count-space RMSLE: {equivalent_rmsle:.6f}")


def main() -> None:
    args = parse_args()
    merged = validate_and_merge(args.submission_path, args.reference_path)
    y_true = merged["next_3m_txn_count_true"].to_numpy(dtype=np.float64)
    y_pred = merged["next_3m_txn_count_pred"].to_numpy(dtype=np.float64)

    if args.submission_mode == "raw":
        print_raw_score(y_true, y_pred)
    elif args.submission_mode == "zindi_log":
        print_zindi_log_score(y_true, y_pred)
    else:
        print_raw_score(y_true, y_pred)
        print()
        print_zindi_log_score(y_true, y_pred)
        print()
        print("Auto mode note: pick the interpretation that matches how the file was generated.")
        print("Use `raw` for development predictions and `zindi_log` for upload-ready competition files.")

    print(f"Rows scored: {len(merged)}")


if __name__ == "__main__":
    main()
