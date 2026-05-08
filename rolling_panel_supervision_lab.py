from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import RepeatedKFold

from anchor_residual_lab import build_current_public_best_anchor
from artifact_blend_lab import load_seed_candidates
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import ModelResult, ModelSpec, build_model_registry, build_run_directory, filter_model_registry
from submission_eval_utils import write_submission_evaluation_artifacts
from temporal_holiday_stack import run_temporal_cv_for_model


FEATURE_PIPELINE_VERSION = "v17_rolling_panel_supervision_lab"
RANDOM_STATE = 42
CURRENT_CUTOFF = (2015, 10)
PANEL_START = (2014, 1)
PANEL_END = (2015, 10)
DEFAULT_INCLUDE_MODELS = "xgb_conservative_v3,catboost_conservative_v1,hgb_conservative_v3"
MONTH_COLUMN_RE = re.compile(r"^(?P<prefix>.+)_(?P<year>\d{4})_(?P<month>\d{2})$")
HOLIDAY_MONTH_SET = {11, 12, 1}


@dataclass(frozen=True)
class PanelArrays:
    unique_ids: np.ndarray
    split: np.ndarray
    target_current: np.ndarray
    demo_frame: pd.DataFrame
    eval_fin_missing_flag: np.ndarray
    count_matrix: np.ndarray
    abs_matrix: np.ndarray
    net_matrix: np.ndarray
    account_matrix: np.ndarray
    type_matrix: np.ndarray
    batch_matrix: np.ndarray


@dataclass(frozen=True)
class CandidatePrediction:
    name: str
    family: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    member_names: str
    mean_abs_log_shift: float
    selection_score: float
    oof_rmsle: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a rolling monthly-panel supervision stack from the cached dense monthly panel, then "
            "blend the new temporal branch against the current public-best anchor."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v5_dense_monthly_panel_stack.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_panel_supervision_lab"))
    parser.add_argument("--run-name", type=str, default="rolling_panel_supervision_lab")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--include-models", type=str, default=DEFAULT_INCLUDE_MODELS)
    parser.add_argument(
        "--pseudo-cutoff-start",
        type=str,
        default="2014-03",
        help="First pseudo cutoff month in YYYY-MM format.",
    )
    parser.add_argument(
        "--pseudo-cutoff-end",
        type=str,
        default="2015-07",
        help="Last pseudo cutoff month in YYYY-MM format.",
    )
    parser.add_argument(
        "--shift-penalty",
        type=float,
        default=0.08,
        help="Penalty multiplier applied to mean absolute log shift from the current public-best anchor.",
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start
    while (year, month) <= end:
        months.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return months


PANEL_MONTHS = month_range(PANEL_START, PANEL_END)
PANEL_MONTH_LABELS = [f"{year}_{month:02d}" for year, month in PANEL_MONTHS]
PANEL_MONTH_TO_INDEX = {label: index for index, label in enumerate(PANEL_MONTH_LABELS)}
CURRENT_CUTOFF_INDEX = PANEL_MONTH_TO_INDEX[f"{CURRENT_CUTOFF[0]}_{CURRENT_CUTOFF[1]:02d}"]


def add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    shifted_year, shifted_month_zero = divmod(total, 12)
    return shifted_year, shifted_month_zero + 1


def parse_month_arg(value: str) -> tuple[int, int]:
    year_text, month_text = value.split("-", maxsplit=1)
    return int(year_text), int(month_text)


def cutoff_name(year: int, month: int) -> str:
    return f"cutoff_{year}_{month:02d}"


def holiday_overlap_count(target_months: list[tuple[int, int]]) -> int:
    return sum(1 for _year, month in target_months if month in HOLIDAY_MONTH_SET)


def is_exact_holiday_window(target_months: list[tuple[int, int]]) -> bool:
    return [month for _year, month in target_months] == [11, 12, 1]


def pseudo_sample_weight(cutoff_year: int, cutoff_month: int) -> float:
    cutoff_index = PANEL_MONTH_TO_INDEX[f"{cutoff_year}_{cutoff_month:02d}"]
    months_ago = CURRENT_CUTOFF_INDEX - cutoff_index
    recency_score = max(0.0, 1.0 - (months_ago / 20.0))
    target_months = [add_months(cutoff_year, cutoff_month, offset) for offset in (1, 2, 3)]
    overlap_score = holiday_overlap_count(target_months) / 3.0
    exact_bonus = 1.0 if is_exact_holiday_window(target_months) else 0.0
    q4_bonus = 1.0 if cutoff_month in {8, 9, 10, 11, 12} else 0.0
    weight = 0.15 + 0.45 * recency_score + 0.20 * overlap_score + 0.10 * exact_bonus + 0.10 * q4_bonus
    return float(np.clip(weight, 0.20, 0.90))


def exact_month_columns(columns: list[str], prefix: str) -> list[str]:
    matched: list[tuple[int, int, str]] = []
    prefix_with_sep = f"{prefix}_"
    for column in columns:
        if not column.startswith(prefix_with_sep):
            continue
        suffix = column[len(prefix_with_sep) :]
        match = re.fullmatch(r"(\d{4})_(\d{2})", suffix)
        if match is None:
            continue
        matched.append((int(match.group(1)), int(match.group(2)), column))
    matched.sort()
    return [column for _year, _month, column in matched]


def offset_block(matrix: np.ndarray, cutoff_index: int, offsets: list[int]) -> np.ndarray:
    row_count = matrix.shape[0]
    output = np.zeros((row_count, len(offsets)), dtype=np.float64)
    for idx, offset in enumerate(offsets):
        source_index = cutoff_index + offset
        if 0 <= source_index < matrix.shape[1]:
            output[:, idx] = matrix[:, source_index]
    return output


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = np.abs(denominator) > 1e-9
    result[valid] = numerator[valid] / denominator[valid]
    return result


def build_panel_arrays(cache_path: Path) -> PanelArrays:
    cache_frame = pl.read_parquet(cache_path).sort("UniqueID").to_pandas()
    panel_count_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_count")
    panel_abs_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_abs_sum")
    panel_net_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_net_amount")
    panel_account_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_account_nunique")
    panel_type_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_type_nunique")
    panel_batch_columns = exact_month_columns(cache_frame.columns.tolist(), "txn_panel_batch_nunique")

    expected_month_count = len(PANEL_MONTH_LABELS)
    for prefix, columns in {
        "txn_panel_count": panel_count_columns,
        "txn_panel_abs_sum": panel_abs_columns,
        "txn_panel_net_amount": panel_net_columns,
        "txn_panel_account_nunique": panel_account_columns,
        "txn_panel_type_nunique": panel_type_columns,
        "txn_panel_batch_nunique": panel_batch_columns,
    }.items():
        if len(columns) != expected_month_count:
            raise ValueError(f"{prefix} expected {expected_month_count} exact monthly columns, found {len(columns)}.")

    demo_columns = [column for column in cache_frame.columns if column.startswith("demo_")]
    demo_frame = cache_frame[["UniqueID"] + demo_columns].copy()

    return PanelArrays(
        unique_ids=cache_frame["UniqueID"].to_numpy(),
        split=cache_frame["__split__"].to_numpy(),
        target_current=cache_frame["next_3m_txn_count"].fillna(0.0).to_numpy(dtype=np.float64),
        demo_frame=demo_frame,
        eval_fin_missing_flag=cache_frame.get("fin_missing_flag", pd.Series(np.zeros(len(cache_frame)))).fillna(0.0).to_numpy(
            dtype=np.float64
        ),
        count_matrix=cache_frame[panel_count_columns].fillna(0.0).to_numpy(dtype=np.float64),
        abs_matrix=cache_frame[panel_abs_columns].fillna(0.0).to_numpy(dtype=np.float64),
        net_matrix=cache_frame[panel_net_columns].fillna(0.0).to_numpy(dtype=np.float64),
        account_matrix=cache_frame[panel_account_columns].fillna(0.0).to_numpy(dtype=np.float64),
        type_matrix=cache_frame[panel_type_columns].fillna(0.0).to_numpy(dtype=np.float64),
        batch_matrix=cache_frame[panel_batch_columns].fillna(0.0).to_numpy(dtype=np.float64),
    )


def build_cutoff_feature_frame(
    arrays: PanelArrays,
    cutoff_index: int,
    row_origin: str,
    sample_weight: float,
    use_actual_target_for_train: bool,
) -> pd.DataFrame:
    cutoff_year, cutoff_month = PANEL_MONTHS[cutoff_index]
    target_months = [add_months(cutoff_year, cutoff_month, offset) for offset in (1, 2, 3)]
    target_start_year, target_start_month = target_months[0]
    months_ago = CURRENT_CUTOFF_INDEX - cutoff_index

    recent1_count = offset_block(arrays.count_matrix, cutoff_index, [0])[:, 0]
    recent2_count = offset_block(arrays.count_matrix, cutoff_index, [-1])[:, 0]
    recent3_count_block = offset_block(arrays.count_matrix, cutoff_index, [-2, -1, 0])
    prev3_count_block = offset_block(arrays.count_matrix, cutoff_index, [-5, -4, -3])
    last6_count_block = offset_block(arrays.count_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])
    prev6_count_block = offset_block(arrays.count_matrix, cutoff_index, [-11, -10, -9, -8, -7, -6])
    last12_count_block = offset_block(arrays.count_matrix, cutoff_index, list(range(-11, 1)))

    recent3_abs_block = offset_block(arrays.abs_matrix, cutoff_index, [-2, -1, 0])
    prev3_abs_block = offset_block(arrays.abs_matrix, cutoff_index, [-5, -4, -3])
    last6_abs_block = offset_block(arrays.abs_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])
    last12_abs_block = offset_block(arrays.abs_matrix, cutoff_index, list(range(-11, 1)))

    recent3_net_block = offset_block(arrays.net_matrix, cutoff_index, [-2, -1, 0])
    prev3_net_block = offset_block(arrays.net_matrix, cutoff_index, [-5, -4, -3])
    last6_net_block = offset_block(arrays.net_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])
    last12_net_block = offset_block(arrays.net_matrix, cutoff_index, list(range(-11, 1)))

    last6_account_block = offset_block(arrays.account_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])
    last6_type_block = offset_block(arrays.type_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])
    last6_batch_block = offset_block(arrays.batch_matrix, cutoff_index, [-5, -4, -3, -2, -1, 0])

    target_prev1_count_block = offset_block(arrays.count_matrix, cutoff_index, [-11, -10, -9])
    target_prev1_abs_block = offset_block(arrays.abs_matrix, cutoff_index, [-11, -10, -9])
    target_prev1_net_block = offset_block(arrays.net_matrix, cutoff_index, [-11, -10, -9])

    yoy_recent3_count_block = offset_block(arrays.count_matrix, cutoff_index, [-14, -13, -12])
    yoy_recent3_abs_block = offset_block(arrays.abs_matrix, cutoff_index, [-14, -13, -12])
    yoy_recent3_net_block = offset_block(arrays.net_matrix, cutoff_index, [-14, -13, -12])

    future_target_block = offset_block(arrays.count_matrix, cutoff_index, [1, 2, 3])

    history_months_available = np.full(len(arrays.unique_ids), cutoff_index + 1, dtype=np.float64)
    holiday_overlap = float(holiday_overlap_count(target_months))
    exact_holiday = 1.0 if is_exact_holiday_window(target_months) else 0.0

    feature_frame = arrays.demo_frame.copy()
    if "demo_age_years" in feature_frame.columns:
        age = pd.to_numeric(feature_frame["demo_age_years"], errors="coerce")
        feature_frame["demo_age_years"] = age - (months_ago / 12.0)

    feature_frame["txn_recent_3m_count"] = recent3_count_block.sum(axis=1)
    feature_frame["txn_recent_3m_mean"] = recent3_count_block.mean(axis=1)
    feature_frame["txn_recent_3m_std"] = recent3_count_block.std(axis=1)
    feature_frame["txn_prev_3m_count"] = prev3_count_block.sum(axis=1)
    feature_frame["txn_last6_count_sum"] = last6_count_block.sum(axis=1)
    feature_frame["txn_last6_count_mean"] = last6_count_block.mean(axis=1)
    feature_frame["txn_last6_count_std"] = last6_count_block.std(axis=1)
    feature_frame["txn_last12_count_sum"] = last12_count_block.sum(axis=1)
    feature_frame["txn_last12_count_mean"] = last12_count_block.mean(axis=1)
    feature_frame["txn_last12_count_std"] = last12_count_block.std(axis=1)
    feature_frame["txn_recent_vs_prev3_count_ratio"] = safe_ratio(
        feature_frame["txn_recent_3m_count"].to_numpy(dtype=np.float64),
        feature_frame["txn_prev_3m_count"].to_numpy(dtype=np.float64),
    )
    feature_frame["txn_recent_share_count"] = safe_ratio(
        feature_frame["txn_recent_3m_count"].to_numpy(dtype=np.float64),
        feature_frame["txn_last12_count_sum"].to_numpy(dtype=np.float64),
    )
    feature_frame["txn_active_months_total"] = (last12_count_block > 0).sum(axis=1).astype(np.float64)
    feature_frame["txn_months_count_eq_1"] = (last12_count_block == 1).sum(axis=1).astype(np.float64)
    feature_frame["txn_months_count_le_2"] = ((last12_count_block > 0) & (last12_count_block <= 2)).sum(axis=1).astype(
        np.float64
    )
    feature_frame["txn_sparse_month_share_eq_1"] = safe_ratio(
        feature_frame["txn_months_count_eq_1"].to_numpy(dtype=np.float64),
        np.maximum(feature_frame["txn_active_months_total"].to_numpy(dtype=np.float64), 1.0),
    )
    feature_frame["txn_sparse_month_share_le_2"] = safe_ratio(
        feature_frame["txn_months_count_le_2"].to_numpy(dtype=np.float64),
        np.maximum(feature_frame["txn_active_months_total"].to_numpy(dtype=np.float64), 1.0),
    )

    feature_frame["txn_recent_3m_abs_sum"] = recent3_abs_block.sum(axis=1)
    feature_frame["txn_prev_3m_abs_sum"] = prev3_abs_block.sum(axis=1)
    feature_frame["txn_last6_abs_sum"] = last6_abs_block.sum(axis=1)
    feature_frame["txn_last12_abs_sum"] = last12_abs_block.sum(axis=1)
    feature_frame["txn_recent_vs_prev3_abs_ratio"] = safe_ratio(
        feature_frame["txn_recent_3m_abs_sum"].to_numpy(dtype=np.float64),
        feature_frame["txn_prev_3m_abs_sum"].to_numpy(dtype=np.float64),
    )

    feature_frame["txn_recent_3m_net_sum"] = recent3_net_block.sum(axis=1)
    feature_frame["txn_prev_3m_net_sum"] = prev3_net_block.sum(axis=1)
    feature_frame["txn_last6_net_sum"] = last6_net_block.sum(axis=1)
    feature_frame["txn_last12_net_sum"] = last12_net_block.sum(axis=1)
    feature_frame["txn_recent_vs_prev3_net_ratio"] = safe_ratio(
        np.abs(feature_frame["txn_recent_3m_net_sum"].to_numpy(dtype=np.float64)),
        np.abs(feature_frame["txn_prev_3m_net_sum"].to_numpy(dtype=np.float64)),
    )

    feature_frame["txn_panel_account_last6_mean"] = last6_account_block.mean(axis=1)
    feature_frame["txn_panel_account_last6_max"] = last6_account_block.max(axis=1)
    feature_frame["txn_panel_type_last6_mean"] = last6_type_block.mean(axis=1)
    feature_frame["txn_panel_type_last6_max"] = last6_type_block.max(axis=1)
    feature_frame["txn_panel_batch_last6_mean"] = last6_batch_block.mean(axis=1)
    feature_frame["txn_panel_batch_last6_max"] = last6_batch_block.max(axis=1)

    feature_frame["txn_target_prev1_count"] = target_prev1_count_block.sum(axis=1)
    feature_frame["txn_target_prev1_abs_sum"] = target_prev1_abs_block.sum(axis=1)
    feature_frame["txn_target_prev1_net_sum"] = target_prev1_net_block.sum(axis=1)
    feature_frame["txn_target_prev1_vs_recent3_ratio"] = safe_ratio(
        feature_frame["txn_target_prev1_count"].to_numpy(dtype=np.float64),
        np.maximum(feature_frame["txn_recent_3m_count"].to_numpy(dtype=np.float64), 1.0),
    )

    feature_frame["txn_recent3_yoy_count_delta"] = recent3_count_block.sum(axis=1) - yoy_recent3_count_block.sum(axis=1)
    feature_frame["txn_recent3_yoy_abs_delta"] = recent3_abs_block.sum(axis=1) - yoy_recent3_abs_block.sum(axis=1)
    feature_frame["txn_recent3_yoy_net_delta"] = recent3_net_block.sum(axis=1) - yoy_recent3_net_block.sum(axis=1)
    feature_frame["txn_recent_count_delta_last1_last2"] = recent1_count - recent2_count
    feature_frame["txn_recent_count_delta_last2_last3"] = recent2_count - recent3_count_block[:, 0]

    for lag in range(12):
        feature_frame[f"panel_count_lag_{lag:02d}"] = offset_block(arrays.count_matrix, cutoff_index, [-lag])[:, 0]
    for lag in range(6):
        feature_frame[f"panel_abs_lag_{lag:02d}"] = offset_block(arrays.abs_matrix, cutoff_index, [-lag])[:, 0]
        feature_frame[f"panel_net_lag_{lag:02d}"] = offset_block(arrays.net_matrix, cutoff_index, [-lag])[:, 0]

    feature_frame["panel_history_months_available"] = history_months_available
    feature_frame["panel_cutoff_year"] = cutoff_year
    feature_frame["panel_cutoff_month"] = cutoff_month
    feature_frame["panel_target_start_year"] = target_start_year
    feature_frame["panel_target_start_month"] = target_start_month
    feature_frame["panel_target_holiday_overlap"] = holiday_overlap
    feature_frame["panel_target_exact_holiday"] = exact_holiday
    feature_frame["panel_target_contains_nov"] = 1.0 if any(month == 11 for _year, month in target_months) else 0.0
    feature_frame["panel_target_contains_dec"] = 1.0 if any(month == 12 for _year, month in target_months) else 0.0
    feature_frame["panel_target_contains_jan"] = 1.0 if any(month == 1 for _year, month in target_months) else 0.0
    feature_frame["panel_cutoff_label"] = f"{cutoff_year}_{cutoff_month:02d}"
    feature_frame["panel_target_start_label"] = f"{target_start_year}_{target_start_month:02d}"
    feature_frame["eval_fin_missing_flag"] = arrays.eval_fin_missing_flag
    feature_frame["__split__"] = arrays.split
    feature_frame["UniqueID"] = arrays.unique_ids
    feature_frame["row_origin"] = row_origin
    feature_frame["sample_weight"] = sample_weight

    if use_actual_target_for_train:
        feature_frame["next_3m_txn_count"] = arrays.target_current
    else:
        feature_frame["next_3m_txn_count"] = future_target_block.sum(axis=1)

    return feature_frame


def build_current_and_pseudo_frames(
    arrays: PanelArrays,
    pseudo_cutoffs: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_full = build_cutoff_feature_frame(
        arrays=arrays,
        cutoff_index=CURRENT_CUTOFF_INDEX,
        row_origin="current_2015_10",
        sample_weight=1.0,
        use_actual_target_for_train=True,
    )

    current_train = (
        current_full.loc[current_full["__split__"] == "train"]
        .drop(columns=["__split__"])
        .sort_values("UniqueID")
        .reset_index(drop=True)
    )
    current_test = (
        current_full.loc[current_full["__split__"] == "test"]
        .drop(columns=["__split__", "next_3m_txn_count"])
        .sort_values("UniqueID")
        .reset_index(drop=True)
    )
    current_test["row_origin"] = "test"

    pseudo_frames: list[pd.DataFrame] = []
    for cutoff_year, cutoff_month in pseudo_cutoffs:
        cutoff_index = PANEL_MONTH_TO_INDEX[f"{cutoff_year}_{cutoff_month:02d}"]
        pseudo_frames.append(
            build_cutoff_feature_frame(
                arrays=arrays,
                cutoff_index=cutoff_index,
                row_origin=cutoff_name(cutoff_year, cutoff_month),
                sample_weight=pseudo_sample_weight(cutoff_year, cutoff_month),
                use_actual_target_for_train=False,
            ).drop(columns=["__split__"])
        )
    pseudo_train = pd.concat(pseudo_frames, axis=0, ignore_index=True) if pseudo_frames else pd.DataFrame()
    return current_train, current_test, pseudo_train


def current_public_best_seed(
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    y_train: np.ndarray,
    train_fin_missing: np.ndarray,
    test_fin_missing: np.ndarray,
    base_dir: Path,
) -> ModelResult:
    seed_train_ids, seed_test_ids, seed_y_train, seeds = load_seed_candidates(base_dir)
    def reorder(values: np.ndarray, source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
        mapping = {value: index for index, value in enumerate(source_ids.tolist())}
        try:
            indices = np.array([mapping[value] for value in target_ids.tolist()], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Missing seed prediction for ID {exc.args[0]}.") from exc
        return values[indices]

    if set(seed_train_ids.tolist()) != set(train_ids.tolist()):
        raise ValueError("Seed train ID set does not align with rolling-panel train IDs.")
    if set(seed_test_ids.tolist()) != set(test_ids.tolist()):
        raise ValueError("Seed test ID set does not align with rolling-panel test IDs.")

    if not np.allclose(reorder(seed_y_train, seed_train_ids, train_ids), y_train):
        raise ValueError("Seed train targets do not align with rolling-panel targets.")

    aligned_seeds = {}
    for name, seed in seeds.items():
        aligned_seeds[name] = type(seed)(
            name=seed.name,
            family=seed.family,
            source_run_dir=seed.source_run_dir,
            oof_predictions=reorder(seed.oof_predictions, seed_train_ids, train_ids),
            test_predictions=reorder(seed.test_predictions, seed_test_ids, test_ids),
        )

    anchor_train, anchor_test, _public_mask_train, _public_mask_test, _threshold = build_current_public_best_anchor(
        aligned_seeds,
        train_fin_missing=train_fin_missing,
        test_fin_missing=test_fin_missing,
    )
    return ModelResult(
        spec=ModelSpec(name="publicbest_feedback_temporal_top10", family="seed_anchor", params={}),
        oof_rmsle=rmsle(y_train, anchor_train),
        fold_scores=[],
        oof_predictions=np.asarray(anchor_train, dtype=np.float64),
        test_predictions=np.asarray(anchor_test, dtype=np.float64),
    )


def blend_anchor_log(anchor_predictions: np.ndarray, model_predictions: np.ndarray, alpha: float) -> np.ndarray:
    anchor_log = np.log1p(np.clip(anchor_predictions, 0.0, None))
    model_log = np.log1p(np.clip(model_predictions, 0.0, None))
    return np.clip(np.expm1((1.0 - alpha) * anchor_log + alpha * model_log), 0.0, None)


def build_anchor_blend_candidates(
    anchor_result: ModelResult,
    model_results: list[ModelResult],
    current_train_df: pd.DataFrame,
    current_test_df: pd.DataFrame,
    y_train: np.ndarray,
    shift_penalty: float,
) -> list[CandidatePrediction]:
    alpha_grid_global = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]
    alpha_grid_masked = [0.10, 0.15, 0.20, 0.25]
    quantiles = [0.80, 0.90]

    candidates: list[CandidatePrediction] = []
    anchor_oof = anchor_result.oof_predictions
    anchor_test = anchor_result.test_predictions

    for result in model_results:
        for alpha in alpha_grid_global:
            oof_predictions = blend_anchor_log(anchor_oof, result.oof_predictions, alpha)
            test_predictions = blend_anchor_log(anchor_test, result.test_predictions, alpha)
            mean_abs_log_shift = float(np.mean(np.abs(np.log1p(oof_predictions) - np.log1p(anchor_oof))))
            oof_value = rmsle(y_train, oof_predictions)
            candidates.append(
                CandidatePrediction(
                    name=f"anchor_global_{result.spec.name}_a{str(alpha).replace('.', 'p')}",
                    family="anchor_global_log_blend",
                    oof_predictions=oof_predictions,
                    test_predictions=test_predictions,
                    member_names=f"{anchor_result.spec.name},{result.spec.name}",
                    mean_abs_log_shift=mean_abs_log_shift,
                    selection_score=float(oof_value + shift_penalty * mean_abs_log_shift),
                    oof_rmsle=float(oof_value),
                )
            )

        disagreement_train = np.abs(np.log1p(result.oof_predictions) - np.log1p(anchor_oof))
        disagreement_test = np.abs(np.log1p(result.test_predictions) - np.log1p(anchor_test))
        for quantile in quantiles:
            threshold = float(np.quantile(disagreement_train, quantile))
            train_mask = disagreement_train >= threshold
            test_mask = disagreement_test >= threshold
            top_pct = int(round((1.0 - quantile) * 100))
            for alpha in alpha_grid_masked:
                oof_predictions = np.array(anchor_oof, copy=True)
                test_predictions = np.array(anchor_test, copy=True)
                oof_predictions[train_mask] = blend_anchor_log(anchor_oof[train_mask], result.oof_predictions[train_mask], alpha)
                test_predictions[test_mask] = blend_anchor_log(anchor_test[test_mask], result.test_predictions[test_mask], alpha)
                mean_abs_log_shift = float(np.mean(np.abs(np.log1p(oof_predictions) - np.log1p(anchor_oof))))
                oof_value = rmsle(y_train, oof_predictions)
                candidates.append(
                    CandidatePrediction(
                        name=f"anchor_top{top_pct:02d}_{result.spec.name}_a{str(alpha).replace('.', 'p')}",
                        family="anchor_quantile_log_blend",
                        oof_predictions=oof_predictions,
                        test_predictions=test_predictions,
                        member_names=f"{anchor_result.spec.name},{result.spec.name}",
                        mean_abs_log_shift=mean_abs_log_shift,
                        selection_score=float(oof_value + shift_penalty * mean_abs_log_shift),
                        oof_rmsle=float(oof_value),
                    )
                )
    return candidates


def build_candidate_metrics(
    current_train_df: pd.DataFrame,
    y_train: np.ndarray,
    model_results: list[ModelResult],
    extra_candidates: list[CandidatePrediction],
    anchor_result: ModelResult,
) -> pd.DataFrame:
    segment_masks: dict[str, np.ndarray] = {"overall": np.ones(len(y_train), dtype=bool)}
    segment_masks["low_recent_3m"] = current_train_df["txn_recent_3m_count"].to_numpy(dtype=np.float64) <= 20.0
    segment_masks["active_months_le_8"] = current_train_df["txn_active_months_total"].to_numpy(dtype=np.float64) <= 8.0
    segment_masks["sparse_month_share_ge_0p5"] = (
        current_train_df["txn_sparse_month_share_le_2"].to_numpy(dtype=np.float64) >= 0.5
    )
    segment_masks["fin_missing"] = current_train_df["eval_fin_missing_flag"].to_numpy(dtype=np.float64) >= 1.0
    segment_masks["target_le_10"] = y_train <= 10.0
    segment_masks["target_le_25"] = y_train <= 25.0

    anchor_log = np.log1p(anchor_result.oof_predictions)
    rows: list[dict[str, object]] = []

    for result in model_results:
        row = {
            "candidate_name": result.spec.name,
            "candidate_type": "base",
            "family": result.spec.family,
            "oof_rmsle": float(result.oof_rmsle),
            "member_names": result.spec.name,
            "mean_abs_log_shift": float(np.mean(np.abs(np.log1p(result.oof_predictions) - anchor_log))),
        }
        for segment_name, mask in segment_masks.items():
            row[f"{segment_name}_rmsle"] = rmsle(y_train[mask], result.oof_predictions[mask]) if mask.sum() >= 50 else np.nan
        rows.append(row)

    for candidate in extra_candidates:
        row = {
            "candidate_name": candidate.name,
            "candidate_type": "anchor_blend",
            "family": candidate.family,
            "oof_rmsle": candidate.oof_rmsle,
            "member_names": candidate.member_names,
            "mean_abs_log_shift": candidate.mean_abs_log_shift,
        }
        for segment_name, mask in segment_masks.items():
            row[f"{segment_name}_rmsle"] = (
                rmsle(y_train[mask], candidate.oof_predictions[mask]) if mask.sum() >= 50 else np.nan
            )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    rank_columns_base = ["oof_rmsle", "mean_abs_log_shift"]
    segment_columns = [column for column in metrics.columns if column.endswith("_rmsle") and column not in {"oof_rmsle"}]
    for column in rank_columns_base + segment_columns:
        ascending = True
        metrics[f"rank_{column}"] = metrics[column].rank(method="average", ascending=ascending, na_option="keep")
    metrics["hedge_rank_mean"] = metrics[[f"rank_{column}" for column in rank_columns_base + segment_columns]].mean(
        axis=1,
        skipna=True,
    )
    return metrics.sort_values(["oof_rmsle", "hedge_rank_mean", "candidate_name"]).reset_index(drop=True)


def write_recommendation_pack(run_dir: Path, submission_map: dict[str, Path], candidate_metrics: pd.DataFrame) -> None:
    selection_dir = run_dir / "recommended_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    selection_sorted = candidate_metrics.sort_values(["selection_score", "oof_rmsle", "candidate_name"])
    top_candidates = selection_sorted.head(4)["candidate_name"].tolist()
    for index, candidate_name in enumerate(top_candidates, start=1):
        target = selection_dir / f"{index:02d}_{candidate_name}.csv"
        target.write_bytes(submission_map[candidate_name].read_bytes())

    lines = [
        "# Recommended Selection",
        "",
        "This pack is ordered by `selection_score = oof_rmsle + shift_penalty * mean_abs_log_shift`.",
        "",
    ]
    for index, candidate_name in enumerate(top_candidates, start=1):
        row = selection_sorted.loc[selection_sorted["candidate_name"] == candidate_name].iloc[0]
        lines.extend(
            [
                f"{index}. `{index:02d}_{candidate_name}.csv`",
                f"   - OOF RMSLE: `{row['oof_rmsle']:.6f}`",
                f"   - mean abs log shift: `{row['mean_abs_log_shift']:.6f}`",
                f"   - selection score: `{row['selection_score']:.6f}`",
                (
                    f"   - local reference score: `{row['local_reference_score']:.6f}`"
                    if "local_reference_score" in row and pd.notna(row["local_reference_score"])
                    else "   - local reference score: unavailable"
                ),
                "",
            ]
        )
    (selection_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = build_run_directory(args.output_dir, args.run_name)

    arrays = build_panel_arrays(args.feature_cache_path)
    pseudo_start = parse_month_arg(args.pseudo_cutoff_start)
    pseudo_end = parse_month_arg(args.pseudo_cutoff_end)
    pseudo_cutoffs = month_range(pseudo_start, pseudo_end)
    if pseudo_cutoffs and pseudo_cutoffs[-1] > (2015, 7):
        raise ValueError("Pseudo cutoffs must not exceed 2015-07 because the next 3 monthly panel targets would be unavailable.")

    current_train_df, current_test_df, pseudo_train_df = build_current_and_pseudo_frames(arrays, pseudo_cutoffs)
    current_train_df = current_train_df.sort_values("UniqueID").reset_index(drop=True)
    current_test_df = current_test_df.sort_values("UniqueID").reset_index(drop=True)
    pseudo_train_df = pseudo_train_df.sort_values(["row_origin", "UniqueID"]).reset_index(drop=True)

    train_ids = current_train_df["UniqueID"].to_numpy()
    test_ids = current_test_df["UniqueID"].to_numpy()
    y_train = current_train_df["next_3m_txn_count"].to_numpy(dtype=np.float64)

    feature_columns = [
        column
        for column in current_train_df.columns
        if column
        not in {
            "UniqueID",
            "next_3m_txn_count",
            "row_origin",
            "sample_weight",
            "eval_fin_missing_flag",
        }
    ]
    numeric_columns = current_train_df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    cv = RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_state)
    cv_splits = list(cv.split(current_train_df, y_train))

    model_registry = filter_model_registry(build_model_registry(), args.include_models)
    trained_results: list[ModelResult] = []
    for spec in model_registry:
        print(f"Running rolling-panel model {spec.name}...")
        result = run_temporal_cv_for_model(
            spec=spec,
            current_train_df=current_train_df,
            current_test_df=current_test_df,
            pseudo_train_df=pseudo_train_df,
            y_train=y_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            cv_splits=cv_splits,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        trained_results.append(result)
        print(f"  OOF RMSLE: {result.oof_rmsle:.6f}")

    train_fin_missing = arrays.eval_fin_missing_flag[arrays.split == "train"] >= 1.0
    test_fin_missing = arrays.eval_fin_missing_flag[arrays.split == "test"] >= 1.0
    anchor_result = current_public_best_seed(
        train_ids=train_ids,
        test_ids=test_ids,
        y_train=y_train,
        train_fin_missing=train_fin_missing,
        test_fin_missing=test_fin_missing,
        base_dir=args.data_dir.resolve(),
    )
    print(f"Current public-best anchor OOF RMSLE: {anchor_result.oof_rmsle:.6f}")

    anchor_blend_candidates = build_anchor_blend_candidates(
        anchor_result=anchor_result,
        model_results=trained_results,
        current_train_df=current_train_df,
        current_test_df=current_test_df,
        y_train=y_train,
        shift_penalty=args.shift_penalty,
    )

    candidate_metrics = build_candidate_metrics(
        current_train_df=current_train_df,
        y_train=y_train,
        model_results=[anchor_result] + trained_results,
        extra_candidates=anchor_blend_candidates,
        anchor_result=anchor_result,
    )
    candidate_metrics["selection_score"] = candidate_metrics["oof_rmsle"] + args.shift_penalty * candidate_metrics[
        "mean_abs_log_shift"
    ]

    model_rows = [
        {
            "model_name": result.spec.name,
            "family": result.spec.family,
            "oof_rmsle": result.oof_rmsle,
            "params_json": json.dumps(result.spec.params, sort_keys=True),
        }
        for result in [anchor_result] + trained_results
    ]
    pd.DataFrame(model_rows).sort_values(["oof_rmsle", "model_name"]).to_csv(run_dir / "base_model_scores.csv", index=False)
    oof_payload: dict[str, object] = {"UniqueID": train_ids, "next_3m_txn_count_true": y_train}
    test_payload: dict[str, object] = {"UniqueID": test_ids}
    submission_map: dict[str, Path] = {}

    submissions_dir = run_dir / "submissions"
    submissions_raw_dir = run_dir / "submissions_raw"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submissions_raw_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[CandidatePrediction] = [
        CandidatePrediction(
            name=result.spec.name,
            family=result.spec.family,
            oof_predictions=result.oof_predictions,
            test_predictions=result.test_predictions,
            member_names=result.spec.name,
            mean_abs_log_shift=float(
                np.mean(np.abs(np.log1p(result.oof_predictions) - np.log1p(anchor_result.oof_predictions)))
            ),
            selection_score=float(
                result.oof_rmsle
                + args.shift_penalty
                * np.mean(np.abs(np.log1p(result.oof_predictions) - np.log1p(anchor_result.oof_predictions)))
            ),
            oof_rmsle=float(result.oof_rmsle),
        )
        for result in [anchor_result] + trained_results
    ] + anchor_blend_candidates

    for candidate in all_candidates:
        oof_payload[f"pred_{candidate.name}"] = candidate.oof_predictions
        test_payload[f"pred_{candidate.name}"] = candidate.test_predictions
        submission_path = submissions_dir / f"{candidate.name}.csv"
        build_submission_frame(
            args.data_dir / "SampleSubmission.csv",
            test_ids,
            candidate.test_predictions,
            submission_format=args.submission_format,
        ).write_csv(submission_path)
        build_submission_frame(
            args.data_dir / "SampleSubmission.csv",
            test_ids,
            candidate.test_predictions,
            submission_format="raw",
        ).write_csv(submissions_raw_dir / f"{candidate.name}.csv")
        submission_map[candidate.name] = submission_path

    pl.DataFrame(oof_payload).write_parquet(run_dir / "oof_predictions.parquet")
    pl.DataFrame(test_payload).write_parquet(run_dir / "test_predictions.parquet")

    evaluation_frame = write_submission_evaluation_artifacts(
        run_dir=run_dir,
        submissions_dir=submissions_dir,
        candidate_names=candidate_metrics["candidate_name"].tolist(),
        data_dir=args.data_dir,
        submission_mode=args.submission_format,
    )
    if evaluation_frame is not None:
        candidate_metrics = candidate_metrics.merge(evaluation_frame, on="candidate_name", how="left")

    candidate_metrics.sort_values(["selection_score", "oof_rmsle", "candidate_name"]).to_csv(
        run_dir / "candidate_metrics.csv",
        index=False,
    )

    write_recommendation_pack(run_dir, submission_map, candidate_metrics)

    best_selection_row = candidate_metrics.sort_values(["selection_score", "oof_rmsle", "candidate_name"]).iloc[0]
    best_oof_row = candidate_metrics.sort_values(["oof_rmsle", "selection_score", "candidate_name"]).iloc[0]

    summary_lines = [
        "# Rolling Panel Supervision Lab",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Feature cache: `{args.feature_cache_path}`",
        f"- Current public-best anchor: `{anchor_result.spec.name}`",
        f"- Current public-best anchor OOF RMSLE: `{anchor_result.oof_rmsle:.6f}`",
        f"- Pseudo cutoffs: `{args.pseudo_cutoff_start}` through `{args.pseudo_cutoff_end}`",
        f"- Repeated CV: `{args.n_splits}` folds x `{args.n_repeats}` repeats",
        f"- Submission format: `{args.submission_format}`",
        f"- Best by selection score: `{best_selection_row['candidate_name']}` with OOF `{best_selection_row['oof_rmsle']:.6f}` and shift `{best_selection_row['mean_abs_log_shift']:.6f}`",
        f"- Best by raw OOF: `{best_oof_row['candidate_name']}` with OOF `{best_oof_row['oof_rmsle']:.6f}` and shift `{best_oof_row['mean_abs_log_shift']:.6f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidate_metrics.sort_values(["selection_score", "oof_rmsle", "candidate_name"]).head(10).to_dict(
        orient="records"
    ):
        summary_lines.append(
            f"- `{row['candidate_name']}` ({row['candidate_type']}): OOF `{row['oof_rmsle']:.6f}`, "
            f"shift `{row['mean_abs_log_shift']:.6f}`, selection `{row['selection_score']:.6f}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config_payload = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "current_cutoff": f"{CURRENT_CUTOFF[0]}-{CURRENT_CUTOFF[1]:02d}",
        "pseudo_cutoffs": [f"{year}-{month:02d}" for year, month in pseudo_cutoffs],
        "include_models": args.include_models,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "shift_penalty": args.shift_penalty,
        "submission_format": args.submission_format,
        "pseudo_rows": int(len(pseudo_train_df)),
        "current_train_rows": int(len(current_train_df)),
        "current_test_rows": int(len(current_test_df)),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
