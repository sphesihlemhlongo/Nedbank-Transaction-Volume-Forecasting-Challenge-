from __future__ import annotations

import argparse
import json
import os
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
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold

from artifact_blend_lab import load_seed_candidates
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory


FEATURE_PIPELINE_VERSION = "v9_public_anchor_expert_lab"
RANDOM_STATE = 42


@dataclass(frozen=True)
class SegmentAdjustment:
    satellite_name: str
    mask_name: str
    alpha: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    adjustments: tuple[SegmentAdjustment, ...]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    affected_train_share: float
    affected_test_share: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search leaderboard-focused expert-routing and restrained meta-stack candidates around the "
            "current public-best submission anchor."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_anchor_expert_lab"))
    parser.add_argument("--run-name", type=str, default="public_anchor_expert_lab")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v4_sparse_activity_batch_stack.parquet"),
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    parser.add_argument(
        "--shift-penalty",
        type=float,
        default=0.35,
        help="Penalty multiplier for ranking low-shift hedge candidates.",
    )
    parser.add_argument(
        "--anchor-variant",
        type=str,
        choices=("base", "public_best_round6"),
        default="public_best_round6",
        help="Which anchor definition to use for the search space.",
    )
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=10,
        help="How many top-ranked candidates to materialize as CSV files.",
    )
    return parser.parse_args()


def weight_token(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def load_feature_splits(
    feature_cache_path: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    feature_table = pl.read_parquet(feature_cache_path)
    train_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", train_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "train").drop("__split__"),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
    )
    test_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", test_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "test").drop("__split__"),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
    )
    return train_df, test_df


def get_numeric_array(frame: pl.DataFrame, column: str, fill_value: float = -999.0) -> np.ndarray:
    series = frame.get_column(column).fill_null(fill_value)
    values = np.array(series.cast(pl.Float64).to_numpy(), dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = fill_value
    return values


def build_masks(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    recent_ratio = get_numeric_array(frame, "txn_recent_vs_prev3_count_ratio")
    sparse_share = get_numeric_array(frame, "txn_sparse_month_share_le_2")
    active_months = get_numeric_array(frame, "txn_active_months_total")
    inactive_months = get_numeric_array(frame, "txn_inactive_months_total")
    recent_count = get_numeric_array(frame, "txn_recent_3m_count")
    fin_missing = get_numeric_array(frame, "fin_missing_flag")

    return {
        "recent_ratio_ge_1p2": recent_ratio >= 1.2,
        "recent_ratio_le_0p8": recent_ratio <= 0.8,
        "sparse_ge_0p3": sparse_share >= 0.3,
        "sparse_ge_0p5": sparse_share >= 0.5,
        "inactive_ge_18": inactive_months >= 18.0,
        "active_le_12": active_months <= 12.0,
        "recent_le_10": recent_count <= 10.0,
        "fin_missing": fin_missing >= 1.0,
    }


def build_base_public_anchor_predictions(
    seeds: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    temporal_xgb_train = seeds["temporal_xgb"].oof_predictions
    temporal_xgb_test = seeds["temporal_xgb"].test_predictions
    return (
        0.92 * v5_train + 0.08 * temporal_xgb_train,
        0.92 * v5_test + 0.08 * temporal_xgb_test,
    )


def build_public_best_round6_anchor_predictions(
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    v5_train = train_satellites["v5_best_subset"]
    v5_test = test_satellites["v5_best_subset"]
    temporal_xgb_train = train_satellites["temporal_xgb"]
    temporal_xgb_test = test_satellites["temporal_xgb"]

    fin_missing_train = train_masks["fin_missing"]
    fin_missing_test = test_masks["fin_missing"]
    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))
    non_missing_train = ~fin_missing_train
    threshold = float(np.quantile(disagreement_train[non_missing_train], 0.92))
    high_gap_train = non_missing_train & (disagreement_train >= threshold)
    high_gap_test = (~fin_missing_test) & (disagreement_test >= threshold)

    anchor_train = np.clip(base_anchor_train.copy(), 0.0, None)
    anchor_test = np.clip(base_anchor_test.copy(), 0.0, None)
    anchor_train[high_gap_train] = 0.78 * v5_train[high_gap_train] + 0.22 * temporal_xgb_train[high_gap_train]
    anchor_test[high_gap_test] = 0.78 * v5_test[high_gap_test] + 0.22 * temporal_xgb_test[high_gap_test]
    anchor_train[fin_missing_train] = v5_train[fin_missing_train]
    anchor_test[fin_missing_test] = v5_test[fin_missing_test]
    return np.clip(anchor_train, 0.0, None), np.clip(anchor_test, 0.0, None)


def apply_adjustments(
    anchor_predictions: np.ndarray,
    masks: dict[str, np.ndarray],
    satellite_predictions: dict[str, np.ndarray],
    adjustments: tuple[SegmentAdjustment, ...],
) -> tuple[np.ndarray, float]:
    adjusted = anchor_predictions.copy()
    affected_union = np.zeros_like(anchor_predictions, dtype=bool)

    for adjustment in adjustments:
        mask = masks[adjustment.mask_name]
        affected_union |= mask
        delta = satellite_predictions[adjustment.satellite_name] - anchor_predictions
        adjusted[mask] = adjusted[mask] + adjustment.alpha * delta[mask]

    return np.clip(adjusted, 0.0, None), float(affected_union.mean())


def build_router_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []

    single_specs: list[SegmentAdjustment] = []
    for alpha in (0.04, 0.06, 0.08, 0.10):
        single_specs.extend(
            [
                SegmentAdjustment("temporal_xgb", "recent_ratio_ge_1p2", alpha),
                SegmentAdjustment("temporal_xgb", "sparse_ge_0p3", alpha),
                SegmentAdjustment("temporal_xgb", "sparse_ge_0p5", alpha),
                SegmentAdjustment("temporal_best", "sparse_ge_0p5", alpha),
                SegmentAdjustment("v4_hgb_single", "inactive_ge_18", alpha),
                SegmentAdjustment("v4_hgb_single", "recent_ratio_le_0p8", alpha),
                SegmentAdjustment("v4_best_subset", "inactive_ge_18", alpha),
            ]
        )

    for adjustment in single_specs:
        adjustments = (adjustment,)
        candidate_name = (
            f"router_{adjustment.satellite_name}_{adjustment.mask_name}_a{weight_token(adjustment.alpha)}"
        )
        train_predictions, affected_train_share = apply_adjustments(
            anchor_train,
            train_masks,
            train_satellites,
            adjustments,
        )
        test_predictions, affected_test_share = apply_adjustments(
            anchor_test,
            test_masks,
            test_satellites,
            adjustments,
        )
        candidates.append(
            CandidateResult(
                name=candidate_name,
                family="router_single",
                adjustments=adjustments,
                oof_predictions=train_predictions,
                test_predictions=test_predictions,
                affected_train_share=affected_train_share,
                affected_test_share=affected_test_share,
            )
        )

    primary_adjustments = [
        SegmentAdjustment("temporal_xgb", "recent_ratio_ge_1p2", alpha)
        for alpha in (0.08, 0.10)
    ] + [
        SegmentAdjustment("temporal_xgb", "sparse_ge_0p3", alpha)
        for alpha in (0.08, 0.10)
    ] + [
        SegmentAdjustment("temporal_xgb", "sparse_ge_0p5", alpha)
        for alpha in (0.08, 0.10)
    ]
    secondary_adjustments = [
        SegmentAdjustment("temporal_best", "sparse_ge_0p5", alpha)
        for alpha in (0.04, 0.06, 0.08, 0.10)
    ] + [
        SegmentAdjustment("v4_hgb_single", "inactive_ge_18", alpha)
        for alpha in (0.04, 0.06, 0.08, 0.10)
    ] + [
        SegmentAdjustment("v4_hgb_single", "recent_ratio_le_0p8", alpha)
        for alpha in (0.04, 0.06, 0.08, 0.10)
    ] + [
        SegmentAdjustment("v4_best_subset", "active_le_12", alpha)
        for alpha in (0.04, 0.06, 0.08, 0.10)
    ]

    for primary in primary_adjustments:
        for secondary in secondary_adjustments:
            if primary.satellite_name == secondary.satellite_name and primary.mask_name == secondary.mask_name:
                continue
            adjustments = (primary, secondary)
            candidate_name = (
                f"router_{primary.satellite_name}_{primary.mask_name}_a{weight_token(primary.alpha)}"
                f"__{secondary.satellite_name}_{secondary.mask_name}_a{weight_token(secondary.alpha)}"
            )
            train_predictions, affected_train_share = apply_adjustments(
                anchor_train,
                train_masks,
                train_satellites,
                adjustments,
            )
            test_predictions, affected_test_share = apply_adjustments(
                anchor_test,
                test_masks,
                test_satellites,
                adjustments,
            )
            candidates.append(
                CandidateResult(
                    name=candidate_name,
                    family="router_dual",
                    adjustments=adjustments,
                    oof_predictions=train_predictions,
                    test_predictions=test_predictions,
                    affected_train_share=affected_train_share,
                    affected_test_share=affected_test_share,
                )
            )

    return candidates


def build_meta_feature_matrices(
    seeds: dict[str, object],
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = np.column_stack(
        [
            np.log1p(seeds["v5_best_subset"].oof_predictions),
            np.log1p(seeds["v4_best_subset"].oof_predictions),
            np.log1p(seeds["temporal_xgb"].oof_predictions),
            np.log1p(seeds["temporal_best"].oof_predictions),
            np.log1p(seeds["v4_hgb_single"].oof_predictions),
            np.log1p(anchor_train),
        ]
    )
    test_matrix = np.column_stack(
        [
            np.log1p(seeds["v5_best_subset"].test_predictions),
            np.log1p(seeds["v4_best_subset"].test_predictions),
            np.log1p(seeds["temporal_xgb"].test_predictions),
            np.log1p(seeds["temporal_best"].test_predictions),
            np.log1p(seeds["v4_hgb_single"].test_predictions),
            np.log1p(anchor_test),
        ]
    )
    return train_matrix, test_matrix


def fit_meta_candidate(
    name: str,
    regressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> CandidateResult:
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    splitter = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(y_train), dtype=np.float64)
    oof_counts = np.zeros(len(y_train), dtype=np.float64)

    for train_idx, valid_idx in splitter.split(X_train, y_train):
        fitted = clone(model)
        fitted.fit(X_train[train_idx], y_train[train_idx])
        fold_predictions = np.clip(fitted.predict(X_train[valid_idx]), 0.0, None)
        oof_predictions[valid_idx] += fold_predictions
        oof_counts[valid_idx] += 1.0

    oof_predictions /= oof_counts
    fitted_full = clone(model)
    fitted_full.fit(X_train, y_train)
    test_predictions = np.clip(fitted_full.predict(X_test), 0.0, None)

    return CandidateResult(
        name=name,
        family="meta",
        adjustments=tuple(),
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        affected_train_share=1.0,
        affected_test_share=1.0,
    )


def build_meta_candidates(
    seeds: dict[str, object],
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
) -> list[CandidateResult]:
    X_train, X_test = build_meta_feature_matrices(seeds, anchor_train, anchor_test)
    return [
        fit_meta_candidate(
            name="meta_positive_linear_predonly",
            regressor=LinearRegression(positive=True),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        ),
        fit_meta_candidate(
            name="meta_ridge_predonly",
            regressor=Ridge(alpha=2.0),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        ),
    ]


def build_disagreement_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    v5_train = train_satellites["v5_best_subset"]
    v5_test = test_satellites["v5_best_subset"]
    temporal_xgb_train = train_satellites["temporal_xgb"]
    temporal_xgb_test = test_satellites["temporal_xgb"]

    fin_missing_train = train_masks["fin_missing"]
    fin_missing_test = test_masks["fin_missing"]
    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))

    # Always include the simplest missing-financial fallback.
    missing_override_train = anchor_train.copy()
    missing_override_test = anchor_test.copy()
    missing_override_train[fin_missing_train] = v5_train[fin_missing_train]
    missing_override_test[fin_missing_test] = v5_test[fin_missing_test]
    candidates.append(
        CandidateResult(
            name="calibration_fin_missing_to_v5",
            family="calibration",
            adjustments=tuple(),
            oof_predictions=np.clip(missing_override_train, 0.0, None),
            test_predictions=np.clip(missing_override_test, 0.0, None),
            affected_train_share=float(fin_missing_train.mean()),
            affected_test_share=float(fin_missing_test.mean()),
        )
    )

    low_count_train = anchor_train <= 10.0
    low_count_test = anchor_test <= 10.0
    shrink_train = missing_override_train.copy()
    shrink_test = missing_override_test.copy()
    shrink_train[low_count_train] = np.clip(shrink_train[low_count_train] * 0.98, 0.0, None)
    shrink_test[low_count_test] = np.clip(shrink_test[low_count_test] * 0.98, 0.0, None)
    candidates.append(
        CandidateResult(
            name="calibration_fin_missing_to_v5__pred_le_10_shrink_0p98",
            family="calibration",
            adjustments=tuple(),
            oof_predictions=np.clip(shrink_train, 0.0, None),
            test_predictions=np.clip(shrink_test, 0.0, None),
            affected_train_share=float((fin_missing_train | low_count_train).mean()),
            affected_test_share=float((fin_missing_test | low_count_test).mean()),
        )
    )

    non_missing_train = ~fin_missing_train
    quantiles = (0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92)
    alphas = (0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28)
    for quantile in quantiles:
        threshold = float(np.quantile(disagreement_train[non_missing_train], quantile))
        high_gap_train = non_missing_train & (disagreement_train >= threshold)
        high_gap_test = (~fin_missing_test) & (disagreement_test >= threshold)
        top_pct = int(round((1.0 - quantile) * 100))
        for alpha in alphas:
            adjusted_train = anchor_train.copy()
            adjusted_test = anchor_test.copy()
            adjusted_train[high_gap_train] = (
                (1.0 - alpha) * v5_train[high_gap_train] + alpha * temporal_xgb_train[high_gap_train]
            )
            adjusted_test[high_gap_test] = (
                (1.0 - alpha) * v5_test[high_gap_test] + alpha * temporal_xgb_test[high_gap_test]
            )
            adjusted_train[fin_missing_train] = v5_train[fin_missing_train]
            adjusted_test[fin_missing_test] = v5_test[fin_missing_test]
            candidate_name = (
                f"disagreement_top{top_pct:02d}_alpha_{weight_token(alpha)}__fin_missing_to_v5"
            )
            candidates.append(
                CandidateResult(
                    name=candidate_name,
                    family="disagreement",
                    adjustments=tuple(),
                    oof_predictions=np.clip(adjusted_train, 0.0, None),
                    test_predictions=np.clip(adjusted_test, 0.0, None),
                    affected_train_share=float((high_gap_train | fin_missing_train).mean()),
                    affected_test_share=float((high_gap_test | fin_missing_test).mean()),
                )
            )

            # A narrow second-order refinement: modest shrinkage on low predicted counts for the stronger,
            # wider disagreement candidates only. This keeps the search disciplined while capturing the
            # only calibration pattern that materially improved offline scores.
            if quantile <= 0.85 and alpha >= 0.24:
                for shrink_factor in (0.98, 0.975):
                    shrink_train = np.clip(adjusted_train.copy(), 0.0, None)
                    shrink_test = np.clip(adjusted_test.copy(), 0.0, None)
                    low_count_train = shrink_train <= 10.0
                    low_count_test = shrink_test <= 10.0
                    shrink_train[low_count_train] = np.clip(
                        shrink_train[low_count_train] * shrink_factor,
                        0.0,
                        None,
                    )
                    shrink_test[low_count_test] = np.clip(
                        shrink_test[low_count_test] * shrink_factor,
                        0.0,
                        None,
                    )
                    candidates.append(
                        CandidateResult(
                            name=(
                                f"disagreement_top{top_pct:02d}_alpha_{weight_token(alpha)}__fin_missing_to_v5"
                                f"__pred_le_10_x{weight_token(shrink_factor)}"
                            ),
                            family="disagreement",
                            adjustments=tuple(),
                            oof_predictions=shrink_train,
                            test_predictions=shrink_test,
                            affected_train_share=float(
                                (high_gap_train | fin_missing_train | low_count_train).mean()
                            ),
                            affected_test_share=float(
                                (high_gap_test | fin_missing_test | low_count_test).mean()
                            ),
                        )
                    )

    return candidates


def build_public_best_base_neighborhood_candidates(
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    v5_train = train_satellites["v5_best_subset"]
    v5_test = test_satellites["v5_best_subset"]
    temporal_xgb_train = train_satellites["temporal_xgb"]
    temporal_xgb_test = test_satellites["temporal_xgb"]
    fin_missing_train = train_masks["fin_missing"]
    fin_missing_test = test_masks["fin_missing"]
    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))
    non_missing_train = ~fin_missing_train

    neighborhood_specs = [
        (0.90, 0.22),
        (0.91, 0.22),
        (0.92, 0.21),
        (0.92, 0.23),
        (0.93, 0.22),
    ]
    for quantile, alpha in neighborhood_specs:
        threshold = float(np.quantile(disagreement_train[non_missing_train], quantile))
        high_gap_train = non_missing_train & (disagreement_train >= threshold)
        high_gap_test = (~fin_missing_test) & (disagreement_test >= threshold)
        top_pct = int(round((1.0 - quantile) * 100))
        adjusted_train = np.clip(base_anchor_train.copy(), 0.0, None)
        adjusted_test = np.clip(base_anchor_test.copy(), 0.0, None)
        adjusted_train[high_gap_train] = (
            (1.0 - alpha) * v5_train[high_gap_train] + alpha * temporal_xgb_train[high_gap_train]
        )
        adjusted_test[high_gap_test] = (
            (1.0 - alpha) * v5_test[high_gap_test] + alpha * temporal_xgb_test[high_gap_test]
        )
        adjusted_train[fin_missing_train] = v5_train[fin_missing_train]
        adjusted_test[fin_missing_test] = v5_test[fin_missing_test]
        candidates.append(
            CandidateResult(
                name=f"publicbest_top{top_pct:02d}_alpha_{weight_token(alpha)}__fin_missing_to_v5",
                family="public_best_neighborhood",
                adjustments=tuple(),
                oof_predictions=np.clip(adjusted_train, 0.0, None),
                test_predictions=np.clip(adjusted_test, 0.0, None),
                affected_train_share=float((high_gap_train | fin_missing_train).mean()),
                affected_test_share=float((high_gap_test | fin_missing_test).mean()),
            )
        )

    piecewise_specs = [
        ("publicbest_piecewise_top04_a0p26__next04_a0p20", ((0.96, 1.00, 0.26), (0.92, 0.96, 0.20))),
        ("publicbest_piecewise_top06_a0p24__next06_a0p18", ((0.94, 1.00, 0.24), (0.88, 0.94, 0.18))),
    ]
    for candidate_name, bands in piecewise_specs:
        adjusted_train = np.clip(base_anchor_train.copy(), 0.0, None)
        adjusted_test = np.clip(base_anchor_test.copy(), 0.0, None)
        affected_train = fin_missing_train.copy()
        affected_test = fin_missing_test.copy()
        for lower_quantile, upper_quantile, alpha in bands:
            lower_threshold = float(np.quantile(disagreement_train[non_missing_train], lower_quantile))
            if upper_quantile >= 1.0:
                band_train = non_missing_train & (disagreement_train >= lower_threshold)
                band_test = (~fin_missing_test) & (disagreement_test >= lower_threshold)
            else:
                upper_threshold = float(np.quantile(disagreement_train[non_missing_train], upper_quantile))
                band_train = non_missing_train & (disagreement_train >= lower_threshold) & (disagreement_train < upper_threshold)
                band_test = (~fin_missing_test) & (disagreement_test >= lower_threshold) & (disagreement_test < upper_threshold)
            adjusted_train[band_train] = (
                (1.0 - alpha) * v5_train[band_train] + alpha * temporal_xgb_train[band_train]
            )
            adjusted_test[band_test] = (
                (1.0 - alpha) * v5_test[band_test] + alpha * temporal_xgb_test[band_test]
            )
            affected_train |= band_train
            affected_test |= band_test
        adjusted_train[fin_missing_train] = v5_train[fin_missing_train]
        adjusted_test[fin_missing_test] = v5_test[fin_missing_test]
        candidates.append(
            CandidateResult(
                name=candidate_name,
                family="public_best_piecewise",
                adjustments=tuple(),
                oof_predictions=np.clip(adjusted_train, 0.0, None),
                test_predictions=np.clip(adjusted_test, 0.0, None),
                affected_train_share=float(affected_train.mean()),
                affected_test_share=float(affected_test.mean()),
            )
        )

    return candidates


def build_public_best_overlay_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    v5_train = train_satellites["v5_best_subset"]
    v5_test = test_satellites["v5_best_subset"]
    temporal_xgb_train = train_satellites["temporal_xgb"]
    temporal_xgb_test = test_satellites["temporal_xgb"]
    temporal_best_train = train_satellites["temporal_best"]
    temporal_best_test = test_satellites["temporal_best"]
    v4_hgb_train = train_satellites["v4_hgb_single"]
    v4_hgb_test = test_satellites["v4_hgb_single"]

    fin_missing_train = train_masks["fin_missing"]
    fin_missing_test = test_masks["fin_missing"]
    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))
    non_missing_train = ~fin_missing_train
    threshold = float(np.quantile(disagreement_train[non_missing_train], 0.92))
    public_mask_train = non_missing_train & (disagreement_train >= threshold)
    public_mask_test = (~fin_missing_test) & (disagreement_test >= threshold)

    overlay_specs = [
        ("publicbest_overlay_recent_growth_a0p03", public_mask_train & train_masks["recent_ratio_ge_1p2"], public_mask_test & test_masks["recent_ratio_ge_1p2"], temporal_xgb_train, temporal_xgb_test, 0.03),
        ("publicbest_overlay_recent_growth_a0p04", public_mask_train & train_masks["recent_ratio_ge_1p2"], public_mask_test & test_masks["recent_ratio_ge_1p2"], temporal_xgb_train, temporal_xgb_test, 0.04),
        ("publicbest_overlay_sparse_temporalbest_a0p03", public_mask_train & train_masks["sparse_ge_0p5"], public_mask_test & test_masks["sparse_ge_0p5"], temporal_best_train, temporal_best_test, 0.03),
        ("publicbest_overlay_inactive_v4hgb_a0p03", (~public_mask_train) & train_masks["inactive_ge_18"], (~public_mask_test) & test_masks["inactive_ge_18"], v4_hgb_train, v4_hgb_test, 0.03),
    ]
    for candidate_name, train_mask, test_mask, train_satellite, test_satellite, alpha in overlay_specs:
        adjusted_train = np.clip(anchor_train.copy(), 0.0, None)
        adjusted_test = np.clip(anchor_test.copy(), 0.0, None)
        adjusted_train[train_mask] = adjusted_train[train_mask] + alpha * (
            train_satellite[train_mask] - adjusted_train[train_mask]
        )
        adjusted_test[test_mask] = adjusted_test[test_mask] + alpha * (
            test_satellite[test_mask] - adjusted_test[test_mask]
        )
        affected_train = train_mask.copy()
        affected_test = test_mask.copy()
        candidates.append(
            CandidateResult(
                name=candidate_name,
                family="public_best_overlay",
                adjustments=tuple(),
                oof_predictions=np.clip(adjusted_train, 0.0, None),
                test_predictions=np.clip(adjusted_test, 0.0, None),
                affected_train_share=float(affected_train.mean()),
                affected_test_share=float(affected_test.mean()),
            )
        )

    mild_shrink_specs = [
        ("publicbest_overlay_m08_recent_le_10_x0p995", public_mask_train & train_masks["recent_le_10"], public_mask_test & test_masks["recent_le_10"], 0.995),
        ("publicbest_overlay_m08_pred_le_8_x0p99", public_mask_train & (anchor_train <= 8.0), public_mask_test & (anchor_test <= 8.0), 0.99),
    ]
    for candidate_name, train_mask, test_mask, shrink_factor in mild_shrink_specs:
        adjusted_train = np.clip(anchor_train.copy(), 0.0, None)
        adjusted_test = np.clip(anchor_test.copy(), 0.0, None)
        adjusted_train[train_mask] = np.clip(adjusted_train[train_mask] * shrink_factor, 0.0, None)
        adjusted_test[test_mask] = np.clip(adjusted_test[test_mask] * shrink_factor, 0.0, None)
        candidates.append(
            CandidateResult(
                name=candidate_name,
                family="public_best_overlay",
                adjustments=tuple(),
                oof_predictions=adjusted_train,
                test_predictions=adjusted_test,
                affected_train_share=float(train_mask.mean()),
                affected_test_share=float(test_mask.mean()),
            )
        )

    return candidates


def build_anchor_microblend_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_satellites: dict[str, np.ndarray],
    test_satellites: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    alpha_grid = (0.01, 0.02, 0.03, 0.04)
    for satellite_name in ("v5_best_subset", "v4_best_subset", "temporal_xgb", "temporal_best", "v4_hgb_single"):
        train_satellite = np.clip(train_satellites[satellite_name], 0.0, None)
        test_satellite = np.clip(test_satellites[satellite_name], 0.0, None)
        for alpha in alpha_grid:
            raw_train = np.clip((1.0 - alpha) * anchor_train + alpha * train_satellite, 0.0, None)
            raw_test = np.clip((1.0 - alpha) * anchor_test + alpha * test_satellite, 0.0, None)
            candidates.append(
                CandidateResult(
                    name=f"microblend_{satellite_name}__raw__a{weight_token(alpha)}",
                    family="microblend",
                    adjustments=tuple(),
                    oof_predictions=raw_train,
                    test_predictions=raw_test,
                    affected_train_share=1.0,
                    affected_test_share=1.0,
                )
            )
            log_train = np.expm1(
                (1.0 - alpha) * np.log1p(np.clip(anchor_train, 0.0, None))
                + alpha * np.log1p(train_satellite)
            )
            log_test = np.expm1(
                (1.0 - alpha) * np.log1p(np.clip(anchor_test, 0.0, None))
                + alpha * np.log1p(test_satellite)
            )
            candidates.append(
                CandidateResult(
                    name=f"microblend_{satellite_name}__log__a{weight_token(alpha)}",
                    family="microblend",
                    adjustments=tuple(),
                    oof_predictions=np.clip(log_train, 0.0, None),
                    test_predictions=np.clip(log_test, 0.0, None),
                    affected_train_share=1.0,
                    affected_test_share=1.0,
                )
            )

    return candidates


def build_metrics(
    anchor_train: np.ndarray,
    y_train: np.ndarray,
    candidates: list[CandidateResult],
    shift_penalty: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "oof_rmsle": float(rmsle(y_train, candidate.oof_predictions)),
                "mean_abs_log_shift_vs_anchor": float(
                    np.mean(np.abs(np.log1p(candidate.oof_predictions) - np.log1p(anchor_train)))
                ),
                "affected_train_share": candidate.affected_train_share,
                "affected_test_share": candidate.affected_test_share,
                "adjustments_json": json.dumps(
                    [
                        {
                            "satellite_name": adjustment.satellite_name,
                            "mask_name": adjustment.mask_name,
                            "alpha": adjustment.alpha,
                        }
                        for adjustment in candidate.adjustments
                    ]
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics["conservative_score"] = metrics["oof_rmsle"] + shift_penalty * metrics["mean_abs_log_shift_vs_anchor"]
    return metrics.sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).reset_index(drop=True)


def build_prediction_lookup(candidates: list[CandidateResult]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {candidate.name: (candidate.oof_predictions, candidate.test_predictions) for candidate in candidates}


def write_artifacts(
    run_dir: Path,
    sample_submission_path: Path,
    test_ids: np.ndarray,
    metrics: pd.DataFrame,
    prediction_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    submission_format: str,
    write_top_k_submissions: int,
) -> None:
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

    top_candidates = metrics.head(write_top_k_submissions)["candidate_name"].tolist()
    best_router = metrics.loc[metrics["family"].str.startswith("router")].sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).iloc[0]
    structured_families = [
        "router_single",
        "router_dual",
        "disagreement",
        "calibration",
        "public_best_neighborhood",
        "public_best_piecewise",
        "public_best_overlay",
        "microblend",
    ]
    best_structured = metrics.loc[metrics["family"].isin(structured_families)].sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).iloc[0]
    best_structured_hedge = metrics.loc[metrics["family"].isin(structured_families)].sort_values(
        ["conservative_score", "oof_rmsle", "candidate_name"]
    ).iloc[0]
    best_structured_alternate = metrics.loc[
        metrics["family"].isin(structured_families)
        & ~metrics["candidate_name"].isin(
            [best_structured["candidate_name"], best_structured_hedge["candidate_name"]]
        )
    ].sort_values(["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]).iloc[0]
    best_meta = metrics.loc[metrics["family"] == "meta"].sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).iloc[0]

    materialized_names = sorted(
        set(
            top_candidates
            + [
                best_router["candidate_name"],
                best_structured["candidate_name"],
                best_structured_hedge["candidate_name"],
                best_structured_alternate["candidate_name"],
                best_meta["candidate_name"],
            ]
        )
    )
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for candidate_name in materialized_names:
        _, test_predictions = prediction_lookup[candidate_name]
        build_submission_frame(
            sample_submission_path=sample_submission_path,
            unique_ids=test_ids,
            predictions=test_predictions,
            submission_format=submission_format,
        ).write_csv(submissions_dir / f"{candidate_name}.csv")

    selection_dir = run_dir / "recommended_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "01_best_structured.csv").write_bytes(
        (submissions_dir / f"{best_structured['candidate_name']}.csv").read_bytes()
    )
    (selection_dir / "02_structured_hedge.csv").write_bytes(
        (submissions_dir / f"{best_structured_hedge['candidate_name']}.csv").read_bytes()
    )
    (selection_dir / "03_structured_alternate.csv").write_bytes(
        (submissions_dir / f"{best_structured_alternate['candidate_name']}.csv").read_bytes()
    )
    selection_lines = [
        "# Recommended Selection",
        "",
        "1. `01_best_structured.csv`",
        f"   - candidate: `{best_structured['candidate_name']}`",
        f"   - OOF RMSLE: `{best_structured['oof_rmsle']:.6f}`",
        f"   - mean abs log shift vs anchor: `{best_structured['mean_abs_log_shift_vs_anchor']:.6f}`",
        "",
        "2. `02_structured_hedge.csv`",
        f"   - candidate: `{best_structured_hedge['candidate_name']}`",
        f"   - OOF RMSLE: `{best_structured_hedge['oof_rmsle']:.6f}`",
        f"   - conservative score: `{best_structured_hedge['conservative_score']:.6f}`",
        "",
        "3. `03_structured_alternate.csv`",
        f"   - candidate: `{best_structured_alternate['candidate_name']}`",
        f"   - OOF RMSLE: `{best_structured_alternate['oof_rmsle']:.6f}`",
        "   - note: higher-upside sibling within the same structured family",
        "",
        f"Meta file left in `submissions/`: `{best_meta['candidate_name']}`",
        (
            "All files are copied from `submissions/` and are upload-ready for Zindi."
            if submission_format == "zindi_log"
            else "This run used raw output format. Do not upload these files without converting them to "
            "`np.log1p(prediction)` first."
        ),
    ]
    (selection_dir / "README.md").write_text("\n".join(selection_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Public Anchor Expert Lab Summary",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Submission format: `{submission_format}`",
        f"- Best structured candidate: `{best_structured['candidate_name']}` at `{best_structured['oof_rmsle']:.6f}`",
        f"- Best structured hedge: `{best_structured_hedge['candidate_name']}` at `{best_structured_hedge['oof_rmsle']:.6f}`",
        f"- Best router candidate: `{best_router['candidate_name']}` at `{best_router['oof_rmsle']:.6f}`",
        f"- Best meta candidate: `{best_meta['candidate_name']}` at `{best_meta['oof_rmsle']:.6f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in metrics.head(12).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}`: OOF `{row['oof_rmsle']:.6f}`, "
            f"shift `{row['mean_abs_log_shift_vs_anchor']:.6f}`, family `{row['family']}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.resolve()
    run_dir = build_run_directory(args.output_dir, args.run_name)
    train_ids, test_ids, y_train, seeds = load_seed_candidates(base_dir)
    base_anchor_train, base_anchor_test = build_base_public_anchor_predictions(seeds)
    train_features, test_features = load_feature_splits(args.feature_cache_path.resolve(), train_ids, test_ids)
    train_masks = build_masks(train_features)
    test_masks = build_masks(test_features)

    train_satellites = {
        "v5_best_subset": seeds["v5_best_subset"].oof_predictions,
        "v4_best_subset": seeds["v4_best_subset"].oof_predictions,
        "temporal_xgb": seeds["temporal_xgb"].oof_predictions,
        "temporal_best": seeds["temporal_best"].oof_predictions,
        "v4_hgb_single": seeds["v4_hgb_single"].oof_predictions,
    }
    test_satellites = {
        "v5_best_subset": seeds["v5_best_subset"].test_predictions,
        "v4_best_subset": seeds["v4_best_subset"].test_predictions,
        "temporal_xgb": seeds["temporal_xgb"].test_predictions,
        "temporal_best": seeds["temporal_best"].test_predictions,
        "v4_hgb_single": seeds["v4_hgb_single"].test_predictions,
    }

    if args.anchor_variant == "base":
        anchor_train, anchor_test = base_anchor_train, base_anchor_test
        anchor_name = "public_best_anchor_v5_temporal_xgb_a0p08"
    else:
        anchor_train, anchor_test = build_public_best_round6_anchor_predictions(
            base_anchor_train=base_anchor_train,
            base_anchor_test=base_anchor_test,
            train_masks=train_masks,
            test_masks=test_masks,
            train_satellites=train_satellites,
            test_satellites=test_satellites,
        )
        anchor_name = "public_best_anchor_round6_top08_alpha0p22"

    anchor_candidate = CandidateResult(
        name=anchor_name,
        family="anchor",
        adjustments=tuple(),
        oof_predictions=anchor_train,
        test_predictions=anchor_test,
        affected_train_share=0.0,
        affected_test_share=0.0,
    )
    router_candidates = build_router_candidates(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        train_masks=train_masks,
        test_masks=test_masks,
        train_satellites=train_satellites,
        test_satellites=test_satellites,
    )
    disagreement_candidates = build_disagreement_candidates(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        train_masks=train_masks,
        test_masks=test_masks,
        train_satellites=train_satellites,
        test_satellites=test_satellites,
    )
    public_best_base_candidates = build_public_best_base_neighborhood_candidates(
        base_anchor_train=base_anchor_train,
        base_anchor_test=base_anchor_test,
        train_masks=train_masks,
        test_masks=test_masks,
        train_satellites=train_satellites,
        test_satellites=test_satellites,
    )
    public_best_overlay_candidates = build_public_best_overlay_candidates(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        train_masks=train_masks,
        test_masks=test_masks,
        train_satellites=train_satellites,
        test_satellites=test_satellites,
    )
    microblend_candidates = build_anchor_microblend_candidates(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        train_satellites=train_satellites,
        test_satellites=test_satellites,
    )
    meta_candidates = build_meta_candidates(
        seeds=seeds,
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        y_train=y_train,
    )
    candidates = (
        [anchor_candidate]
        + public_best_base_candidates
        + public_best_overlay_candidates
        + microblend_candidates
        + router_candidates
        + disagreement_candidates
        + meta_candidates
    )
    metrics = build_metrics(
        anchor_train=anchor_train,
        y_train=y_train,
        candidates=candidates,
        shift_penalty=args.shift_penalty,
    )
    prediction_lookup = build_prediction_lookup(candidates)
    write_artifacts(
        run_dir=run_dir,
        sample_submission_path=base_dir / "SampleSubmission.csv",
        test_ids=test_ids,
        metrics=metrics,
        prediction_lookup=prediction_lookup,
        submission_format=args.submission_format,
        write_top_k_submissions=args.write_top_k_submissions,
    )
    config = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "anchor_variant": args.anchor_variant,
        "submission_format": args.submission_format,
        "shift_penalty": args.shift_penalty,
        "write_top_k_submissions": args.write_top_k_submissions,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
