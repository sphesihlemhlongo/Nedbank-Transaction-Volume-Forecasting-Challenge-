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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from artifact_blend_lab import align_predictions_by_id, load_seed_candidates
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import SklearnCatBoostRegressor, build_run_directory


FEATURE_PIPELINE_VERSION = "v11_anchor_residual_lab"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ResidualModelResult:
    name: str
    family: str
    oof_residuals: np.ndarray
    test_residuals: np.ndarray
    residual_rmse: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    source_name: str
    mask_name: str
    alpha: float
    clip_value: float
    confidence_quantile: float | None
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    affected_train_share: float
    affected_test_share: float
    mean_abs_log_shift: float
    oof_rmsle: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit conservative log-residual correction models around the current public-best "
            "anchor and materialize leaderboard-focused candidate submissions."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/anchor_residual_lab"))
    parser.add_argument("--run-name", type=str, default="anchor_residual_lab")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v5_dense_monthly_panel_stack.parquet"),
        help="Dense monthly-panel feature cache used for residual features and mask construction.",
    )
    parser.add_argument(
        "--monthly-run-dir",
        type=Path,
        default=Path("outputs/experiments/v5_monthly_panel_full1_20260430_091349_503223_78ac19f8"),
        help="Monthly-panel experiment artifact directory that contains OOF/test prediction parquets.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--shift-penalty",
        type=float,
        default=0.30,
        help="Penalty multiplier for ranking low-drift hedge candidates.",
    )
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=16,
        help="How many top-ranked candidates to materialize as submission CSV files.",
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def weight_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def maybe_token(value: float | None) -> str:
    if value is None:
        return "all"
    return f"q{int(round(value * 100))}"


def load_feature_splits(
    feature_cache_path: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_table = pl.read_parquet(feature_cache_path)
    train_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", train_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "train").drop("__split__"),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
        .to_pandas()
    )
    test_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", test_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "test").drop(["__split__", "next_3m_txn_count"]),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
        .to_pandas()
    )
    return train_df, test_df


def load_monthly_panel_satellites(
    monthly_run_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    oof_predictions = pl.read_parquet(monthly_run_dir / "oof_predictions.parquet")
    test_predictions = pl.read_parquet(monthly_run_dir / "test_predictions.parquet")
    return {
        "panel_blend_top2": (
            align_predictions_by_id(oof_predictions, train_ids, "pred_blend_log_top2_equal"),
            align_predictions_by_id(test_predictions, test_ids, "pred_blend_log_top2_equal"),
        ),
        "panel_catboost_v1": (
            align_predictions_by_id(oof_predictions, train_ids, "pred_catboost_conservative_v1"),
            align_predictions_by_id(test_predictions, test_ids, "pred_catboost_conservative_v1"),
        ),
        "panel_xgb_v3": (
            align_predictions_by_id(oof_predictions, train_ids, "pred_xgb_conservative_v3"),
            align_predictions_by_id(test_predictions, test_ids, "pred_xgb_conservative_v3"),
        ),
    }


def build_current_public_best_anchor(
    seeds: dict[str, object],
    train_fin_missing: np.ndarray,
    test_fin_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    temporal_xgb_train = seeds["temporal_xgb"].oof_predictions
    temporal_xgb_test = seeds["temporal_xgb"].test_predictions

    base_anchor_train = 0.92 * v5_train + 0.08 * temporal_xgb_train
    base_anchor_test = 0.92 * v5_test + 0.08 * temporal_xgb_test

    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))
    non_missing_train = ~train_fin_missing
    threshold = float(np.quantile(disagreement_train[non_missing_train], 0.92))
    public_mask_train = non_missing_train & (disagreement_train >= threshold)
    public_mask_test = (~test_fin_missing) & (disagreement_test >= threshold)

    anchor_train = np.clip(base_anchor_train.copy(), 0.0, None)
    anchor_test = np.clip(base_anchor_test.copy(), 0.0, None)
    anchor_train[public_mask_train] = 0.77 * v5_train[public_mask_train] + 0.23 * temporal_xgb_train[public_mask_train]
    anchor_test[public_mask_test] = 0.77 * v5_test[public_mask_test] + 0.23 * temporal_xgb_test[public_mask_test]
    anchor_train[train_fin_missing] = v5_train[train_fin_missing]
    anchor_test[test_fin_missing] = v5_test[test_fin_missing]
    return np.clip(anchor_train, 0.0, None), np.clip(anchor_test, 0.0, None), public_mask_train, public_mask_test, threshold


def numeric_array(frame: pd.DataFrame, column: str, fill_value: float = -999.0) -> np.ndarray:
    values = np.array(
        pd.to_numeric(frame[column], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
    )
    values[~np.isfinite(values)] = fill_value
    return values


def build_masks(
    frame: pd.DataFrame,
    public_gap_mask: np.ndarray,
    fin_missing_mask: np.ndarray,
    anchor_predictions: np.ndarray,
) -> dict[str, np.ndarray]:
    sparse_share = numeric_array(frame, "txn_sparse_month_share_le_2")
    active_months = numeric_array(frame, "txn_active_months_total")
    inactive_months = numeric_array(frame, "txn_inactive_months_total")
    recent_count = numeric_array(frame, "txn_recent_3m_count")
    recent_ratio = numeric_array(frame, "txn_recent_vs_prev3_count_ratio")

    sparse_ge_0p5 = sparse_share >= 0.5
    active_le_12 = active_months <= 12.0
    inactive_ge_18 = inactive_months >= 18.0
    recent_le_10 = recent_count <= 10.0
    recent_ratio_ge_1p2 = recent_ratio >= 1.2
    anchor_le_8 = anchor_predictions <= 8.0
    inactive_or_sparse = inactive_ge_18 | sparse_ge_0p5

    return {
        "all_rows": np.ones(len(frame), dtype=bool),
        "sparse_ge_0p5": sparse_ge_0p5,
        "active_le_12": active_le_12,
        "inactive_ge_18": inactive_ge_18,
        "recent_le_10": recent_le_10,
        "recent_ratio_ge_1p2": recent_ratio_ge_1p2,
        "inactive_or_sparse": inactive_or_sparse,
        "inactive_and_sparse": inactive_ge_18 & sparse_ge_0p5,
        "public_gap_top08": public_gap_mask,
        "public_gap_sparse": public_gap_mask & sparse_ge_0p5,
        "public_gap_inactive_or_sparse": public_gap_mask & inactive_or_sparse,
        "public_gap_anchor_le_8": public_gap_mask & anchor_le_8,
        "fin_missing": fin_missing_mask,
    }


def add_prediction_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    seeds: dict[str, object],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_sources: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "anchor_publicbest": (anchor_train, anchor_test),
        "v5_best_subset": (seeds["v5_best_subset"].oof_predictions, seeds["v5_best_subset"].test_predictions),
        "temporal_xgb": (seeds["temporal_xgb"].oof_predictions, seeds["temporal_xgb"].test_predictions),
        "temporal_best": (seeds["temporal_best"].oof_predictions, seeds["temporal_best"].test_predictions),
        "v4_best_subset": (seeds["v4_best_subset"].oof_predictions, seeds["v4_best_subset"].test_predictions),
        "v4_hgb_single": (seeds["v4_hgb_single"].oof_predictions, seeds["v4_hgb_single"].test_predictions),
        "panel_blend_top2": monthly_satellites["panel_blend_top2"],
        "panel_catboost_v1": monthly_satellites["panel_catboost_v1"],
        "panel_xgb_v3": monthly_satellites["panel_xgb_v3"],
    }

    train_feature_map: dict[str, np.ndarray] = {}
    test_feature_map: dict[str, np.ndarray] = {}
    for source_name, (train_values, test_values) in prediction_sources.items():
        clipped_train = np.clip(train_values, 0.0, None)
        clipped_test = np.clip(test_values, 0.0, None)
        train_feature_map[f"pred_{source_name}"] = clipped_train
        test_feature_map[f"pred_{source_name}"] = clipped_test
        train_feature_map[f"log_pred_{source_name}"] = np.log1p(clipped_train)
        test_feature_map[f"log_pred_{source_name}"] = np.log1p(clipped_test)

    train_feature_map["log_gap_temporal_vs_v5"] = np.abs(
        train_feature_map["log_pred_temporal_xgb"] - train_feature_map["log_pred_v5_best_subset"]
    )
    test_feature_map["log_gap_temporal_vs_v5"] = np.abs(
        test_feature_map["log_pred_temporal_xgb"] - test_feature_map["log_pred_v5_best_subset"]
    )
    train_feature_map["log_gap_panel_cat_vs_anchor"] = np.abs(
        train_feature_map["log_pred_panel_catboost_v1"] - train_feature_map["log_pred_anchor_publicbest"]
    )
    test_feature_map["log_gap_panel_cat_vs_anchor"] = np.abs(
        test_feature_map["log_pred_panel_catboost_v1"] - test_feature_map["log_pred_anchor_publicbest"]
    )
    train_feature_map["log_gap_panel_blend_vs_anchor"] = np.abs(
        train_feature_map["log_pred_panel_blend_top2"] - train_feature_map["log_pred_anchor_publicbest"]
    )
    test_feature_map["log_gap_panel_blend_vs_anchor"] = np.abs(
        test_feature_map["log_pred_panel_blend_top2"] - test_feature_map["log_pred_anchor_publicbest"]
    )
    train_feature_map["log_delta_panel_cat_vs_anchor"] = (
        train_feature_map["log_pred_panel_catboost_v1"] - train_feature_map["log_pred_anchor_publicbest"]
    )
    test_feature_map["log_delta_panel_cat_vs_anchor"] = (
        test_feature_map["log_pred_panel_catboost_v1"] - test_feature_map["log_pred_anchor_publicbest"]
    )
    train_feature_map["log_delta_temporal_vs_anchor"] = (
        train_feature_map["log_pred_temporal_xgb"] - train_feature_map["log_pred_anchor_publicbest"]
    )
    test_feature_map["log_delta_temporal_vs_anchor"] = (
        test_feature_map["log_pred_temporal_xgb"] - test_feature_map["log_pred_anchor_publicbest"]
    )

    train_feature_df = pd.DataFrame(train_feature_map, index=train_df.index)
    test_feature_df = pd.DataFrame(test_feature_map, index=test_df.index)
    train_df = pd.concat([train_df, train_feature_df], axis=1).copy()
    test_df = pd.concat([test_df, test_feature_df], axis=1).copy()
    return train_df, test_df


def build_model_frames(
    feature_cache_path: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    seeds: dict[str, object],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_df, test_df = load_feature_splits(feature_cache_path, train_ids, test_ids)
    y_train = train_df.pop("next_3m_txn_count").to_numpy(dtype=np.float64)

    train_fin_missing = numeric_array(train_df, "fin_missing_flag", fill_value=1.0) >= 1.0
    test_fin_missing = numeric_array(test_df, "fin_missing_flag", fill_value=1.0) >= 1.0

    anchor_train, anchor_test, public_mask_train, public_mask_test, _ = build_current_public_best_anchor(
        seeds,
        train_fin_missing,
        test_fin_missing,
    )
    train_df, test_df = add_prediction_features(
        train_df,
        test_df,
        anchor_train,
        anchor_test,
        seeds,
        monthly_satellites,
    )

    train_masks = build_masks(train_df, public_mask_train, train_fin_missing, anchor_train)
    test_masks = build_masks(test_df, public_mask_test, test_fin_missing, anchor_test)
    return train_df, test_df, y_train, anchor_train, anchor_test, train_masks, test_masks


def feature_columns_for_model(train_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_columns = [column for column in train_df.columns if column != "UniqueID"]
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(train_df[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return feature_columns, numeric_columns, categorical_columns


def build_ridge_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", Ridge(alpha=18.0)),
        ]
    )


def build_hgb_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("transformer", transformer),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.03,
                    max_iter=240,
                    max_depth=3,
                    min_samples_leaf=80,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_catboost_model(categorical_columns: list[str]) -> SklearnCatBoostRegressor:
    return SklearnCatBoostRegressor(
        categorical_columns=tuple(categorical_columns),
        params={
            "iterations": 320,
            "depth": 4,
            "learning_rate": 0.04,
            "l2_leaf_reg": 10.0,
            "subsample": 0.80,
            "bootstrap_type": "Bernoulli",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
        },
        random_state=RANDOM_STATE,
    )


def cross_fit_residual_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    residual_target: np.ndarray,
    n_splits: int,
) -> list[ResidualModelResult]:
    feature_columns, numeric_columns, categorical_columns = feature_columns_for_model(train_df)
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    for column in categorical_columns:
        X_train[column] = X_train[column].astype("string").fillna("__missing__")
        X_test[column] = X_test[column].astype("string").fillna("__missing__")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    model_builders = [
        ("ridge_resid_v1", "linear", lambda: build_ridge_model(numeric_columns, categorical_columns)),
        ("hgb_resid_v1", "tree", lambda: build_hgb_model(numeric_columns, categorical_columns)),
        ("catboost_resid_v1", "tree_cat", lambda: build_catboost_model(categorical_columns)),
    ]

    results: list[ResidualModelResult] = []
    for model_name, family, builder in model_builders:
        oof_predictions = np.zeros(len(X_train), dtype=np.float64)
        test_fold_predictions: list[np.ndarray] = []
        for train_idx, valid_idx in splitter.split(X_train):
            model = builder()
            model.fit(X_train.iloc[train_idx], residual_target[train_idx])
            oof_predictions[valid_idx] = np.asarray(model.predict(X_train.iloc[valid_idx]), dtype=np.float64)
            test_fold_predictions.append(np.asarray(model.predict(X_test), dtype=np.float64))
        test_predictions = np.mean(np.column_stack(test_fold_predictions), axis=1)
        residual_rmse = float(np.sqrt(np.mean(np.square(oof_predictions - residual_target))))
        results.append(
            ResidualModelResult(
                name=model_name,
                family=family,
                oof_residuals=oof_predictions,
                test_residuals=test_predictions,
                residual_rmse=residual_rmse,
            )
        )

    top2 = sorted(results, key=lambda result: result.residual_rmse)[:2]
    inv_rmse = np.array([1.0 / max(result.residual_rmse, 1e-9) for result in top2], dtype=np.float64)
    weights = inv_rmse / inv_rmse.sum()
    results.append(
        ResidualModelResult(
            name="blend_inv_rmse_top2",
            family="blend",
            oof_residuals=weights[0] * top2[0].oof_residuals + weights[1] * top2[1].oof_residuals,
            test_residuals=weights[0] * top2[0].test_residuals + weights[1] * top2[1].test_residuals,
            residual_rmse=float(
                np.sqrt(
                    np.mean(
                        np.square(
                            weights[0] * top2[0].oof_residuals + weights[1] * top2[1].oof_residuals - residual_target
                        )
                    )
                )
            ),
        )
    )

    return results


def build_delta_sources(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    seeds: dict[str, object],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
    residual_target: np.ndarray,
) -> list[ResidualModelResult]:
    anchor_train_log = np.log1p(np.clip(anchor_train, 0.0, None))
    anchor_test_log = np.log1p(np.clip(anchor_test, 0.0, None))
    source_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "delta_panel_catboost_v1": monthly_satellites["panel_catboost_v1"],
        "delta_panel_blend_top2": monthly_satellites["panel_blend_top2"],
        "delta_panel_xgb_v3": monthly_satellites["panel_xgb_v3"],
        "delta_temporal_xgb": (seeds["temporal_xgb"].oof_predictions, seeds["temporal_xgb"].test_predictions),
        "delta_temporal_best": (seeds["temporal_best"].oof_predictions, seeds["temporal_best"].test_predictions),
    }
    results: list[ResidualModelResult] = []
    for source_name, (train_raw, test_raw) in source_map.items():
        train_residuals = np.log1p(np.clip(train_raw, 0.0, None)) - anchor_train_log
        test_residuals = np.log1p(np.clip(test_raw, 0.0, None)) - anchor_test_log
        results.append(
            ResidualModelResult(
                name=source_name,
                family="satellite_delta",
                oof_residuals=train_residuals,
                test_residuals=test_residuals,
                residual_rmse=float(np.sqrt(np.mean(np.square(train_residuals - residual_target)))),
            )
        )
    return results


def apply_residual_correction(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_residuals: np.ndarray,
    test_residuals: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_train_log = np.log1p(np.clip(anchor_train, 0.0, None))
    corrected_test_log = np.log1p(np.clip(anchor_test, 0.0, None))
    clipped_train = np.clip(np.asarray(train_residuals, dtype=np.float64), -clip_value, clip_value)
    clipped_test = np.clip(np.asarray(test_residuals, dtype=np.float64), -clip_value, clip_value)
    corrected_train_log[train_mask] = corrected_train_log[train_mask] + alpha * clipped_train[train_mask]
    corrected_test_log[test_mask] = corrected_test_log[test_mask] + alpha * clipped_test[test_mask]
    return np.clip(np.expm1(corrected_train_log), 0.0, None), np.clip(np.expm1(corrected_test_log), 0.0, None)


def candidate_from_source(
    source: ResidualModelResult,
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    mask_name: str,
    alpha: float,
    clip_value: float,
    confidence_quantile: float | None,
    y_train: np.ndarray,
) -> CandidateResult | None:
    effective_train_mask = train_mask.copy()
    effective_test_mask = test_mask.copy()
    if confidence_quantile is not None:
        train_segment = np.abs(source.oof_residuals[train_mask])
        if train_segment.size == 0:
            return None
        threshold = float(np.quantile(train_segment, confidence_quantile))
        effective_train_mask &= np.abs(source.oof_residuals) >= threshold
        effective_test_mask &= np.abs(source.test_residuals) >= threshold
        if not effective_train_mask.any() or not effective_test_mask.any():
            return None

    corrected_train, corrected_test = apply_residual_correction(
        anchor_train,
        anchor_test,
        source.oof_residuals,
        source.test_residuals,
        effective_train_mask,
        effective_test_mask,
        alpha,
        clip_value,
    )
    mean_abs_log_shift = float(
        np.mean(
            np.abs(
                np.log1p(np.clip(corrected_test, 0.0, None)) - np.log1p(np.clip(anchor_test, 0.0, None))
            )
        )
    )
    candidate_name = (
        f"{source.name}__{mask_name}__{maybe_token(confidence_quantile)}"
        f"__a{weight_token(alpha)}__c{weight_token(clip_value)}"
    )
    return CandidateResult(
        name=candidate_name,
        family="residual_correction",
        source_name=source.name,
        mask_name=mask_name,
        alpha=alpha,
        clip_value=clip_value,
        confidence_quantile=confidence_quantile,
        oof_predictions=corrected_train,
        test_predictions=corrected_test,
        affected_train_share=float(effective_train_mask.mean()),
        affected_test_share=float(effective_test_mask.mean()),
        mean_abs_log_shift=mean_abs_log_shift,
        oof_rmsle=rmsle(y_train, corrected_train),
    )


def build_candidates(
    residual_sources: list[ResidualModelResult],
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []

    global_alphas = (0.04, 0.06, 0.08, 0.10, 0.12)
    global_clips = (0.08, 0.12, 0.16)
    for source in residual_sources:
        for alpha in global_alphas:
            for clip_value in global_clips:
                candidate = candidate_from_source(
                    source,
                    anchor_train,
                    anchor_test,
                    train_masks["all_rows"],
                    test_masks["all_rows"],
                    "all_rows",
                    alpha,
                    clip_value,
                    None,
                    y_train,
                )
                if candidate is not None:
                    candidates.append(candidate)

    gated_sources = [source for source in residual_sources if source.family in {"tree_cat", "linear", "blend", "satellite_delta"}]
    gated_masks = [
        "inactive_or_sparse",
        "inactive_ge_18",
        "sparse_ge_0p5",
        "recent_le_10",
        "public_gap_top08",
        "public_gap_sparse",
        "public_gap_inactive_or_sparse",
        "fin_missing",
    ]
    gated_alphas = (0.10, 0.14, 0.18, 0.22, 0.26)
    gated_clips = (0.08, 0.12, 0.16, 0.20)
    confidence_grid: tuple[float | None, ...] = (None, 0.75, 0.85)
    for source in gated_sources:
        for mask_name in gated_masks:
            for alpha in gated_alphas:
                for clip_value in gated_clips:
                    for confidence_quantile in confidence_grid:
                        candidate = candidate_from_source(
                            source,
                            anchor_train,
                            anchor_test,
                            train_masks[mask_name],
                            test_masks[mask_name],
                            mask_name,
                            alpha,
                            clip_value,
                            confidence_quantile,
                            y_train,
                        )
                        if candidate is not None:
                            candidates.append(candidate)

    return candidates


def build_metrics_frame(candidates: list[CandidateResult], shift_penalty: float) -> pd.DataFrame:
    metrics_rows = []
    for candidate in candidates:
        selection_score = candidate.oof_rmsle + shift_penalty * candidate.mean_abs_log_shift
        metrics_rows.append(
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "source_name": candidate.source_name,
                "mask_name": candidate.mask_name,
                "alpha": candidate.alpha,
                "clip_value": candidate.clip_value,
                "confidence_quantile": candidate.confidence_quantile,
                "oof_rmsle": candidate.oof_rmsle,
                "mean_abs_log_shift": candidate.mean_abs_log_shift,
                "affected_train_share": candidate.affected_train_share,
                "affected_test_share": candidate.affected_test_share,
                "selection_score": selection_score,
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values(["selection_score", "oof_rmsle", "mean_abs_log_shift", "candidate_name"]).reset_index(
        drop=True
    )
    metrics["selection_rank"] = np.arange(1, len(metrics) + 1)
    metrics["oof_rank"] = metrics["oof_rmsle"].rank(method="dense").astype(int)
    return metrics


def write_outputs(
    run_dir: Path,
    sample_submission_path: Path,
    test_ids: np.ndarray,
    residual_sources: list[ResidualModelResult],
    candidates: list[CandidateResult],
    metrics: pd.DataFrame,
    submission_format: str,
    write_top_k_submissions: int,
    args: argparse.Namespace,
) -> None:
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    residual_metrics = pd.DataFrame(
        [
            {
                "model_name": result.name,
                "family": result.family,
                "residual_rmse": result.residual_rmse,
            }
            for result in sorted(residual_sources, key=lambda result: result.residual_rmse)
        ]
    )
    residual_metrics.to_csv(run_dir / "residual_model_metrics.csv", index=False)
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    top_selection_names = metrics.head(write_top_k_submissions)["candidate_name"].tolist()
    top_oof_names = (
        metrics.sort_values(["oof_rmsle", "mean_abs_log_shift", "candidate_name"])
        .head(max(8, write_top_k_submissions // 3))["candidate_name"]
        .tolist()
    )
    materialized_names = list(dict.fromkeys(top_selection_names + top_oof_names))
    for candidate_name in materialized_names:
        candidate = candidate_lookup[candidate_name]
        submission = build_submission_frame(
            sample_submission_path=sample_submission_path,
            unique_ids=test_ids,
            predictions=candidate.test_predictions,
            submission_format=submission_format,
        )
        submission.write_csv(submissions_dir / f"{candidate_name}.csv")

    best_by_selection = metrics.iloc[0]
    best_by_oof = metrics.sort_values(["oof_rmsle", "mean_abs_log_shift"]).iloc[0]
    summary_lines = [
        "# Anchor Residual Lab",
        "",
        f"Run directory: `{run_dir}`",
        f"Feature cache: `{args.feature_cache_path}`",
        f"Monthly run dir: `{args.monthly_run_dir}`",
        f"Submission format: `{submission_format}`",
        "",
        "## Residual models",
        "",
    ]
    for row in residual_metrics.itertuples(index=False):
        summary_lines.append(
            f"- `{row.model_name}` ({row.family}) residual RMSE: `{row.residual_rmse:.6f}`"
        )
    summary_lines.extend(
        [
            "",
            "## Best candidate by selection score",
            "",
            f"- name: `{best_by_selection.candidate_name}`",
            f"- OOF RMSLE: `{best_by_selection.oof_rmsle:.6f}`",
            f"- mean abs log shift: `{best_by_selection.mean_abs_log_shift:.6f}`",
            f"- affected test share: `{best_by_selection.affected_test_share:.6f}`",
            "",
            "## Best candidate by raw OOF RMSLE",
            "",
            f"- name: `{best_by_oof.candidate_name}`",
            f"- OOF RMSLE: `{best_by_oof.oof_rmsle:.6f}`",
            f"- mean abs log shift: `{best_by_oof.mean_abs_log_shift:.6f}`",
            f"- affected test share: `{best_by_oof.affected_test_share:.6f}`",
            "",
            "## Materialized submissions",
            "",
        ]
    )
    for candidate_name in materialized_names:
        summary_lines.append(f"- `{candidate_name}.csv`")
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="ascii")
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = build_run_directory(args.output_dir, args.run_name)

    sample_submission_path = data_dir / "SampleSubmission.csv"
    train_ids, test_ids, y_train, seeds = load_seed_candidates(data_dir)
    monthly_satellites = load_monthly_panel_satellites(args.monthly_run_dir.resolve(), train_ids, test_ids)
    train_df, test_df, y_train, anchor_train, anchor_test, train_masks, test_masks = build_model_frames(
        args.feature_cache_path.resolve(),
        train_ids,
        test_ids,
        seeds,
        monthly_satellites,
    )

    anchor_log = np.log1p(np.clip(anchor_train, 0.0, None))
    residual_target = np.log1p(y_train) - anchor_log

    learned_sources = cross_fit_residual_models(train_df, test_df, residual_target, args.n_splits)
    delta_sources = build_delta_sources(anchor_train, anchor_test, seeds, monthly_satellites, residual_target)
    residual_sources = learned_sources + delta_sources
    candidates = build_candidates(residual_sources, anchor_train, anchor_test, train_masks, test_masks, y_train)
    metrics = build_metrics_frame(candidates, shift_penalty=args.shift_penalty)

    write_outputs(
        run_dir=run_dir,
        sample_submission_path=sample_submission_path,
        test_ids=test_ids,
        residual_sources=residual_sources,
        candidates=candidates,
        metrics=metrics,
        submission_format=args.submission_format,
        write_top_k_submissions=args.write_top_k_submissions,
        args=args,
    )


if __name__ == "__main__":
    main()
