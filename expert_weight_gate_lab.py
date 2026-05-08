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
from sklearn.model_selection import KFold

from anchor_residual_lab import (
    add_prediction_features,
    build_catboost_model,
    build_hgb_model,
    build_ridge_model,
    feature_columns_for_model,
    numeric_array,
)
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from public_feedback_calibration_lab import build_extended_masks
from public_feedback_round2_lab import build_current_public_feedback_anchor
from public_feedback_temporal_lab import build_disagreement_masks, build_feedback_anchor, load_context


FEATURE_PIPELINE_VERSION = "v16_expert_weight_gate_lab"
RANDOM_STATE = 42


@dataclass(frozen=True)
class GateModelResult:
    name: str
    family: str
    alpha_oof: np.ndarray
    alpha_test: np.ndarray
    alpha_rmse: float
    source_oof_predictions: np.ndarray
    source_test_predictions: np.ndarray
    source_oof_rmsle: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    source_name: str
    mask_name: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_rmsle: float
    mean_abs_log_shift: float
    affected_train_share: float
    affected_test_share: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train supervised soft-gating models that learn customer-level blend weights between "
            "v5_best_subset and temporal_xgb, then deploy them conservatively around the current public-best anchor."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/expert_weight_gate_lab"))
    parser.add_argument("--run-name", type=str, default="expert_weight_gate_lab")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v5_dense_monthly_panel_stack.parquet"),
    )
    parser.add_argument(
        "--monthly-run-dir",
        type=Path,
        default=Path("outputs/experiments/v5_monthly_panel_full1_20260430_091349_503223_78ac19f8"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--shift-penalty",
        type=float,
        default=0.10,
        help="Penalty multiplier for drift relative to the current public-best anchor.",
    )
    parser.add_argument("--write-top-k-submissions", type=int, default=16)
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def weight_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def optimal_alpha_target(
    y_train: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
) -> np.ndarray:
    y_log = np.log1p(np.clip(y_train, 0.0, None))
    left_log = np.log1p(np.clip(left_predictions, 0.0, None))
    right_log = np.log1p(np.clip(right_predictions, 0.0, None))
    delta = right_log - left_log
    alpha = np.zeros_like(y_log, dtype=np.float64)
    valid = np.abs(delta) > 1e-8
    alpha[valid] = (y_log[valid] - left_log[valid]) / delta[valid]
    return np.clip(alpha, 0.0, 1.0)


def prepare_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    feature_columns, numeric_columns, categorical_columns = feature_columns_for_model(train_df)
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    for column in categorical_columns:
        X_train[column] = X_train[column].astype("string").fillna("__missing__")
        X_test[column] = X_test[column].astype("string").fillna("__missing__")
    return X_train, X_test, feature_columns, numeric_columns, categorical_columns


def fit_gate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    alpha_target: np.ndarray,
    left_train: np.ndarray,
    left_test: np.ndarray,
    right_train: np.ndarray,
    right_test: np.ndarray,
    train_fin_missing: np.ndarray,
    test_fin_missing: np.ndarray,
    n_splits: int,
) -> list[GateModelResult]:
    X_train, X_test, _feature_columns, numeric_columns, categorical_columns = prepare_frames(train_df, test_df)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model_builders = [
        ("gate_ridge_v1", "linear", lambda: build_ridge_model(numeric_columns, categorical_columns)),
        ("gate_hgb_v1", "tree", lambda: build_hgb_model(numeric_columns, categorical_columns)),
        ("gate_catboost_v1", "tree_cat", lambda: build_catboost_model(categorical_columns)),
    ]

    results: list[GateModelResult] = []
    left_train_log = np.log1p(np.clip(left_train, 0.0, None))
    left_test_log = np.log1p(np.clip(left_test, 0.0, None))
    right_train_log = np.log1p(np.clip(right_train, 0.0, None))
    right_test_log = np.log1p(np.clip(right_test, 0.0, None))

    for model_name, family, builder in model_builders:
        alpha_oof = np.zeros(len(X_train), dtype=np.float64)
        test_fold_predictions: list[np.ndarray] = []
        for train_idx, valid_idx in splitter.split(X_train):
            model = builder()
            model.fit(X_train.iloc[train_idx], alpha_target[train_idx])
            alpha_oof[valid_idx] = np.asarray(model.predict(X_train.iloc[valid_idx]), dtype=np.float64)
            test_fold_predictions.append(np.asarray(model.predict(X_test), dtype=np.float64))
        alpha_test = np.mean(np.column_stack(test_fold_predictions), axis=1)
        alpha_oof = np.clip(alpha_oof, 0.0, 1.0)
        alpha_test = np.clip(alpha_test, 0.0, 1.0)

        # Preserve the known stronger fallback on missing-financial rows.
        alpha_oof[train_fin_missing] = 0.0
        alpha_test[test_fin_missing] = 0.0

        source_oof_log = (1.0 - alpha_oof) * left_train_log + alpha_oof * right_train_log
        source_test_log = (1.0 - alpha_test) * left_test_log + alpha_test * right_test_log
        source_oof_predictions = np.clip(np.expm1(source_oof_log), 0.0, None)
        source_test_predictions = np.clip(np.expm1(source_test_log), 0.0, None)
        results.append(
            GateModelResult(
                name=model_name,
                family=family,
                alpha_oof=alpha_oof,
                alpha_test=alpha_test,
                alpha_rmse=float(np.sqrt(np.mean(np.square(alpha_oof - alpha_target)))),
                source_oof_predictions=source_oof_predictions,
                source_test_predictions=source_test_predictions,
                source_oof_rmsle=rmsle(y_train, source_oof_predictions),
            )
        )

    top2 = sorted(results, key=lambda result: result.alpha_rmse)[:2]
    inv_rmse = np.array([1.0 / max(result.alpha_rmse, 1e-9) for result in top2], dtype=np.float64)
    weights = inv_rmse / inv_rmse.sum()
    alpha_oof = np.clip(weights[0] * top2[0].alpha_oof + weights[1] * top2[1].alpha_oof, 0.0, 1.0)
    alpha_test = np.clip(weights[0] * top2[0].alpha_test + weights[1] * top2[1].alpha_test, 0.0, 1.0)
    alpha_oof[train_fin_missing] = 0.0
    alpha_test[test_fin_missing] = 0.0
    source_oof_log = (1.0 - alpha_oof) * left_train_log + alpha_oof * right_train_log
    source_test_log = (1.0 - alpha_test) * left_test_log + alpha_test * right_test_log
    results.append(
        GateModelResult(
            name="gate_blend_top2_inv_rmse",
            family="blend",
            alpha_oof=alpha_oof,
            alpha_test=alpha_test,
            alpha_rmse=float(np.sqrt(np.mean(np.square(alpha_oof - alpha_target)))),
            source_oof_predictions=np.clip(np.expm1(source_oof_log), 0.0, None),
            source_test_predictions=np.clip(np.expm1(source_test_log), 0.0, None),
            source_oof_rmsle=rmsle(y_train, np.expm1(source_oof_log)),
        )
    )
    return results


def apply_log_blend(
    base_train: np.ndarray,
    base_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(base_train.copy(), 0.0, None)
    test_predictions = np.clip(base_test.copy(), 0.0, None)
    train_predictions[train_mask] = np.expm1(
        (1.0 - alpha) * np.log1p(np.clip(train_predictions[train_mask], 0.0, None))
        + alpha * np.log1p(np.clip(source_train[train_mask], 0.0, None))
    )
    test_predictions[test_mask] = np.expm1(
        (1.0 - alpha) * np.log1p(np.clip(test_predictions[test_mask], 0.0, None))
        + alpha * np.log1p(np.clip(source_test[test_mask], 0.0, None))
    )
    return np.clip(train_predictions, 0.0, None), np.clip(test_predictions, 0.0, None)


def scaled_source_predictions(
    left_train: np.ndarray,
    left_test: np.ndarray,
    right_train: np.ndarray,
    right_test: np.ndarray,
    alpha_oof: np.ndarray,
    alpha_test: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_train_log = np.log1p(np.clip(left_train, 0.0, None))
    left_test_log = np.log1p(np.clip(left_test, 0.0, None))
    right_train_log = np.log1p(np.clip(right_train, 0.0, None))
    right_test_log = np.log1p(np.clip(right_test, 0.0, None))
    alpha_oof_scaled = np.clip(scale * alpha_oof, 0.0, 1.0)
    alpha_test_scaled = np.clip(scale * alpha_test, 0.0, 1.0)
    train_log = (1.0 - alpha_oof_scaled) * left_train_log + alpha_oof_scaled * right_train_log
    test_log = (1.0 - alpha_test_scaled) * left_test_log + alpha_test_scaled * right_test_log
    return np.clip(np.expm1(train_log), 0.0, None), np.clip(np.expm1(test_log), 0.0, None)


def score_candidate(
    name: str,
    family: str,
    source_name: str,
    mask_name: str,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    y_train: np.ndarray,
    anchor_test: np.ndarray,
    affected_train_mask: np.ndarray,
    affected_test_mask: np.ndarray,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        family=family,
        source_name=source_name,
        mask_name=mask_name,
        oof_predictions=np.clip(train_predictions, 0.0, None),
        test_predictions=np.clip(test_predictions, 0.0, None),
        oof_rmsle=rmsle(y_train, train_predictions),
        mean_abs_log_shift=float(
            np.mean(np.abs(np.log1p(np.clip(test_predictions, 0.0, None)) - np.log1p(np.clip(anchor_test, 0.0, None))))
        ),
        affected_train_share=float(affected_train_mask.mean()),
        affected_test_share=float(affected_test_mask.mean()),
    )


def build_candidates(
    gate_models: list[GateModelResult],
    current_anchor_train: np.ndarray,
    current_anchor_test: np.ndarray,
    left_train: np.ndarray,
    left_test: np.ndarray,
    right_train: np.ndarray,
    right_test: np.ndarray,
    y_train: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    full_mask_train = np.ones(len(y_train), dtype=bool)
    full_mask_test = np.ones(len(current_anchor_test), dtype=bool)
    gated_masks = (
        "public_gap_top10",
        "top10_recent_le_10",
        "recent_le_10_pred_le_12",
        "top10_recent_le_10_pred_le_12",
        "fin_missing_pred_le_12",
    )
    for gate_model in gate_models:
        for scale in (0.60, 0.80, 1.00):
            source_train, source_test = scaled_source_predictions(
                left_train,
                left_test,
                right_train,
                right_test,
                gate_model.alpha_oof,
                gate_model.alpha_test,
                scale,
            )
            candidates.append(
                score_candidate(
                    name=f"{gate_model.name}__softgate_full__s{weight_token(scale)}",
                    family="softgate_full",
                    source_name=gate_model.name,
                    mask_name="all_rows",
                    train_predictions=source_train,
                    test_predictions=source_test,
                    y_train=y_train,
                    anchor_test=current_anchor_test,
                    affected_train_mask=full_mask_train,
                    affected_test_mask=full_mask_test,
                )
            )
            for alpha_global in (0.25, 0.40, 0.55, 0.70):
                train_predictions, test_predictions = apply_log_blend(
                    current_anchor_train,
                    current_anchor_test,
                    source_train,
                    source_test,
                    full_mask_train,
                    full_mask_test,
                    alpha_global,
                )
                candidates.append(
                    score_candidate(
                        name=(
                            f"{gate_model.name}__softgate_global__s{weight_token(scale)}"
                            f"__a{weight_token(alpha_global)}"
                        ),
                        family="softgate_global_anchorblend",
                        source_name=gate_model.name,
                        mask_name="all_rows",
                        train_predictions=train_predictions,
                        test_predictions=test_predictions,
                        y_train=y_train,
                        anchor_test=current_anchor_test,
                        affected_train_mask=full_mask_train,
                        affected_test_mask=full_mask_test,
                    )
                )
            for mask_name in gated_masks:
                for alpha in (0.25, 0.40, 0.55):
                    train_predictions, test_predictions = apply_log_blend(
                        current_anchor_train,
                        current_anchor_test,
                        source_train,
                        source_test,
                        train_masks[mask_name],
                        test_masks[mask_name],
                        alpha,
                    )
                    candidates.append(
                        score_candidate(
                            name=(
                                f"{gate_model.name}__{mask_name}__s{weight_token(scale)}"
                                f"__a{weight_token(alpha)}"
                            ),
                            family="softgate_anchorblend",
                            source_name=gate_model.name,
                            mask_name=mask_name,
                            train_predictions=train_predictions,
                            test_predictions=test_predictions,
                            y_train=y_train,
                            anchor_test=current_anchor_test,
                            affected_train_mask=train_masks[mask_name],
                            affected_test_mask=test_masks[mask_name],
                        )
                    )
    return candidates


def build_metrics_frame(candidates: list[CandidateResult], shift_penalty: float) -> pd.DataFrame:
    metrics = pd.DataFrame(
        [
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "source_name": candidate.source_name,
                "mask_name": candidate.mask_name,
                "oof_rmsle": candidate.oof_rmsle,
                "mean_abs_log_shift": candidate.mean_abs_log_shift,
                "affected_train_share": candidate.affected_train_share,
                "affected_test_share": candidate.affected_test_share,
                "selection_score": candidate.oof_rmsle + shift_penalty * candidate.mean_abs_log_shift,
            }
            for candidate in candidates
        ]
    )
    metrics = metrics.sort_values(["selection_score", "oof_rmsle", "mean_abs_log_shift", "candidate_name"]).reset_index(drop=True)
    metrics["selection_rank"] = np.arange(1, len(metrics) + 1)
    metrics["oof_rank"] = metrics["oof_rmsle"].rank(method="dense").astype(int)
    return metrics


def build_residual_segment_report(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    candidate_train: np.ndarray,
) -> pd.DataFrame:
    truth_log = np.log1p(np.clip(y_train, 0.0, None))
    anchor_log = np.log1p(np.clip(anchor_train, 0.0, None))
    candidate_log = np.log1p(np.clip(candidate_train, 0.0, None))
    segment_frame = pd.DataFrame(
        {
            "y": y_train,
            "anchor_pred": anchor_train,
            "candidate_pred": candidate_train,
            "abs_log_err_anchor": np.abs(truth_log - anchor_log),
            "abs_log_err_candidate": np.abs(truth_log - candidate_log),
            "recent_3m_count": numeric_array(train_df, "txn_recent_3m_count"),
            "active_months": numeric_array(train_df, "txn_active_months_total"),
            "fin_missing": (numeric_array(train_df, "fin_missing_flag", fill_value=1.0) >= 1.0).astype(int),
        }
    )
    segment_frame["recent_bin"] = pd.cut(
        segment_frame["recent_3m_count"],
        bins=[-1, 3, 10, 25, 60, 1e9],
        labels=["0_3", "4_10", "11_25", "26_60", "61_plus"],
        include_lowest=True,
    )
    segment_frame["active_bin"] = pd.cut(
        segment_frame["active_months"],
        bins=[-1, 6, 12, 18, 24, 40],
        labels=["0_6", "7_12", "13_18", "19_24", "25_plus"],
        include_lowest=True,
    )
    segment_frame["target_bin"] = pd.cut(
        segment_frame["y"],
        bins=[-1, 3, 10, 25, 60, 1e9],
        labels=["1_3", "4_10", "11_25", "26_60", "61_plus"],
        include_lowest=True,
    )
    reports: list[pd.DataFrame] = []
    for segment_name in ("recent_bin", "active_bin", "target_bin", "fin_missing"):
        report = (
            segment_frame.groupby(segment_name, dropna=False)
            .agg(
                n=("y", "size"),
                abs_log_err_anchor=("abs_log_err_anchor", "mean"),
                abs_log_err_candidate=("abs_log_err_candidate", "mean"),
                y_mean=("y", "mean"),
                pred_mean=("anchor_pred", "mean"),
                candidate_mean=("candidate_pred", "mean"),
            )
            .reset_index()
        )
        report.insert(0, "segment_name", segment_name)
        report.rename(columns={segment_name: "segment_value"}, inplace=True)
        report["candidate_gain"] = report["abs_log_err_anchor"] - report["abs_log_err_candidate"]
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def write_outputs(
    run_dir: Path,
    sample_submission_path: Path,
    test_ids: np.ndarray,
    gate_models: list[GateModelResult],
    candidates: list[CandidateResult],
    metrics: pd.DataFrame,
    residual_report: pd.DataFrame,
    baseline_rmsle: float,
    args: argparse.Namespace,
) -> None:
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    residual_report.to_csv(run_dir / "residual_segment_report.csv", index=False)
    gate_frame = pd.DataFrame(
        [
            {
                "model_name": gate_model.name,
                "family": gate_model.family,
                "alpha_rmse": gate_model.alpha_rmse,
                "source_oof_rmsle": gate_model.source_oof_rmsle,
                "mean_alpha_oof": float(gate_model.alpha_oof.mean()),
                "mean_alpha_test": float(gate_model.alpha_test.mean()),
            }
            for gate_model in gate_models
        ]
    ).sort_values(["alpha_rmse", "model_name"])
    gate_frame.to_csv(run_dir / "gate_model_metrics.csv", index=False)

    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    top_selection_names = metrics.head(args.write_top_k_submissions)["candidate_name"].tolist()
    top_oof_names = metrics.sort_values(["oof_rmsle", "mean_abs_log_shift", "candidate_name"]).head(
        max(10, args.write_top_k_submissions // 2)
    )["candidate_name"].tolist()
    materialized_names = list(dict.fromkeys(top_selection_names + top_oof_names))

    for candidate_name in materialized_names:
        candidate = candidate_lookup[candidate_name]
        submission = build_submission_frame(
            sample_submission_path=sample_submission_path,
            unique_ids=test_ids,
            predictions=candidate.test_predictions,
            submission_format=args.submission_format,
        )
        submission.write_csv(submissions_dir / f"{candidate_name}.csv")

    best_by_selection = metrics.iloc[0]
    best_by_oof = metrics.sort_values(["oof_rmsle", "mean_abs_log_shift"]).iloc[0]
    summary_lines = [
        "# Expert Weight Gate Lab",
        "",
        f"Run directory: `{run_dir}`",
        f"Feature cache: `{args.feature_cache_path}`",
        f"Monthly run dir: `{args.monthly_run_dir}`",
        f"Submission format: `{args.submission_format}`",
        f"Current public-best anchor OOF RMSLE: `{baseline_rmsle:.6f}`",
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
    for candidate_name in materialized_names:
        summary_lines.append(f"- `{candidate_name}.csv`")
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="ascii")
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = build_run_directory(args.output_dir, args.run_name)

    (
        _train_ids,
        test_ids,
        y_train,
        seeds,
        train_df,
        test_df,
        base_anchor_train,
        base_anchor_test,
        _train_masks_unused,
        _test_masks_unused,
        monthly_satellites,
    ) = load_context(
        feature_cache_path=args.feature_cache_path.resolve(),
        monthly_run_dir=args.monthly_run_dir.resolve(),
        data_dir=data_dir,
    )

    train_fin_missing = numeric_array(train_df, "fin_missing_flag", fill_value=1.0) >= 1.0
    test_fin_missing = numeric_array(test_df, "fin_missing_flag", fill_value=1.0) >= 1.0
    disagreement_masks_train, disagreement_masks_test = build_disagreement_masks(seeds, train_fin_missing, test_fin_missing)

    feedback_anchor_train, feedback_anchor_test = build_feedback_anchor(
        base_anchor_train,
        base_anchor_test,
        seeds["temporal_xgb"].oof_predictions,
        seeds["temporal_xgb"].test_predictions,
        disagreement_masks_train["public_gap_top08"],
        disagreement_masks_test["public_gap_top08"],
    )
    current_anchor_train, current_anchor_test = build_current_public_feedback_anchor(
        feedback_anchor_train,
        feedback_anchor_test,
        seeds["temporal_xgb"].oof_predictions,
        seeds["temporal_xgb"].test_predictions,
        disagreement_masks_train["public_gap_top10"],
        disagreement_masks_test["public_gap_top10"],
    )
    baseline_rmsle = rmsle(y_train, current_anchor_train)
    train_masks, test_masks = build_extended_masks(
        train_df,
        test_df,
        current_anchor_train,
        current_anchor_test,
        disagreement_masks_train,
        disagreement_masks_test,
    )

    train_df, test_df = add_prediction_features(
        train_df,
        test_df,
        current_anchor_train,
        current_anchor_test,
        seeds,
        monthly_satellites,
    )

    left_train = seeds["v5_best_subset"].oof_predictions
    left_test = seeds["v5_best_subset"].test_predictions
    right_train = seeds["temporal_xgb"].oof_predictions
    right_test = seeds["temporal_xgb"].test_predictions
    alpha_target = optimal_alpha_target(y_train, left_train, right_train)

    gate_models = fit_gate_models(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        alpha_target=alpha_target,
        left_train=left_train,
        left_test=left_test,
        right_train=right_train,
        right_test=right_test,
        train_fin_missing=train_fin_missing,
        test_fin_missing=test_fin_missing,
        n_splits=args.n_splits,
    )

    candidates = build_candidates(
        gate_models=gate_models,
        current_anchor_train=current_anchor_train,
        current_anchor_test=current_anchor_test,
        left_train=left_train,
        left_test=left_test,
        right_train=right_train,
        right_test=right_test,
        y_train=y_train,
        train_masks=train_masks,
        test_masks=test_masks,
    )
    metrics = build_metrics_frame(candidates, args.shift_penalty)
    best_candidate_name = metrics.iloc[0]["candidate_name"]
    best_candidate = next(candidate for candidate in candidates if candidate.name == best_candidate_name)
    residual_report = build_residual_segment_report(
        train_df,
        y_train,
        current_anchor_train,
        best_candidate.oof_predictions,
    )

    write_outputs(
        run_dir=run_dir,
        sample_submission_path=data_dir / "SampleSubmission.csv",
        test_ids=test_ids,
        gate_models=gate_models,
        candidates=candidates,
        metrics=metrics,
        residual_report=residual_report,
        baseline_rmsle=baseline_rmsle,
        args=args,
    )


if __name__ == "__main__":
    main()
