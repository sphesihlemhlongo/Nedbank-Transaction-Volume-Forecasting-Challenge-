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
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

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


FEATURE_PIPELINE_VERSION = "v15_feature_aware_target_stack_lab"
RANDOM_STATE = 42


@dataclass(frozen=True)
class StackSource:
    name: str
    family: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    log_rmse: float


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
            "Train cross-fitted feature-aware target stackers around the current public-best anchor and "
            "materialize full, anchored, and gated leaderboard candidates."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_aware_target_stack_lab"))
    parser.add_argument("--run-name", type=str, default="feature_aware_target_stack_lab")
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


def fit_target_stack_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int,
) -> list[StackSource]:
    y_train_log = np.log1p(np.clip(y_train, 0.0, None))
    feature_columns, numeric_columns, categorical_columns = feature_columns_for_model(train_df)
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    for column in categorical_columns:
        X_train[column] = X_train[column].astype("string").fillna("__missing__")
        X_test[column] = X_test[column].astype("string").fillna("__missing__")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model_builders: list[tuple[str, str, object]] = [
        ("target_ridge_stack_v1", "linear", lambda: build_ridge_model(numeric_columns, categorical_columns)),
        ("target_hgb_stack_v1", "tree", lambda: build_hgb_model(numeric_columns, categorical_columns)),
        ("target_catboost_stack_v1", "tree_cat", lambda: build_catboost_model(categorical_columns)),
        (
            "target_ridge_stack_v2",
            "linear",
            lambda: Pipeline(
                steps=[
                    ("transformer", build_ridge_model(numeric_columns, categorical_columns).named_steps["transformer"]),
                    ("model", Ridge(alpha=8.0, positive=True)),
                ]
            ),
        ),
    ]

    results: list[StackSource] = []
    for model_name, family, builder in model_builders:
        oof_log = np.zeros(len(X_train), dtype=np.float64)
        test_fold_predictions: list[np.ndarray] = []
        for train_idx, valid_idx in splitter.split(X_train):
            model = builder()
            model.fit(X_train.iloc[train_idx], y_train_log[train_idx])
            oof_log[valid_idx] = np.asarray(model.predict(X_train.iloc[valid_idx]), dtype=np.float64)
            test_fold_predictions.append(np.asarray(model.predict(X_test), dtype=np.float64))
        test_log = np.mean(np.column_stack(test_fold_predictions), axis=1)
        oof_predictions = np.clip(np.expm1(oof_log), 0.0, None)
        test_predictions = np.clip(np.expm1(test_log), 0.0, None)
        results.append(
            StackSource(
                name=model_name,
                family=family,
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                log_rmse=float(np.sqrt(np.mean(np.square(oof_log - y_train_log)))),
            )
        )

    top2 = sorted(results, key=lambda result: result.log_rmse)[:2]
    inv_rmse = np.array([1.0 / max(result.log_rmse, 1e-9) for result in top2], dtype=np.float64)
    weights = inv_rmse / inv_rmse.sum()
    blend_oof_log = (
        weights[0] * np.log1p(np.clip(top2[0].oof_predictions, 0.0, None))
        + weights[1] * np.log1p(np.clip(top2[1].oof_predictions, 0.0, None))
    )
    blend_test_log = (
        weights[0] * np.log1p(np.clip(top2[0].test_predictions, 0.0, None))
        + weights[1] * np.log1p(np.clip(top2[1].test_predictions, 0.0, None))
    )
    results.append(
        StackSource(
            name="target_stack_blend_top2_inv_rmse",
            family="blend",
            oof_predictions=np.clip(np.expm1(blend_oof_log), 0.0, None),
            test_predictions=np.clip(np.expm1(blend_test_log), 0.0, None),
            log_rmse=float(np.sqrt(np.mean(np.square(blend_oof_log - y_train_log)))),
        )
    )
    return results


def apply_log_blend(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(anchor_train.copy(), 0.0, None)
    test_predictions = np.clip(anchor_test.copy(), 0.0, None)
    train_predictions[train_mask] = np.expm1(
        (1.0 - alpha) * np.log1p(np.clip(train_predictions[train_mask], 0.0, None))
        + alpha * np.log1p(np.clip(source_train[train_mask], 0.0, None))
    )
    test_predictions[test_mask] = np.expm1(
        (1.0 - alpha) * np.log1p(np.clip(test_predictions[test_mask], 0.0, None))
        + alpha * np.log1p(np.clip(source_test[test_mask], 0.0, None))
    )
    return np.clip(train_predictions, 0.0, None), np.clip(test_predictions, 0.0, None)


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


def add_round11_expert_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    panel_train: np.ndarray,
    panel_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_specs = {
        "pair_recent10_active6": [("recent_le_10", 0.26), ("active_le_6", 0.22)],
        "pair_recent10_inactive18": [("recent_le_10", 0.26), ("inactive_ge_18", 0.22)],
        "pair_recent10_top10recent10": [("recent_le_10", 0.26), ("top10_recent_le_10", 0.22)],
    }
    extra_train: dict[str, np.ndarray] = {}
    extra_test: dict[str, np.ndarray] = {}
    for feature_name, steps in candidate_specs.items():
        candidate_train = anchor_train.copy()
        candidate_test = anchor_test.copy()
        for mask_name, alpha in steps:
            candidate_train, candidate_test = apply_log_blend(
                candidate_train,
                candidate_test,
                panel_train,
                panel_test,
                train_masks[mask_name],
                test_masks[mask_name],
                alpha,
            )
        extra_train[f"pred_{feature_name}"] = candidate_train
        extra_test[f"pred_{feature_name}"] = candidate_test
        extra_train[f"log_pred_{feature_name}"] = np.log1p(np.clip(candidate_train, 0.0, None))
        extra_test[f"log_pred_{feature_name}"] = np.log1p(np.clip(candidate_test, 0.0, None))
    return (
        pd.concat([train_df, pd.DataFrame(extra_train, index=train_df.index)], axis=1).copy(),
        pd.concat([test_df, pd.DataFrame(extra_test, index=test_df.index)], axis=1).copy(),
    )


def build_candidates(
    stack_sources: list[StackSource],
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    full_mask_train = np.ones_like(anchor_train, dtype=bool)
    full_mask_test = np.ones_like(anchor_test, dtype=bool)
    gated_mask_names = (
        "recent_le_10",
        "active_le_6",
        "inactive_ge_18",
        "fin_missing",
        "top10_recent_le_10",
        "recent_le_10_pred_le_12",
        "top10_recent_le_10_pred_le_12",
        "fin_missing_pred_le_12",
    )

    for source in stack_sources:
        candidates.append(
            score_candidate(
                name=f"{source.name}__full",
                family="stack_full",
                source_name=source.name,
                mask_name="all_rows",
                train_predictions=source.oof_predictions,
                test_predictions=source.test_predictions,
                y_train=y_train,
                anchor_test=anchor_test,
                affected_train_mask=full_mask_train,
                affected_test_mask=full_mask_test,
            )
        )
        for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
            train_predictions, test_predictions = apply_log_blend(
                anchor_train,
                anchor_test,
                source.oof_predictions,
                source.test_predictions,
                full_mask_train,
                full_mask_test,
                alpha,
            )
            candidates.append(
                score_candidate(
                    name=f"{source.name}__anchorblend_a{weight_token(alpha)}",
                    family="stack_anchorblend",
                    source_name=source.name,
                    mask_name="all_rows",
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    y_train=y_train,
                    anchor_test=anchor_test,
                    affected_train_mask=full_mask_train,
                    affected_test_mask=full_mask_test,
                )
            )
        for mask_name in gated_mask_names:
            for alpha in (0.25, 0.40, 0.55, 0.70):
                train_predictions, test_predictions = apply_log_blend(
                    anchor_train,
                    anchor_test,
                    source.oof_predictions,
                    source.test_predictions,
                    train_masks[mask_name],
                    test_masks[mask_name],
                    alpha,
                )
                candidates.append(
                    score_candidate(
                        name=f"{source.name}__{mask_name}__a{weight_token(alpha)}",
                        family="stack_gated",
                        source_name=source.name,
                        mask_name=mask_name,
                        train_predictions=train_predictions,
                        test_predictions=test_predictions,
                        y_train=y_train,
                        anchor_test=anchor_test,
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
    stack_sources: list[StackSource],
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

    stack_frame = pd.DataFrame(
        [
            {
                "source_name": source.name,
                "family": source.family,
                "log_rmse": source.log_rmse,
            }
            for source in stack_sources
        ]
    ).sort_values(["log_rmse", "source_name"])
    stack_frame.to_csv(run_dir / "stack_source_metrics.csv", index=False)

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
        "# Feature Aware Target Stack Lab",
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
    train_df, test_df = add_round11_expert_features(
        train_df,
        test_df,
        current_anchor_train,
        current_anchor_test,
        train_masks,
        test_masks,
        monthly_satellites["panel_catboost_v1"][0],
        monthly_satellites["panel_catboost_v1"][1],
    )

    stack_sources = fit_target_stack_models(train_df, test_df, y_train, args.n_splits)
    candidates = build_candidates(
        stack_sources=stack_sources,
        anchor_train=current_anchor_train,
        anchor_test=current_anchor_test,
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
        stack_sources=stack_sources,
        candidates=candidates,
        metrics=metrics,
        residual_report=residual_report,
        baseline_rmsle=baseline_rmsle,
        args=args,
    )


if __name__ == "__main__":
    main()
