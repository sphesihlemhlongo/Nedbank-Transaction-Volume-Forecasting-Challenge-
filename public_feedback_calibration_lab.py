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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from public_feedback_round2_lab import build_current_public_feedback_anchor, build_masks
from public_feedback_temporal_lab import build_disagreement_masks, build_feedback_anchor, load_context
from anchor_residual_lab import numeric_array


FEATURE_PIPELINE_VERSION = "v14_public_feedback_calibration_lab"
RANDOM_STATE = 42


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
            "Cross-fitted deterministic calibration search around the current public-best feedback "
            "temporal winner. Focuses on monotone low-count calibration and low-activity shrink schedules."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_feedback_calibration_lab"))
    parser.add_argument("--run-name", type=str, default="public_feedback_calibration_lab")
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
        help="Penalty multiplier for drift relative to the current public-best feedback anchor.",
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


def build_extended_masks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    disagreement_masks_train: dict[str, np.ndarray],
    disagreement_masks_test: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_masks, test_masks = build_masks(
        train_df,
        test_df,
        anchor_train,
        anchor_test,
        disagreement_masks_train,
        disagreement_masks_test,
    )
    train_masks["recent_le_10_pred_le_12"] = train_masks["recent_le_10"] & (anchor_train <= 12.0)
    test_masks["recent_le_10_pred_le_12"] = test_masks["recent_le_10"] & (anchor_test <= 12.0)
    train_masks["recent_le_3_pred_le_8"] = train_masks["recent_le_3"] & (anchor_train <= 8.0)
    test_masks["recent_le_3_pred_le_8"] = test_masks["recent_le_3"] & (anchor_test <= 8.0)
    train_masks["active_le_6_pred_le_12"] = train_masks["active_le_6"] & (anchor_train <= 12.0)
    test_masks["active_le_6_pred_le_12"] = test_masks["active_le_6"] & (anchor_test <= 12.0)
    train_masks["fin_missing_pred_le_12"] = train_masks["fin_missing"] & (anchor_train <= 12.0)
    test_masks["fin_missing_pred_le_12"] = test_masks["fin_missing"] & (anchor_test <= 12.0)
    train_masks["top10_recent_le_10_pred_le_12"] = train_masks["top10_recent_le_10"] & (anchor_train <= 12.0)
    test_masks["top10_recent_le_10_pred_le_12"] = test_masks["top10_recent_le_10"] & (anchor_test <= 12.0)
    return train_masks, test_masks


def apply_log_overlay(
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


def apply_piecewise_shrink(
    base_train: np.ndarray,
    base_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    thresholds: tuple[float, float, float],
    factors: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(base_train.copy(), 0.0, None)
    test_predictions = np.clip(base_test.copy(), 0.0, None)

    def scaled(values: np.ndarray) -> np.ndarray:
        output = values.copy()
        output = np.where(output <= thresholds[0], output * factors[0], output)
        output = np.where((output > thresholds[0]) & (output <= thresholds[1]), output * factors[1], output)
        output = np.where((output > thresholds[1]) & (output <= thresholds[2]), output * factors[2], output)
        output = np.where(output > thresholds[2], output * factors[3], output)
        return np.clip(output, 0.0, None)

    train_predictions[train_mask] = scaled(train_predictions[train_mask])
    test_predictions[test_mask] = scaled(test_predictions[test_mask])
    return train_predictions, test_predictions


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


def cross_fit_isotonic_source(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if int(train_mask.sum()) < 80:
        return None

    x_full = np.log1p(np.clip(anchor_train, 0.0, None))
    y_full = np.log1p(np.clip(y_train, 0.0, None))
    x_test_full = np.log1p(np.clip(anchor_test, 0.0, None))

    oof_log = x_full.copy()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    successful_folds = 0
    for train_index, valid_index in kfold.split(x_full):
        fit_index = train_index[train_mask[train_index]]
        predict_index = valid_index[train_mask[valid_index]]
        if len(fit_index) < 50 or len(predict_index) == 0:
            continue
        x_fit = x_full[fit_index]
        y_fit = y_full[fit_index]
        if np.unique(x_fit).size < 10:
            continue
        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(x_fit, y_fit)
        oof_log[predict_index] = model.predict(x_full[predict_index])
        successful_folds += 1

    if successful_folds == 0:
        return None

    fit_index = np.flatnonzero(train_mask)
    x_fit = x_full[fit_index]
    y_fit = y_full[fit_index]
    if np.unique(x_fit).size < 10:
        return None
    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(x_fit, y_fit)
    test_log = x_test_full.copy()
    predict_test_index = np.flatnonzero(test_mask)
    if len(predict_test_index) > 0:
        test_log[predict_test_index] = model.predict(x_test_full[predict_test_index])

    return np.clip(np.expm1(oof_log), 0.0, None), np.clip(np.expm1(test_log), 0.0, None)


def build_isotonic_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    n_splits: int,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    mask_names = (
        "all_rows",
        "recent_le_3",
        "recent_le_10",
        "active_le_6",
        "fin_missing",
        "top10_recent_le_10",
        "recent_le_3_pred_le_8",
        "recent_le_10_pred_le_12",
        "top10_recent_le_10_pred_le_12",
    )
    alpha_grid = {
        "all_rows": (0.20, 0.35, 0.50),
        "recent_le_3": (0.35, 0.50, 0.65, 0.80, 1.00),
        "recent_le_10": (0.25, 0.40, 0.55, 0.70, 1.00),
        "active_le_6": (0.25, 0.40, 0.55, 0.70, 1.00),
        "fin_missing": (0.25, 0.40, 0.55, 0.70, 1.00),
        "top10_recent_le_10": (0.35, 0.50, 0.65, 0.80, 1.00),
        "recent_le_3_pred_le_8": (0.35, 0.50, 0.65, 0.80, 1.00),
        "recent_le_10_pred_le_12": (0.25, 0.40, 0.55, 0.70, 1.00),
        "top10_recent_le_10_pred_le_12": (0.35, 0.50, 0.65, 0.80, 1.00),
    }

    for mask_name in mask_names:
        source = cross_fit_isotonic_source(
            anchor_train=anchor_train,
            anchor_test=anchor_test,
            y_train=y_train,
            train_mask=train_masks[mask_name],
            test_mask=test_masks[mask_name],
            n_splits=n_splits,
        )
        if source is None:
            continue
        source_train, source_test = source
        for alpha in alpha_grid[mask_name]:
            train_predictions, test_predictions = apply_log_overlay(
                anchor_train,
                anchor_test,
                source_train,
                source_test,
                train_masks[mask_name],
                test_masks[mask_name],
                alpha,
            )
            candidates.append(
                score_candidate(
                    name=f"isotonic__{mask_name}__a{weight_token(alpha)}",
                    family="isotonic",
                    source_name="isotonic",
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


def build_piecewise_shrink_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    schedule_specs = {
        "s1": ((4.0, 10.0, 20.0), (0.88, 0.94, 0.98, 1.00)),
        "s2": ((4.0, 10.0, 20.0), (0.90, 0.95, 0.985, 1.00)),
        "s3": ((4.0, 8.0, 15.0), (0.92, 0.96, 0.985, 1.00)),
        "s4": ((3.0, 8.0, 12.0), (0.90, 0.955, 0.985, 1.00)),
        "s5": ((4.0, 12.0, 20.0), (0.94, 0.97, 0.99, 1.00)),
        "s6": ((6.0, 12.0, 20.0), (0.95, 0.975, 0.99, 1.00)),
    }
    mask_names = (
        "recent_le_3",
        "recent_le_10",
        "active_le_6",
        "fin_missing",
        "top10_recent_le_10",
        "recent_le_3_pred_le_8",
        "recent_le_10_pred_le_12",
        "top10_recent_le_10_pred_le_12",
        "fin_missing_pred_le_12",
    )
    for mask_name in mask_names:
        for schedule_name, (thresholds, factors) in schedule_specs.items():
            train_predictions, test_predictions = apply_piecewise_shrink(
                anchor_train,
                anchor_test,
                train_masks[mask_name],
                test_masks[mask_name],
                thresholds,
                factors,
            )
            candidates.append(
                score_candidate(
                    name=f"piecewise_shrink__{mask_name}__{schedule_name}",
                    family="piecewise_shrink",
                    source_name=schedule_name,
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


def build_hybrid_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    n_splits: int,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    recent_source = cross_fit_isotonic_source(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        y_train=y_train,
        train_mask=train_masks["recent_le_10_pred_le_12"],
        test_mask=test_masks["recent_le_10_pred_le_12"],
        n_splits=n_splits,
    )
    if recent_source is not None:
        source_train, source_test = recent_source
        for alpha in (0.35, 0.50, 0.65):
            base_train, base_test = apply_log_overlay(
                anchor_train,
                anchor_test,
                source_train,
                source_test,
                train_masks["recent_le_10_pred_le_12"],
                test_masks["recent_le_10_pred_le_12"],
                alpha,
            )
            for schedule_name, (thresholds, factors) in {
                "lite": ((3.0, 8.0, 12.0), (0.95, 0.98, 0.995, 1.00)),
                "mid": ((3.0, 8.0, 12.0), (0.93, 0.97, 0.99, 1.00)),
                "focused": ((2.0, 6.0, 10.0), (0.90, 0.96, 0.99, 1.00)),
            }.items():
                train_predictions, test_predictions = apply_piecewise_shrink(
                    base_train,
                    base_test,
                    train_masks["top10_recent_le_10_pred_le_12"],
                    test_masks["top10_recent_le_10_pred_le_12"],
                    thresholds,
                    factors,
                )
                affected_train_mask = train_masks["recent_le_10_pred_le_12"] | train_masks["top10_recent_le_10_pred_le_12"]
                affected_test_mask = test_masks["recent_le_10_pred_le_12"] | test_masks["top10_recent_le_10_pred_le_12"]
                candidates.append(
                    score_candidate(
                        name=f"hybrid_isotonic_recent_pred12__a{weight_token(alpha)}__{schedule_name}",
                        family="hybrid",
                        source_name="isotonic_plus_shrink",
                        mask_name="recent_le_10_pred_le_12__top10_recent_le_10_pred_le_12",
                        train_predictions=train_predictions,
                        test_predictions=test_predictions,
                        y_train=y_train,
                        anchor_test=anchor_test,
                        affected_train_mask=affected_train_mask,
                        affected_test_mask=affected_test_mask,
                    )
                )

    fin_source = cross_fit_isotonic_source(
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        y_train=y_train,
        train_mask=train_masks["fin_missing"],
        test_mask=test_masks["fin_missing"],
        n_splits=n_splits,
    )
    if fin_source is None:
        return candidates

    fin_source_train, fin_source_test = fin_source
    fin_schedule_specs = {
        "fin_s3": ("fin_missing_pred_le_12", (4.0, 8.0, 12.0), (0.94, 0.97, 0.99, 1.00)),
        "fin_s5": ("fin_missing_pred_le_12", (4.0, 12.0, 20.0), (0.94, 0.97, 0.99, 1.00)),
        "top10_s5": ("top10_recent_le_10_pred_le_12", (4.0, 12.0, 20.0), (0.94, 0.97, 0.99, 1.00)),
        "recent3_s5": ("recent_le_3_pred_le_8", (4.0, 12.0, 20.0), (0.94, 0.97, 0.99, 1.00)),
    }
    for alpha in (0.25, 0.40, 0.55):
        base_train, base_test = apply_log_overlay(
            anchor_train,
            anchor_test,
            fin_source_train,
            fin_source_test,
            train_masks["fin_missing"],
            test_masks["fin_missing"],
            alpha,
        )
        for schedule_name, (mask_name, thresholds, factors) in fin_schedule_specs.items():
            train_predictions, test_predictions = apply_piecewise_shrink(
                base_train,
                base_test,
                train_masks[mask_name],
                test_masks[mask_name],
                thresholds,
                factors,
            )
            affected_train_mask = train_masks["fin_missing"] | train_masks[mask_name]
            affected_test_mask = test_masks["fin_missing"] | test_masks[mask_name]
            candidates.append(
                score_candidate(
                    name=f"hybrid_isotonic_fin_missing__a{weight_token(alpha)}__{schedule_name}",
                    family="hybrid",
                    source_name="fin_missing_isotonic_plus_shrink",
                    mask_name=f"fin_missing__{mask_name}",
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    y_train=y_train,
                    anchor_test=anchor_test,
                    affected_train_mask=affected_train_mask,
                    affected_test_mask=affected_test_mask,
                )
            )
    return candidates


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


def write_outputs(
    run_dir: Path,
    sample_submission_path: Path,
    test_ids: np.ndarray,
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
        "# Public Feedback Calibration Lab",
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
        _monthly_satellites_unused,
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

    candidates: list[CandidateResult] = []
    candidates.extend(
        build_isotonic_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            train_masks,
            test_masks,
            args.n_splits,
        )
    )
    candidates.extend(
        build_piecewise_shrink_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            train_masks,
            test_masks,
        )
    )
    candidates.extend(
        build_hybrid_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            train_masks,
            test_masks,
            args.n_splits,
        )
    )
    if not candidates:
        raise ValueError("No calibration candidates were produced.")

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
        candidates=candidates,
        metrics=metrics,
        residual_report=residual_report,
        baseline_rmsle=baseline_rmsle,
        args=args,
    )


if __name__ == "__main__":
    main()
