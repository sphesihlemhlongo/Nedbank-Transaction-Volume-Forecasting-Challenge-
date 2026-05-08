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

from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from public_feedback_temporal_lab import (
    build_disagreement_masks,
    build_feedback_anchor,
    load_context,
)
from anchor_residual_lab import numeric_array


FEATURE_PIPELINE_VERSION = "v13_public_feedback_round2_lab"


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    source_name: str
    primary_mask: str
    secondary_mask: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_rmsle: float
    mean_abs_log_shift: float
    affected_train_share: float
    affected_test_share: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round-2 deterministic patch search around the public-feedback temporal winner, "
            "focused on the identified low-activity weak points."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_feedback_round2_lab"))
    parser.add_argument("--run-name", type=str, default="public_feedback_round2_lab")
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
    parser.add_argument(
        "--shift-penalty",
        type=float,
        default=0.08,
        help="Penalty multiplier for drift relative to the current public-feedback winner.",
    )
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=16,
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


def build_current_public_feedback_anchor(
    feedback_anchor_train: np.ndarray,
    feedback_anchor_test: np.ndarray,
    temporal_xgb_train: np.ndarray,
    temporal_xgb_test: np.ndarray,
    public_gap_top10_train: np.ndarray,
    public_gap_top10_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(feedback_anchor_train.copy(), 0.0, None)
    test_predictions = np.clip(feedback_anchor_test.copy(), 0.0, None)
    train_predictions[public_gap_top10_train] = np.expm1(
        0.78 * np.log1p(np.clip(train_predictions[public_gap_top10_train], 0.0, None))
        + 0.22 * np.log1p(np.clip(temporal_xgb_train[public_gap_top10_train], 0.0, None))
    )
    test_predictions[public_gap_top10_test] = np.expm1(
        0.78 * np.log1p(np.clip(test_predictions[public_gap_top10_test], 0.0, None))
        + 0.22 * np.log1p(np.clip(temporal_xgb_test[public_gap_top10_test], 0.0, None))
    )
    return np.clip(train_predictions, 0.0, None), np.clip(test_predictions, 0.0, None)


def build_masks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    disagreement_masks_train: dict[str, np.ndarray],
    disagreement_masks_test: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    recent_train = numeric_array(train_df, "txn_recent_3m_count")
    recent_test = numeric_array(test_df, "txn_recent_3m_count")
    active_train = numeric_array(train_df, "txn_active_months_total")
    active_test = numeric_array(test_df, "txn_active_months_total")
    inactive_train = numeric_array(train_df, "txn_inactive_months_total")
    inactive_test = numeric_array(test_df, "txn_inactive_months_total")
    sparse_train = numeric_array(train_df, "txn_sparse_month_share_le_2")
    sparse_test = numeric_array(test_df, "txn_sparse_month_share_le_2")
    fin_missing_train = numeric_array(train_df, "fin_missing_flag", fill_value=1.0) >= 1.0
    fin_missing_test = numeric_array(test_df, "fin_missing_flag", fill_value=1.0) >= 1.0

    train_masks = {
        "all_rows": np.ones(len(train_df), dtype=bool),
        "recent_le_3": recent_train <= 3.0,
        "recent_le_10": recent_train <= 10.0,
        "active_le_6": active_train <= 6.0,
        "active_le_12": active_train <= 12.0,
        "inactive_ge_18": inactive_train >= 18.0,
        "sparse_ge_0p5": sparse_train >= 0.5,
        "sparse_ge_0p8": sparse_train >= 0.8,
        "fin_missing": fin_missing_train,
        "pred_le_8": anchor_train <= 8.0,
        "pred_le_12": anchor_train <= 12.0,
        "public_gap_top10": disagreement_masks_train["public_gap_top10"],
    }
    test_masks = {
        "all_rows": np.ones(len(test_df), dtype=bool),
        "recent_le_3": recent_test <= 3.0,
        "recent_le_10": recent_test <= 10.0,
        "active_le_6": active_test <= 6.0,
        "active_le_12": active_test <= 12.0,
        "inactive_ge_18": inactive_test >= 18.0,
        "sparse_ge_0p5": sparse_test >= 0.5,
        "sparse_ge_0p8": sparse_test >= 0.8,
        "fin_missing": fin_missing_test,
        "pred_le_8": anchor_test <= 8.0,
        "pred_le_12": anchor_test <= 12.0,
        "public_gap_top10": disagreement_masks_test["public_gap_top10"],
    }
    train_masks["top10_recent_le_10"] = train_masks["public_gap_top10"] & train_masks["recent_le_10"]
    test_masks["top10_recent_le_10"] = test_masks["public_gap_top10"] & test_masks["recent_le_10"]
    train_masks["top10_active_le_12"] = train_masks["public_gap_top10"] & train_masks["active_le_12"]
    test_masks["top10_active_le_12"] = test_masks["public_gap_top10"] & test_masks["active_le_12"]
    train_masks["top10_fin_missing"] = train_masks["public_gap_top10"] & train_masks["fin_missing"]
    test_masks["top10_fin_missing"] = test_masks["public_gap_top10"] & test_masks["fin_missing"]
    train_masks["top10_sparse_ge_0p5"] = train_masks["public_gap_top10"] & train_masks["sparse_ge_0p5"]
    test_masks["top10_sparse_ge_0p5"] = test_masks["public_gap_top10"] & test_masks["sparse_ge_0p5"]
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


def apply_shrink(
    base_train: np.ndarray,
    base_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    shrink_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(base_train.copy(), 0.0, None)
    test_predictions = np.clip(base_test.copy(), 0.0, None)
    train_predictions[train_mask] = np.clip(train_predictions[train_mask] * shrink_factor, 0.0, None)
    test_predictions[test_mask] = np.clip(test_predictions[test_mask] * shrink_factor, 0.0, None)
    return train_predictions, test_predictions


def score_candidate(
    name: str,
    family: str,
    source_name: str,
    primary_mask: str,
    secondary_mask: str,
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
        primary_mask=primary_mask,
        secondary_mask=secondary_mask,
        oof_predictions=np.clip(train_predictions, 0.0, None),
        test_predictions=np.clip(test_predictions, 0.0, None),
        oof_rmsle=rmsle(y_train, train_predictions),
        mean_abs_log_shift=float(
            np.mean(np.abs(np.log1p(np.clip(test_predictions, 0.0, None)) - np.log1p(np.clip(anchor_test, 0.0, None))))
        ),
        affected_train_share=float(affected_train_mask.mean()),
        affected_test_share=float(affected_test_mask.mean()),
    )


def build_single_overlay_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    anchor_test_reference: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    source_specs = {
        "panel_catboost_v1": (monthly_satellites["panel_catboost_v1"][0], monthly_satellites["panel_catboost_v1"][1]),
        "panel_blend_top2": (monthly_satellites["panel_blend_top2"][0], monthly_satellites["panel_blend_top2"][1]),
    }
    alpha_grid = {
        "panel_catboost_v1": (0.06, 0.10, 0.14, 0.18, 0.22, 0.26),
        "panel_blend_top2": (0.06, 0.10, 0.14, 0.18, 0.22),
    }
    mask_names = [
        "recent_le_3",
        "recent_le_10",
        "active_le_6",
        "active_le_12",
        "inactive_ge_18",
        "sparse_ge_0p5",
        "sparse_ge_0p8",
        "fin_missing",
        "pred_le_8",
        "pred_le_12",
        "top10_recent_le_10",
        "top10_active_le_12",
        "top10_fin_missing",
        "top10_sparse_ge_0p5",
    ]

    for source_name, (source_train, source_test) in source_specs.items():
        for mask_name in mask_names:
            for alpha in alpha_grid[source_name]:
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
                        name=f"single_{source_name}__{mask_name}__a{weight_token(alpha)}",
                        family="single_overlay",
                        source_name=source_name,
                        primary_mask=mask_name,
                        secondary_mask="",
                        train_predictions=train_predictions,
                        test_predictions=test_predictions,
                        y_train=y_train,
                        anchor_test=anchor_test_reference,
                        affected_train_mask=train_masks[mask_name],
                        affected_test_mask=test_masks[mask_name],
                    )
                )
    return candidates


def build_pair_overlay_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    anchor_test_reference: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    source_train, source_test = monthly_satellites["panel_catboost_v1"]
    pair_specs = [
        ("recent_le_10", "inactive_ge_18", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("recent_le_10", "active_le_6", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("recent_le_10", "pred_le_12", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("recent_le_10", "fin_missing", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("recent_le_10", "top10_recent_le_10", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("inactive_ge_18", "fin_missing", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("top10_recent_le_10", "active_le_6", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("top10_recent_le_10", "fin_missing", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
        ("top10_recent_le_10", "top10_sparse_ge_0p5", (0.18, 0.22, 0.26), (0.10, 0.14, 0.18, 0.22)),
    ]

    for primary_mask, secondary_mask, primary_grid, secondary_grid in pair_specs:
        for alpha_primary in primary_grid:
            for alpha_secondary in secondary_grid:
                train_predictions, test_predictions = apply_log_overlay(
                    anchor_train,
                    anchor_test,
                    source_train,
                    source_test,
                    train_masks[primary_mask],
                    test_masks[primary_mask],
                    alpha_primary,
                )
                train_predictions, test_predictions = apply_log_overlay(
                    train_predictions,
                    test_predictions,
                    source_train,
                    source_test,
                    train_masks[secondary_mask],
                    test_masks[secondary_mask],
                    alpha_secondary,
                )
                affected_train_mask = train_masks[primary_mask] | train_masks[secondary_mask]
                affected_test_mask = test_masks[primary_mask] | test_masks[secondary_mask]
                candidates.append(
                    score_candidate(
                        name=(
                            f"pair_panel_catboost_v1__{primary_mask}__a{weight_token(alpha_primary)}"
                            f"__{secondary_mask}__a{weight_token(alpha_secondary)}"
                        ),
                        family="pair_overlay",
                        source_name="panel_catboost_v1",
                        primary_mask=primary_mask,
                        secondary_mask=secondary_mask,
                        train_predictions=train_predictions,
                        test_predictions=test_predictions,
                        y_train=y_train,
                        anchor_test=anchor_test_reference,
                        affected_train_mask=affected_train_mask,
                        affected_test_mask=affected_test_mask,
                    )
                )
    return candidates


def build_shrink_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    y_train: np.ndarray,
    anchor_test_reference: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    for mask_name in (
        "recent_le_3",
        "recent_le_10",
        "active_le_6",
        "active_le_12",
        "fin_missing",
        "pred_le_8",
        "pred_le_12",
        "top10_recent_le_10",
        "top10_active_le_12",
    ):
        for shrink_factor in (0.94, 0.96, 0.97, 0.98, 0.99):
            train_predictions, test_predictions = apply_shrink(
                anchor_train,
                anchor_test,
                train_masks[mask_name],
                test_masks[mask_name],
                shrink_factor,
            )
            candidates.append(
                score_candidate(
                    name=f"shrink__{mask_name}__x{str(shrink_factor).replace('.', 'p')}",
                    family="shrink",
                    source_name="shrink",
                    primary_mask=mask_name,
                    secondary_mask="",
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    y_train=y_train,
                    anchor_test=anchor_test_reference,
                    affected_train_mask=train_masks[mask_name],
                    affected_test_mask=test_masks[mask_name],
                )
            )
    return candidates


def build_residual_segment_report(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    panel_single_train: np.ndarray,
) -> pd.DataFrame:
    truth_log = np.log1p(np.clip(y_train, 0.0, None))
    anchor_log = np.log1p(np.clip(anchor_train, 0.0, None))
    panel_log = np.log1p(np.clip(panel_single_train, 0.0, None))

    segment_frame = pd.DataFrame(
        {
            "y": y_train,
            "anchor_pred": anchor_train,
            "abs_log_err_anchor": np.abs(truth_log - anchor_log),
            "abs_log_err_panel_recent10": np.abs(truth_log - panel_log),
            "residual_anchor": truth_log - anchor_log,
            "recent_3m_count": numeric_array(train_df, "txn_recent_3m_count"),
            "active_months": numeric_array(train_df, "txn_active_months_total"),
            "inactive_months": numeric_array(train_df, "txn_inactive_months_total"),
            "sparse_share": numeric_array(train_df, "txn_sparse_month_share_le_2"),
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
                abs_log_err_panel_recent10=("abs_log_err_panel_recent10", "mean"),
                residual_anchor=("residual_anchor", "mean"),
                y_mean=("y", "mean"),
                pred_mean=("anchor_pred", "mean"),
            )
            .reset_index()
        )
        report.insert(0, "segment_name", segment_name)
        report.rename(columns={segment_name: "segment_value"}, inplace=True)
        report["panel_gain"] = report["abs_log_err_anchor"] - report["abs_log_err_panel_recent10"]
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def build_metrics_frame(candidates: list[CandidateResult], shift_penalty: float) -> pd.DataFrame:
    metrics = pd.DataFrame(
        [
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "source_name": candidate.source_name,
                "primary_mask": candidate.primary_mask,
                "secondary_mask": candidate.secondary_mask,
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
        "# Public Feedback Round 2 Lab",
        "",
        f"Run directory: `{run_dir}`",
        f"Feature cache: `{args.feature_cache_path}`",
        f"Monthly run dir: `{args.monthly_run_dir}`",
        f"Submission format: `{args.submission_format}`",
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
        train_ids,
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
    train_masks, test_masks = build_masks(
        train_df,
        test_df,
        current_anchor_train,
        current_anchor_test,
        disagreement_masks_train,
        disagreement_masks_test,
    )

    candidates: list[CandidateResult] = []
    candidates.extend(
        build_single_overlay_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            current_anchor_test,
            train_masks,
            test_masks,
            monthly_satellites,
        )
    )
    candidates.extend(
        build_pair_overlay_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            current_anchor_test,
            train_masks,
            test_masks,
            monthly_satellites,
        )
    )
    candidates.extend(
        build_shrink_candidates(
            current_anchor_train,
            current_anchor_test,
            y_train,
            current_anchor_test,
            train_masks,
            test_masks,
        )
    )

    panel_recent_train, panel_recent_test = apply_log_overlay(
        current_anchor_train,
        current_anchor_test,
        monthly_satellites["panel_catboost_v1"][0],
        monthly_satellites["panel_catboost_v1"][1],
        train_masks["recent_le_10"],
        test_masks["recent_le_10"],
        0.22,
    )
    residual_report = build_residual_segment_report(train_df, y_train, current_anchor_train, panel_recent_train)
    metrics = build_metrics_frame(candidates, shift_penalty=args.shift_penalty)

    write_outputs(
        run_dir=run_dir,
        sample_submission_path=data_dir / "SampleSubmission.csv",
        test_ids=test_ids,
        candidates=candidates,
        metrics=metrics,
        residual_report=residual_report,
        args=args,
    )


if __name__ == "__main__":
    main()
