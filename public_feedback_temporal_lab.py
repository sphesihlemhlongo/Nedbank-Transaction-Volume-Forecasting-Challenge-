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

from anchor_residual_lab import (
    build_current_public_best_anchor,
    build_masks,
    load_feature_splits,
    load_monthly_panel_satellites,
    numeric_array,
)
from artifact_blend_lab import load_seed_candidates
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory


FEATURE_PIPELINE_VERSION = "v12_public_feedback_temporal_lab"


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
            "Search reproducible temporal-heavy candidate schedules around the current public-best "
            "leaderboard branch."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_feedback_temporal_lab"))
    parser.add_argument("--run-name", type=str, default="public_feedback_temporal_lab")
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
        default=0.10,
        help="Drift penalty relative to the new temporal public-feedback anchor.",
    )
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=18,
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


def load_context(
    feature_cache_path: Path,
    monthly_run_dir: Path,
    data_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    train_ids, test_ids, y_train, seeds = load_seed_candidates(data_dir)
    monthly_satellites = load_monthly_panel_satellites(monthly_run_dir, train_ids, test_ids)
    train_df, test_df = load_feature_splits(feature_cache_path, train_ids, test_ids)
    y_train_from_frame = train_df.pop("next_3m_txn_count").to_numpy(dtype=np.float64)
    if not np.allclose(y_train, y_train_from_frame):
        raise ValueError("Target mismatch between seed artifacts and feature cache.")

    train_fin_missing = numeric_array(train_df, "fin_missing_flag", fill_value=1.0) >= 1.0
    test_fin_missing = numeric_array(test_df, "fin_missing_flag", fill_value=1.0) >= 1.0

    base_anchor_train, base_anchor_test, public_gap_top08_train, public_gap_top08_test, _ = build_current_public_best_anchor(
        seeds,
        train_fin_missing,
        test_fin_missing,
    )

    train_masks = build_masks(train_df, public_gap_top08_train, train_fin_missing, base_anchor_train)
    test_masks = build_masks(test_df, public_gap_top08_test, test_fin_missing, base_anchor_test)
    return (
        train_ids,
        test_ids,
        y_train,
        seeds,
        train_df,
        test_df,
        base_anchor_train,
        base_anchor_test,
        train_masks,
        test_masks,
        monthly_satellites,
    )


def build_disagreement_masks(
    seeds: dict[str, object],
    train_fin_missing: np.ndarray,
    test_fin_missing: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    temporal_train = seeds["temporal_xgb"].oof_predictions
    temporal_test = seeds["temporal_xgb"].test_predictions
    disagreement_train = np.abs(np.log1p(temporal_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_test) - np.log1p(v5_test))
    non_missing_train = ~train_fin_missing
    non_missing_test = ~test_fin_missing

    quantile_specs = {
        "public_gap_top12": 0.88,
        "public_gap_top10": 0.90,
        "public_gap_top08": 0.92,
        "public_gap_top06": 0.94,
        "public_gap_top04": 0.96,
    }

    train_masks: dict[str, np.ndarray] = {}
    test_masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, quantile in quantile_specs.items():
        threshold = float(np.quantile(disagreement_train[non_missing_train], quantile))
        thresholds[name] = threshold
        train_masks[name] = non_missing_train & (disagreement_train >= threshold)
        test_masks[name] = non_missing_test & (disagreement_test >= threshold)

    train_masks["public_gap_band_04_08"] = train_masks["public_gap_top08"] & ~train_masks["public_gap_top04"]
    test_masks["public_gap_band_04_08"] = test_masks["public_gap_top08"] & ~test_masks["public_gap_top04"]
    train_masks["public_gap_band_06_12"] = train_masks["public_gap_top12"] & ~train_masks["public_gap_top06"]
    test_masks["public_gap_band_06_12"] = test_masks["public_gap_top12"] & ~test_masks["public_gap_top06"]
    return train_masks, test_masks


def apply_log_delta(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_train_log = np.log1p(np.clip(anchor_train, 0.0, None))
    corrected_test_log = np.log1p(np.clip(anchor_test, 0.0, None))
    train_delta = np.clip(
        np.log1p(np.clip(source_train, 0.0, None)) - corrected_train_log,
        -clip_value,
        clip_value,
    )
    test_delta = np.clip(
        np.log1p(np.clip(source_test, 0.0, None)) - corrected_test_log,
        -clip_value,
        clip_value,
    )
    corrected_train_log[train_mask] = corrected_train_log[train_mask] + alpha * train_delta[train_mask]
    corrected_test_log[test_mask] = corrected_test_log[test_mask] + alpha * test_delta[test_mask]
    return np.clip(np.expm1(corrected_train_log), 0.0, None), np.clip(np.expm1(corrected_test_log), 0.0, None)


def apply_direct_mask_blend(
    base_train: np.ndarray,
    base_test: np.ndarray,
    source_left_train: np.ndarray,
    source_left_test: np.ndarray,
    source_right_train: np.ndarray,
    source_right_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    blend_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = np.clip(base_train.copy(), 0.0, None)
    test_predictions = np.clip(base_test.copy(), 0.0, None)
    if blend_space == "raw":
        train_predictions[train_mask] = (
            (1.0 - alpha) * np.clip(source_left_train[train_mask], 0.0, None)
            + alpha * np.clip(source_right_train[train_mask], 0.0, None)
        )
        test_predictions[test_mask] = (
            (1.0 - alpha) * np.clip(source_left_test[test_mask], 0.0, None)
            + alpha * np.clip(source_right_test[test_mask], 0.0, None)
        )
    elif blend_space == "log":
        train_predictions[train_mask] = np.expm1(
            (1.0 - alpha) * np.log1p(np.clip(source_left_train[train_mask], 0.0, None))
            + alpha * np.log1p(np.clip(source_right_train[train_mask], 0.0, None))
        )
        test_predictions[test_mask] = np.expm1(
            (1.0 - alpha) * np.log1p(np.clip(source_left_test[test_mask], 0.0, None))
            + alpha * np.log1p(np.clip(source_right_test[test_mask], 0.0, None))
        )
    else:
        raise ValueError(f"Unsupported blend space: {blend_space}")
    return train_predictions, test_predictions


def candidate_metrics(
    name: str,
    family: str,
    source_name: str,
    mask_name: str,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    y_train: np.ndarray,
    anchor_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> CandidateResult:
    mean_abs_log_shift = float(
        np.mean(np.abs(np.log1p(np.clip(test_predictions, 0.0, None)) - np.log1p(np.clip(anchor_test, 0.0, None))))
    )
    return CandidateResult(
        name=name,
        family=family,
        source_name=source_name,
        mask_name=mask_name,
        oof_predictions=np.clip(train_predictions, 0.0, None),
        test_predictions=np.clip(test_predictions, 0.0, None),
        oof_rmsle=rmsle(y_train, train_predictions),
        mean_abs_log_shift=mean_abs_log_shift,
        affected_train_share=float(train_mask.mean()),
        affected_test_share=float(test_mask.mean()),
    )


def build_feedback_anchor(
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    temporal_xgb_train: np.ndarray,
    temporal_xgb_test: np.ndarray,
    public_gap_top08_train: np.ndarray,
    public_gap_top08_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_log_delta(
        base_anchor_train,
        base_anchor_test,
        temporal_xgb_train,
        temporal_xgb_test,
        public_gap_top08_train,
        public_gap_top08_test,
        alpha=0.26,
        clip_value=0.20,
    )


def build_direct_schedule_candidates(
    seeds: dict[str, object],
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    y_train: np.ndarray,
    anchor_test: np.ndarray,
    disagreement_masks_train: dict[str, np.ndarray],
    disagreement_masks_test: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    source_specs = {
        "temporal_xgb": (
            seeds["temporal_xgb"].oof_predictions,
            seeds["temporal_xgb"].test_predictions,
            (0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54),
        ),
        "temporal_best": (
            seeds["temporal_best"].oof_predictions,
            seeds["temporal_best"].test_predictions,
            (0.26, 0.30, 0.34, 0.38, 0.42, 0.46),
        ),
    }
    mask_names = ["public_gap_top12", "public_gap_top10", "public_gap_top08", "public_gap_top06", "public_gap_top04"]
    for source_name, (source_train, source_test, alpha_grid) in source_specs.items():
        for mask_name in mask_names:
            train_mask = disagreement_masks_train[mask_name]
            test_mask = disagreement_masks_test[mask_name]
            for blend_space in ("raw", "log"):
                for alpha in alpha_grid:
                    train_predictions, test_predictions = apply_direct_mask_blend(
                        base_anchor_train,
                        base_anchor_test,
                        v5_train,
                        v5_test,
                        source_train,
                        source_test,
                        train_mask,
                        test_mask,
                        alpha,
                        blend_space,
                    )
                    candidates.append(
                        candidate_metrics(
                            name=f"direct_{blend_space}_{source_name}__{mask_name}__a{weight_token(alpha)}",
                            family=f"direct_{blend_space}_schedule",
                            source_name=source_name,
                            mask_name=mask_name,
                            train_predictions=train_predictions,
                            test_predictions=test_predictions,
                            y_train=y_train,
                            anchor_test=anchor_test,
                            train_mask=train_mask,
                            test_mask=test_mask,
                        )
                    )
    return candidates


def build_piecewise_candidates(
    seeds: dict[str, object],
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    y_train: np.ndarray,
    anchor_test: np.ndarray,
    disagreement_masks_train: dict[str, np.ndarray],
    disagreement_masks_test: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    source_specs = {
        "temporal_xgb": (seeds["temporal_xgb"].oof_predictions, seeds["temporal_xgb"].test_predictions),
        "temporal_best": (seeds["temporal_best"].oof_predictions, seeds["temporal_best"].test_predictions),
    }
    schedule_specs = [
        ("top04_next04", "public_gap_top04", "public_gap_band_04_08", (0.46, 0.50, 0.54, 0.58), (0.30, 0.34, 0.38, 0.42)),
        ("top06_next06", "public_gap_top06", "public_gap_band_06_12", (0.42, 0.46, 0.50), (0.26, 0.30, 0.34)),
    ]
    for source_name, (source_train, source_test) in source_specs.items():
        for schedule_name, high_mask_name, low_mask_name, high_grid, low_grid in schedule_specs:
            high_train_mask = disagreement_masks_train[high_mask_name]
            high_test_mask = disagreement_masks_test[high_mask_name]
            low_train_mask = disagreement_masks_train[low_mask_name]
            low_test_mask = disagreement_masks_test[low_mask_name]
            for blend_space in ("raw", "log"):
                for high_alpha in high_grid:
                    for low_alpha in low_grid:
                        train_predictions = np.clip(base_anchor_train.copy(), 0.0, None)
                        test_predictions = np.clip(base_anchor_test.copy(), 0.0, None)
                        train_predictions, test_predictions = apply_direct_mask_blend(
                            train_predictions,
                            test_predictions,
                            v5_train,
                            v5_test,
                            source_train,
                            source_test,
                            low_train_mask,
                            low_test_mask,
                            low_alpha,
                            blend_space,
                        )
                        train_predictions, test_predictions = apply_direct_mask_blend(
                            train_predictions,
                            test_predictions,
                            v5_train,
                            v5_test,
                            source_train,
                            source_test,
                            high_train_mask,
                            high_test_mask,
                            high_alpha,
                            blend_space,
                        )
                        full_train_mask = high_train_mask | low_train_mask
                        full_test_mask = high_test_mask | low_test_mask
                        candidates.append(
                            candidate_metrics(
                                name=(
                                    f"piecewise_{blend_space}_{source_name}__{schedule_name}"
                                    f"__hi{weight_token(high_alpha)}__lo{weight_token(low_alpha)}"
                                ),
                                family=f"piecewise_{blend_space}_schedule",
                                source_name=source_name,
                                mask_name=schedule_name,
                                train_predictions=train_predictions,
                                test_predictions=test_predictions,
                                y_train=y_train,
                                anchor_test=anchor_test,
                                train_mask=full_train_mask,
                                test_mask=full_test_mask,
                            )
                        )
    return candidates


def build_feedback_overlay_candidates(
    feedback_anchor_train: np.ndarray,
    feedback_anchor_test: np.ndarray,
    y_train: np.ndarray,
    seeds: dict[str, object],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    disagreement_masks_train: dict[str, np.ndarray],
    disagreement_masks_test: dict[str, np.ndarray],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    overlay_sources = {
        "temporal_xgb": (seeds["temporal_xgb"].oof_predictions, seeds["temporal_xgb"].test_predictions, (0.06, 0.10, 0.14, 0.18, 0.22), (0.08, 0.12, 0.16, 0.20, 0.24, 0.28)),
        "temporal_best": (seeds["temporal_best"].oof_predictions, seeds["temporal_best"].test_predictions, (0.06, 0.10, 0.14, 0.18), (0.08, 0.12, 0.16, 0.20)),
        "panel_catboost_v1": (monthly_satellites["panel_catboost_v1"][0], monthly_satellites["panel_catboost_v1"][1], (0.06, 0.10, 0.14, 0.18), (0.08, 0.12, 0.16, 0.20)),
        "panel_blend_top2": (monthly_satellites["panel_blend_top2"][0], monthly_satellites["panel_blend_top2"][1], (0.06, 0.10, 0.14), (0.08, 0.12, 0.16)),
    }

    combined_train_masks = {
        **train_masks,
        **disagreement_masks_train,
        "public_gap_top08_recent_le_10": disagreement_masks_train["public_gap_top08"] & train_masks["recent_le_10"],
        "public_gap_top08_sparse": disagreement_masks_train["public_gap_top08"] & train_masks["sparse_ge_0p5"],
        "public_gap_top08_inactive_or_sparse": disagreement_masks_train["public_gap_top08"] & train_masks["inactive_or_sparse"],
    }
    combined_test_masks = {
        **test_masks,
        **disagreement_masks_test,
        "public_gap_top08_recent_le_10": disagreement_masks_test["public_gap_top08"] & test_masks["recent_le_10"],
        "public_gap_top08_sparse": disagreement_masks_test["public_gap_top08"] & test_masks["sparse_ge_0p5"],
        "public_gap_top08_inactive_or_sparse": disagreement_masks_test["public_gap_top08"] & test_masks["inactive_or_sparse"],
    }

    overlay_masks = [
        "public_gap_top08",
        "public_gap_top10",
        "public_gap_top06",
        "public_gap_top08_recent_le_10",
        "public_gap_top08_sparse",
        "public_gap_top08_inactive_or_sparse",
        "recent_le_10",
        "inactive_or_sparse",
    ]
    for source_name, (source_train, source_test, alpha_grid, clip_grid) in overlay_sources.items():
        for mask_name in overlay_masks:
            train_mask = combined_train_masks[mask_name]
            test_mask = combined_test_masks[mask_name]
            for alpha in alpha_grid:
                for clip_value in clip_grid:
                    train_predictions, test_predictions = apply_log_delta(
                        feedback_anchor_train,
                        feedback_anchor_test,
                        source_train,
                        source_test,
                        train_mask,
                        test_mask,
                        alpha,
                        clip_value,
                    )
                    candidates.append(
                        candidate_metrics(
                            name=f"feedback_overlay_{source_name}__{mask_name}__a{weight_token(alpha)}__c{weight_token(clip_value)}",
                            family="feedback_overlay",
                            source_name=source_name,
                            mask_name=mask_name,
                            train_predictions=train_predictions,
                            test_predictions=test_predictions,
                            y_train=y_train,
                            anchor_test=feedback_anchor_test,
                            train_mask=train_mask,
                            test_mask=test_mask,
                        )
                    )
    return candidates


def build_global_feedback_microblends(
    feedback_anchor_train: np.ndarray,
    feedback_anchor_test: np.ndarray,
    y_train: np.ndarray,
    seeds: dict[str, object],
    monthly_satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    sources = {
        "temporal_xgb": (seeds["temporal_xgb"].oof_predictions, seeds["temporal_xgb"].test_predictions),
        "temporal_best": (seeds["temporal_best"].oof_predictions, seeds["temporal_best"].test_predictions),
        "panel_catboost_v1": monthly_satellites["panel_catboost_v1"],
        "panel_blend_top2": monthly_satellites["panel_blend_top2"],
    }
    full_train_mask = np.ones_like(feedback_anchor_train, dtype=bool)
    full_test_mask = np.ones_like(feedback_anchor_test, dtype=bool)
    for source_name, (source_train, source_test) in sources.items():
        for alpha in (0.02, 0.04, 0.06, 0.08):
            train_predictions, test_predictions = apply_log_delta(
                feedback_anchor_train,
                feedback_anchor_test,
                source_train,
                source_test,
                full_train_mask,
                full_test_mask,
                alpha,
                clip_value=0.16,
            )
            candidates.append(
                candidate_metrics(
                    name=f"feedback_global_{source_name}__a{weight_token(alpha)}",
                    family="feedback_global",
                    source_name=source_name,
                    mask_name="all_rows",
                    train_predictions=train_predictions,
                    test_predictions=test_predictions,
                    y_train=y_train,
                    anchor_test=feedback_anchor_test,
                    train_mask=full_train_mask,
                    test_mask=full_test_mask,
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


def write_outputs(
    run_dir: Path,
    sample_submission_path: Path,
    test_ids: np.ndarray,
    candidates: list[CandidateResult],
    metrics: pd.DataFrame,
    submission_format: str,
    args: argparse.Namespace,
) -> None:
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

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
            submission_format=submission_format,
        )
        submission.write_csv(submissions_dir / f"{candidate_name}.csv")

    best_by_selection = metrics.iloc[0]
    best_by_oof = metrics.sort_values(["oof_rmsle", "mean_abs_log_shift"]).iloc[0]
    summary_lines = [
        "# Public Feedback Temporal Lab",
        "",
        f"Run directory: `{run_dir}`",
        f"Feature cache: `{args.feature_cache_path}`",
        f"Monthly run dir: `{args.monthly_run_dir}`",
        f"Submission format: `{submission_format}`",
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
        train_masks,
        test_masks,
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

    candidates = []
    candidates.extend(
        build_direct_schedule_candidates(
            seeds,
            base_anchor_train,
            base_anchor_test,
            y_train,
            feedback_anchor_test,
            disagreement_masks_train,
            disagreement_masks_test,
        )
    )
    candidates.extend(
        build_piecewise_candidates(
            seeds,
            base_anchor_train,
            base_anchor_test,
            y_train,
            feedback_anchor_test,
            disagreement_masks_train,
            disagreement_masks_test,
        )
    )
    candidates.extend(
        build_feedback_overlay_candidates(
            feedback_anchor_train,
            feedback_anchor_test,
            y_train,
            seeds,
            monthly_satellites,
            train_masks,
            test_masks,
            disagreement_masks_train,
            disagreement_masks_test,
        )
    )
    candidates.extend(
        build_global_feedback_microblends(
            feedback_anchor_train,
            feedback_anchor_test,
            y_train,
            seeds,
            monthly_satellites,
        )
    )

    metrics = build_metrics_frame(candidates, shift_penalty=args.shift_penalty)
    write_outputs(
        run_dir=run_dir,
        sample_submission_path=data_dir / "SampleSubmission.csv",
        test_ids=test_ids,
        candidates=candidates,
        metrics=metrics,
        submission_format=args.submission_format,
        args=args,
    )


if __name__ == "__main__":
    main()
