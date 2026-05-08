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

from artifact_blend_lab import align_predictions_by_id, load_seed_candidates
from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory


FEATURE_PIPELINE_VERSION = "v10_monthly_panel_anchor_lab"


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    affected_train_share: float
    affected_test_share: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend the current public-best disagreement anchor with improved monthly-panel model "
            "artifacts and write leaderboard-focused candidate submissions."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/monthly_panel_anchor_lab"))
    parser.add_argument("--run-name", type=str, default="monthly_panel_anchor_lab")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v4_sparse_activity_batch_stack.parquet"),
        help="Mask source cache used by the public-best anchor family.",
    )
    parser.add_argument(
        "--monthly-run-dir",
        type=Path,
        default=Path("outputs/experiments/v5_monthly_panel_full1_20260430_091349_503223_78ac19f8"),
        help="Experiment harness run directory containing monthly-panel OOF/test prediction parquets.",
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
        default=0.30,
        help="Penalty on prediction-surface drift when ranking hedge candidates.",
    )
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=20,
        help="How many top-ranked candidates to materialize as submission files.",
    )
    return parser.parse_args()


def weight_token(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def load_masks(
    feature_cache_path: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_table = pl.read_parquet(feature_cache_path)
    train_frame = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", train_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "train").select(
                [
                    "UniqueID",
                    "fin_missing_flag",
                    "txn_sparse_month_share_le_2",
                    "txn_active_months_total",
                    "txn_inactive_months_total",
                    "txn_recent_vs_prev3_count_ratio",
                ]
            ),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
    )
    test_frame = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", test_ids)})
        .join(
            feature_table.filter(pl.col("__split__") == "test").select(
                [
                    "UniqueID",
                    "fin_missing_flag",
                    "txn_sparse_month_share_le_2",
                    "txn_active_months_total",
                    "txn_inactive_months_total",
                    "txn_recent_vs_prev3_count_ratio",
                ]
            ),
            on="UniqueID",
            how="left",
        )
        .sort("UniqueID")
    )

    def frame_masks(frame: pl.DataFrame) -> dict[str, np.ndarray]:
        sparse = frame.get_column("txn_sparse_month_share_le_2").fill_null(-999.0).cast(pl.Float64).to_numpy() >= 0.5
        active = frame.get_column("txn_active_months_total").fill_null(-999.0).cast(pl.Float64).to_numpy() <= 12.0
        inactive = frame.get_column("txn_inactive_months_total").fill_null(-999.0).cast(pl.Float64).to_numpy() >= 18.0
        recent_growth = (
            frame.get_column("txn_recent_vs_prev3_count_ratio").fill_null(-999.0).cast(pl.Float64).to_numpy() >= 1.2
        )
        fin_missing = frame.get_column("fin_missing_flag").fill_null(0).cast(pl.Int8).to_numpy() >= 1
        return {
            "sparse_ge_0p5": sparse,
            "active_le_12": active,
            "inactive_ge_18": inactive,
            "recent_ratio_ge_1p2": recent_growth,
            "inactive_or_sparse": inactive | sparse,
            "inactive_and_sparse": inactive & sparse,
            "active_and_sparse": active & sparse,
            "fin_missing": fin_missing,
        }

    return frame_masks(train_frame), frame_masks(test_frame)


def build_current_public_best_anchor(
    seeds: dict[str, object],
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    v5_train = seeds["v5_best_subset"].oof_predictions
    v5_test = seeds["v5_best_subset"].test_predictions
    temporal_xgb_train = seeds["temporal_xgb"].oof_predictions
    temporal_xgb_test = seeds["temporal_xgb"].test_predictions

    base_anchor_train = 0.92 * v5_train + 0.08 * temporal_xgb_train
    base_anchor_test = 0.92 * v5_test + 0.08 * temporal_xgb_test

    fin_missing_train = train_masks["fin_missing"]
    fin_missing_test = test_masks["fin_missing"]
    disagreement_train = np.abs(np.log1p(temporal_xgb_train) - np.log1p(v5_train))
    disagreement_test = np.abs(np.log1p(temporal_xgb_test) - np.log1p(v5_test))
    threshold = float(np.quantile(disagreement_train[~fin_missing_train], 0.92))
    high_gap_train = (~fin_missing_train) & (disagreement_train >= threshold)
    high_gap_test = (~fin_missing_test) & (disagreement_test >= threshold)

    anchor_train = np.clip(base_anchor_train.copy(), 0.0, None)
    anchor_test = np.clip(base_anchor_test.copy(), 0.0, None)
    anchor_train[high_gap_train] = 0.77 * v5_train[high_gap_train] + 0.23 * temporal_xgb_train[high_gap_train]
    anchor_test[high_gap_test] = 0.77 * v5_test[high_gap_test] + 0.23 * temporal_xgb_test[high_gap_test]
    anchor_train[fin_missing_train] = v5_train[fin_missing_train]
    anchor_test[fin_missing_test] = v5_test[fin_missing_test]
    return np.clip(anchor_train, 0.0, None), np.clip(anchor_test, 0.0, None)


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


def build_global_microblend_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    for satellite_name in ("panel_blend_top2", "panel_catboost_v1", "panel_xgb_v3"):
        satellite_train, satellite_test = satellites[satellite_name]
        for alpha in (0.02, 0.04, 0.06, 0.08, 0.10):
            train_predictions = np.expm1(
                (1.0 - alpha) * np.log1p(np.clip(anchor_train, 0.0, None))
                + alpha * np.log1p(np.clip(satellite_train, 0.0, None))
            )
            test_predictions = np.expm1(
                (1.0 - alpha) * np.log1p(np.clip(anchor_test, 0.0, None))
                + alpha * np.log1p(np.clip(satellite_test, 0.0, None))
            )
            candidates.append(
                CandidateResult(
                    name=f"{satellite_name}__global_log_a{weight_token(alpha)}",
                    family="global_microblend",
                    oof_predictions=np.clip(train_predictions, 0.0, None),
                    test_predictions=np.clip(test_predictions, 0.0, None),
                    affected_train_share=1.0,
                    affected_test_share=1.0,
                )
            )
    return candidates


def build_segment_overlay_candidates(
    anchor_train: np.ndarray,
    anchor_test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    satellites: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    candidate_specs = [
        ("panel_catboost_v1", "inactive_ge_18", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_catboost_v1", "active_le_12", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_catboost_v1", "sparse_ge_0p5", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_catboost_v1", "inactive_or_sparse", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_catboost_v1", "inactive_and_sparse", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_catboost_v1", "active_and_sparse", (0.08, 0.10, 0.12, 0.15, 0.18)),
        ("panel_blend_top2", "inactive_ge_18", (0.08, 0.10, 0.12, 0.15)),
        ("panel_blend_top2", "active_le_12", (0.08, 0.10, 0.12, 0.15)),
        ("panel_blend_top2", "sparse_ge_0p5", (0.08, 0.10, 0.12, 0.15)),
        ("panel_blend_top2", "inactive_or_sparse", (0.08, 0.10, 0.12, 0.15)),
    ]

    for satellite_name, mask_name, alpha_grid in candidate_specs:
        satellite_train, satellite_test = satellites[satellite_name]
        train_mask = train_masks[mask_name]
        test_mask = test_masks[mask_name]
        for alpha in alpha_grid:
            train_predictions = np.clip(anchor_train.copy(), 0.0, None)
            test_predictions = np.clip(anchor_test.copy(), 0.0, None)
            train_predictions[train_mask] = np.expm1(
                (1.0 - alpha) * np.log1p(train_predictions[train_mask])
                + alpha * np.log1p(np.clip(satellite_train[train_mask], 0.0, None))
            )
            test_predictions[test_mask] = np.expm1(
                (1.0 - alpha) * np.log1p(test_predictions[test_mask])
                + alpha * np.log1p(np.clip(satellite_test[test_mask], 0.0, None))
            )
            candidates.append(
                CandidateResult(
                    name=f"{satellite_name}__{mask_name}__a{weight_token(alpha)}",
                    family="segment_overlay",
                    oof_predictions=np.clip(train_predictions, 0.0, None),
                    test_predictions=np.clip(test_predictions, 0.0, None),
                    affected_train_share=float(train_mask.mean()),
                    affected_test_share=float(test_mask.mean()),
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
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["conservative_score"] = metrics["oof_rmsle"] + shift_penalty * metrics["mean_abs_log_shift_vs_anchor"]
    return metrics.sort_values(["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]).reset_index(drop=True)


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

    best_oof = metrics.iloc[0]
    best_hedge = metrics.sort_values(["conservative_score", "oof_rmsle", "candidate_name"]).iloc[0]
    best_segment = metrics.loc[metrics["family"] == "segment_overlay"].sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).iloc[0]
    best_global = metrics.loc[metrics["family"] == "global_microblend"].sort_values(
        ["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]
    ).iloc[0]

    materialized = sorted(
        set(
            metrics.head(write_top_k_submissions)["candidate_name"].tolist()
            + [
                best_oof["candidate_name"],
                best_hedge["candidate_name"],
                best_segment["candidate_name"],
                best_global["candidate_name"],
            ]
        )
    )
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for candidate_name in materialized:
        _, test_predictions = prediction_lookup[candidate_name]
        build_submission_frame(
            sample_submission_path=sample_submission_path,
            unique_ids=test_ids,
            predictions=test_predictions,
            submission_format=submission_format,
        ).write_csv(submissions_dir / f"{candidate_name}.csv")

    recommended_dir = run_dir / "recommended_selection"
    recommended_dir.mkdir(parents=True, exist_ok=True)
    selections = [
        ("01_best_oof.csv", best_oof["candidate_name"]),
        ("02_best_hedge.csv", best_hedge["candidate_name"]),
        ("03_best_segment.csv", best_segment["candidate_name"]),
        ("04_best_global.csv", best_global["candidate_name"]),
    ]
    for filename, candidate_name in selections:
        (recommended_dir / filename).write_bytes((submissions_dir / f"{candidate_name}.csv").read_bytes())

    readme_lines = [
        "# Monthly Panel Anchor Lab",
        "",
        "Current anchor:",
        "",
        "- `publicbest_top08_alpha_0p23__fin_missing_to_v5`",
        "",
        "Recommended files:",
        "",
        f"1. `01_best_oof.csv` -> `{best_oof['candidate_name']}`",
        f"   - OOF RMSLE: `{best_oof['oof_rmsle']:.6f}`",
        f"   - shift vs anchor: `{best_oof['mean_abs_log_shift_vs_anchor']:.6f}`",
        "",
        f"2. `02_best_hedge.csv` -> `{best_hedge['candidate_name']}`",
        f"   - OOF RMSLE: `{best_hedge['oof_rmsle']:.6f}`",
        f"   - conservative score: `{best_hedge['conservative_score']:.6f}`",
        "",
        f"3. `03_best_segment.csv` -> `{best_segment['candidate_name']}`",
        f"   - OOF RMSLE: `{best_segment['oof_rmsle']:.6f}`",
        f"   - affected test share: `{best_segment['affected_test_share']:.6f}`",
        "",
        f"4. `04_best_global.csv` -> `{best_global['candidate_name']}`",
        f"   - OOF RMSLE: `{best_global['oof_rmsle']:.6f}`",
        f"   - shift vs anchor: `{best_global['mean_abs_log_shift_vs_anchor']:.6f}`",
        "",
        "All files are copied from `submissions/` and are upload-ready for Zindi."
        if submission_format == "zindi_log"
        else "This run used raw output format. Convert these files to `np.log1p(prediction)` before upload.",
    ]
    (recommended_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Monthly Panel Anchor Lab Summary",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Submission format: `{submission_format}`",
        f"- Best OOF candidate: `{best_oof['candidate_name']}` at `{best_oof['oof_rmsle']:.6f}`",
        f"- Best hedge candidate: `{best_hedge['candidate_name']}` at `{best_hedge['oof_rmsle']:.6f}`",
        f"- Best segment candidate: `{best_segment['candidate_name']}` at `{best_segment['oof_rmsle']:.6f}`",
        f"- Best global candidate: `{best_global['candidate_name']}` at `{best_global['oof_rmsle']:.6f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in metrics.head(15).to_dict(orient="records"):
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
    train_masks, test_masks = load_masks(args.feature_cache_path.resolve(), train_ids, test_ids)
    anchor_train, anchor_test = build_current_public_best_anchor(seeds, train_masks, test_masks)
    satellites = load_monthly_panel_satellites(args.monthly_run_dir.resolve(), train_ids, test_ids)

    anchor_candidate = CandidateResult(
        name="publicbest_top08_alpha_0p23__fin_missing_to_v5",
        family="anchor",
        oof_predictions=anchor_train,
        test_predictions=anchor_test,
        affected_train_share=0.0,
        affected_test_share=0.0,
    )
    candidates = (
        [anchor_candidate]
        + build_global_microblend_candidates(anchor_train, anchor_test, satellites)
        + build_segment_overlay_candidates(anchor_train, anchor_test, train_masks, test_masks, satellites)
    )

    metrics = build_metrics(anchor_train, y_train, candidates, shift_penalty=args.shift_penalty)
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
        "monthly_run_dir": str(args.monthly_run_dir.resolve()),
        "feature_cache_path": str(args.feature_cache_path.resolve()),
        "shift_penalty": args.shift_penalty,
        "submission_format": args.submission_format,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
