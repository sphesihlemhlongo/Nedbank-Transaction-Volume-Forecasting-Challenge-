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

from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, RANDOM_STATE, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from rolling_panel_public_feedback_lab import align_feature_splits
from rolling_panel_public_stability_lab import build_pseudo_public_subsets, evaluate_public_stability
from submission_eval_utils import write_submission_evaluation_artifacts


FEATURE_PIPELINE_VERSION = "v21_rolling_panel_public_calibration_feedback_lab"


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_rmsle: float
    mean_abs_log_shift: float
    pseudo_public_win_rate: float
    pseudo_public_mean_delta: float
    pseudo_public_median_delta: float
    pseudo_public_p75_delta: float
    pseudo_public_p90_delta: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search microblends around the current public-best isotonic calibration candidate."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--calibration-run-dir",
        type=Path,
        default=Path(
            "outputs/rolling_panel_public_calibration_lab/rolling_panel_public_calibration_lab_refined_20260503_071742_969261_1349fb40"
        ),
    )
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v5_dense_monthly_panel_stack.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_panel_public_calibration_feedback_lab"))
    parser.add_argument("--run-name", type=str, default="rolling_panel_public_calibration_feedback_lab")
    parser.add_argument("--public-fraction", type=float, default=0.30)
    parser.add_argument("--pseudo-public-splits", type=int, default=240)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--shift-penalty", type=float, default=0.04)
    parser.add_argument("--win-rate-bonus", type=float, default=0.0015)
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def log_blend(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left_log = np.log1p(np.clip(left, 0.0, None))
    right_log = np.log1p(np.clip(right, 0.0, None))
    return np.clip(np.expm1((1.0 - alpha) * left_log + alpha * right_log), 0.0, None)


def build_candidate_metrics(candidates: list[CandidateResult]) -> pd.DataFrame:
    rows = [
        {
            "candidate_name": candidate.name,
            "family": candidate.family,
            "oof_rmsle": candidate.oof_rmsle,
            "mean_abs_log_shift": candidate.mean_abs_log_shift,
            "pseudo_public_win_rate": candidate.pseudo_public_win_rate,
            "pseudo_public_mean_delta": candidate.pseudo_public_mean_delta,
            "pseudo_public_median_delta": candidate.pseudo_public_median_delta,
            "pseudo_public_p75_delta": candidate.pseudo_public_p75_delta,
            "pseudo_public_p90_delta": candidate.pseudo_public_p90_delta,
            "selection_score": candidate.selection_score,
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "pseudo_public_median_delta", "oof_rmsle", "candidate_name"]
    ).reset_index(drop=True)


def write_recommendation_pack(run_dir: Path, submissions_dir: Path, candidate_metrics: pd.DataFrame) -> None:
    selection_dir = run_dir / "recommended_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    top_rows = candidate_metrics.head(4)

    for index, row in enumerate(top_rows.to_dict(orient="records"), start=1):
        source = submissions_dir / f"{row['candidate_name']}.csv"
        target = selection_dir / f"{index:02d}_{row['candidate_name']}.csv"
        target.write_bytes(source.read_bytes())

    lines = [
        "# Recommended Selection",
        "",
        "Candidates are ordered by stability-aware microblends around the current public-best isotonic anchor.",
        "",
    ]
    for index, row in enumerate(top_rows.to_dict(orient="records"), start=1):
        lines.extend(
            [
                f"{index}. `{index:02d}_{row['candidate_name']}.csv`",
                f"   - OOF RMSLE: `{row['oof_rmsle']:.6f}`",
                f"   - pseudo-public win rate: `{row['pseudo_public_win_rate']:.4f}`",
                f"   - pseudo-public median delta vs anchor: `{row['pseudo_public_median_delta']:.6f}`",
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


def add_candidate(
    candidates: list[CandidateResult],
    y_train: np.ndarray,
    current_anchor_train: np.ndarray,
    pseudo_public_subsets: list[np.ndarray],
    shift_penalty: float,
    win_rate_bonus: float,
    name: str,
    family: str,
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
) -> None:
    stability = evaluate_public_stability(y_train, current_anchor_train, oof_predictions, pseudo_public_subsets)
    mean_abs_log_shift = float(
        np.mean(
            np.abs(
                np.log1p(np.clip(oof_predictions, 0.0, None))
                - np.log1p(np.clip(current_anchor_train, 0.0, None))
            )
        )
    )
    oof_value = float(rmsle(y_train, oof_predictions))
    selection_score = float(
        oof_value
        + shift_penalty * mean_abs_log_shift
        + max(stability["pseudo_public_p75_delta"], 0.0)
        - win_rate_bonus * stability["pseudo_public_win_rate"]
    )
    candidates.append(
        CandidateResult(
            name=name,
            family=family,
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            oof_rmsle=oof_value,
            mean_abs_log_shift=mean_abs_log_shift,
            pseudo_public_win_rate=stability["pseudo_public_win_rate"],
            pseudo_public_mean_delta=stability["pseudo_public_mean_delta"],
            pseudo_public_median_delta=stability["pseudo_public_median_delta"],
            pseudo_public_p75_delta=stability["pseudo_public_p75_delta"],
            pseudo_public_p90_delta=stability["pseudo_public_p90_delta"],
            selection_score=selection_score,
        )
    )


def main() -> None:
    args = parse_args()
    run_dir = build_run_directory(args.output_dir, args.run_name)
    calibration_run_dir = args.calibration_run_dir.resolve()

    oof_frame = pl.read_parquet(calibration_run_dir / "oof_predictions.parquet")
    test_frame = pl.read_parquet(calibration_run_dir / "test_predictions.parquet")
    train_ids = oof_frame["UniqueID"].to_numpy()
    test_ids = test_frame["UniqueID"].to_numpy()
    y_train = oof_frame["next_3m_txn_count_true"].to_numpy().astype(np.float64)

    current_anchor_name = "isotonic_anchor_b0p08"
    current_anchor_train = oof_frame[f"pred_{current_anchor_name}"].to_numpy().astype(np.float64)
    current_anchor_test = test_frame[f"pred_{current_anchor_name}"].to_numpy().astype(np.float64)

    train_features, _ = align_feature_splits(args.feature_cache_path, train_ids, test_ids)
    pseudo_public_subsets = build_pseudo_public_subsets(
        y_train=y_train,
        train_features=train_features,
        public_fraction=args.public_fraction,
        n_splits=args.pseudo_public_splits,
        random_state=args.random_state,
    )

    candidate_sources = [
        "isotonic_anchor_b0p05",
        "isotonic_anchor_b0p1",
        "isotonic_anchor_b0p12",
        "isotonic_anchor_b0p15",
        "direct_global_a0p255",
        "direct_global_a0p258",
        "direct_global_a0p26",
        "direct_global_a0p263",
        "direct_global_a0p265",
        "direct_global_a0p268",
        "direct_global_a0p27",
    ]
    available_sources = [
        source for source in candidate_sources if f"pred_{source}" in oof_frame.columns and f"pred_{source}" in test_frame.columns
    ]

    candidates: list[CandidateResult] = []
    for source_name in available_sources:
        source_train = oof_frame[f"pred_{source_name}"].to_numpy().astype(np.float64)
        source_test = test_frame[f"pred_{source_name}"].to_numpy().astype(np.float64)
        for alpha in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
            add_candidate(
                candidates=candidates,
                y_train=y_train,
                current_anchor_train=current_anchor_train,
                pseudo_public_subsets=pseudo_public_subsets,
                shift_penalty=args.shift_penalty,
                win_rate_bonus=args.win_rate_bonus,
                name=f"{current_anchor_name}__to__{source_name}__a{str(alpha).replace('.', 'p')}",
                family="microblend",
                oof_predictions=log_blend(current_anchor_train, source_train, alpha),
                test_predictions=log_blend(current_anchor_test, source_test, alpha),
            )

    candidate_metrics = build_candidate_metrics(candidates)

    submissions_dir = run_dir / "submissions"
    submissions_raw_dir = run_dir / "submissions_raw"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submissions_raw_dir.mkdir(parents=True, exist_ok=True)
    sample_submission_path = args.data_dir / "SampleSubmission.csv"

    oof_payload: dict[str, object] = {"UniqueID": train_ids, "next_3m_txn_count_true": y_train}
    test_payload: dict[str, object] = {"UniqueID": test_ids}

    for candidate in candidates:
        oof_payload[f"pred_{candidate.name}"] = candidate.oof_predictions
        test_payload[f"pred_{candidate.name}"] = candidate.test_predictions
        build_submission_frame(
            sample_submission_path,
            test_ids,
            candidate.test_predictions,
            submission_format=args.submission_format,
        ).write_csv(submissions_dir / f"{candidate.name}.csv")
        build_submission_frame(
            sample_submission_path,
            test_ids,
            candidate.test_predictions,
            submission_format="raw",
        ).write_csv(submissions_raw_dir / f"{candidate.name}.csv")

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
    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

    write_recommendation_pack(run_dir, submissions_dir, candidate_metrics)

    best_selection = candidate_metrics.iloc[0]
    best_raw = candidate_metrics.sort_values(["oof_rmsle", "selection_score"]).iloc[0]
    summary_lines = [
        "# Rolling Panel Public Calibration Feedback Lab",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Calibration source run: `{calibration_run_dir}`",
        f"- Current public-best anchor: `{current_anchor_name}`",
        f"- Current public-best anchor OOF RMSLE: `{rmsle(y_train, current_anchor_train):.6f}`",
        (
            f"- Best candidate by stability selection: `{best_selection['candidate_name']}` "
            f"with OOF `{best_selection['oof_rmsle']:.6f}` and pseudo-public win rate "
            f"`{best_selection['pseudo_public_win_rate']:.4f}`"
        ),
        f"- Best candidate by raw OOF: `{best_raw['candidate_name']}` with OOF `{best_raw['oof_rmsle']:.6f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidate_metrics.head(12).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}` ({row['family']}): OOF `{row['oof_rmsle']:.6f}`, "
            f"win rate `{row['pseudo_public_win_rate']:.4f}`, median delta `{row['pseudo_public_median_delta']:.6f}`, "
            f"shift `{row['mean_abs_log_shift']:.6f}`, selection `{row['selection_score']:.6f}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config_payload = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "calibration_run_dir": str(calibration_run_dir),
        "public_fraction": args.public_fraction,
        "pseudo_public_splits": args.pseudo_public_splits,
        "random_state": args.random_state,
        "shift_penalty": args.shift_penalty,
        "win_rate_bonus": args.win_rate_bonus,
        "submission_format": args.submission_format,
        "current_public_best_anchor": current_anchor_name,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
