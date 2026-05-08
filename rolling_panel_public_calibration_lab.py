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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import KFold

from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, RANDOM_STATE, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from rolling_panel_public_feedback_lab import align_feature_splits, log_blend
from rolling_panel_public_stability_lab import (
    build_pseudo_public_subsets,
    evaluate_public_stability,
    precise_weight_token,
)
from submission_eval_utils import write_submission_evaluation_artifacts


FEATURE_PIPELINE_VERSION = "v20_rolling_panel_public_calibration_lab"


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
        description=(
            "Search smooth global recalibration candidates around the current best rolling-panel public anchor."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--rolling-run-dir",
        type=Path,
        default=Path(
            "outputs/rolling_panel_supervision_lab/rolling_panel_supervision_xgb_strict_20260502_185351_716878_1fb481b5"
        ),
    )
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v5_dense_monthly_panel_stack.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_panel_public_calibration_lab"))
    parser.add_argument("--run-name", type=str, default="rolling_panel_public_calibration_lab")
    parser.add_argument("--public-fraction", type=float, default=0.30)
    parser.add_argument("--pseudo-public-splits", type=int, default=240)
    parser.add_argument("--n-calibration-folds", type=int, default=5)
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


def logs_to_raw(log_values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(log_values, dtype=np.float64)), 0.0, None)


def blend_anchor_with_log_calibration(
    anchor_raw: np.ndarray,
    calibrated_log: np.ndarray,
    beta: float,
) -> np.ndarray:
    anchor_log = np.log1p(np.clip(anchor_raw, 0.0, None))
    blended_log = (1.0 - beta) * anchor_log + beta * np.asarray(calibrated_log, dtype=np.float64)
    return logs_to_raw(blended_log)


def cross_fit_regression_model(
    train_x: np.ndarray,
    train_y_log: np.ndarray,
    test_x: np.ndarray,
    fit_predictor,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_log = np.zeros(len(train_y_log), dtype=np.float64)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(train_x), start=1):
        predictor = fit_predictor(train_x[fit_idx], train_y_log[fit_idx], fold_idx)
        oof_log[val_idx] = predictor(train_x[val_idx])
    full_predictor = fit_predictor(train_x, train_y_log, 0)
    test_log = full_predictor(test_x)
    return oof_log, np.asarray(test_log, dtype=np.float64)


def fit_linear_positive(train_x: np.ndarray, train_y_log: np.ndarray, _: int):
    model = LinearRegression(positive=True)
    model.fit(train_x, train_y_log)
    return model.predict


def fit_huber(train_x: np.ndarray, train_y_log: np.ndarray, _: int):
    model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=200)
    model.fit(train_x, train_y_log)
    return model.predict


def fit_isotonic(train_x: np.ndarray, train_y_log: np.ndarray, _: int):
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    model.fit(train_x.reshape(-1), train_y_log)
    return lambda values: model.predict(values.reshape(-1))


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
        (
            "Candidates are ordered by a stability-aware score built from full-train OOF, "
            "pseudo-public win rate, and drift from the current public-best rolling-panel anchor."
        ),
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
    name: str,
    family: str,
    y_train: np.ndarray,
    current_anchor_train: np.ndarray,
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
    pseudo_public_subsets: list[np.ndarray],
    shift_penalty: float,
    win_rate_bonus: float,
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
    rolling_run_dir = args.rolling_run_dir.resolve()

    oof_frame = pl.read_parquet(rolling_run_dir / "oof_predictions.parquet")
    test_frame = pl.read_parquet(rolling_run_dir / "test_predictions.parquet")
    train_ids = oof_frame["UniqueID"].to_numpy()
    test_ids = test_frame["UniqueID"].to_numpy()
    y_train = oof_frame["next_3m_txn_count_true"].to_numpy().astype(np.float64)
    y_train_log = np.log1p(np.clip(y_train, 0.0, None))

    base_anchor_train = oof_frame["pred_publicbest_feedback_temporal_top10"].to_numpy().astype(np.float64)
    base_anchor_test = test_frame["pred_publicbest_feedback_temporal_top10"].to_numpy().astype(np.float64)
    source_train = oof_frame["pred_xgb_conservative_v3"].to_numpy().astype(np.float64)
    source_test = test_frame["pred_xgb_conservative_v3"].to_numpy().astype(np.float64)
    current_anchor_train = oof_frame["pred_anchor_global_xgb_conservative_v3_a0p25"].to_numpy().astype(np.float64)
    current_anchor_test = test_frame["pred_anchor_global_xgb_conservative_v3_a0p25"].to_numpy().astype(np.float64)

    train_features, _ = align_feature_splits(args.feature_cache_path, train_ids, test_ids)
    pseudo_public_subsets = build_pseudo_public_subsets(
        y_train=y_train,
        train_features=train_features,
        public_fraction=args.public_fraction,
        n_splits=args.pseudo_public_splits,
        random_state=args.random_state,
    )

    anchor_train_log = np.log1p(np.clip(current_anchor_train, 0.0, None)).reshape(-1, 1)
    anchor_test_log = np.log1p(np.clip(current_anchor_test, 0.0, None)).reshape(-1, 1)
    source_train_log = np.log1p(np.clip(source_train, 0.0, None)).reshape(-1, 1)
    source_test_log = np.log1p(np.clip(source_test, 0.0, None)).reshape(-1, 1)
    dual_train_log = np.hstack([anchor_train_log, source_train_log])
    dual_test_log = np.hstack([anchor_test_log, source_test_log])

    candidates: list[CandidateResult] = []

    for alpha in (0.25, 0.2525, 0.255, 0.2575, 0.26, 0.2625, 0.265, 0.2675, 0.27):
        oof_predictions = log_blend(base_anchor_train, source_train, alpha)
        test_predictions = log_blend(base_anchor_test, source_test, alpha)
        add_candidate(
            candidates=candidates,
            name=f"direct_global_a{precise_weight_token(alpha)}",
            family="direct_global_alpha",
            y_train=y_train,
            current_anchor_train=current_anchor_train,
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            pseudo_public_subsets=pseudo_public_subsets,
            shift_penalty=args.shift_penalty,
            win_rate_bonus=args.win_rate_bonus,
        )

    linear_oof_log, linear_test_log = cross_fit_regression_model(
        train_x=anchor_train_log,
        train_y_log=y_train_log,
        test_x=anchor_test_log,
        fit_predictor=fit_linear_positive,
        n_splits=args.n_calibration_folds,
        random_state=args.random_state,
    )
    for beta in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        add_candidate(
            candidates=candidates,
            name=f"affine_anchor_b{precise_weight_token(beta)}",
            family="anchor_affine",
            y_train=y_train,
            current_anchor_train=current_anchor_train,
            oof_predictions=blend_anchor_with_log_calibration(current_anchor_train, linear_oof_log, beta),
            test_predictions=blend_anchor_with_log_calibration(current_anchor_test, linear_test_log, beta),
            pseudo_public_subsets=pseudo_public_subsets,
            shift_penalty=args.shift_penalty,
            win_rate_bonus=args.win_rate_bonus,
        )

    huber_oof_log, huber_test_log = cross_fit_regression_model(
        train_x=anchor_train_log,
        train_y_log=y_train_log,
        test_x=anchor_test_log,
        fit_predictor=fit_huber,
        n_splits=args.n_calibration_folds,
        random_state=args.random_state,
    )
    for beta in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        add_candidate(
            candidates=candidates,
            name=f"huber_anchor_b{precise_weight_token(beta)}",
            family="anchor_huber",
            y_train=y_train,
            current_anchor_train=current_anchor_train,
            oof_predictions=blend_anchor_with_log_calibration(current_anchor_train, huber_oof_log, beta),
            test_predictions=blend_anchor_with_log_calibration(current_anchor_test, huber_test_log, beta),
            pseudo_public_subsets=pseudo_public_subsets,
            shift_penalty=args.shift_penalty,
            win_rate_bonus=args.win_rate_bonus,
        )

    isotonic_oof_log, isotonic_test_log = cross_fit_regression_model(
        train_x=anchor_train_log,
        train_y_log=y_train_log,
        test_x=anchor_test_log,
        fit_predictor=fit_isotonic,
        n_splits=args.n_calibration_folds,
        random_state=args.random_state,
    )
    for beta in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40):
        add_candidate(
            candidates=candidates,
            name=f"isotonic_anchor_b{precise_weight_token(beta)}",
            family="anchor_isotonic",
            y_train=y_train,
            current_anchor_train=current_anchor_train,
            oof_predictions=blend_anchor_with_log_calibration(current_anchor_train, isotonic_oof_log, beta),
            test_predictions=blend_anchor_with_log_calibration(current_anchor_test, isotonic_test_log, beta),
            pseudo_public_subsets=pseudo_public_subsets,
            shift_penalty=args.shift_penalty,
            win_rate_bonus=args.win_rate_bonus,
        )

    dual_oof_log, dual_test_log = cross_fit_regression_model(
        train_x=dual_train_log,
        train_y_log=y_train_log,
        test_x=dual_test_log,
        fit_predictor=fit_linear_positive,
        n_splits=args.n_calibration_folds,
        random_state=args.random_state,
    )
    for beta in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        add_candidate(
            candidates=candidates,
            name=f"dual_positive_b{precise_weight_token(beta)}",
            family="dual_positive",
            y_train=y_train,
            current_anchor_train=current_anchor_train,
            oof_predictions=blend_anchor_with_log_calibration(current_anchor_train, dual_oof_log, beta),
            test_predictions=blend_anchor_with_log_calibration(current_anchor_test, dual_test_log, beta),
            pseudo_public_subsets=pseudo_public_subsets,
            shift_penalty=args.shift_penalty,
            win_rate_bonus=args.win_rate_bonus,
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
        "# Rolling Panel Public Calibration Lab",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Rolling source run: `{rolling_run_dir}`",
        f"- Current public-best anchor: `anchor_global_xgb_conservative_v3_a0p25`",
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
        "rolling_run_dir": str(rolling_run_dir),
        "public_fraction": args.public_fraction,
        "pseudo_public_splits": args.pseudo_public_splits,
        "n_calibration_folds": args.n_calibration_folds,
        "random_state": args.random_state,
        "shift_penalty": args.shift_penalty,
        "win_rate_bonus": args.win_rate_bonus,
        "submission_format": args.submission_format,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
