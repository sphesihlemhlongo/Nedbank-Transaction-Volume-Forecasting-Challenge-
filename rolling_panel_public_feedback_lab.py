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

from baseline_polars_pipeline import DEFAULT_SUBMISSION_FORMAT, build_submission_frame, rmsle
from experiment_harness import build_run_directory
from submission_eval_utils import write_submission_evaluation_artifacts


FEATURE_PIPELINE_VERSION = "v18_rolling_panel_public_feedback_lab"


@dataclass(frozen=True)
class CandidateResult:
    name: str
    family: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_rmsle: float
    mean_abs_log_shift: float
    affected_train_share: float
    affected_test_share: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine the public-best rolling-panel XGBoost branch with neighborhood, disagreement-boost, "
            "and fin-missing fallback candidates."
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_panel_public_feedback_lab"))
    parser.add_argument("--run-name", type=str, default="rolling_panel_public_feedback_lab")
    parser.add_argument("--base-alpha", type=float, default=0.25)
    parser.add_argument("--shift-penalty", type=float, default=0.06)
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def weight_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def align_feature_splits(
    feature_cache_path: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frame = pl.read_parquet(feature_cache_path)
    keep_columns = [
        "UniqueID",
        "txn_recent_3m_count",
        "txn_active_months_total",
        "txn_sparse_month_share_le_2",
        "fin_missing_flag",
    ]
    train_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", train_ids)})
        .join(
            feature_frame.filter(pl.col("__split__") == "train").select(keep_columns),
            on="UniqueID",
            how="left",
        )
        .to_pandas()
    )
    test_df = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", test_ids)})
        .join(
            feature_frame.filter(pl.col("__split__") == "test").select(keep_columns),
            on="UniqueID",
            how="left",
        )
        .to_pandas()
    )
    return train_df, test_df


def numeric_array(frame: pd.DataFrame, column: str, fill_value: float = -999.0) -> np.ndarray:
    values = np.array(
        pd.to_numeric(frame[column], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
    )
    values[~np.isfinite(values)] = fill_value
    return values


def build_masks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pbest_train: np.ndarray,
    pbest_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    recent_train = numeric_array(train_df, "txn_recent_3m_count")
    recent_test = numeric_array(test_df, "txn_recent_3m_count")
    active_train = numeric_array(train_df, "txn_active_months_total")
    active_test = numeric_array(test_df, "txn_active_months_total")
    sparse_train = numeric_array(train_df, "txn_sparse_month_share_le_2")
    sparse_test = numeric_array(test_df, "txn_sparse_month_share_le_2")
    fin_missing_train = numeric_array(train_df, "fin_missing_flag") >= 1.0
    fin_missing_test = numeric_array(test_df, "fin_missing_flag") >= 1.0

    disagreement_train = np.abs(np.log1p(source_train) - np.log1p(pbest_train))
    disagreement_test = np.abs(np.log1p(source_test) - np.log1p(pbest_test))

    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "recent_le_20": (recent_train <= 20.0, recent_test <= 20.0),
        "recent_le_30": (recent_train <= 30.0, recent_test <= 30.0),
        "active_le_8": (active_train <= 8.0, active_test <= 8.0),
        "active_le_12": (active_train <= 12.0, active_test <= 12.0),
        "sparse_ge_0p5": (sparse_train >= 0.5, sparse_test >= 0.5),
        "low_recent_or_sparse": (
            (recent_train <= 20.0) | (sparse_train >= 0.5),
            (recent_test <= 20.0) | (sparse_test >= 0.5),
        ),
        "fin_missing": (fin_missing_train, fin_missing_test),
    }

    for quantile in (0.75, 0.80, 0.85, 0.90):
        threshold = float(np.quantile(disagreement_train, quantile))
        top_pct = int(round((1.0 - quantile) * 100))
        masks[f"top{top_pct:02d}_disagreement"] = (disagreement_train >= threshold, disagreement_test >= threshold)

    return masks


def log_blend(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left_log = np.log1p(np.clip(left, 0.0, None))
    right_log = np.log1p(np.clip(right, 0.0, None))
    return np.clip(np.expm1((1.0 - alpha) * left_log + alpha * right_log), 0.0, None)


def build_candidates(
    y_train: np.ndarray,
    current_anchor_train: np.ndarray,
    current_anchor_test: np.ndarray,
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    masks: dict[str, tuple[np.ndarray, np.ndarray]],
    shift_penalty: float,
    base_alpha: float,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    current_anchor_log = np.log1p(current_anchor_train)

    def add_candidate(
        name: str,
        family: str,
        oof_predictions: np.ndarray,
        test_predictions: np.ndarray,
        affected_train_mask: np.ndarray,
        affected_test_mask: np.ndarray,
    ) -> None:
        mean_abs_log_shift = float(np.mean(np.abs(np.log1p(oof_predictions) - current_anchor_log)))
        oof_value = float(rmsle(y_train, oof_predictions))
        candidates.append(
            CandidateResult(
                name=name,
                family=family,
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                oof_rmsle=oof_value,
                mean_abs_log_shift=mean_abs_log_shift,
                affected_train_share=float(affected_train_mask.mean()),
                affected_test_share=float(affected_test_mask.mean()),
                selection_score=float(oof_value + shift_penalty * mean_abs_log_shift),
            )
        )

    # Global neighborhood around the validated public-best effective alpha.
    for effective_alpha in (0.22, 0.23, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35):
        oof_predictions = log_blend(base_anchor_train, source_train, effective_alpha)
        test_predictions = log_blend(base_anchor_test, source_test, effective_alpha)
        add_candidate(
            name=f"global_effective_a{weight_token(effective_alpha)}",
            family="global_neighborhood",
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            affected_train_mask=np.ones(len(y_train), dtype=bool),
            affected_test_mask=np.ones(len(test_predictions), dtype=bool),
        )

    # Piecewise uplift on top-disagreement rows starting from the current public-best candidate.
    for mask_name in ("top25_disagreement", "top20_disagreement", "top15_disagreement", "top10_disagreement"):
        train_mask, test_mask = masks[mask_name]
        for extra_alpha in (0.03, 0.05, 0.07, 0.10):
            oof_predictions = np.array(current_anchor_train, copy=True)
            test_predictions = np.array(current_anchor_test, copy=True)
            oof_predictions[train_mask] = log_blend(oof_predictions[train_mask], source_train[train_mask], extra_alpha)
            test_predictions[test_mask] = log_blend(test_predictions[test_mask], source_test[test_mask], extra_alpha)
            add_candidate(
                name=f"{mask_name}_extra_a{weight_token(extra_alpha)}",
                family="disagreement_uplift",
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                affected_train_mask=train_mask,
                affected_test_mask=test_mask,
            )

    # Segment-specific uplift on low-activity groups.
    for mask_name in ("recent_le_20", "recent_le_30", "active_le_8", "active_le_12", "sparse_ge_0p5", "low_recent_or_sparse"):
        train_mask, test_mask = masks[mask_name]
        for extra_alpha in (0.03, 0.05, 0.07):
            oof_predictions = np.array(current_anchor_train, copy=True)
            test_predictions = np.array(current_anchor_test, copy=True)
            oof_predictions[train_mask] = log_blend(oof_predictions[train_mask], source_train[train_mask], extra_alpha)
            test_predictions[test_mask] = log_blend(test_predictions[test_mask], source_test[test_mask], extra_alpha)
            add_candidate(
                name=f"{mask_name}_extra_a{weight_token(extra_alpha)}",
                family="segment_uplift",
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                affected_train_mask=train_mask,
                affected_test_mask=test_mask,
            )

    # Stabilize fin-missing customers by pulling them back toward the original anchor.
    fin_missing_train, fin_missing_test = masks["fin_missing"]
    for fallback_alpha in (0.00, 0.05, 0.10, 0.15, 0.20):
        oof_predictions = np.array(current_anchor_train, copy=True)
        test_predictions = np.array(current_anchor_test, copy=True)
        oof_predictions[fin_missing_train] = log_blend(
            base_anchor_train[fin_missing_train],
            source_train[fin_missing_train],
            fallback_alpha,
        )
        test_predictions[fin_missing_test] = log_blend(
            base_anchor_test[fin_missing_test],
            source_test[fin_missing_test],
            fallback_alpha,
        )
        add_candidate(
            name=f"finmissing_fallback_a{weight_token(fallback_alpha)}",
            family="finmissing_fallback",
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            affected_train_mask=fin_missing_train,
            affected_test_mask=fin_missing_test,
        )

    return candidates


def build_candidate_metrics(
    y_train: np.ndarray,
    candidates: list[CandidateResult],
) -> pd.DataFrame:
    rows = [
        {
            "candidate_name": candidate.name,
            "family": candidate.family,
            "oof_rmsle": candidate.oof_rmsle,
            "mean_abs_log_shift": candidate.mean_abs_log_shift,
            "affected_train_share": candidate.affected_train_share,
            "affected_test_share": candidate.affected_test_share,
            "selection_score": candidate.selection_score,
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows).sort_values(["selection_score", "oof_rmsle", "candidate_name"]).reset_index(drop=True)


def write_recommendation_pack(
    run_dir: Path,
    submissions_dir: Path,
    candidate_metrics: pd.DataFrame,
) -> None:
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
        "Candidates are ordered by `selection_score = oof_rmsle + shift_penalty * mean_abs_log_shift` around the current public-best file.",
        "",
    ]
    for index, row in enumerate(top_rows.to_dict(orient="records"), start=1):
        lines.extend(
            [
                f"{index}. `{index:02d}_{row['candidate_name']}.csv`",
                f"   - OOF RMSLE: `{row['oof_rmsle']:.6f}`",
                f"   - mean abs log shift: `{row['mean_abs_log_shift']:.6f}`",
                f"   - affected test share: `{row['affected_test_share']:.6f}`",
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


def main() -> None:
    args = parse_args()
    run_dir = build_run_directory(args.output_dir, args.run_name)
    rolling_run_dir = args.rolling_run_dir.resolve()

    oof_frame = pl.read_parquet(rolling_run_dir / "oof_predictions.parquet")
    test_frame = pl.read_parquet(rolling_run_dir / "test_predictions.parquet")
    train_ids = oof_frame["UniqueID"].to_numpy()
    test_ids = test_frame["UniqueID"].to_numpy()
    y_train = oof_frame["next_3m_txn_count_true"].to_numpy().astype(np.float64)

    base_anchor_train = oof_frame["pred_publicbest_feedback_temporal_top10"].to_numpy().astype(np.float64)
    base_anchor_test = test_frame["pred_publicbest_feedback_temporal_top10"].to_numpy().astype(np.float64)
    source_train = oof_frame["pred_xgb_conservative_v3"].to_numpy().astype(np.float64)
    source_test = test_frame["pred_xgb_conservative_v3"].to_numpy().astype(np.float64)
    current_anchor_train = oof_frame["pred_anchor_global_xgb_conservative_v3_a0p25"].to_numpy().astype(np.float64)
    current_anchor_test = test_frame["pred_anchor_global_xgb_conservative_v3_a0p25"].to_numpy().astype(np.float64)

    train_features, test_features = align_feature_splits(args.feature_cache_path, train_ids, test_ids)
    masks = build_masks(
        train_df=train_features,
        test_df=test_features,
        pbest_train=current_anchor_train,
        pbest_test=current_anchor_test,
        source_train=source_train,
        source_test=source_test,
    )

    candidates = build_candidates(
        y_train=y_train,
        current_anchor_train=current_anchor_train,
        current_anchor_test=current_anchor_test,
        base_anchor_train=base_anchor_train,
        base_anchor_test=base_anchor_test,
        source_train=source_train,
        source_test=source_test,
        masks=masks,
        shift_penalty=args.shift_penalty,
        base_alpha=args.base_alpha,
    )

    candidate_metrics = build_candidate_metrics(y_train, candidates)

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

    summary_lines = [
        "# Rolling Panel Public Feedback Lab",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Rolling source run: `{rolling_run_dir}`",
        f"- Current public-best anchor: `anchor_global_xgb_conservative_v3_a0p25`",
        f"- Current public-best anchor OOF RMSLE: `{rmsle(y_train, current_anchor_train):.6f}`",
        f"- Best candidate by selection score: `{candidate_metrics.iloc[0]['candidate_name']}` with OOF `{candidate_metrics.iloc[0]['oof_rmsle']:.6f}`",
        f"- Best candidate by raw OOF: `{candidate_metrics.sort_values(['oof_rmsle', 'selection_score']).iloc[0]['candidate_name']}` with OOF `{candidate_metrics.sort_values(['oof_rmsle', 'selection_score']).iloc[0]['oof_rmsle']:.6f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidate_metrics.head(10).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}` ({row['family']}): OOF `{row['oof_rmsle']:.6f}`, shift `{row['mean_abs_log_shift']:.6f}`, selection `{row['selection_score']:.6f}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config_payload = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "rolling_run_dir": str(rolling_run_dir),
        "base_alpha": args.base_alpha,
        "shift_penalty": args.shift_penalty,
        "submission_format": args.submission_format,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
