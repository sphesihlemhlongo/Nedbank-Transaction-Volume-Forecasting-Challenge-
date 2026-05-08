from __future__ import annotations

import argparse
import itertools
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


FEATURE_PIPELINE_VERSION = "v7_artifact_anchor_blend_lab"


@dataclass(frozen=True)
class SeedCandidate:
    name: str
    family: str
    source_run_dir: Path
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass(frozen=True)
class AnchorBlendCandidate:
    name: str
    family: str
    members: tuple[str, ...]
    space: str
    weights: tuple[float, ...]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search conservative blends around the prior best submission using stored OOF/test predictions and "
            "write a small shortlist of candidate CSVs. They are upload-ready only when "
            "--submission-format zindi_log is used."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/artifact_blend_lab"))
    parser.add_argument("--run-name", type=str, default="artifact_anchor_blend")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path("outputs/cache/feature_table_v4_sparse_activity_batch_stack.parquet"),
    )
    parser.add_argument("--anchor-name", type=str, default="v4_best_subset")
    parser.add_argument(
        "--pair-satellites",
        type=str,
        default="v5_best_subset,temporal_best,temporal_xgb,v4_hgb_single",
    )
    parser.add_argument(
        "--triple-satellites",
        type=str,
        default="v5_best_subset,temporal_best,temporal_xgb,v4_hgb_single",
    )
    parser.add_argument(
        "--pair-alpha-grid",
        type=str,
        default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20",
    )
    parser.add_argument(
        "--triple-alpha-grid",
        type=str,
        default="0.02,0.04,0.06,0.08,0.10,0.12",
    )
    parser.add_argument("--max-total-triple-alpha", type=float, default=0.20)
    parser.add_argument("--shift-penalty", type=float, default=0.10)
    parser.add_argument(
        "--write-top-k-submissions",
        type=int,
        default=12,
        help="How many top-ranked anchor blends to materialize as CSV files.",
    )
    parser.add_argument(
        "--force-materialize-candidates",
        type=str,
        default="",
        help=(
            "Comma-separated candidate names to always materialize, even if they are not in the "
            "top-k shortlist."
        ),
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    return parser.parse_args()


def parse_float_grid(grid: str) -> list[float]:
    return [float(value.strip()) for value in grid.split(",") if value.strip()]


def align_predictions_by_id(
    frame: pl.DataFrame,
    ordered_ids: np.ndarray,
    prediction_column: str,
) -> np.ndarray:
    aligned = (
        pl.DataFrame({"UniqueID": pl.Series("UniqueID", ordered_ids)})
        .join(frame.select(["UniqueID", prediction_column]), on="UniqueID", how="left")
        .select(prediction_column)
        .to_series()
        .to_numpy()
    )
    if np.isnan(aligned).any():
        raise ValueError(f"Missing values while aligning {prediction_column}.")
    return aligned.astype(np.float64)


def reconstruct_log_equal_blend(member_predictions: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack([np.clip(prediction, 0.0, None) for prediction in member_predictions])
    return np.expm1(np.log1p(matrix).mean(axis=1))


def load_seed_candidates(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, SeedCandidate]]:
    v4_dir = base_dir / "outputs/experiments/v4_full_stack_20260428_180309_465571_b7834006"
    v5_dir = base_dir / "outputs/experiments/v5_hybrid_full_stack_20260428_210442_752267_61bb4bac"
    temporal_dir = base_dir / "outputs/temporal_holiday_stack/temporal_holiday_xgb_full_20260429_083025_656575_c2f6d028"

    v4_oof = pl.read_parquet(v4_dir / "oof_predictions.parquet")
    v4_test = pl.read_parquet(v4_dir / "test_predictions.parquet")
    v5_oof = pl.read_parquet(v5_dir / "oof_predictions.parquet")
    v5_test = pl.read_parquet(v5_dir / "test_predictions.parquet")
    temporal_oof = pl.read_parquet(temporal_dir / "oof_predictions.parquet")
    temporal_test = pl.read_parquet(temporal_dir / "test_predictions.parquet")

    train_ids = v4_oof["UniqueID"].to_numpy()
    test_ids = v4_test["UniqueID"].to_numpy()
    y_train = v4_oof["next_3m_txn_count_true"].to_numpy().astype(np.float64)

    v4_hgb_oof = align_predictions_by_id(v4_oof, train_ids, "pred_hgb_conservative_v3")
    v4_hgb_test = align_predictions_by_id(v4_test, test_ids, "pred_hgb_conservative_v3")
    v4_xgb_oof = align_predictions_by_id(v4_oof, train_ids, "pred_xgb_conservative_v3")
    v4_xgb_test = align_predictions_by_id(v4_test, test_ids, "pred_xgb_conservative_v3")
    v4_cat_oof = align_predictions_by_id(v4_oof, train_ids, "pred_catboost_conservative_v1")
    v4_cat_test = align_predictions_by_id(v4_test, test_ids, "pred_catboost_conservative_v1")

    seeds = {
        "v4_best_subset": SeedCandidate(
            name="v4_best_subset",
            family="seed",
            source_run_dir=v4_dir,
            oof_predictions=reconstruct_log_equal_blend([v4_hgb_oof, v4_xgb_oof, v4_cat_oof]),
            test_predictions=reconstruct_log_equal_blend([v4_hgb_test, v4_xgb_test, v4_cat_test]),
        ),
        "v4_hgb_single": SeedCandidate(
            name="v4_hgb_single",
            family="single",
            source_run_dir=v4_dir,
            oof_predictions=v4_hgb_oof,
            test_predictions=v4_hgb_test,
        ),
        "v4_xgb_single": SeedCandidate(
            name="v4_xgb_single",
            family="single",
            source_run_dir=v4_dir,
            oof_predictions=v4_xgb_oof,
            test_predictions=v4_xgb_test,
        ),
        "v5_best_subset": SeedCandidate(
            name="v5_best_subset",
            family="seed",
            source_run_dir=v5_dir,
            oof_predictions=align_predictions_by_id(v5_oof, train_ids, "pred_blend_log_equal_search_best_k3"),
            test_predictions=align_predictions_by_id(v5_test, test_ids, "pred_blend_log_equal_search_best_k3"),
        ),
        "temporal_best": SeedCandidate(
            name="temporal_best",
            family="temporal",
            source_run_dir=temporal_dir,
            oof_predictions=align_predictions_by_id(temporal_oof, train_ids, "pred_blend_greedy_forward"),
            test_predictions=align_predictions_by_id(temporal_test, test_ids, "pred_blend_greedy_forward"),
        ),
        "temporal_xgb": SeedCandidate(
            name="temporal_xgb",
            family="temporal",
            source_run_dir=temporal_dir,
            oof_predictions=align_predictions_by_id(temporal_oof, train_ids, "pred_xgb_conservative_v3"),
            test_predictions=align_predictions_by_id(temporal_test, test_ids, "pred_xgb_conservative_v3"),
        ),
    }
    return train_ids, test_ids, y_train, seeds


def blend_predictions(predictions: list[np.ndarray], weights: list[float], space: str) -> np.ndarray:
    matrix = np.column_stack([np.clip(prediction, 0.0, None) for prediction in predictions])
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weight_array.sum(), 1.0):
        weight_array = weight_array / weight_array.sum()
    if space == "raw":
        return matrix @ weight_array
    if space == "log":
        return np.expm1(np.log1p(matrix) @ weight_array)
    raise ValueError(f"Unsupported blend space: {space}")


def weight_token(weight: float) -> str:
    return f"{weight:.3f}".replace(".", "p")


def build_anchor_blend_candidates(
    seeds: dict[str, SeedCandidate],
    anchor_name: str,
    pair_satellites: list[str],
    triple_satellites: list[str],
    pair_alphas: list[float],
    triple_alphas: list[float],
    max_total_triple_alpha: float,
) -> list[AnchorBlendCandidate]:
    anchor = seeds[anchor_name]
    candidates: list[AnchorBlendCandidate] = []

    for satellite_name in pair_satellites:
        if satellite_name == anchor_name:
            continue
        satellite = seeds[satellite_name]
        for space in ("raw", "log"):
            for alpha in pair_alphas:
                weights = [1.0 - alpha, alpha]
                candidates.append(
                    AnchorBlendCandidate(
                        name=f"anchor_{anchor_name}__{space}__{satellite_name}_a{weight_token(alpha)}",
                        family="anchor_pair",
                        members=(anchor_name, satellite_name),
                        space=space,
                        weights=tuple(weights),
                        oof_predictions=blend_predictions(
                            [anchor.oof_predictions, satellite.oof_predictions],
                            weights,
                            space,
                        ),
                        test_predictions=blend_predictions(
                            [anchor.test_predictions, satellite.test_predictions],
                            weights,
                            space,
                        ),
                    )
                )

    valid_triple_satellites = [name for name in triple_satellites if name != anchor_name]
    for first_name, second_name in itertools.combinations(valid_triple_satellites, 2):
        first = seeds[first_name]
        second = seeds[second_name]
        for space in ("raw", "log"):
            for first_alpha in triple_alphas:
                for second_alpha in triple_alphas:
                    total_alpha = first_alpha + second_alpha
                    if total_alpha > max_total_triple_alpha:
                        continue
                    weights = [1.0 - total_alpha, first_alpha, second_alpha]
                    candidates.append(
                        AnchorBlendCandidate(
                            name=(
                                f"anchor_{anchor_name}__{space}__{first_name}_a{weight_token(first_alpha)}"
                                f"__{second_name}_a{weight_token(second_alpha)}"
                            ),
                            family="anchor_triple",
                            members=(anchor_name, first_name, second_name),
                            space=space,
                            weights=tuple(weights),
                            oof_predictions=blend_predictions(
                                [anchor.oof_predictions, first.oof_predictions, second.oof_predictions],
                                weights,
                                space,
                            ),
                            test_predictions=blend_predictions(
                                [anchor.test_predictions, first.test_predictions, second.test_predictions],
                                weights,
                                space,
                            ),
                        )
                    )

    return candidates


def compute_candidate_metrics(
    y_train: np.ndarray,
    anchor_predictions: np.ndarray,
    feature_cache_path: Path,
    direct_seeds: dict[str, SeedCandidate],
    blend_candidates: list[AnchorBlendCandidate],
    shift_penalty: float,
) -> pd.DataFrame:
    segment_frame = pl.read_parquet(feature_cache_path).filter(pl.col("__split__") == "train").to_pandas()
    masks: dict[str, np.ndarray] = {
        "low_recent_3m": segment_frame["txn_recent_3m_count"].to_numpy(dtype=np.float64) <= 20,
        "active_months_le_12": segment_frame["txn_active_months_total"].to_numpy(dtype=np.float64) <= 12,
        "sparse_months_ge_2": segment_frame["txn_months_count_le_2"].to_numpy(dtype=np.float64) >= 2,
        "fin_missing": segment_frame["fin_missing_flag"].to_numpy(dtype=np.float64) >= 1,
        "target_le_3": y_train <= 3,
    }

    rows: list[dict[str, object]] = []

    def build_row(
        candidate_name: str,
        candidate_type: str,
        family: str,
        space: str,
        members: str,
        weights_json: str,
        predictions: np.ndarray,
    ) -> dict[str, object]:
        row = {
            "candidate_name": candidate_name,
            "candidate_type": candidate_type,
            "family": family,
            "space": space,
            "members": members,
            "weights_json": weights_json,
            "oof_rmsle": rmsle(y_train, predictions),
            "mean_abs_log_shift_vs_anchor": float(
                np.mean(np.abs(np.log1p(np.clip(predictions, 0.0, None)) - np.log1p(anchor_predictions)))
            ),
        }
        for mask_name, mask in masks.items():
            row[f"{mask_name}_rmsle"] = rmsle(y_train[mask], predictions[mask])
        return row

    for seed in direct_seeds.values():
        rows.append(
            build_row(
                candidate_name=seed.name,
                candidate_type="direct_seed",
                family=seed.family,
                space="raw",
                members=seed.name,
                weights_json=json.dumps([1.0]),
                predictions=seed.oof_predictions,
            )
        )

    for candidate in blend_candidates:
        rows.append(
            build_row(
                candidate_name=candidate.name,
                candidate_type="anchor_blend",
                family=candidate.family,
                space=candidate.space,
                members=",".join(candidate.members),
                weights_json=json.dumps(candidate.weights),
                predictions=candidate.oof_predictions,
            )
        )

    metrics = pd.DataFrame(rows)
    metric_columns = [
        "oof_rmsle",
        "mean_abs_log_shift_vs_anchor",
        "low_recent_3m_rmsle",
        "active_months_le_12_rmsle",
        "sparse_months_ge_2_rmsle",
        "fin_missing_rmsle",
        "target_le_3_rmsle",
    ]
    rank_columns: list[str] = []
    for column in metric_columns:
        rank_column = f"rank_{column}"
        metrics[rank_column] = metrics[column].rank(method="average", ascending=True)
        rank_columns.append(rank_column)
    metrics["hedge_rank_mean"] = metrics[rank_columns].mean(axis=1)
    metrics["conservative_score"] = metrics["oof_rmsle"] + shift_penalty * metrics["mean_abs_log_shift_vs_anchor"]
    return metrics.sort_values(["candidate_type", "oof_rmsle", "candidate_name"]).reset_index(drop=True)


def build_prediction_lookup(
    direct_seeds: dict[str, SeedCandidate],
    blend_candidates: list[AnchorBlendCandidate],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seed in direct_seeds.values():
        lookup[seed.name] = (seed.oof_predictions, seed.test_predictions)
    for candidate in blend_candidates:
        lookup[candidate.name] = (candidate.oof_predictions, candidate.test_predictions)
    return lookup


def write_artifacts(
    run_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    y_train: np.ndarray,
    sample_submission_path: Path,
    metrics: pd.DataFrame,
    prediction_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

    anchor_blends = metrics.loc[metrics["candidate_type"] == "anchor_blend"].copy()
    best_oof = anchor_blends.sort_values(["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"]).iloc[0]
    best_hedge = anchor_blends.sort_values(
        ["conservative_score", "hedge_rank_mean", "oof_rmsle", "candidate_name"]
    ).iloc[0]

    direct_seed_names = metrics.loc[metrics["candidate_type"] == "direct_seed", "candidate_name"].tolist()
    top_anchor_names = (
        anchor_blends.sort_values(["oof_rmsle", "mean_abs_log_shift_vs_anchor", "candidate_name"])
        .head(args.write_top_k_submissions)["candidate_name"]
        .tolist()
    )
    forced_candidate_names = [
        value.strip() for value in args.force_materialize_candidates.split(",") if value.strip()
    ]
    missing_forced = sorted(set(forced_candidate_names).difference(prediction_lookup))
    if missing_forced:
        missing_text = ", ".join(missing_forced)
        raise ValueError(f"Unknown forced candidate name(s): {missing_text}")

    materialized_candidates = sorted(
        set(
            direct_seed_names
            + top_anchor_names
            + forced_candidate_names
            + [best_oof["candidate_name"], best_hedge["candidate_name"]]
        )
    )

    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for candidate_name in materialized_candidates:
        _, test_predictions = prediction_lookup[candidate_name]
        build_submission_frame(
            sample_submission_path,
            test_ids,
            test_predictions,
            submission_format=args.submission_format,
        ).write_csv(submissions_dir / f"{candidate_name}.csv")

    selection_dir = run_dir / "recommended_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "01_anchor_best_oof.csv").write_bytes(
        (submissions_dir / f"{best_oof['candidate_name']}.csv").read_bytes()
    )
    (selection_dir / "02_anchor_best_hedge.csv").write_bytes(
        (submissions_dir / f"{best_hedge['candidate_name']}.csv").read_bytes()
    )
    (selection_dir / "README.md").write_text(
        "\n".join(
            [
                "# Recommended Selection",
                "",
                "1. `01_anchor_best_oof.csv`",
                f"   - candidate: `{best_oof['candidate_name']}`",
                f"   - OOF RMSLE: `{best_oof['oof_rmsle']:.6f}`",
                f"   - mean abs log shift vs anchor: `{best_oof['mean_abs_log_shift_vs_anchor']:.6f}`",
                "",
                "2. `02_anchor_best_hedge.csv`",
                f"   - candidate: `{best_hedge['candidate_name']}`",
                f"   - OOF RMSLE: `{best_hedge['oof_rmsle']:.6f}`",
                f"   - conservative score: `{best_hedge['conservative_score']:.6f}`",
                "",
                "Candidate names containing `__raw__` or `__log__` refer to blend space, not CSV serialization.",
                (
                    "Both files are copied from `submissions/` and are upload-ready for Zindi."
                    if args.submission_format == "zindi_log"
                    else "Both files are copied from `submissions/`, but this run used raw output format. "
                    "Do not upload them to Zindi without converting to `np.log1p(prediction)` first."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Artifact Blend Lab Summary",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Anchor candidate: `{args.anchor_name}`",
        f"- Submission format: `{args.submission_format}`",
        f"- Best anchor-blend OOF candidate: `{best_oof['candidate_name']}` at `{best_oof['oof_rmsle']:.6f}`",
        f"- Best hedge candidate: `{best_hedge['candidate_name']}` with conservative score `{best_hedge['conservative_score']:.6f}`",
        "",
        "## Top Anchor Blends",
        "",
    ]
    for row in anchor_blends.sort_values(["oof_rmsle", "mean_abs_log_shift_vs_anchor"]).head(12).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}`: OOF `{row['oof_rmsle']:.6f}`, "
            f"shift `{row['mean_abs_log_shift_vs_anchor']:.6f}`, members `{row['members']}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "anchor_name": args.anchor_name,
        "pair_satellites": args.pair_satellites,
        "triple_satellites": args.triple_satellites,
        "pair_alpha_grid": args.pair_alpha_grid,
        "triple_alpha_grid": args.triple_alpha_grid,
        "max_total_triple_alpha": args.max_total_triple_alpha,
        "shift_penalty": args.shift_penalty,
        "submission_format": args.submission_format,
        "write_top_k_submissions": args.write_top_k_submissions,
        "force_materialize_candidates": args.force_materialize_candidates,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.resolve()
    run_dir = build_run_directory(args.output_dir, args.run_name)
    train_ids, test_ids, y_train, seeds = load_seed_candidates(base_dir)

    pair_satellites = [value.strip() for value in args.pair_satellites.split(",") if value.strip()]
    triple_satellites = [value.strip() for value in args.triple_satellites.split(",") if value.strip()]
    pair_alphas = parse_float_grid(args.pair_alpha_grid)
    triple_alphas = parse_float_grid(args.triple_alpha_grid)

    blend_candidates = build_anchor_blend_candidates(
        seeds=seeds,
        anchor_name=args.anchor_name,
        pair_satellites=pair_satellites,
        triple_satellites=triple_satellites,
        pair_alphas=pair_alphas,
        triple_alphas=triple_alphas,
        max_total_triple_alpha=args.max_total_triple_alpha,
    )

    metrics = compute_candidate_metrics(
        y_train=y_train,
        anchor_predictions=seeds[args.anchor_name].oof_predictions,
        feature_cache_path=args.feature_cache_path.resolve(),
        direct_seeds=seeds,
        blend_candidates=blend_candidates,
        shift_penalty=args.shift_penalty,
    )

    prediction_lookup = build_prediction_lookup(seeds, blend_candidates)
    write_artifacts(
        run_dir=run_dir,
        train_ids=train_ids,
        test_ids=test_ids,
        y_train=y_train,
        sample_submission_path=base_dir / "SampleSubmission.csv",
        metrics=metrics,
        prediction_lookup=prediction_lookup,
        args=args,
    )
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
