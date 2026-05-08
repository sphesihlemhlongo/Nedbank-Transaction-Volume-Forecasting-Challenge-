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
from rolling_panel_public_feedback_lab import align_feature_splits, build_masks, log_blend
from submission_eval_utils import write_submission_evaluation_artifacts


FEATURE_PIPELINE_VERSION = "v19_rolling_panel_public_stability_lab"


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
    pseudo_public_win_rate: float
    pseudo_public_mean_delta: float
    pseudo_public_median_delta: float
    pseudo_public_p75_delta: float
    pseudo_public_p90_delta: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search public-safe anchored refinements around the current best rolling-panel XGBoost submission "
            "and rank them using deterministic pseudo-public stability."
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_panel_public_stability_lab"))
    parser.add_argument("--run-name", type=str, default="rolling_panel_public_stability_lab")
    parser.add_argument("--public-fraction", type=float, default=0.30)
    parser.add_argument("--pseudo-public-splits", type=int, default=80)
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


def precise_weight_token(value: float) -> str:
    token = f"{value:.3f}".rstrip("0").rstrip(".")
    return token.replace(".", "p")


def numeric_array(frame: pd.DataFrame, column: str, fill_value: float = -999.0) -> np.ndarray:
    values = np.array(
        pd.to_numeric(frame[column], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
    )
    values[~np.isfinite(values)] = fill_value
    return values


def apply_masked_blend(
    anchor_predictions: np.ndarray,
    source_predictions: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    anchor_test_predictions: np.ndarray,
    source_test_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof_predictions = np.array(anchor_predictions, copy=True)
    test_predictions = np.array(anchor_test_predictions, copy=True)
    oof_predictions[train_mask] = log_blend(oof_predictions[train_mask], source_predictions[train_mask], alpha)
    test_predictions[test_mask] = log_blend(test_predictions[test_mask], source_test_predictions[test_mask], alpha)
    return oof_predictions, test_predictions


def apply_weighted_blend(
    anchor_predictions: np.ndarray,
    source_predictions: np.ndarray,
    anchor_test_predictions: np.ndarray,
    source_test_predictions: np.ndarray,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_alpha = np.clip(alpha * train_weights, 0.0, 1.0)
    test_alpha = np.clip(alpha * test_weights, 0.0, 1.0)
    oof_predictions = np.clip(
        np.expm1(
            (1.0 - train_alpha) * np.log1p(np.clip(anchor_predictions, 0.0, None))
            + train_alpha * np.log1p(np.clip(source_predictions, 0.0, None))
        ),
        0.0,
        None,
    )
    test_predictions = np.clip(
        np.expm1(
            (1.0 - test_alpha) * np.log1p(np.clip(anchor_test_predictions, 0.0, None))
            + test_alpha * np.log1p(np.clip(source_test_predictions, 0.0, None))
        ),
        0.0,
        None,
    )
    return oof_predictions, test_predictions


def build_pseudo_public_subsets(
    y_train: np.ndarray,
    train_features: pd.DataFrame,
    public_fraction: float,
    n_splits: int,
    random_state: int,
) -> list[np.ndarray]:
    y_log = np.log1p(np.clip(y_train, 0.0, None))
    y_bins = pd.qcut(pd.Series(y_log), q=10, labels=False, duplicates="drop").astype(int).to_numpy()
    recent = numeric_array(train_features, "txn_recent_3m_count")
    sparse = numeric_array(train_features, "txn_sparse_month_share_le_2")
    fin_missing = numeric_array(train_features, "fin_missing_flag") >= 1.0

    recent_band = np.where(recent <= 10.0, 0, np.where(recent <= 20.0, 1, 2))
    sparse_band = np.where(sparse >= 0.5, 1, 0)
    fin_band = fin_missing.astype(int)
    stratum_code = y_bins * 12 + recent_band * 4 + sparse_band * 2 + fin_band

    indices = np.arange(len(y_train), dtype=np.int32)
    rng = np.random.default_rng(random_state)
    subsets: list[np.ndarray] = []
    for split_idx in range(n_splits):
        chosen: list[np.ndarray] = []
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + split_idx)
        for code in np.unique(stratum_code):
            group = indices[stratum_code == code]
            if len(group) == 0:
                continue
            sample_size = max(1, int(round(len(group) * public_fraction)))
            sample_size = min(sample_size, len(group))
            chosen.append(np.sort(local_rng.choice(group, size=sample_size, replace=False)))
        subsets.append(np.sort(np.concatenate(chosen)))
    return subsets


def evaluate_public_stability(
    y_train: np.ndarray,
    anchor_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    pseudo_public_subsets: list[np.ndarray],
) -> dict[str, float]:
    deltas = []
    for subset in pseudo_public_subsets:
        anchor_score = rmsle(y_train[subset], anchor_predictions[subset])
        candidate_score = rmsle(y_train[subset], candidate_predictions[subset])
        deltas.append(float(candidate_score - anchor_score))
    deltas_array = np.array(deltas, dtype=np.float64)
    return {
        "pseudo_public_win_rate": float(np.mean(deltas_array < 0.0)),
        "pseudo_public_mean_delta": float(np.mean(deltas_array)),
        "pseudo_public_median_delta": float(np.median(deltas_array)),
        "pseudo_public_p75_delta": float(np.quantile(deltas_array, 0.75)),
        "pseudo_public_p90_delta": float(np.quantile(deltas_array, 0.90)),
    }


def build_sparse_weights(values: np.ndarray, threshold: float) -> np.ndarray:
    weights = (values - threshold) / max(1e-6, 1.0 - threshold)
    return np.clip(weights, 0.0, 1.0)


def build_disagreement_weights(
    anchor_predictions: np.ndarray,
    source_predictions: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    reference_values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    disagreement = np.abs(np.log1p(np.clip(source_predictions, 0.0, None)) - np.log1p(np.clip(anchor_predictions, 0.0, None)))
    lower = float(np.quantile(reference_values, lower_quantile))
    upper = float(np.quantile(reference_values, upper_quantile))
    weights = (disagreement - lower) / max(1e-6, upper - lower)
    return np.clip(weights, 0.0, 1.0), lower, upper


def build_candidates(
    y_train: np.ndarray,
    base_anchor_train: np.ndarray,
    base_anchor_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    current_anchor_train: np.ndarray,
    current_anchor_test: np.ndarray,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    pseudo_public_subsets: list[np.ndarray],
    shift_penalty: float,
    win_rate_bonus: float,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    current_anchor_log = np.log1p(np.clip(current_anchor_train, 0.0, None))
    masks = build_masks(
        train_df=train_features,
        test_df=test_features,
        pbest_train=current_anchor_train,
        pbest_test=current_anchor_test,
        source_train=source_train,
        source_test=source_test,
    )
    sparse_train = numeric_array(train_features, "txn_sparse_month_share_le_2")
    sparse_test = numeric_array(test_features, "txn_sparse_month_share_le_2")
    disagreement_reference = np.abs(
        np.log1p(np.clip(source_train, 0.0, None)) - np.log1p(np.clip(current_anchor_train, 0.0, None))
    )

    def add_candidate(
        name: str,
        family: str,
        oof_predictions: np.ndarray,
        test_predictions: np.ndarray,
        affected_train_mask: np.ndarray,
        affected_test_mask: np.ndarray,
    ) -> None:
        stability = evaluate_public_stability(y_train, current_anchor_train, oof_predictions, pseudo_public_subsets)
        mean_abs_log_shift = float(np.mean(np.abs(np.log1p(np.clip(oof_predictions, 0.0, None)) - current_anchor_log)))
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
                affected_train_share=float(affected_train_mask.mean()),
                affected_test_share=float(affected_test_mask.mean()),
                pseudo_public_win_rate=stability["pseudo_public_win_rate"],
                pseudo_public_mean_delta=stability["pseudo_public_mean_delta"],
                pseudo_public_median_delta=stability["pseudo_public_median_delta"],
                pseudo_public_p75_delta=stability["pseudo_public_p75_delta"],
                pseudo_public_p90_delta=stability["pseudo_public_p90_delta"],
                selection_score=selection_score,
            )
        )

    global_alphas = (0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.30, 0.32, 0.35)
    extra_alphas = (0.02, 0.03, 0.05, 0.07)
    combo_masks = ("sparse_ge_0p5", "top20_disagreement", "top15_disagreement", "low_recent_or_sparse")

    for alpha in global_alphas:
        global_train = log_blend(base_anchor_train, source_train, alpha)
        global_test = log_blend(base_anchor_test, source_test, alpha)
        add_candidate(
            name=f"global_effective_a{precise_weight_token(alpha)}",
            family="global_neighborhood",
            oof_predictions=global_train,
            test_predictions=global_test,
            affected_train_mask=np.ones(len(y_train), dtype=bool),
            affected_test_mask=np.ones(len(global_test), dtype=bool),
        )

        for mask_name in combo_masks:
            train_mask, test_mask = masks[mask_name]
            for extra_alpha in extra_alphas:
                combo_train, combo_test = apply_masked_blend(
                    anchor_predictions=global_train,
                    source_predictions=source_train,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    alpha=extra_alpha,
                    anchor_test_predictions=global_test,
                    source_test_predictions=source_test,
                )
                add_candidate(
                    name=f"g{precise_weight_token(alpha)}__{mask_name}__a{precise_weight_token(extra_alpha)}",
                    family="global_plus_mask",
                    oof_predictions=combo_train,
                    test_predictions=combo_test,
                    affected_train_mask=train_mask,
                    affected_test_mask=test_mask,
                )

    for threshold in (0.40, 0.50, 0.60):
        train_weights = build_sparse_weights(sparse_train, threshold)
        test_weights = build_sparse_weights(sparse_test, threshold)
        for alpha in (0.04, 0.06, 0.08, 0.10):
            oof_predictions, test_predictions = apply_weighted_blend(
                anchor_predictions=current_anchor_train,
                source_predictions=source_train,
                anchor_test_predictions=current_anchor_test,
                source_test_predictions=source_test,
                train_weights=train_weights,
                test_weights=test_weights,
                alpha=alpha,
            )
            add_candidate(
                name=f"soft_sparse_t{precise_weight_token(threshold)}__a{precise_weight_token(alpha)}",
                family="soft_sparse_gate",
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                affected_train_mask=train_weights > 0.0,
                affected_test_mask=test_weights > 0.0,
            )

    for lower_quantile, upper_quantile in ((0.75, 0.95), (0.80, 0.95), (0.85, 0.97)):
        train_weights, lower, upper = build_disagreement_weights(
            current_anchor_train,
            source_train,
            lower_quantile,
            upper_quantile,
            disagreement_reference,
        )
        test_disagreement = np.abs(
            np.log1p(np.clip(source_test, 0.0, None)) - np.log1p(np.clip(current_anchor_test, 0.0, None))
        )
        test_weights = np.clip((test_disagreement - lower) / max(1e-6, upper - lower), 0.0, 1.0)
        for alpha in (0.04, 0.06, 0.08):
            oof_predictions, test_predictions = apply_weighted_blend(
                anchor_predictions=current_anchor_train,
                source_predictions=source_train,
                anchor_test_predictions=current_anchor_test,
                source_test_predictions=source_test,
                train_weights=train_weights,
                test_weights=test_weights,
                alpha=alpha,
            )
            add_candidate(
                name=(
                    f"soft_disagreement_q{int(lower_quantile * 100):02d}"
                    f"_{int(upper_quantile * 100):02d}__a{precise_weight_token(alpha)}"
                ),
                family="soft_disagreement_gate",
                oof_predictions=oof_predictions,
                test_predictions=test_predictions,
                affected_train_mask=train_weights > 0.0,
                affected_test_mask=test_weights > 0.0,
            )

    train_sparse_soft = build_sparse_weights(sparse_train, 0.50)
    test_sparse_soft = build_sparse_weights(sparse_test, 0.50)
    train_disagreement_soft, lower, upper = build_disagreement_weights(
        current_anchor_train,
        source_train,
        0.80,
        0.95,
        disagreement_reference,
    )
    test_disagreement = np.abs(
        np.log1p(np.clip(source_test, 0.0, None)) - np.log1p(np.clip(current_anchor_test, 0.0, None))
    )
    test_disagreement_soft = np.clip((test_disagreement - lower) / max(1e-6, upper - lower), 0.0, 1.0)
    for alpha in (0.04, 0.06, 0.08):
        train_weights = np.maximum(train_sparse_soft, train_disagreement_soft)
        test_weights = np.maximum(test_sparse_soft, test_disagreement_soft)
        oof_predictions, test_predictions = apply_weighted_blend(
            anchor_predictions=current_anchor_train,
            source_predictions=source_train,
            anchor_test_predictions=current_anchor_test,
            source_test_predictions=source_test,
            train_weights=train_weights,
            test_weights=test_weights,
            alpha=alpha,
        )
        add_candidate(
            name=f"soft_union_sparse_disagreement__a{precise_weight_token(alpha)}",
            family="soft_union_gate",
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            affected_train_mask=train_weights > 0.0,
            affected_test_mask=test_weights > 0.0,
        )

    return candidates


def build_candidate_metrics(candidates: list[CandidateResult]) -> pd.DataFrame:
    rows = [
        {
            "candidate_name": candidate.name,
            "family": candidate.family,
            "oof_rmsle": candidate.oof_rmsle,
            "mean_abs_log_shift": candidate.mean_abs_log_shift,
            "affected_train_share": candidate.affected_train_share,
            "affected_test_share": candidate.affected_test_share,
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
        (
            "Candidates are ordered by a pseudo-public stability score built from full-train OOF, "
            "public-split win rate, and drift from the current public-best rolling-panel anchor."
        ),
        "",
    ]
    for index, row in enumerate(top_rows.to_dict(orient="records"), start=1):
        lines.extend(
            [
                f"{index}. `{index:02d}_{row['candidate_name']}.csv`",
                f"   - OOF RMSLE: `{row['oof_rmsle']:.6f}`",
                f"   - mean abs log shift: `{row['mean_abs_log_shift']:.6f}`",
                f"   - pseudo-public win rate: `{row['pseudo_public_win_rate']:.4f}`",
                f"   - pseudo-public median delta vs anchor: `{row['pseudo_public_median_delta']:.6f}`",
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
    pseudo_public_subsets = build_pseudo_public_subsets(
        y_train=y_train,
        train_features=train_features,
        public_fraction=args.public_fraction,
        n_splits=args.pseudo_public_splits,
        random_state=args.random_state,
    )

    candidates = build_candidates(
        y_train=y_train,
        base_anchor_train=base_anchor_train,
        base_anchor_test=base_anchor_test,
        source_train=source_train,
        source_test=source_test,
        current_anchor_train=current_anchor_train,
        current_anchor_test=current_anchor_test,
        train_features=train_features,
        test_features=test_features,
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
        "# Rolling Panel Public Stability Lab",
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
        (
            f"- Best candidate by raw OOF: `{best_raw['candidate_name']}` "
            f"with OOF `{best_raw['oof_rmsle']:.6f}`"
        ),
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidate_metrics.head(12).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}` ({row['family']}): "
            f"OOF `{row['oof_rmsle']:.6f}`, win rate `{row['pseudo_public_win_rate']:.4f}`, "
            f"median delta `{row['pseudo_public_median_delta']:.6f}`, "
            f"shift `{row['mean_abs_log_shift']:.6f}`, selection `{row['selection_score']:.6f}`"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config_payload = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "rolling_run_dir": str(rolling_run_dir),
        "public_fraction": args.public_fraction,
        "pseudo_public_splits": args.pseudo_public_splits,
        "random_state": args.random_state,
        "shift_penalty": args.shift_penalty,
        "win_rate_bonus": args.win_rate_bonus,
        "submission_format": args.submission_format,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
