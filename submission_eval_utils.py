from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate import rmsle_from_raw, zindi_log_rmse


REFERENCE_CANDIDATE_NAMES = (
    "PublicReference.csv",
    "reference.csv",
    "public_reference.csv",
)


def discover_reference_path(data_dir: Path) -> Path | None:
    for name in REFERENCE_CANDIDATE_NAMES:
        candidate = (data_dir / name).resolve()
        if candidate.exists():
            return candidate
    return None


def _merge_submission_with_reference(submission_path: Path, reference_path: Path) -> pd.DataFrame:
    submission_frame = pd.read_csv(submission_path)
    reference_frame = pd.read_csv(reference_path)
    required_columns = {"UniqueID", "next_3m_txn_count"}

    if not required_columns.issubset(submission_frame.columns):
        missing = sorted(required_columns - set(submission_frame.columns))
        raise ValueError(f"Submission is missing required columns: {missing}")
    if not required_columns.issubset(reference_frame.columns):
        missing = sorted(required_columns - set(reference_frame.columns))
        raise ValueError(f"Reference is missing required columns: {missing}")

    merged = reference_frame.merge(
        submission_frame,
        on="UniqueID",
        suffixes=("_true", "_pred"),
        how="left",
    )
    if len(merged) != len(reference_frame):
        raise ValueError(
            f"Reference row count mismatch after merge: expected {len(reference_frame)}, got {len(merged)}."
        )
    if merged["next_3m_txn_count_pred"].isna().any():
        raise ValueError("Submission contains missing predictions after merging with reference.")
    return merged


def score_submission_file(
    submission_path: Path,
    reference_path: Path,
    submission_mode: str,
) -> dict[str, object]:
    merged = _merge_submission_with_reference(submission_path, reference_path)
    y_true = merged["next_3m_txn_count_true"].to_numpy(dtype=float)
    y_pred = merged["next_3m_txn_count_pred"].to_numpy(dtype=float)

    if submission_mode == "zindi_log":
        score = zindi_log_rmse(y_true, y_pred)
    elif submission_mode == "raw":
        score = rmsle_from_raw(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported submission mode: {submission_mode}")

    return {
        "submission_path": str(submission_path),
        "rows_scored": int(len(merged)),
        "submission_mode": submission_mode,
        "local_reference_score": float(score),
    }


def write_submission_evaluation_artifacts(
    run_dir: Path,
    submissions_dir: Path,
    candidate_names: list[str],
    data_dir: Path,
    submission_mode: str,
) -> pd.DataFrame | None:
    reference_path = discover_reference_path(data_dir)
    note_path = run_dir / "submission_evaluation.md"
    if reference_path is None:
        note_lines = [
            "# Submission Evaluation",
            "",
            "- No local reference file was found.",
            "- Checked names:",
        ]
        note_lines.extend([f"  - `{name}`" for name in REFERENCE_CANDIDATE_NAMES])
        note_lines.extend(
            [
                "",
                "- Packaging still completed successfully.",
                "- If you add `PublicReference.csv` later, rerun the lab to populate local reference scores.",
            ]
        )
        note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
        return None

    rows = []
    for candidate_name in candidate_names:
        submission_path = submissions_dir / f"{candidate_name}.csv"
        score_payload = score_submission_file(submission_path, reference_path, submission_mode)
        rows.append(
            {
                "candidate_name": candidate_name,
                "local_reference_score": score_payload["local_reference_score"],
                "rows_scored": score_payload["rows_scored"],
                "submission_mode": submission_mode,
                "reference_path": str(reference_path),
            }
        )

    score_frame = pd.DataFrame(rows).sort_values(["local_reference_score", "candidate_name"]).reset_index(drop=True)
    score_frame.to_csv(run_dir / "submission_evaluation.csv", index=False)

    top_rows = score_frame.head(min(10, len(score_frame)))
    note_lines = [
        "# Submission Evaluation",
        "",
        f"- Reference file: `{reference_path}`",
        f"- Submission interpretation: `{submission_mode}`",
        "",
        "## Best Local Reference Scores",
        "",
    ]
    for row in top_rows.to_dict(orient="records"):
        note_lines.append(f"- `{row['candidate_name']}`: `{row['local_reference_score']:.6f}`")
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    return score_frame
