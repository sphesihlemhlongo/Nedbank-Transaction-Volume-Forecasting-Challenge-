from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current rolling-panel search chain end-to-end on a higher-compute machine."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pseudo-public-splits", type=int, default=480)
    parser.add_argument("--skip-supervision", action="store_true")
    parser.add_argument("--rolling-run-dir", type=Path, default=None)
    parser.add_argument("--calibration-run-dir", type=Path, default=None)
    parser.add_argument("--supervision-run-name", type=str, default="dgx_rolling_panel_xgb_strict")
    parser.add_argument("--stability-run-name", type=str, default="dgx_public_stability")
    parser.add_argument("--calibration-run-name", type=str, default="dgx_public_calibration")
    parser.add_argument("--feedback-run-name", type=str, default="dgx_public_calibration_feedback")
    parser.add_argument("--include-models", type=str, default="xgb_conservative_v3")
    return parser.parse_args()


def run_command(command: list[str], workdir: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def latest_run(root: Path, prefix: str) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(f"{prefix}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {root} for prefix {prefix}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_summary(
    summary_dir: Path,
    rolling_run_dir: Path,
    stability_run_dir: Path,
    calibration_run_dir: Path,
    feedback_run_dir: Path,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DGX Search Runner Summary",
        "",
        f"- Generated: `{datetime.utcnow().isoformat()}Z`",
        f"- Rolling supervision run: `{rolling_run_dir}`",
        f"- Stability run: `{stability_run_dir}`",
        f"- Calibration run: `{calibration_run_dir}`",
        f"- Calibration feedback run: `{feedback_run_dir}`",
        "",
        "## Recommended Selection Folders",
        "",
        f"- Stability: `{stability_run_dir / 'recommended_selection'}`",
        f"- Calibration: `{calibration_run_dir / 'recommended_selection'}`",
        f"- Calibration feedback: `{feedback_run_dir / 'recommended_selection'}`",
        "",
        "Use the calibration-feedback folder first when the current public-best file is an isotonic calibration candidate.",
    ]
    (summary_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.data_dir.resolve()
    python_executable = sys.executable

    rolling_root = repo_root / "outputs" / "rolling_panel_supervision_lab"
    stability_root = repo_root / "outputs" / "rolling_panel_public_stability_lab"
    calibration_root = repo_root / "outputs" / "rolling_panel_public_calibration_lab"
    feedback_root = repo_root / "outputs" / "rolling_panel_public_calibration_feedback_lab"

    if args.skip_supervision:
        if args.rolling_run_dir is None:
            raise ValueError("--skip-supervision requires --rolling-run-dir.")
        rolling_run_dir = args.rolling_run_dir.resolve()
    else:
        run_command(
            [
                python_executable,
                "rolling_panel_supervision_lab.py",
                "--data-dir",
                ".",
                "--run-name",
                args.supervision_run_name,
                "--include-models",
                args.include_models,
                "--n-splits",
                "5",
                "--n-repeats",
                "1",
                "--pseudo-cutoff-start",
                "2014-08",
                "--pseudo-cutoff-end",
                "2015-07",
            ],
            repo_root,
        )
        rolling_run_dir = latest_run(rolling_root, args.supervision_run_name)

    run_command(
        [
            python_executable,
            "rolling_panel_public_stability_lab.py",
            "--data-dir",
            ".",
            "--run-name",
            args.stability_run_name,
            "--rolling-run-dir",
            str(rolling_run_dir),
            "--pseudo-public-splits",
            str(args.pseudo_public_splits),
        ],
        repo_root,
    )
    stability_run_dir = latest_run(stability_root, args.stability_run_name)

    run_command(
        [
            python_executable,
            "rolling_panel_public_calibration_lab.py",
            "--data-dir",
            ".",
            "--run-name",
            args.calibration_run_name,
            "--rolling-run-dir",
            str(rolling_run_dir),
            "--pseudo-public-splits",
            str(args.pseudo_public_splits),
        ],
        repo_root,
    )
    calibration_run_dir = latest_run(calibration_root, args.calibration_run_name)

    if args.calibration_run_dir is not None:
        calibration_run_dir = args.calibration_run_dir.resolve()

    run_command(
        [
            python_executable,
            "rolling_panel_public_calibration_feedback_lab.py",
            "--data-dir",
            ".",
            "--run-name",
            args.feedback_run_name,
            "--calibration-run-dir",
            str(calibration_run_dir),
            "--pseudo-public-splits",
            str(args.pseudo_public_splits),
        ],
        repo_root,
    )
    feedback_run_dir = latest_run(feedback_root, args.feedback_run_name)

    summary_dir = repo_root / "outputs" / "dgx_search_runner" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    write_summary(summary_dir, rolling_run_dir, stability_run_dir, calibration_run_dir, feedback_run_dir)
    print(f"DGX search summary written to {summary_dir}")


if __name__ == "__main__":
    main()
