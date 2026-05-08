from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


FEATURE_PIPELINE_VERSION = "v5_dense_monthly_panel_stack"
RANDOM_STATE = 42
DEFAULT_SUBMISSION_FORMAT = "zindi_log"
PREDICTION_CUTOFF = datetime(2015, 10, 31, 0, 0, 0)
TOTAL_OBSERVED_MONTHS = 35
DENSE_PANEL_MONTHS = [(year, month) for year in (2014, 2015) for month in range(1, 13) if (year, month) <= (2015, 10)]
HOLIDAY_MONTHS = {
    "nov_2013": (2013, 11),
    "dec_2013": (2013, 12),
    "jan_2014": (2014, 1),
    "nov_2014": (2014, 11),
    "dec_2014": (2014, 12),
    "jan_2015": (2015, 1),
}
RECENT_MONTHS = {
    "aug_2015": (2015, 8),
    "sep_2015": (2015, 9),
    "oct_2015": (2015, 10),
}
TRANSACTION_TYPE_FEATURES = {
    "account_maintenance": "Account Maintenance",
    "card_transactions": "Card Transactions",
    "charges_fees": "Charges & Fees",
    "debit_orders_standing_orders": "Debit Orders & Standing Orders",
    "deposits": "Deposits",
    "foreign_exchange": "Foreign Exchange",
    "interest_investments": "Interest & Investments",
    "other_unclassified": "Other / Unclassified",
    "reversals_adjustments": "Reversals & Adjustments",
    "teller_branch_transactions": "Teller & Branch Transactions",
    "transfers_payments": "Transfers & Payments",
    "unpaid_returned_items": "Unpaid / Returned Items",
    "withdrawals": "Withdrawals",
}
TRANSACTION_BATCH_FEATURES = {
    "credit_debit_service": "Credit/Debit Service",
    "digital_banking_fees": "Digital Banking Fees",
    "not_disclosed_unknown": "Not Disclosed / Unknown",
    "other_unclassified": "Other / Unclassified",
    "other_charges": "Other Charges",
    "system_defined": "System Defined",
    "transaction_service_fees": "Transaction Service Fees",
    "unallocated": "Unallocated",
}
REVERSAL_TYPE_FEATURES = {
    "manual": "Manual",
    "not_applicable": "Not Applicable",
    "system": "Sytem",
}
TRANSACTION_BATCH_FOCUS = {
    "system_defined": "System Defined",
    "other_charges": "Other Charges",
    "not_disclosed_unknown": "Not Disclosed / Unknown",
    "digital_banking_fees": "Digital Banking Fees",
    "transaction_service_fees": "Transaction Service Fees",
    "credit_debit_service": "Credit/Debit Service",
    "unallocated": "Unallocated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a polars-first baseline for the Nedbank transaction forecasting challenge "
            "and write predictions in SampleSubmission format."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing Train.csv, Test.csv, and the feature parquet files.",
    )
    parser.add_argument(
        "--transactions-path",
        type=Path,
        default=None,
        help="Optional explicit path to transactions_features.parquet.",
    )
    parser.add_argument(
        "--financials-path",
        type=Path,
        default=None,
        help="Optional explicit path to financials_features.parquet.",
    )
    parser.add_argument(
        "--demographics-path",
        type=Path,
        default=None,
        help="Optional explicit path to demographics_clean.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/submission_baseline_polars.csv"),
        help=(
            "Base destination for the final submission file. The pipeline always appends a unique "
            "timestamp so repeated runs never overwrite prior outputs."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for an in-sample RMSLE estimate. Set to 0 or 1 to skip.",
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
        help=(
            "Submission export format. `zindi_log` writes np.log1p(prediction) as required by the competition. "
            "`raw` is only for offline analysis."
        ),
    )
    return parser.parse_args()


def iter_project_parquet_files(data_dir: Path) -> list[Path]:
    excluded_parts = {".git", ".venv", "__pycache__", "__MACOSX"}
    parquet_files: list[Path] = []

    for path in data_dir.rglob("*.parquet"):
        if any(part in excluded_parts for part in path.parts):
            continue
        parquet_files.append(path)

    return sorted(parquet_files)


def parquet_has_required_columns(path: Path, required_columns: set[str]) -> bool:
    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception:
        return False
    return required_columns.issubset(set(schema.names()))


def resolve_existing_path(
    explicit_path: Path | None,
    data_dir: Path,
    candidates: list[Path],
    label: str,
    required_columns: set[str],
    name_hints: tuple[str, ...],
    cli_arg_name: str,
) -> Path:
    checked_paths: list[Path] = []

    if explicit_path is not None:
        checked_paths.append(explicit_path)
        if explicit_path.exists():
            if explicit_path.is_dir():
                direct_dir_matches = sorted(explicit_path.rglob("*.parquet"))
                for match in direct_dir_matches:
                    checked_paths.append(match)
                    if parquet_has_required_columns(match, required_columns):
                        return match
            elif explicit_path.is_file() and parquet_has_required_columns(explicit_path, required_columns):
                return explicit_path

    for candidate in candidates:
        checked_paths.append(candidate)
        if candidate.exists() and parquet_has_required_columns(candidate, required_columns):
            return candidate

    discovered_parquets = iter_project_parquet_files(data_dir)
    name_filtered_matches = [
        path
        for path in discovered_parquets
        if any(hint in str(path).lower() for hint in name_hints) and parquet_has_required_columns(path, required_columns)
    ]
    if len(name_filtered_matches) == 1:
        return name_filtered_matches[0]
    if len(name_filtered_matches) > 1:
        matches = "\n".join(f"  - {path}" for path in name_filtered_matches)
        raise FileNotFoundError(f"Multiple possible matches found for {label}. Please pass --{cli_arg_name}.\n{matches}")

    schema_matches = [path for path in discovered_parquets if parquet_has_required_columns(path, required_columns)]
    if len(schema_matches) == 1:
        return schema_matches[0]
    if len(schema_matches) > 1:
        matches = "\n".join(f"  - {path}" for path in schema_matches)
        raise FileNotFoundError(
            f"Multiple parquet files match the required schema for {label}. Please pass --{cli_arg_name}.\n{matches}"
        )

    checked = "\n".join(f"  - {path}" for path in checked_paths)
    discovered = "\n".join(f"  - {path}" for path in discovered_parquets) or "  - <none>"
    raise FileNotFoundError(
        f"Could not locate {label}. Checked:\n{checked}\nDiscovered parquet files under {data_dir}:\n{discovered}"
    )


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clipped = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    truth = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(np.log1p(clipped) - np.log1p(truth)))))


def format_submission_predictions(
    predictions: np.ndarray,
    submission_format: str = DEFAULT_SUBMISSION_FORMAT,
) -> np.ndarray:
    raw_predictions = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, None)
    if submission_format == "raw":
        return raw_predictions
    if submission_format == "zindi_log":
        return np.log1p(raw_predictions)
    raise ValueError(f"Unsupported submission format: {submission_format}")


def build_submission_frame(
    sample_submission_path: Path,
    unique_ids: np.ndarray,
    predictions: np.ndarray,
    submission_format: str = DEFAULT_SUBMISSION_FORMAT,
) -> pl.DataFrame:
    formatted_predictions = format_submission_predictions(predictions, submission_format=submission_format)
    prediction_frame = pl.DataFrame(
        {
            "UniqueID": pl.Series("UniqueID", unique_ids),
            "next_3m_txn_count": pl.Series("next_3m_txn_count", formatted_predictions),
        }
    )
    sample_submission = pl.read_csv(sample_submission_path).with_row_index("__row__")
    final_submission = (
        sample_submission.drop("next_3m_txn_count")
        .join(prediction_frame, on="UniqueID", how="left")
        .sort("__row__")
        .drop("__row__")
    )

    if final_submission["next_3m_txn_count"].null_count() != 0:
        raise ValueError("Submission contains missing predictions after joining back to SampleSubmission.csv.")
    if (final_submission["next_3m_txn_count"] < 0).sum() != 0:
        raise ValueError("Submission contains negative predictions.")
    return final_submission


def build_versioned_output_path(requested_output_path: Path) -> Path:
    if requested_output_path.suffix:
        parent = requested_output_path.parent
        stem = requested_output_path.stem
        suffix = requested_output_path.suffix
    else:
        parent = requested_output_path
        stem = "submission_baseline_polars"
        suffix = ".csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_token = uuid4().hex[:8]
    candidate = parent / f"{stem}_{timestamp}_{run_token}{suffix}"
    collision_index = 1

    while candidate.exists():
        candidate = parent / f"{stem}_{timestamp}_{run_token}_{collision_index}{suffix}"
        collision_index += 1

    return candidate


def month_mask(year: int, month: int) -> pl.Expr:
    return (pl.col("txn_year") == year) & (pl.col("txn_month") == month)


def count_expr(mask: pl.Expr, alias: str) -> pl.Expr:
    return pl.when(mask).then(1).otherwise(0).sum().cast(pl.Int64).alias(alias)


def sum_expr(value_expr: pl.Expr, mask: pl.Expr, alias: str) -> pl.Expr:
    return pl.when(mask).then(value_expr).otherwise(0.0).sum().alias(alias)


def ratio_expr(numerator: str, denominator: str, alias: str, default: float = 0.0) -> pl.Expr:
    return (
        pl.when(pl.col(denominator).abs() > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(default)
        .alias(alias)
    )


def horizontal_sum_expr(columns: list[str], alias: str) -> pl.Expr:
    return pl.sum_horizontal([pl.col(column) for column in columns]).alias(alias)


def horizontal_mean_expr(columns: list[str], alias: str) -> pl.Expr:
    return (pl.sum_horizontal([pl.col(column) for column in columns]) / float(len(columns))).alias(alias)


def horizontal_std_expr(columns: list[str], alias: str) -> pl.Expr:
    return pl.concat_list([pl.col(column) for column in columns]).list.std().alias(alias)


def horizontal_max_expr(columns: list[str], alias: str) -> pl.Expr:
    return pl.max_horizontal([pl.col(column) for column in columns]).alias(alias)


def horizontal_min_expr(columns: list[str], alias: str) -> pl.Expr:
    return pl.min_horizontal([pl.col(column) for column in columns]).alias(alias)


def horizontal_active_count_expr(columns: list[str], alias: str) -> pl.Expr:
    return pl.sum_horizontal([(pl.col(column) > 0).cast(pl.Int64) for column in columns]).alias(alias)


def month_label(year: int, month: int) -> str:
    return f"{year}_{month:02d}"


def build_monthly_panel_features(customer_month: pl.LazyFrame, base_ids: pl.LazyFrame) -> pl.LazyFrame:
    panel_month_labels = [month_label(year, month) for year, month in DENSE_PANEL_MONTHS]
    panel_month_frame = (
        customer_month.filter(
            pl.any_horizontal(
                [month_mask(year, month) for year, month in DENSE_PANEL_MONTHS]
            )
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("txn_year").cast(pl.String),
                    pl.lit("_"),
                    pl.col("txn_month").cast(pl.String).str.zfill(2),
                ]
            ).alias("panel_month")
        )
        .collect()
    )
    base_ids_df = base_ids.collect()

    metric_specs = [
        ("month_txn_count", "txn_panel_count", pl.Int64),
        ("month_abs_sum", "txn_panel_abs_sum", pl.Float64),
        ("month_net_amount", "txn_panel_net_amount", pl.Float64),
        ("month_account_nunique", "txn_panel_account_nunique", pl.Int64),
        ("month_type_nunique", "txn_panel_type_nunique", pl.Int64),
        ("month_batch_nunique", "txn_panel_batch_nunique", pl.Int64),
    ]

    panel_frame: pl.DataFrame | None = None
    for value_column, feature_prefix, fill_dtype in metric_specs:
        pivot_frame = panel_month_frame.select(["UniqueID", "panel_month", value_column]).pivot(
            index="UniqueID",
            on="panel_month",
            values=value_column,
            aggregate_function="first",
        )
        if "UniqueID" not in pivot_frame.columns:
            pivot_frame = base_ids_df.clone()
        for label in panel_month_labels:
            if label not in pivot_frame.columns:
                pivot_frame = pivot_frame.with_columns(pl.lit(0).cast(fill_dtype).alias(label))
        pivot_frame = (
            base_ids_df.join(pivot_frame, on="UniqueID", how="left")
            .with_columns([pl.col(label).fill_null(0).cast(fill_dtype).alias(label) for label in panel_month_labels])
            .select(["UniqueID"] + panel_month_labels)
            .rename({label: f"{feature_prefix}_{label}" for label in panel_month_labels})
        )
        panel_frame = pivot_frame if panel_frame is None else panel_frame.join(pivot_frame, on="UniqueID", how="left")

    if panel_frame is None:
        return base_ids_df.lazy()

    count_cols = [f"txn_panel_count_{label}" for label in panel_month_labels]
    abs_cols = [f"txn_panel_abs_sum_{label}" for label in panel_month_labels]
    net_cols = [f"txn_panel_net_amount_{label}" for label in panel_month_labels]
    account_cols = [f"txn_panel_account_nunique_{label}" for label in panel_month_labels]
    type_cols = [f"txn_panel_type_nunique_{label}" for label in panel_month_labels]
    batch_cols = [f"txn_panel_batch_nunique_{label}" for label in panel_month_labels]

    last_3_count = count_cols[-3:]
    prev_3_count = count_cols[-6:-3]
    last_6_count = count_cols[-6:]
    prev_6_count = count_cols[-12:-6]
    last_12_count = count_cols[-12:]
    first_6_count = count_cols[:6]

    last_3_abs = abs_cols[-3:]
    prev_3_abs = abs_cols[-6:-3]
    last_6_abs = abs_cols[-6:]
    prev_6_abs = abs_cols[-12:-6]
    last_12_abs = abs_cols[-12:]

    last_3_net = net_cols[-3:]
    prev_3_net = net_cols[-6:-3]
    last_6_net = net_cols[-6:]
    prev_6_net = net_cols[-12:-6]
    last_12_net = net_cols[-12:]

    last_6_account = account_cols[-6:]
    last_12_account = account_cols[-12:]
    last_6_type = type_cols[-6:]
    last_12_type = type_cols[-12:]
    last_6_batch = batch_cols[-6:]
    last_12_batch = batch_cols[-12:]

    primary_derived = [
        horizontal_sum_expr(last_3_count, "txn_panel_count_last3_sum"),
        horizontal_sum_expr(prev_3_count, "txn_panel_count_prev3_sum"),
        horizontal_sum_expr(last_6_count, "txn_panel_count_last6_sum"),
        horizontal_sum_expr(prev_6_count, "txn_panel_count_prev6_sum"),
        horizontal_sum_expr(last_12_count, "txn_panel_count_last12_sum"),
        horizontal_mean_expr(last_3_count, "txn_panel_count_last3_mean"),
        horizontal_mean_expr(last_6_count, "txn_panel_count_last6_mean"),
        horizontal_mean_expr(last_12_count, "txn_panel_count_last12_mean"),
        horizontal_mean_expr(first_6_count, "txn_panel_count_first6_mean"),
        horizontal_std_expr(last_3_count, "txn_panel_count_last3_std"),
        horizontal_std_expr(last_6_count, "txn_panel_count_last6_std"),
        horizontal_std_expr(last_12_count, "txn_panel_count_last12_std"),
        horizontal_max_expr(last_6_count, "txn_panel_count_last6_max"),
        horizontal_max_expr(last_12_count, "txn_panel_count_last12_max"),
        horizontal_min_expr(last_6_count, "txn_panel_count_last6_min"),
        horizontal_active_count_expr(last_3_count, "txn_panel_count_last3_active_months"),
        horizontal_active_count_expr(last_6_count, "txn_panel_count_last6_active_months"),
        horizontal_active_count_expr(last_12_count, "txn_panel_count_last12_active_months"),
        horizontal_sum_expr(last_3_abs, "txn_panel_abs_last3_sum"),
        horizontal_sum_expr(prev_3_abs, "txn_panel_abs_prev3_sum"),
        horizontal_sum_expr(last_6_abs, "txn_panel_abs_last6_sum"),
        horizontal_sum_expr(prev_6_abs, "txn_panel_abs_prev6_sum"),
        horizontal_sum_expr(last_12_abs, "txn_panel_abs_last12_sum"),
        horizontal_mean_expr(last_3_abs, "txn_panel_abs_last3_mean"),
        horizontal_mean_expr(last_6_abs, "txn_panel_abs_last6_mean"),
        horizontal_mean_expr(last_12_abs, "txn_panel_abs_last12_mean"),
        horizontal_std_expr(last_6_abs, "txn_panel_abs_last6_std"),
        horizontal_std_expr(last_12_abs, "txn_panel_abs_last12_std"),
        horizontal_sum_expr(last_3_net, "txn_panel_net_last3_sum"),
        horizontal_sum_expr(prev_3_net, "txn_panel_net_prev3_sum"),
        horizontal_sum_expr(last_6_net, "txn_panel_net_last6_sum"),
        horizontal_sum_expr(prev_6_net, "txn_panel_net_prev6_sum"),
        horizontal_sum_expr(last_12_net, "txn_panel_net_last12_sum"),
        horizontal_mean_expr(last_3_net, "txn_panel_net_last3_mean"),
        horizontal_mean_expr(last_6_net, "txn_panel_net_last6_mean"),
        horizontal_mean_expr(last_12_net, "txn_panel_net_last12_mean"),
        horizontal_std_expr(last_6_net, "txn_panel_net_last6_std"),
        horizontal_std_expr(last_12_net, "txn_panel_net_last12_std"),
        horizontal_mean_expr(last_6_account, "txn_panel_account_last6_mean"),
        horizontal_mean_expr(last_12_account, "txn_panel_account_last12_mean"),
        horizontal_max_expr(last_6_account, "txn_panel_account_last6_max"),
        horizontal_max_expr(last_12_account, "txn_panel_account_last12_max"),
        horizontal_mean_expr(last_6_type, "txn_panel_type_last6_mean"),
        horizontal_mean_expr(last_12_type, "txn_panel_type_last12_mean"),
        horizontal_max_expr(last_6_type, "txn_panel_type_last6_max"),
        horizontal_max_expr(last_12_type, "txn_panel_type_last12_max"),
        horizontal_mean_expr(last_6_batch, "txn_panel_batch_last6_mean"),
        horizontal_mean_expr(last_12_batch, "txn_panel_batch_last12_mean"),
        horizontal_max_expr(last_6_batch, "txn_panel_batch_last6_max"),
        horizontal_max_expr(last_12_batch, "txn_panel_batch_last12_max"),
        (pl.col(count_cols[-1]) - pl.col(count_cols[-2])).alias("txn_panel_count_delta_oct_sep"),
        (pl.col(count_cols[-2]) - pl.col(count_cols[-3])).alias("txn_panel_count_delta_sep_aug"),
        (pl.col(abs_cols[-1]) - pl.col(abs_cols[-2])).alias("txn_panel_abs_delta_oct_sep"),
        (pl.col(abs_cols[-2]) - pl.col(abs_cols[-3])).alias("txn_panel_abs_delta_sep_aug"),
        (pl.col(net_cols[-1]) - pl.col(net_cols[-2])).alias("txn_panel_net_delta_oct_sep"),
        (pl.col(net_cols[-2]) - pl.col(net_cols[-3])).alias("txn_panel_net_delta_sep_aug"),
        (pl.col("txn_panel_count_2015_10") - pl.col("txn_panel_count_2014_10")).alias(
            "txn_panel_count_oct_yoy_delta"
        ),
        (pl.col("txn_panel_count_2015_09") - pl.col("txn_panel_count_2014_09")).alias(
            "txn_panel_count_sep_yoy_delta"
        ),
        (pl.col("txn_panel_count_2015_08") - pl.col("txn_panel_count_2014_08")).alias(
            "txn_panel_count_aug_yoy_delta"
        ),
        (pl.col("txn_panel_abs_sum_2015_10") - pl.col("txn_panel_abs_sum_2014_10")).alias(
            "txn_panel_abs_oct_yoy_delta"
        ),
        (pl.col("txn_panel_abs_sum_2015_09") - pl.col("txn_panel_abs_sum_2014_09")).alias(
            "txn_panel_abs_sep_yoy_delta"
        ),
        (pl.col("txn_panel_abs_sum_2015_08") - pl.col("txn_panel_abs_sum_2014_08")).alias(
            "txn_panel_abs_aug_yoy_delta"
        ),
    ]

    secondary_derived = [
        ratio_expr("txn_panel_count_last3_sum", "txn_panel_count_prev3_sum", "txn_panel_count_last3_vs_prev3_ratio"),
        ratio_expr("txn_panel_count_last6_sum", "txn_panel_count_prev6_sum", "txn_panel_count_last6_vs_prev6_ratio"),
        ratio_expr("txn_panel_count_last3_sum", "txn_panel_count_last12_sum", "txn_panel_count_last3_share_of_last12"),
        ratio_expr("txn_panel_abs_last3_sum", "txn_panel_abs_prev3_sum", "txn_panel_abs_last3_vs_prev3_ratio"),
        ratio_expr("txn_panel_abs_last6_sum", "txn_panel_abs_prev6_sum", "txn_panel_abs_last6_vs_prev6_ratio"),
        ratio_expr("txn_panel_net_last3_sum", "txn_panel_net_prev3_sum", "txn_panel_net_last3_vs_prev3_ratio"),
        ratio_expr("txn_panel_count_2015_10", "txn_panel_count_2014_10", "txn_panel_count_oct_yoy_ratio"),
        ratio_expr("txn_panel_count_2015_09", "txn_panel_count_2014_09", "txn_panel_count_sep_yoy_ratio"),
        ratio_expr("txn_panel_count_2015_08", "txn_panel_count_2014_08", "txn_panel_count_aug_yoy_ratio"),
        ratio_expr("txn_panel_abs_sum_2015_10", "txn_panel_abs_sum_2014_10", "txn_panel_abs_oct_yoy_ratio"),
        ratio_expr("txn_panel_abs_sum_2015_09", "txn_panel_abs_sum_2014_09", "txn_panel_abs_sep_yoy_ratio"),
        ratio_expr("txn_panel_abs_sum_2015_08", "txn_panel_abs_sum_2014_08", "txn_panel_abs_aug_yoy_ratio"),
        (pl.col("txn_panel_count_last3_mean") - pl.col("txn_panel_count_first6_mean")).alias(
            "txn_panel_count_recent_vs_early_mean_delta"
        ),
    ]

    derived = panel_frame.lazy().with_columns(primary_derived).with_columns(secondary_derived)

    return derived


def build_transaction_features(transactions_path: Path, base_ids: pl.LazyFrame) -> pl.LazyFrame:
    direction = pl.col("IsDebitCredit").cast(pl.String).str.to_uppercase().fill_null("UNKNOWN")
    absolute_amount = pl.col("TransactionAmount").abs()
    is_credit = direction.is_in(["C", "CREDIT"])
    is_debit = direction.is_in(["D", "DEBIT"])
    credit_amount = (
        pl.when(is_credit)
        .then(absolute_amount)
        .when((~is_credit) & (~is_debit) & (pl.col("TransactionAmount") < 0))
        .then(absolute_amount)
        .otherwise(0.0)
    )
    debit_amount = (
        pl.when(is_debit)
        .then(absolute_amount)
        .when((~is_credit) & (~is_debit) & (pl.col("TransactionAmount") > 0))
        .then(absolute_amount)
        .otherwise(0.0)
    )

    txn = (
        pl.scan_parquet(transactions_path)
        .join(base_ids, on="UniqueID", how="inner")
        .with_columns(
            [
                pl.col("UniqueID").cast(pl.String),
                pl.col("AccountID").cast(pl.String),
                pl.col("TransactionTypeDescription").fill_null("Unknown"),
                pl.col("TransactionBatchDescription").fill_null("Unknown"),
                pl.col("ReversalTypeDescription").fill_null("Unknown"),
                pl.col("TransactionDate").dt.year().alias("txn_year"),
                pl.col("TransactionDate").dt.month().alias("txn_month"),
                direction.alias("txn_direction"),
            ]
        )
    )

    holiday_2013_2014_mask = month_mask(2013, 11) | month_mask(2013, 12) | month_mask(2014, 1)
    holiday_2014_2015_mask = month_mask(2014, 11) | month_mask(2014, 12) | month_mask(2015, 1)
    recent_3m_mask = month_mask(2015, 8) | month_mask(2015, 9) | month_mask(2015, 10)
    prev_3m_mask = month_mask(2015, 5) | month_mask(2015, 6) | month_mask(2015, 7)

    aggregations: list[pl.Expr] = [
        pl.len().alias("txn_total_count"),
        pl.col("AccountID").n_unique().alias("txn_account_nunique"),
        pl.col("TransactionDate").n_unique().alias("txn_active_day_count"),
        credit_amount.sum().alias("txn_credit_sum"),
        debit_amount.sum().alias("txn_debit_sum"),
        pl.col("TransactionAmount").mean().alias("txn_amount_mean"),
        pl.col("TransactionAmount").std().alias("txn_amount_std"),
        pl.col("TransactionAmount").min().alias("txn_amount_min"),
        pl.col("TransactionAmount").max().alias("txn_amount_max"),
        absolute_amount.mean().alias("txn_amount_abs_mean"),
        absolute_amount.max().alias("txn_amount_abs_max"),
        pl.col("StatementBalance").mean().alias("txn_statement_balance_mean"),
        pl.col("StatementBalance").std().alias("txn_statement_balance_std"),
        pl.col("StatementBalance").min().alias("txn_statement_balance_min"),
        pl.col("StatementBalance").max().alias("txn_statement_balance_max"),
        pl.col("TransactionTypeDescription").n_unique().alias("txn_type_nunique"),
        pl.col("TransactionBatchDescription").n_unique().alias("txn_batch_nunique"),
        pl.col("ReversalTypeDescription").n_unique().alias("txn_reversal_type_nunique"),
        (
            (pl.lit(PREDICTION_CUTOFF) - pl.col("TransactionDate").max())
            .dt.total_days()
            .alias("txn_days_since_last")
        ),
        count_expr(holiday_2013_2014_mask, "txn_holiday_2013_2014_count"),
        sum_expr(credit_amount, holiday_2013_2014_mask, "txn_holiday_2013_2014_credit_sum"),
        sum_expr(debit_amount, holiday_2013_2014_mask, "txn_holiday_2013_2014_debit_sum"),
        count_expr(holiday_2014_2015_mask, "txn_holiday_2014_2015_count"),
        sum_expr(credit_amount, holiday_2014_2015_mask, "txn_holiday_2014_2015_credit_sum"),
        sum_expr(debit_amount, holiday_2014_2015_mask, "txn_holiday_2014_2015_debit_sum"),
        count_expr(recent_3m_mask, "txn_recent_3m_count"),
        sum_expr(credit_amount, recent_3m_mask, "txn_recent_3m_credit_sum"),
        sum_expr(debit_amount, recent_3m_mask, "txn_recent_3m_debit_sum"),
        count_expr(prev_3m_mask, "txn_prev_3m_count"),
        sum_expr(credit_amount, prev_3m_mask, "txn_prev_3m_credit_sum"),
        sum_expr(debit_amount, prev_3m_mask, "txn_prev_3m_debit_sum"),
        pl.col("TransactionAmount")
        .filter(recent_3m_mask)
        .std()
        .alias("txn_recent_3m_amount_std"),
    ]

    for period_name, (year, month) in HOLIDAY_MONTHS.items():
        mask = month_mask(year, month)
        aggregations.extend(
            [
                count_expr(mask, f"txn_{period_name}_count"),
                sum_expr(credit_amount, mask, f"txn_{period_name}_credit_sum"),
                sum_expr(debit_amount, mask, f"txn_{period_name}_debit_sum"),
            ]
        )

    for period_name, (year, month) in RECENT_MONTHS.items():
        mask = month_mask(year, month)
        aggregations.extend(
            [
                count_expr(mask, f"txn_{period_name}_count"),
                sum_expr(credit_amount, mask, f"txn_{period_name}_credit_sum"),
                sum_expr(debit_amount, mask, f"txn_{period_name}_debit_sum"),
            ]
        )

    for feature_alias, feature_name in TRANSACTION_TYPE_FEATURES.items():
        feature_mask = pl.col("TransactionTypeDescription") == feature_name
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_type_{feature_alias}_count"),
                sum_expr(absolute_amount, feature_mask, f"txn_type_{feature_alias}_abs_sum"),
                count_expr(feature_mask & recent_3m_mask, f"txn_recent_type_{feature_alias}_count"),
            ]
        )

    for feature_alias, feature_name in TRANSACTION_BATCH_FEATURES.items():
        feature_mask = pl.col("TransactionBatchDescription") == feature_name
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_batch_{feature_alias}_count"),
                sum_expr(absolute_amount, feature_mask, f"txn_batch_{feature_alias}_abs_sum"),
                count_expr(feature_mask & recent_3m_mask, f"txn_recent_batch_{feature_alias}_count"),
            ]
        )

    for feature_alias, feature_name in REVERSAL_TYPE_FEATURES.items():
        feature_mask = pl.col("ReversalTypeDescription") == feature_name
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_reversal_{feature_alias}_count"),
                count_expr(feature_mask & recent_3m_mask, f"txn_recent_reversal_{feature_alias}_count"),
            ]
        )

    customer_month = (
        txn.group_by(["UniqueID", "txn_year", "txn_month"])
        .agg(
            [
                pl.len().alias("month_txn_count"),
                credit_amount.sum().alias("month_credit_sum"),
                debit_amount.sum().alias("month_debit_sum"),
                absolute_amount.sum().alias("month_abs_sum"),
                pl.col("TransactionAmount").sum().alias("month_net_amount"),
                pl.col("AccountID").n_unique().alias("month_account_nunique"),
                pl.col("TransactionTypeDescription").n_unique().alias("month_type_nunique"),
                pl.col("TransactionBatchDescription").n_unique().alias("month_batch_nunique"),
            ]
        )
    )

    monthly_features = customer_month.group_by("UniqueID").agg(
        [
            pl.len().alias("txn_active_months_total"),
            (pl.col("month_txn_count") == 1).sum().cast(pl.Int64).alias("txn_months_count_eq_1"),
            (pl.col("month_txn_count") <= 1).sum().cast(pl.Int64).alias("txn_months_count_le_1"),
            (pl.col("month_txn_count") <= 2).sum().cast(pl.Int64).alias("txn_months_count_le_2"),
            (pl.col("month_txn_count") <= 5).sum().cast(pl.Int64).alias("txn_months_count_le_5"),
            pl.col("month_txn_count").mean().alias("txn_monthly_count_mean"),
            pl.col("month_txn_count").std().alias("txn_monthly_count_std"),
            pl.col("month_txn_count").max().alias("txn_monthly_count_max"),
            pl.col("month_txn_count").min().alias("txn_monthly_count_min"),
            pl.col("month_abs_sum").mean().alias("txn_monthly_abs_sum_mean"),
            pl.col("month_abs_sum").std().alias("txn_monthly_abs_sum_std"),
            pl.col("month_abs_sum").max().alias("txn_monthly_abs_sum_max"),
            pl.col("month_net_amount").mean().alias("txn_monthly_net_amount_mean"),
            pl.col("month_net_amount").std().alias("txn_monthly_net_amount_std"),
            pl.col("month_account_nunique").mean().alias("txn_monthly_account_nunique_mean"),
            pl.col("month_account_nunique").max().alias("txn_monthly_account_nunique_max"),
            pl.col("month_type_nunique").mean().alias("txn_monthly_type_nunique_mean"),
            pl.col("month_batch_nunique").mean().alias("txn_monthly_batch_nunique_mean"),
        ]
    )
    monthly_panel_features = build_monthly_panel_features(customer_month, base_ids)

    account_features = (
        txn.group_by(["UniqueID", "AccountID"])
        .agg(
            [
                pl.len().alias("account_txn_count"),
                credit_amount.sum().alias("account_credit_sum"),
                debit_amount.sum().alias("account_debit_sum"),
                absolute_amount.sum().alias("account_abs_sum"),
                count_expr(recent_3m_mask, "account_recent_3m_count"),
            ]
        )
        .group_by("UniqueID")
        .agg(
            [
                pl.col("account_txn_count").mean().alias("txn_account_txn_mean"),
                pl.col("account_txn_count").std().alias("txn_account_txn_std"),
                pl.col("account_txn_count").max().alias("txn_account_txn_max"),
                pl.col("account_abs_sum").mean().alias("txn_account_abs_sum_mean"),
                pl.col("account_abs_sum").max().alias("txn_account_abs_sum_max"),
                (pl.col("account_recent_3m_count") > 0)
                .sum()
                .cast(pl.Int64)
                .alias("txn_recent_active_account_count"),
            ]
        )
    )

    return (
        txn.group_by("UniqueID")
        .agg(aggregations)
        .join(monthly_features, on="UniqueID", how="left")
        .join(monthly_panel_features, on="UniqueID", how="left")
        .join(account_features, on="UniqueID", how="left")
        .with_columns(
            [
                (pl.lit(TOTAL_OBSERVED_MONTHS) - pl.col("txn_active_months_total")).alias("txn_inactive_months_total"),
                ratio_expr("txn_active_months_total", "txn_total_count", "txn_active_month_to_txn_ratio"),
                ratio_expr("txn_months_count_eq_1", "txn_active_months_total", "txn_sparse_month_share_eq_1"),
                ratio_expr("txn_months_count_le_2", "txn_active_months_total", "txn_sparse_month_share_le_2"),
                ratio_expr("txn_recent_3m_count", "txn_total_count", "txn_recent_share_count"),
                ratio_expr("txn_recent_3m_credit_sum", "txn_credit_sum", "txn_recent_share_credit"),
                ratio_expr("txn_recent_3m_debit_sum", "txn_debit_sum", "txn_recent_share_debit"),
                ratio_expr("txn_recent_3m_count", "txn_prev_3m_count", "txn_recent_vs_prev3_count_ratio"),
                ratio_expr(
                    "txn_recent_3m_credit_sum",
                    "txn_prev_3m_credit_sum",
                    "txn_recent_vs_prev3_credit_ratio",
                ),
                ratio_expr(
                    "txn_recent_3m_debit_sum",
                    "txn_prev_3m_debit_sum",
                    "txn_recent_vs_prev3_debit_ratio",
                ),
                ratio_expr(
                    "txn_holiday_2014_2015_count",
                    "txn_holiday_2013_2014_count",
                    "txn_holiday_count_yoy_ratio",
                ),
                ratio_expr("txn_credit_sum", "txn_debit_sum", "txn_credit_debit_ratio"),
                ratio_expr("txn_account_txn_max", "txn_total_count", "txn_account_concentration_ratio"),
                (pl.col("txn_recent_3m_count") - pl.col("txn_prev_3m_count")).alias("txn_recent_vs_prev3_count_delta"),
                (pl.col("txn_recent_3m_credit_sum") - pl.col("txn_prev_3m_credit_sum")).alias(
                    "txn_recent_vs_prev3_credit_delta"
                ),
                (pl.col("txn_recent_3m_debit_sum") - pl.col("txn_prev_3m_debit_sum")).alias(
                    "txn_recent_vs_prev3_debit_delta"
                ),
                (pl.col("txn_oct_2015_count") - pl.col("txn_sep_2015_count")).alias("txn_recent_count_delta_oct_sep"),
                (pl.col("txn_sep_2015_count") - pl.col("txn_aug_2015_count")).alias("txn_recent_count_delta_sep_aug"),
                (pl.col("txn_oct_2015_credit_sum") - pl.col("txn_sep_2015_credit_sum")).alias(
                    "txn_recent_credit_delta_oct_sep"
                ),
                (pl.col("txn_oct_2015_debit_sum") - pl.col("txn_sep_2015_debit_sum")).alias(
                    "txn_recent_debit_delta_oct_sep"
                ),
                (pl.col("txn_holiday_2014_2015_count") - pl.col("txn_holiday_2013_2014_count")).alias(
                    "txn_holiday_count_yoy_delta"
                ),
                (
                    pl.col("txn_statement_balance_std")
                    / (pl.col("txn_statement_balance_mean").abs() + pl.lit(1.0))
                ).alias("txn_statement_balance_cv"),
            ]
        )
    )


def build_financial_features(financials_path: Path, base_ids: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
    financials = (
        pl.scan_parquet(financials_path)
        .join(base_ids, on="UniqueID", how="inner")
        .with_columns([pl.col("UniqueID").cast(pl.String), pl.col("Product").fill_null("Unknown")])
    )

    numeric_feature_names = [
        "fin_record_count",
        "fin_snapshot_count",
        "fin_product_nunique",
        "fin_nii_sum",
        "fin_nii_mean",
        "fin_nii_std",
        "fin_nii_min",
        "fin_nii_max",
        "fin_nir_sum",
        "fin_nir_mean",
        "fin_nir_std",
        "fin_nir_min",
        "fin_nir_max",
        "fin_last_snapshot_days_ago",
        "fin_transactional_count",
        "fin_transactional_nii_sum",
        "fin_transactional_nir_sum",
        "fin_investments_count",
        "fin_investments_nii_sum",
        "fin_investments_nir_sum",
        "fin_mortgages_count",
        "fin_mortgages_nii_sum",
        "fin_mortgages_nir_sum",
    ]

    features = financials.group_by("UniqueID").agg(
        [
            pl.len().alias("fin_record_count"),
            pl.col("RunDate").n_unique().alias("fin_snapshot_count"),
            pl.col("Product").n_unique().alias("fin_product_nunique"),
            pl.col("NetInterestIncome").sum().alias("fin_nii_sum"),
            pl.col("NetInterestIncome").mean().alias("fin_nii_mean"),
            pl.col("NetInterestIncome").std().alias("fin_nii_std"),
            pl.col("NetInterestIncome").min().alias("fin_nii_min"),
            pl.col("NetInterestIncome").max().alias("fin_nii_max"),
            pl.col("NetInterestRevenue").sum().alias("fin_nir_sum"),
            pl.col("NetInterestRevenue").mean().alias("fin_nir_mean"),
            pl.col("NetInterestRevenue").std().alias("fin_nir_std"),
            pl.col("NetInterestRevenue").min().alias("fin_nir_min"),
            pl.col("NetInterestRevenue").max().alias("fin_nir_max"),
            (
                (pl.lit(PREDICTION_CUTOFF) - pl.col("RunDate").max())
                .dt.total_days()
                .alias("fin_last_snapshot_days_ago")
            ),
            count_expr(pl.col("Product") == "Transactional", "fin_transactional_count"),
            sum_expr(pl.col("NetInterestIncome"), pl.col("Product") == "Transactional", "fin_transactional_nii_sum"),
            sum_expr(pl.col("NetInterestRevenue"), pl.col("Product") == "Transactional", "fin_transactional_nir_sum"),
            count_expr(pl.col("Product") == "Investments", "fin_investments_count"),
            sum_expr(pl.col("NetInterestIncome"), pl.col("Product") == "Investments", "fin_investments_nii_sum"),
            sum_expr(pl.col("NetInterestRevenue"), pl.col("Product") == "Investments", "fin_investments_nir_sum"),
            count_expr(pl.col("Product") == "Mortgages", "fin_mortgages_count"),
            sum_expr(pl.col("NetInterestIncome"), pl.col("Product") == "Mortgages", "fin_mortgages_nii_sum"),
            sum_expr(pl.col("NetInterestRevenue"), pl.col("Product") == "Mortgages", "fin_mortgages_nir_sum"),
        ]
    )

    return features, numeric_feature_names


def build_demographic_features(demographics_path: Path, base_ids: pl.LazyFrame) -> pl.LazyFrame:
    age_years = ((pl.lit(PREDICTION_CUTOFF) - pl.col("BirthDate")).dt.total_days()).truediv(365.25)

    return (
        pl.scan_parquet(demographics_path)
        .join(base_ids, on="UniqueID", how="inner")
        .with_columns([pl.col("UniqueID").cast(pl.String)])
        .unique(subset=["UniqueID"], keep="first")
        .select(
            [
                "UniqueID",
                pl.when(age_years.is_between(10, 110, closed="both"))
                .then(age_years)
                .otherwise(None)
                .alias("demo_age_years"),
                pl.col("AnnualGrossIncome").alias("demo_annual_gross_income"),
                pl.when(pl.col("AnnualGrossIncome").is_not_null())
                .then(pl.col("AnnualGrossIncome").clip(lower_bound=0.0).log1p())
                .otherwise(None)
                .alias("demo_annual_gross_income_log1p"),
                pl.col("BirthDate").is_null().cast(pl.Int8).alias("demo_birthdate_missing"),
                pl.col("AnnualGrossIncome").is_null().cast(pl.Int8).alias("demo_income_missing"),
                pl.col("CustomerBankingType").is_null().cast(pl.Int8).alias("demo_banking_type_missing"),
                pl.col("Gender").alias("demo_gender"),
                pl.col("IncomeCategory").alias("demo_income_category"),
                pl.col("CustomerStatus").alias("demo_customer_status"),
                pl.col("ClientType").alias("demo_client_type"),
                pl.col("MaritalStatus").alias("demo_marital_status"),
                pl.col("OccupationCategory").alias("demo_occupation_category"),
                pl.col("IndustryCategory").alias("demo_industry_category"),
                pl.col("CustomerBankingType").alias("demo_customer_banking_type"),
                pl.col("CustomerOnboardingChannel").alias("demo_onboarding_channel"),
                pl.col("ResidentialCityName").alias("demo_residential_city"),
                pl.col("CountryCodeNationality").alias("demo_country_code_nationality"),
                pl.col("LowIncomeFlag").alias("demo_low_income_flag"),
                pl.col("CertificationTypeDescription").alias("demo_certification_type"),
                pl.col("ContactPreference").alias("demo_contact_preference"),
            ]
        )
    )


def build_feature_table(
    train_path: Path,
    test_path: Path,
    transactions_path: Path,
    financials_path: Path,
    demographics_path: Path,
) -> pl.DataFrame:
    train_df = pl.read_csv(train_path).with_columns(
        [pl.col("UniqueID").cast(pl.String), pl.col("next_3m_txn_count").cast(pl.Float64)]
    )
    test_df = pl.read_csv(test_path).with_columns([pl.col("UniqueID").cast(pl.String)])

    if train_df["UniqueID"].n_unique() != train_df.height:
        raise ValueError("Train.csv contains duplicate UniqueID values.")
    if test_df["UniqueID"].n_unique() != test_df.height:
        raise ValueError("Test.csv contains duplicate UniqueID values.")

    base_df = pl.concat(
        [
            train_df.with_columns(pl.lit("train").alias("__split__")),
            test_df.with_columns(
                [pl.lit(None).cast(pl.Float64).alias("next_3m_txn_count"), pl.lit("test").alias("__split__")]
            ),
        ],
        how="diagonal_relaxed",
    )
    base_ids = base_df.lazy().select("UniqueID").unique()

    txn_features = build_transaction_features(transactions_path, base_ids)
    fin_features, fin_numeric_cols = build_financial_features(financials_path, base_ids)
    demo_features = build_demographic_features(demographics_path, base_ids)

    full_feature_table = (
        base_df.lazy()
        .join(txn_features, on="UniqueID", how="left")
        .join(fin_features, on="UniqueID", how="left")
        .join(demo_features, on="UniqueID", how="left")
        .with_columns(
            [pl.col("fin_record_count").is_null().cast(pl.Int8).alias("fin_missing_flag")]
            + [pl.col(column).fill_null(-999.0).alias(column) for column in fin_numeric_cols]
        )
        .collect()
    )

    expected_rows = train_df.height + test_df.height
    if full_feature_table.height != expected_rows:
        raise ValueError(
            f"Feature assembly changed the row count. Expected {expected_rows}, got {full_feature_table.height}."
        )

    return full_feature_table


def build_model() -> TransformedTargetRegressor:
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float64,
    )

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.015,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.3,
        max_iter=900,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("numeric", "passthrough", make_numeric_selector),
                        ("categorical", categorical_encoder, make_categorical_selector),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("regressor", regressor),
        ]
    )

    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def make_numeric_selector(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def make_categorical_selector(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


def fit_predict_and_save(
    feature_table: pl.DataFrame,
    sample_submission_path: Path,
    output_path: Path,
    cv_folds: int,
    submission_format: str,
) -> None:
    resolved_output_path = build_versioned_output_path(output_path)

    train_features = feature_table.filter(pl.col("__split__") == "train")
    test_features = feature_table.filter(pl.col("__split__") == "test")

    train_pd = train_features.drop(["__split__"]).to_pandas()
    test_pd = test_features.drop(["__split__", "next_3m_txn_count"]).to_pandas()

    y_train = train_pd.pop("next_3m_txn_count").to_numpy(dtype=np.float64)
    train_pd.pop("UniqueID")
    test_ids = test_pd.pop("UniqueID").to_numpy()

    model = build_model()

    if cv_folds and cv_folds > 1:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            model,
            train_pd,
            y_train,
            cv=cv,
            scoring=make_scorer(rmsle, greater_is_better=False),
            n_jobs=1,
        )
        print(f"CV RMSLE mean: {-scores.mean():.6f}")
        print(f"CV RMSLE std: {scores.std():.6f}")

    model.fit(train_pd, y_train)
    predictions = np.clip(model.predict(test_pd), 0.0, None)

    final_submission = build_submission_frame(
        sample_submission_path=sample_submission_path,
        unique_ids=test_ids,
        predictions=predictions,
        submission_format=submission_format,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_submission.write_csv(resolved_output_path)
    print(f"Saved {submission_format} submission to {resolved_output_path}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    train_path = data_dir / "Train.csv"
    test_path = data_dir / "Test.csv"
    sample_submission_path = data_dir / "SampleSubmission.csv"

    transactions_path = resolve_existing_path(
        args.transactions_path,
        data_dir,
        [
            data_dir / "transactions_features.parquet",
            data_dir / "transactions_features" / "transactions_features.parquet",
        ],
        "transactions_features.parquet",
        {"UniqueID", "TransactionDate", "TransactionAmount"},
        ("transactions", "transaction", "txn"),
        "transactions-path",
    )
    financials_path = resolve_existing_path(
        args.financials_path,
        data_dir,
        [
            data_dir / "financials_features.parquet",
            data_dir / "financials_features" / "financials_features.parquet",
        ],
        "financials_features.parquet",
        {"UniqueID", "RunDate", "NetInterestIncome", "NetInterestRevenue"},
        ("financials", "financial"),
        "financials-path",
    )
    demographics_path = resolve_existing_path(
        args.demographics_path,
        data_dir,
        [
            data_dir / "demographics_clean.parquet",
            data_dir / "demographics_clean" / "demographics_clean.parquet",
        ],
        "demographics_clean.parquet",
        {"UniqueID", "BirthDate", "AnnualGrossIncome"},
        ("demographics", "demo"),
        "demographics-path",
    )

    feature_table = build_feature_table(
        train_path=train_path,
        test_path=test_path,
        transactions_path=transactions_path,
        financials_path=financials_path,
        demographics_path=demographics_path,
    )

    fit_predict_and_save(
        feature_table=feature_table,
        sample_submission_path=sample_submission_path,
        output_path=args.output_path.resolve(),
        cv_folds=args.cv_folds,
        submission_format=args.submission_format,
    )


if __name__ == "__main__":
    main()
