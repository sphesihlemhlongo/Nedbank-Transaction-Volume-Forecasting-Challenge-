from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RepeatedKFold

from baseline_polars_pipeline import (
    DEFAULT_SUBMISSION_FORMAT,
    build_submission_frame,
    resolve_existing_path,
    rmsle,
)
from experiment_harness import (
    BlendCandidate,
    DatasetPaths,
    ModelResult,
    ModelSpec,
    build_blend_candidates,
    build_estimator,
    build_model_registry,
    build_run_directory,
    evaluate_blend_candidates,
    filter_model_registry,
)


FEATURE_PIPELINE_VERSION = "v6_temporal_holiday_augmentation"
RANDOM_STATE = 42
DATA_START_YEAR = 2012
DATA_START_MONTH = 12
TRANSACTION_TYPE_FOCUS = {
    "transfers_payments": "Transfers & Payments",
    "charges_fees": "Charges & Fees",
    "interest_investments": "Interest & Investments",
    "debit_orders_standing_orders": "Debit Orders & Standing Orders",
    "other_unclassified": "Other / Unclassified",
    "withdrawals": "Withdrawals",
    "foreign_exchange": "Foreign Exchange",
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
REVERSAL_TYPE_FOCUS = {
    "manual": "Manual",
    "system": "Sytem",
    "not_applicable": "Not Applicable",
}
DEFAULT_INCLUDE_MODELS = "hgb_conservative_v3,xgb_conservative_v3,catboost_conservative_v1,catboost_conservative_v3"


@dataclass(frozen=True)
class CutoffConfig:
    name: str
    cutoff_year: int
    cutoff_month: int
    cutoff_date: datetime
    target_months: list[tuple[int, int]]
    recent_months: dict[str, tuple[int, int]]
    prev_3m_months: list[tuple[int, int]]
    holiday_prev1_months: dict[str, tuple[int, int]]
    holiday_prev2_months: dict[str, tuple[int, int]]
    observed_months_total: int
    sample_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a holiday-aware temporal augmentation stack for the Nedbank challenge with leak-safe CV, "
            "repeatable artifacts, and zindi_log submissions by default."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--transactions-path", type=Path, default=None)
    parser.add_argument("--financials-path", type=Path, default=None)
    parser.add_argument("--demographics-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temporal_holiday_stack"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/temporal_holiday_stack"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--top-k-blend-models", type=int, default=4)
    parser.add_argument("--equal-blend-search-models", type=int, default=8)
    parser.add_argument("--max-equal-blend-size", type=int, default=5)
    parser.add_argument("--include-models", type=str, default=DEFAULT_INCLUDE_MODELS)
    parser.add_argument("--run-name", type=str, default="temporal_holiday_stack")
    parser.add_argument("--current-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-2014-weight", type=float, default=0.75)
    parser.add_argument("--pseudo-2013-weight", type=float, default=0.50)
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
    )
    parser.add_argument(
        "--seed-artifact-dir",
        type=Path,
        default=Path("outputs/experiments/v4_full_stack_20260428_180309_465571_b7834006"),
        help="Optional prior run directory used to seed blend search with the proven v4 subset candidate.",
    )
    parser.add_argument("--disable-seed-candidates", action="store_true")
    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> DatasetPaths:
    data_dir = args.data_dir.resolve()
    return DatasetPaths(
        train_path=data_dir / "Train.csv",
        test_path=data_dir / "Test.csv",
        sample_submission_path=data_dir / "SampleSubmission.csv",
        transactions_path=resolve_existing_path(
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
        ),
        financials_path=resolve_existing_path(
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
        ),
        demographics_path=resolve_existing_path(
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
        ),
    )


def add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total_month_index = (year * 12 + (month - 1)) + delta
    shifted_year, shifted_month_zero = divmod(total_month_index, 12)
    return shifted_year, shifted_month_zero + 1


def months_between_inclusive(start_year: int, start_month: int, end_year: int, end_month: int) -> int:
    return (end_year - start_year) * 12 + (end_month - start_month) + 1


def build_cutoff_config(name: str, cutoff_year: int, sample_weight: float) -> CutoffConfig:
    cutoff_month = 10
    cutoff_date = datetime(cutoff_year, cutoff_month, 31, 0, 0, 0)
    recent_months = {
        "recent_m3": add_months(cutoff_year, cutoff_month, -2),
        "recent_m2": add_months(cutoff_year, cutoff_month, -1),
        "recent_m1": (cutoff_year, cutoff_month),
    }
    prev_3m_months = [add_months(cutoff_year, cutoff_month, delta) for delta in (-5, -4, -3)]
    holiday_prev1_months = (
        {
            "holiday_prev1_nov": (cutoff_year - 1, 11),
            "holiday_prev1_dec": (cutoff_year - 1, 12),
            "holiday_prev1_jan": (cutoff_year, 1),
        }
        if cutoff_year >= 2014
        else {}
    )
    holiday_prev2_months = (
        {
            "holiday_prev2_nov": (cutoff_year - 2, 11),
            "holiday_prev2_dec": (cutoff_year - 2, 12),
            "holiday_prev2_jan": (cutoff_year - 1, 1),
        }
        if cutoff_year >= 2015
        else {}
    )
    return CutoffConfig(
        name=name,
        cutoff_year=cutoff_year,
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
        target_months=[(cutoff_year, 11), (cutoff_year, 12), (cutoff_year + 1, 1)],
        recent_months=recent_months,
        prev_3m_months=prev_3m_months,
        holiday_prev1_months=holiday_prev1_months,
        holiday_prev2_months=holiday_prev2_months,
        observed_months_total=months_between_inclusive(
            DATA_START_YEAR,
            DATA_START_MONTH,
            cutoff_year,
            cutoff_month,
        ),
        sample_weight=sample_weight,
    )


def month_mask(year: int, month: int) -> pl.Expr:
    return (pl.col("txn_year") == year) & (pl.col("txn_month") == month)


def months_mask(month_pairs: list[tuple[int, int]]) -> pl.Expr:
    if not month_pairs:
        return pl.lit(False)

    mask = month_mask(month_pairs[0][0], month_pairs[0][1])
    for year, month in month_pairs[1:]:
        mask = mask | month_mask(year, month)
    return mask


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


def build_customer_feature_matrix(
    dataset_paths: DatasetPaths,
    base_ids: pl.LazyFrame,
    config: CutoffConfig,
) -> pl.DataFrame:
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

    recent_3m_mask = months_mask(list(config.recent_months.values()))
    prev_3m_mask = months_mask(config.prev_3m_months)
    holiday_prev1_mask = months_mask(list(config.holiday_prev1_months.values()))
    holiday_prev2_mask = months_mask(list(config.holiday_prev2_months.values()))

    txn = (
        pl.scan_parquet(dataset_paths.transactions_path)
        .join(base_ids, on="UniqueID", how="inner")
        .filter(pl.col("TransactionDate") <= pl.lit(config.cutoff_date))
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
        ((pl.lit(config.cutoff_date) - pl.col("TransactionDate").max()).dt.total_days()).alias("txn_days_since_last"),
        count_expr(holiday_prev2_mask, "txn_holiday_prev2_count"),
        sum_expr(credit_amount, holiday_prev2_mask, "txn_holiday_prev2_credit_sum"),
        sum_expr(debit_amount, holiday_prev2_mask, "txn_holiday_prev2_debit_sum"),
        count_expr(holiday_prev1_mask, "txn_holiday_prev1_count"),
        sum_expr(credit_amount, holiday_prev1_mask, "txn_holiday_prev1_credit_sum"),
        sum_expr(debit_amount, holiday_prev1_mask, "txn_holiday_prev1_debit_sum"),
        count_expr(recent_3m_mask, "txn_recent_3m_count"),
        sum_expr(credit_amount, recent_3m_mask, "txn_recent_3m_credit_sum"),
        sum_expr(debit_amount, recent_3m_mask, "txn_recent_3m_debit_sum"),
        count_expr(prev_3m_mask, "txn_prev_3m_count"),
        sum_expr(credit_amount, prev_3m_mask, "txn_prev_3m_credit_sum"),
        sum_expr(debit_amount, prev_3m_mask, "txn_prev_3m_debit_sum"),
        pl.col("TransactionAmount").filter(recent_3m_mask).std().alias("txn_recent_3m_amount_std"),
    ]

    for feature_name, (year, month) in config.holiday_prev2_months.items():
        feature_mask = month_mask(year, month)
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_{feature_name}_count"),
                sum_expr(credit_amount, feature_mask, f"txn_{feature_name}_credit_sum"),
                sum_expr(debit_amount, feature_mask, f"txn_{feature_name}_debit_sum"),
            ]
        )

    for feature_name, (year, month) in config.holiday_prev1_months.items():
        feature_mask = month_mask(year, month)
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_{feature_name}_count"),
                sum_expr(credit_amount, feature_mask, f"txn_{feature_name}_credit_sum"),
                sum_expr(debit_amount, feature_mask, f"txn_{feature_name}_debit_sum"),
            ]
        )

    for feature_name, (year, month) in config.recent_months.items():
        feature_mask = month_mask(year, month)
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_{feature_name}_count"),
                sum_expr(credit_amount, feature_mask, f"txn_{feature_name}_credit_sum"),
                sum_expr(debit_amount, feature_mask, f"txn_{feature_name}_debit_sum"),
            ]
        )

    for feature_alias, feature_name in TRANSACTION_TYPE_FOCUS.items():
        feature_mask = pl.col("TransactionTypeDescription") == feature_name
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_type_{feature_alias}_count"),
                sum_expr(absolute_amount, feature_mask, f"txn_type_{feature_alias}_abs_sum"),
                count_expr(feature_mask & recent_3m_mask, f"txn_recent_type_{feature_alias}_count"),
            ]
        )

    for feature_alias, feature_name in TRANSACTION_BATCH_FOCUS.items():
        feature_mask = pl.col("TransactionBatchDescription") == feature_name
        aggregations.extend(
            [
                count_expr(feature_mask, f"txn_batch_{feature_alias}_count"),
                sum_expr(absolute_amount, feature_mask, f"txn_batch_{feature_alias}_abs_sum"),
                count_expr(feature_mask & recent_3m_mask, f"txn_recent_batch_{feature_alias}_count"),
            ]
        )

    for feature_alias, feature_name in REVERSAL_TYPE_FOCUS.items():
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
                (pl.col("account_recent_3m_count") > 0).sum().cast(pl.Int64).alias("txn_recent_active_account_count"),
            ]
        )
    )

    financials = (
        pl.scan_parquet(dataset_paths.financials_path)
        .join(base_ids, on="UniqueID", how="inner")
        .filter(pl.col("RunDate") <= pl.lit(config.cutoff_date))
        .with_columns([pl.col("UniqueID").cast(pl.String), pl.col("Product").fill_null("Unknown")])
    )
    fin_numeric_cols = [
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
    fin_features = financials.group_by("UniqueID").agg(
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
            ((pl.lit(config.cutoff_date) - pl.col("RunDate").max()).dt.total_days()).alias("fin_last_snapshot_days_ago"),
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

    age_years = ((pl.lit(config.cutoff_date) - pl.col("BirthDate")).dt.total_days()).truediv(365.25)
    demo_features = (
        pl.scan_parquet(dataset_paths.demographics_path)
        .join(base_ids, on="UniqueID", how="inner")
        .with_columns([pl.col("UniqueID").cast(pl.String)])
        .unique(subset=["UniqueID"], keep="first")
        .select(
            [
                "UniqueID",
                pl.when(age_years.is_between(10, 110, closed="both")).then(age_years).otherwise(None).alias(
                    "demo_age_years"
                ),
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

    customer_features = (
        base_ids.join(
            txn.group_by("UniqueID")
            .agg(aggregations)
            .join(monthly_features, on="UniqueID", how="left")
            .join(account_features, on="UniqueID", how="left"),
            on="UniqueID",
            how="left",
        )
        .join(fin_features, on="UniqueID", how="left")
        .join(demo_features, on="UniqueID", how="left")
        .with_columns(
            [pl.col("fin_record_count").is_null().cast(pl.Int8).alias("fin_missing_flag")]
            + [pl.col(column).fill_null(-999.0).alias(column) for column in fin_numeric_cols]
        )
        .with_columns(
            [
                (pl.lit(config.observed_months_total) - pl.col("txn_active_months_total")).alias(
                    "txn_inactive_months_total"
                ),
                ratio_expr("txn_active_months_total", "txn_total_count", "txn_active_month_to_txn_ratio"),
                ratio_expr("txn_months_count_eq_1", "txn_active_months_total", "txn_sparse_month_share_eq_1"),
                ratio_expr("txn_months_count_le_2", "txn_active_months_total", "txn_sparse_month_share_le_2"),
                ratio_expr("txn_recent_3m_count", "txn_total_count", "txn_recent_share_count"),
                ratio_expr("txn_recent_3m_credit_sum", "txn_credit_sum", "txn_recent_share_credit"),
                ratio_expr("txn_recent_3m_debit_sum", "txn_debit_sum", "txn_recent_share_debit"),
                ratio_expr("txn_recent_3m_count", "txn_prev_3m_count", "txn_recent_vs_prev3_count_ratio"),
                ratio_expr("txn_recent_3m_credit_sum", "txn_prev_3m_credit_sum", "txn_recent_vs_prev3_credit_ratio"),
                ratio_expr("txn_recent_3m_debit_sum", "txn_prev_3m_debit_sum", "txn_recent_vs_prev3_debit_ratio"),
                ratio_expr("txn_holiday_prev1_count", "txn_holiday_prev2_count", "txn_holiday_count_yoy_ratio"),
                ratio_expr("txn_credit_sum", "txn_debit_sum", "txn_credit_debit_ratio"),
                ratio_expr("txn_account_txn_max", "txn_total_count", "txn_account_concentration_ratio"),
                (pl.col("txn_recent_3m_count") - pl.col("txn_prev_3m_count")).alias("txn_recent_vs_prev3_count_delta"),
                (pl.col("txn_recent_3m_credit_sum") - pl.col("txn_prev_3m_credit_sum")).alias(
                    "txn_recent_vs_prev3_credit_delta"
                ),
                (pl.col("txn_recent_3m_debit_sum") - pl.col("txn_prev_3m_debit_sum")).alias(
                    "txn_recent_vs_prev3_debit_delta"
                ),
                (pl.col("txn_recent_m1_count") - pl.col("txn_recent_m2_count")).alias("txn_recent_count_delta_m1_m2"),
                (pl.col("txn_recent_m2_count") - pl.col("txn_recent_m3_count")).alias("txn_recent_count_delta_m2_m3"),
                (pl.col("txn_recent_m1_credit_sum") - pl.col("txn_recent_m2_credit_sum")).alias(
                    "txn_recent_credit_delta_m1_m2"
                ),
                (pl.col("txn_recent_m1_debit_sum") - pl.col("txn_recent_m2_debit_sum")).alias(
                    "txn_recent_debit_delta_m1_m2"
                ),
                (pl.col("txn_holiday_prev1_count") - pl.col("txn_holiday_prev2_count")).alias(
                    "txn_holiday_count_yoy_delta"
                ),
                (pl.col("txn_statement_balance_std") / (pl.col("txn_statement_balance_mean").abs() + pl.lit(1.0))).alias(
                    "txn_statement_balance_cv"
                ),
                pl.lit(config.observed_months_total).alias("meta_history_months_available"),
                pl.lit(int(bool(config.holiday_prev1_months))).alias("meta_prev1_holiday_available"),
                pl.lit(int(bool(config.holiday_prev2_months))).alias("meta_prev2_holiday_available"),
            ]
        )
        .collect()
    )

    return customer_features


def build_or_load_customer_feature_matrix(
    dataset_paths: DatasetPaths,
    base_ids: pl.LazyFrame,
    config: CutoffConfig,
    cache_dir: Path,
    refresh_cache: bool,
) -> pl.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"customer_features_{config.name}_{FEATURE_PIPELINE_VERSION}.parquet"
    if cache_path.exists() and not refresh_cache:
        return pl.read_parquet(cache_path)

    feature_matrix = build_customer_feature_matrix(dataset_paths, base_ids, config)
    feature_matrix.write_parquet(cache_path)
    return feature_matrix


def assemble_frame(base_df: pl.DataFrame, customer_features: pl.DataFrame) -> pl.DataFrame:
    assembled = base_df.join(customer_features, on="UniqueID", how="left")
    if assembled.height != base_df.height:
        raise ValueError(f"Row count changed during feature assembly. Expected {base_df.height}, got {assembled.height}.")
    return assembled


def build_pseudo_target_frame(
    transactions_path: Path,
    base_ids_df: pl.DataFrame,
    config: CutoffConfig,
) -> pl.DataFrame:
    target_mask = months_mask(config.target_months)
    counts = (
        pl.scan_parquet(transactions_path)
        .join(base_ids_df.lazy(), on="UniqueID", how="inner")
        .with_columns(
            [
                pl.col("UniqueID").cast(pl.String),
                pl.col("TransactionDate").dt.year().alias("txn_year"),
                pl.col("TransactionDate").dt.month().alias("txn_month"),
            ]
        )
        .filter(target_mask)
        .group_by("UniqueID")
        .agg(pl.len().alias("next_3m_txn_count"))
        .collect()
    )

    return (
        base_ids_df.join(counts, on="UniqueID", how="left")
        .with_columns(pl.col("next_3m_txn_count").fill_null(0).cast(pl.Float64))
        .select(["UniqueID", "next_3m_txn_count"])
    )


def prepare_augmented_frames(
    current_feature_table: pl.DataFrame,
    pseudo_feature_tables: list[tuple[CutoffConfig, pl.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    current_train_df = current_feature_table.filter(pl.col("__split__") == "train").drop("__split__").to_pandas()
    current_test_df = (
        current_feature_table.filter(pl.col("__split__") == "test").drop(["__split__", "next_3m_txn_count"]).to_pandas()
    )

    current_train_df["row_origin"] = "current"
    current_train_df["sample_weight"] = 1.0
    current_test_df["row_origin"] = "test"

    pseudo_frames: list[pd.DataFrame] = []
    for config, feature_table in pseudo_feature_tables:
        pseudo_df = feature_table.drop("__split__").to_pandas()
        pseudo_df["row_origin"] = config.name
        pseudo_df["sample_weight"] = config.sample_weight
        pseudo_frames.append(pseudo_df)

    pseudo_train_df = pd.concat(pseudo_frames, axis=0, ignore_index=True) if pseudo_frames else pd.DataFrame()

    train_ids = current_train_df["UniqueID"].to_numpy()
    test_ids = current_test_df["UniqueID"].to_numpy()
    y_train = current_train_df["next_3m_txn_count"].to_numpy(dtype=np.float64)

    feature_columns = [
        column
        for column in current_train_df.columns
        if column not in {"UniqueID", "next_3m_txn_count", "row_origin", "sample_weight"}
    ]
    numeric_columns = current_train_df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return current_train_df, current_test_df, pseudo_train_df, y_train, train_ids, test_ids, numeric_columns, categorical_columns


def fit_with_sample_weight(
    estimator: TransformedTargetRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
) -> TransformedTargetRegressor:
    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        fit_kwargs["regressor__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    estimator.fit(X_train, y_train, **fit_kwargs)
    return estimator


def run_temporal_cv_for_model(
    spec: ModelSpec,
    current_train_df: pd.DataFrame,
    current_test_df: pd.DataFrame,
    pseudo_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int,
    random_state: int,
) -> ModelResult:
    feature_columns = [
        column
        for column in current_train_df.columns
        if column not in {"UniqueID", "next_3m_txn_count", "row_origin", "sample_weight"}
    ]

    oof_sum = np.zeros(len(current_train_df), dtype=np.float64)
    oof_count = np.zeros(len(current_train_df), dtype=np.int32)
    fold_scores: list[dict[str, object]] = []

    for split_index, (train_idx, valid_idx) in enumerate(cv_splits):
        valid_ids = set(current_train_df.iloc[valid_idx]["UniqueID"].tolist())
        train_current = current_train_df.iloc[train_idx].copy()
        if pseudo_train_df.empty:
            train_augmented = train_current
        else:
            train_pseudo = pseudo_train_df.loc[~pseudo_train_df["UniqueID"].isin(valid_ids)].copy()
            train_augmented = pd.concat([train_current, train_pseudo], axis=0, ignore_index=True)

        estimator = build_estimator(spec, numeric_columns, categorical_columns, random_state)
        fit_with_sample_weight(
            estimator=estimator,
            X_train=train_augmented[feature_columns],
            y_train=train_augmented["next_3m_txn_count"].to_numpy(dtype=np.float64),
            sample_weight=train_augmented["sample_weight"].to_numpy(dtype=np.float64),
        )

        valid_pred = np.clip(estimator.predict(current_train_df.iloc[valid_idx][feature_columns]), 0.0, None)
        oof_sum[valid_idx] += valid_pred
        oof_count[valid_idx] += 1

        fold_rmsle = rmsle(y_train[valid_idx], valid_pred)
        fold_scores.append(
            {
                "model_name": spec.name,
                "family": spec.family,
                "repeat_index": split_index // n_splits,
                "fold_index": split_index % n_splits,
                "train_current_rows": int(len(train_current)),
                "train_pseudo_rows": int(len(train_augmented) - len(train_current)),
                "valid_rows": int(len(valid_idx)),
                "fold_rmsle": float(fold_rmsle),
            }
        )

    if np.any(oof_count == 0):
        raise ValueError(f"Model {spec.name} has rows without out-of-fold coverage.")

    full_train = pd.concat([current_train_df, pseudo_train_df], axis=0, ignore_index=True)
    final_estimator = build_estimator(spec, numeric_columns, categorical_columns, random_state)
    fit_with_sample_weight(
        estimator=final_estimator,
        X_train=full_train[feature_columns],
        y_train=full_train["next_3m_txn_count"].to_numpy(dtype=np.float64),
        sample_weight=full_train["sample_weight"].to_numpy(dtype=np.float64),
    )
    test_predictions = np.clip(final_estimator.predict(current_test_df[feature_columns]), 0.0, None)

    oof_predictions = oof_sum / oof_count
    return ModelResult(
        spec=spec,
        oof_rmsle=rmsle(y_train, oof_predictions),
        fold_scores=fold_scores,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
    )


def align_predictions_by_id(
    prediction_frame: pl.DataFrame,
    id_column: str,
    ordered_ids: np.ndarray,
    prediction_column: str,
) -> np.ndarray:
    aligned = (
        pl.DataFrame({id_column: pl.Series(id_column, ordered_ids)})
        .join(prediction_frame.select([id_column, prediction_column]), on=id_column, how="left")
        .select(prediction_column)
        .to_series()
        .to_numpy()
    )
    if np.isnan(aligned).any():
        raise ValueError(f"Missing values while aligning {prediction_column} by ID.")
    return aligned.astype(np.float64)


def build_seed_candidate_from_members(
    seed_artifact_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    member_names: list[str],
    candidate_name: str,
) -> ModelResult | None:
    oof_path = seed_artifact_dir / "oof_predictions.parquet"
    test_path = seed_artifact_dir / "test_predictions.parquet"
    if not oof_path.exists() or not test_path.exists():
        return None

    oof_predictions = pl.read_parquet(oof_path)
    test_predictions = pl.read_parquet(test_path)
    member_columns = [f"pred_{name}" for name in member_names]
    if any(column not in oof_predictions.columns for column in member_columns):
        return None
    if any(column not in test_predictions.columns for column in member_columns):
        return None

    oof_member_matrix = np.column_stack(
        [align_predictions_by_id(oof_predictions, "UniqueID", train_ids, column) for column in member_columns]
    )
    test_member_matrix = np.column_stack(
        [align_predictions_by_id(test_predictions, "UniqueID", test_ids, column) for column in member_columns]
    )
    oof_blended = np.expm1(np.log1p(np.clip(oof_member_matrix, 0.0, None)).mean(axis=1))
    test_blended = np.expm1(np.log1p(np.clip(test_member_matrix, 0.0, None)).mean(axis=1))
    return ModelResult(
        spec=ModelSpec(
            name=candidate_name,
            family="external_seed",
            params={"source_run_dir": str(seed_artifact_dir), "members": member_names, "blend_space": "log_equal"},
        ),
        oof_rmsle=float("nan"),
        fold_scores=[],
        oof_predictions=oof_blended,
        test_predictions=test_blended,
    )


def load_seed_candidates(
    seed_artifact_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    y_train: np.ndarray,
) -> list[ModelResult]:
    if not seed_artifact_dir.exists():
        return []

    seed_candidates: list[ModelResult] = []
    best_subset = build_seed_candidate_from_members(
        seed_artifact_dir=seed_artifact_dir,
        train_ids=train_ids,
        test_ids=test_ids,
        member_names=["hgb_conservative_v3", "xgb_conservative_v3", "catboost_conservative_v1"],
        candidate_name="seed_v4_best_subset",
    )
    if best_subset is not None:
        seed_candidates.append(
            ModelResult(
                spec=best_subset.spec,
                oof_rmsle=rmsle(y_train, best_subset.oof_predictions),
                fold_scores=[],
                oof_predictions=best_subset.oof_predictions,
                test_predictions=best_subset.test_predictions,
            )
        )
    return seed_candidates


def compute_candidate_metrics(
    current_train_df: pd.DataFrame,
    y_train: np.ndarray,
    model_results: list[ModelResult],
    blend_results: list[dict[str, object]],
) -> pd.DataFrame:
    segment_masks: dict[str, np.ndarray] = {"overall": np.ones(len(y_train), dtype=bool)}
    if "txn_recent_3m_count" in current_train_df.columns:
        segment_masks["low_recent_3m"] = current_train_df["txn_recent_3m_count"].to_numpy(dtype=np.float64) <= 20
    if "txn_active_months_total" in current_train_df.columns:
        segment_masks["active_months_le_12"] = (
            current_train_df["txn_active_months_total"].to_numpy(dtype=np.float64) <= 12
        )
    if "txn_months_count_le_2" in current_train_df.columns:
        segment_masks["sparse_months_ge_2"] = current_train_df["txn_months_count_le_2"].to_numpy(dtype=np.float64) >= 2
    if "fin_missing_flag" in current_train_df.columns:
        segment_masks["fin_missing"] = current_train_df["fin_missing_flag"].to_numpy(dtype=np.float64) >= 1
    segment_masks["target_le_3"] = y_train <= 3

    candidate_rows: list[dict[str, object]] = []
    for result in model_results:
        row = {
            "candidate_name": result.spec.name,
            "candidate_type": "base",
            "family": result.spec.family,
            "oof_rmsle": result.oof_rmsle,
            "member_names": result.spec.name,
        }
        for segment_name, mask in segment_masks.items():
            if mask.sum() < 50:
                row[f"{segment_name}_rmsle"] = np.nan
            else:
                row[f"{segment_name}_rmsle"] = rmsle(y_train[mask], result.oof_predictions[mask])
        candidate_rows.append(row)

    for blend in blend_results:
        row = {
            "candidate_name": str(blend["name"]),
            "candidate_type": "blend",
            "family": "blend",
            "oof_rmsle": float(blend["oof_rmsle"]),
            "member_names": ",".join(blend["members"]),
        }
        for segment_name, mask in segment_masks.items():
            if mask.sum() < 50:
                row[f"{segment_name}_rmsle"] = np.nan
            else:
                row[f"{segment_name}_rmsle"] = rmsle(y_train[mask], np.asarray(blend["oof_predictions"])[mask])
        candidate_rows.append(row)

    metrics = pd.DataFrame(candidate_rows)
    rank_metric_columns = [
        column
        for column in metrics.columns
        if column.endswith("_rmsle")
        and column not in {"overall_rmsle"}
        and column != "oof_rmsle"
    ]
    metrics["overall_rmsle"] = metrics["oof_rmsle"]
    rank_metric_columns = ["overall_rmsle"] + rank_metric_columns
    for column in rank_metric_columns:
        metrics[f"rank_{column}"] = metrics[column].rank(method="average", ascending=True, na_option="keep")
    rank_columns = [f"rank_{column}" for column in rank_metric_columns]
    metrics["hedge_rank_mean"] = metrics[rank_columns].mean(axis=1, skipna=True)
    return metrics.sort_values(["oof_rmsle", "hedge_rank_mean", "candidate_name"]).reset_index(drop=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def copy_submission(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def write_recommendation_pack(
    run_dir: Path,
    submissions_dir: Path,
    candidate_metrics: pd.DataFrame,
) -> None:
    selection_dir = run_dir / "recommended_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    best_oof_name = candidate_metrics.sort_values(["oof_rmsle", "hedge_rank_mean"]).iloc[0]["candidate_name"]
    hedge_candidates = candidate_metrics.sort_values(["hedge_rank_mean", "oof_rmsle", "candidate_name"])
    best_hedge_name = hedge_candidates.iloc[0]["candidate_name"]
    if best_hedge_name == best_oof_name and len(hedge_candidates) > 1:
        best_hedge_name = hedge_candidates.iloc[1]["candidate_name"]

    copy_submission(submissions_dir / f"{best_oof_name}.csv", selection_dir / "01_best_oof.csv")
    copy_submission(submissions_dir / f"{best_hedge_name}.csv", selection_dir / "02_best_hedge.csv")

    selection_readme = [
        "# Recommended Selection",
        "",
        "1. `01_best_oof.csv`",
        f"   - candidate: `{best_oof_name}`",
        "   - rationale: strongest overall out-of-fold RMSLE in this temporal augmentation run",
        "",
        "2. `02_best_hedge.csv`",
        f"   - candidate: `{best_hedge_name}`",
        "   - rationale: strongest average rank across overall error and hard customer segments",
        "",
        (
            "Both files are copied from the `submissions/` directory and are upload-ready for Zindi."
            if args.submission_format == "zindi_log"
            else "Both files are copied from the `submissions/` directory, but this run used raw output format. "
            "Do not upload them to Zindi without converting to `np.log1p(prediction)` first."
        ),
    ]
    (selection_dir / "README.md").write_text("\n".join(selection_readme) + "\n", encoding="utf-8")


def write_artifacts(
    run_dir: Path,
    dataset_paths: DatasetPaths,
    current_config: CutoffConfig,
    pseudo_configs: list[CutoffConfig],
    current_train_df: pd.DataFrame,
    current_test_df: pd.DataFrame,
    pseudo_train_df: pd.DataFrame,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    model_results: list[ModelResult],
    blend_results: list[dict[str, object]],
    candidate_metrics: pd.DataFrame,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    write_json(
        run_dir / "config.json",
        {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "data_paths": asdict(dataset_paths),
            "cache_dir": str(cache_dir.resolve()),
            "current_cutoff": asdict(current_config),
            "pseudo_cutoffs": [asdict(config) for config in pseudo_configs],
            "cv": {
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "random_state": args.random_state,
            },
            "blend_search": {
                "top_k_blend_models": args.top_k_blend_models,
                "equal_blend_search_models": args.equal_blend_search_models,
                "max_equal_blend_size": args.max_equal_blend_size,
            },
            "submission_format": args.submission_format,
            "seed_artifact_dir": None if args.disable_seed_candidates else str(args.seed_artifact_dir.resolve()),
            "model_registry": [asdict(result.spec) for result in model_results],
        },
    )

    write_json(
        run_dir / "dataset_manifest.json",
        {
            "current_train_rows": int(len(current_train_df)),
            "current_test_rows": int(len(current_test_df)),
            "pseudo_rows": int(len(pseudo_train_df)),
            "train_unique_ids": int(pd.Series(train_ids).nunique()),
            "test_unique_ids": int(pd.Series(test_ids).nunique()),
        },
    )

    fold_rows = [row for result in model_results for row in result.fold_scores]
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_scores.csv", index=False)

    base_rows: list[dict[str, object]] = []
    for result in model_results:
        fold_rmsles = [row["fold_rmsle"] for row in result.fold_scores]
        base_rows.append(
            {
                "model_name": result.spec.name,
                "family": result.spec.family,
                "oof_rmsle": result.oof_rmsle,
                "fold_rmsle_mean": float(np.mean(fold_rmsles)) if fold_rmsles else np.nan,
                "fold_rmsle_std": float(np.std(fold_rmsles)) if fold_rmsles else np.nan,
                "params_json": json.dumps(result.spec.params, sort_keys=True),
            }
        )
    pd.DataFrame(base_rows).sort_values(["oof_rmsle", "model_name"]).to_csv(run_dir / "base_model_scores.csv", index=False)

    blend_rows = [
        {
            "blend_name": str(blend["name"]),
            "space": str(blend["space"]),
            "oof_rmsle": float(blend["oof_rmsle"]),
            "members": ",".join(blend["members"]),
            "weights_json": json.dumps(blend["weights"]),
        }
        for blend in blend_results
    ]
    pd.DataFrame(blend_rows).sort_values(["oof_rmsle", "blend_name"]).to_csv(run_dir / "blend_scores.csv", index=False)

    candidate_metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)

    oof_payload: dict[str, object] = {"UniqueID": train_ids, "next_3m_txn_count_true": y_train}
    test_payload: dict[str, object] = {"UniqueID": test_ids}
    for result in model_results:
        oof_payload[f"pred_{result.spec.name}"] = result.oof_predictions
        test_payload[f"pred_{result.spec.name}"] = result.test_predictions
    for blend in blend_results:
        oof_payload[f"pred_{blend['name']}"] = blend["oof_predictions"]
        test_payload[f"pred_{blend['name']}"] = blend["test_predictions"]
    pl.DataFrame(oof_payload).write_parquet(run_dir / "oof_predictions.parquet")
    pl.DataFrame(test_payload).write_parquet(run_dir / "test_predictions.parquet")

    submissions_dir = run_dir / "submissions"
    submissions_raw_dir = run_dir / "submissions_raw"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submissions_raw_dir.mkdir(parents=True, exist_ok=True)

    for result in model_results:
        build_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            result.test_predictions,
            submission_format=args.submission_format,
        ).write_csv(submissions_dir / f"{result.spec.name}.csv")
        build_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            result.test_predictions,
            submission_format="raw",
        ).write_csv(submissions_raw_dir / f"{result.spec.name}.csv")

    for blend in blend_results:
        build_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            np.asarray(blend["test_predictions"]),
            submission_format=args.submission_format,
        ).write_csv(submissions_dir / f"{blend['name']}.csv")
        build_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            np.asarray(blend["test_predictions"]),
            submission_format="raw",
        ).write_csv(submissions_raw_dir / f"{blend['name']}.csv")

    write_recommendation_pack(run_dir, submissions_dir, candidate_metrics)

    best_oof_row = candidate_metrics.sort_values(["oof_rmsle", "hedge_rank_mean"]).iloc[0]
    best_hedge_row = candidate_metrics.sort_values(["hedge_rank_mean", "oof_rmsle", "candidate_name"]).iloc[0]
    summary_lines = [
        "# Temporal Holiday Stack Summary",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Current cutoff: `{current_config.name}`",
        f"- Pseudo cutoffs: `{', '.join(config.name for config in pseudo_configs)}`",
        f"- Repeated CV: `{args.n_splits}` folds x `{args.n_repeats}` repeats",
        f"- Submission artifact format: `{args.submission_format}`",
        f"- Best overall candidate: `{best_oof_row['candidate_name']}` with OOF RMSLE `{best_oof_row['oof_rmsle']:.6f}`",
        f"- Best hedge candidate: `{best_hedge_row['candidate_name']}` with hedge rank `{best_hedge_row['hedge_rank_mean']:.3f}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidate_metrics.head(8).to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['candidate_name']}` ({row['candidate_type']}): OOF `{row['oof_rmsle']:.6f}`, "
            f"hedge rank `{row['hedge_rank_mean']:.3f}`, members `{row['member_names']}`"
        )
    summary_lines.extend(
        [
            "",
            "## Hard-Segment Focus",
            "",
            "- `candidate_metrics.csv` ranks candidates on overall RMSLE plus low-recent, sparse, missing-financial, and low-target slices.",
            "- `recommended_selection/01_best_oof.csv` is the best pure-score submission.",
            "- `recommended_selection/02_best_hedge.csv` is the safer private-leaderboard hedge.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_paths = resolve_dataset_paths(args)
    run_dir = build_run_directory(args.output_dir, args.run_name)
    cache_dir = args.cache_dir.resolve()

    train_df = pl.read_csv(dataset_paths.train_path).with_columns(
        [pl.col("UniqueID").cast(pl.String), pl.col("next_3m_txn_count").cast(pl.Float64)]
    )
    test_df = pl.read_csv(dataset_paths.test_path).with_columns(pl.col("UniqueID").cast(pl.String))
    full_ids_df = (
        pl.concat([train_df.select("UniqueID"), test_df.select("UniqueID")], how="vertical")
        .unique(subset=["UniqueID"], keep="first")
        .sort("UniqueID")
    )
    base_ids = full_ids_df.lazy()

    current_config = build_cutoff_config("current_2015_10", 2015, args.current_weight)
    pseudo_configs = [
        build_cutoff_config("pseudo_2014_10", 2014, args.pseudo_2014_weight),
        build_cutoff_config("pseudo_2013_10", 2013, args.pseudo_2013_weight),
    ]

    current_customer_features = build_or_load_customer_feature_matrix(
        dataset_paths=dataset_paths,
        base_ids=base_ids,
        config=current_config,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
    )
    current_base = pl.concat(
        [
            train_df.with_columns(pl.lit("train").alias("__split__")),
            test_df.with_columns(
                [pl.lit(None).cast(pl.Float64).alias("next_3m_txn_count"), pl.lit("test").alias("__split__")]
            ),
        ],
        how="diagonal_relaxed",
    )
    current_feature_table = assemble_frame(current_base, current_customer_features)

    pseudo_feature_tables: list[tuple[CutoffConfig, pl.DataFrame]] = []
    for config in pseudo_configs:
        customer_features = build_or_load_customer_feature_matrix(
            dataset_paths=dataset_paths,
            base_ids=base_ids,
            config=config,
            cache_dir=cache_dir,
            refresh_cache=args.refresh_cache,
        )
        pseudo_target = build_pseudo_target_frame(dataset_paths.transactions_path, full_ids_df, config)
        pseudo_base = pseudo_target.with_columns(pl.lit("pseudo").alias("__split__"))
        pseudo_feature_tables.append((config, assemble_frame(pseudo_base, customer_features)))

    current_train_df, current_test_df, pseudo_train_df, y_train, train_ids, test_ids, numeric_columns, categorical_columns = (
        prepare_augmented_frames(current_feature_table, pseudo_feature_tables)
    )

    cv = RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_state)
    cv_splits = list(cv.split(current_train_df, y_train))

    model_registry = filter_model_registry(build_model_registry(), args.include_models)
    trained_results: list[ModelResult] = []
    for spec in model_registry:
        print(f"Running temporal model {spec.name}...")
        result = run_temporal_cv_for_model(
            spec=spec,
            current_train_df=current_train_df,
            current_test_df=current_test_df,
            pseudo_train_df=pseudo_train_df,
            y_train=y_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            cv_splits=cv_splits,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        trained_results.append(result)
        print(f"  OOF RMSLE: {result.oof_rmsle:.6f}")

    seed_results: list[ModelResult] = []
    if not args.disable_seed_candidates:
        seed_results = load_seed_candidates(
            seed_artifact_dir=args.seed_artifact_dir.resolve(),
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
        )
        for result in seed_results:
            print(f"Loaded seed candidate {result.spec.name} with OOF RMSLE {result.oof_rmsle:.6f}")

    all_results = trained_results + seed_results
    blend_candidates = build_blend_candidates(
        all_results,
        args.top_k_blend_models,
        y_train,
        args.equal_blend_search_models,
        args.max_equal_blend_size,
    )
    blend_results = evaluate_blend_candidates(blend_candidates, y_train)
    for blend in blend_results[:10]:
        print(f"Blend {blend['name']}: {blend['oof_rmsle']:.6f}")

    candidate_metrics = compute_candidate_metrics(current_train_df, y_train, all_results, blend_results)
    write_artifacts(
        run_dir=run_dir,
        dataset_paths=dataset_paths,
        current_config=current_config,
        pseudo_configs=pseudo_configs,
        current_train_df=current_train_df,
        current_test_df=current_test_df,
        pseudo_train_df=pseudo_train_df,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
        model_results=all_results,
        blend_results=blend_results,
        candidate_metrics=candidate_metrics,
        cache_dir=cache_dir,
        args=args,
    )
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
