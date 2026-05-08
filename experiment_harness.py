from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from baseline_polars_pipeline import (
    DEFAULT_SUBMISSION_FORMAT,
    FEATURE_PIPELINE_VERSION,
    RANDOM_STATE,
    build_submission_frame as build_baseline_submission_frame,
    build_feature_table,
    resolve_existing_path,
    rmsle,
)


@dataclass(frozen=True)
class DatasetPaths:
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    transactions_path: Path
    financials_path: Path
    demographics_path: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict[str, object]


@dataclass
class ModelResult:
    spec: ModelSpec
    oof_rmsle: float
    fold_scores: list[dict[str, object]]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass
class BlendCandidate:
    name: str
    members: list[ModelResult]
    weights: np.ndarray
    space: str


class SklearnCatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        categorical_columns: tuple[str, ...],
        params: dict[str, object],
        random_state: int,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.params = params
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SklearnCatBoostRegressor":
        params = dict(self.params)
        loss_function = str(params.pop("loss_function", "RMSE"))
        eval_metric = str(params.pop("eval_metric", loss_function))
        self.model_ = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=self.random_state,
            allow_writing_files=False,
            thread_count=1,
            verbose=False,
            **params,
        )
        self.model_.fit(X, y, cat_features=list(self.categorical_columns), sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable repeated-CV experiments for the Nedbank challenge, log artifacts, "
            "benchmark conservative models, and generate blend submissions."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--transactions-path", type=Path, default=None)
    parser.add_argument("--financials-path", type=Path, default=None)
    parser.add_argument("--demographics-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=Path(f"outputs/cache/feature_table_{FEATURE_PIPELINE_VERSION}.parquet"),
        help="Parquet cache for the fully assembled customer-level feature table.",
    )
    parser.add_argument(
        "--refresh-feature-cache",
        action="store_true",
        help="Rebuild the feature cache instead of reusing an existing parquet artifact.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--top-k-blend-models",
        type=int,
        default=3,
        help="Maximum number of top-ranked models to use when forming blend candidates.",
    )
    parser.add_argument(
        "--equal-blend-search-models",
        type=int,
        default=8,
        help="How many top-ranked base models to consider in the constrained equal-weight subset search.",
    )
    parser.add_argument(
        "--max-equal-blend-size",
        type=int,
        default=5,
        help="Maximum number of members to allow in the equal-weight subset search.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="conservative_stack",
        help="Human-readable prefix for the experiment artifact directory.",
    )
    parser.add_argument(
        "--include-models",
        type=str,
        default="",
        help="Optional comma-separated list of model names to run from the registry.",
    )
    parser.add_argument(
        "--submission-format",
        type=str,
        choices=("zindi_log", "raw"),
        default=DEFAULT_SUBMISSION_FORMAT,
        help="Format for the main submission artifacts. `zindi_log` is the competition-ready default.",
    )
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


def build_run_directory(output_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    token = uuid4().hex[:8]
    run_dir = output_dir.resolve() / f"{run_name}_{timestamp}_{token}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_or_build_feature_table(
    dataset_paths: DatasetPaths,
    cache_path: Path,
    refresh_feature_cache: bool,
) -> tuple[pl.DataFrame, str]:
    resolved_cache_path = cache_path.resolve()
    if resolved_cache_path.exists() and not refresh_feature_cache:
        return pl.read_parquet(resolved_cache_path), "cache"

    feature_table = build_feature_table(
        train_path=dataset_paths.train_path,
        test_path=dataset_paths.test_path,
        transactions_path=dataset_paths.transactions_path,
        financials_path=dataset_paths.financials_path,
        demographics_path=dataset_paths.demographics_path,
    )
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.write_parquet(resolved_cache_path)
    return feature_table, "rebuilt"


def prepare_model_frames(
    feature_table: pl.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    train_df = feature_table.filter(pl.col("__split__") == "train").drop(["__split__"]).to_pandas()
    test_df = feature_table.filter(pl.col("__split__") == "test").drop(["__split__", "next_3m_txn_count"]).to_pandas()

    y_train = train_df.pop("next_3m_txn_count").to_numpy(dtype=np.float64)
    train_ids = train_df.pop("UniqueID").to_numpy()
    test_ids = test_df.pop("UniqueID").to_numpy()

    numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return train_df, test_df, y_train, train_ids, test_ids, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_glm_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def build_hgb_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
) -> TransformedTargetRegressor:
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = HistGradientBoostingRegressor(loss="squared_error", random_state=random_state, **params)
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", preprocessor), ("regressor", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_hgb_count_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
):
    params = dict(params)
    loss = str(params.pop("loss", "poisson"))
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = HistGradientBoostingRegressor(loss=loss, random_state=random_state, **params)
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def build_xgb_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
) -> TransformedTargetRegressor:
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
        verbosity=0,
        **params,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", preprocessor), ("regressor", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_xgb_direct_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
):
    params = dict(params)
    objective = str(params.pop("objective"))
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = XGBRegressor(
        objective=objective,
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
        verbosity=0,
        **params,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def build_lgbm_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
) -> TransformedTargetRegressor:
    params = dict(params)
    objective = str(params.pop("objective", "regression"))
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = LGBMRegressor(
        objective=objective,
        n_jobs=1,
        random_state=random_state,
        verbosity=-1,
        **params,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", preprocessor), ("regressor", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_lgbm_direct_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
):
    params = dict(params)
    objective = str(params.pop("objective"))
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = LGBMRegressor(
        objective=objective,
        n_jobs=1,
        random_state=random_state,
        verbosity=-1,
        **params,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def prepare_catboost_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def build_catboost_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
) -> TransformedTargetRegressor:
    del numeric_columns
    transformer = FunctionTransformer(
        func=lambda frame: prepare_catboost_frame(frame, categorical_columns),
        validate=False,
    )
    regressor = SklearnCatBoostRegressor(
        categorical_columns=tuple(categorical_columns),
        params=params,
        random_state=random_state,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("catboost_input", transformer), ("regressor", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_catboost_direct_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
):
    del numeric_columns
    transformer = FunctionTransformer(
        func=lambda frame: prepare_catboost_frame(frame, categorical_columns),
        validate=False,
    )
    regressor = SklearnCatBoostRegressor(
        categorical_columns=tuple(categorical_columns),
        params=params,
        random_state=random_state,
    )
    return Pipeline([("catboost_input", transformer), ("regressor", regressor)])


def build_extra_trees_estimator(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, object],
    random_state: int,
) -> TransformedTargetRegressor:
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    regressor = ExtraTreesRegressor(
        random_state=random_state,
        n_jobs=1,
        **params,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", preprocessor), ("regressor", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_glm_count_estimator(
    spec: ModelSpec,
    numeric_columns: list[str],
    categorical_columns: list[str],
):
    preprocessor = build_glm_preprocessor(numeric_columns, categorical_columns)
    params = dict(spec.params)
    if spec.family == "poisson_glm":
        regressor = PoissonRegressor(**params)
    elif spec.family == "tweedie_glm":
        regressor = TweedieRegressor(**params)
    else:
        raise ValueError(f"Unsupported GLM family: {spec.family}")
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def build_estimator(
    spec: ModelSpec,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
) -> BaseEstimator:
    if spec.family == "hist_gradient_boosting":
        return build_hgb_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "hist_gradient_boosting_count":
        return build_hgb_count_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "lightgbm":
        return build_lgbm_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "lightgbm_direct":
        return build_lgbm_direct_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "xgboost":
        return build_xgb_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "xgboost_direct":
        return build_xgb_direct_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "catboost":
        return build_catboost_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "catboost_direct":
        return build_catboost_direct_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family == "extra_trees":
        return build_extra_trees_estimator(numeric_columns, categorical_columns, spec.params, random_state)
    if spec.family in {"poisson_glm", "tweedie_glm"}:
        return build_glm_count_estimator(spec, numeric_columns, categorical_columns)
    raise ValueError(f"Unsupported model family: {spec.family}")


def build_model_registry() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="hgb_conservative_v1",
            family="hist_gradient_boosting",
            params={
                "learning_rate": 0.03,
                "max_iter": 500,
                "max_depth": 5,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
            },
        ),
        ModelSpec(
            name="hgb_conservative_v2",
            family="hist_gradient_boosting",
            params={
                "learning_rate": 0.02,
                "max_iter": 650,
                "max_depth": 4,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 25,
                "l2_regularization": 0.2,
            },
        ),
        ModelSpec(
            name="hgb_conservative_v3",
            family="hist_gradient_boosting",
            params={
                "learning_rate": 0.015,
                "max_iter": 900,
                "max_depth": 4,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 30,
                "l2_regularization": 0.3,
            },
        ),
        ModelSpec(
            name="xgb_conservative_v1",
            family="xgboost",
            params={
                "n_estimators": 900,
                "learning_rate": 0.025,
                "max_depth": 5,
                "min_child_weight": 30,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
            },
        ),
        ModelSpec(
            name="xgb_conservative_v2",
            family="xgboost",
            params={
                "n_estimators": 600,
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 25,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
            },
        ),
        ModelSpec(
            name="xgb_conservative_v3",
            family="xgboost",
            params={
                "n_estimators": 1200,
                "learning_rate": 0.02,
                "max_depth": 5,
                "min_child_weight": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.65,
                "reg_lambda": 4.0,
                "reg_alpha": 0.2,
                "gamma": 0.05,
            },
        ),
        ModelSpec(
            name="hgb_poisson_v1",
            family="hist_gradient_boosting_count",
            params={
                "loss": "poisson",
                "learning_rate": 0.03,
                "max_iter": 600,
                "max_depth": 4,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 30,
                "l2_regularization": 0.3,
            },
        ),
        ModelSpec(
            name="hgb_poisson_v2",
            family="hist_gradient_boosting_count",
            params={
                "loss": "poisson",
                "learning_rate": 0.02,
                "max_iter": 900,
                "max_depth": 4,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 40,
                "l2_regularization": 0.5,
            },
        ),
        ModelSpec(
            name="lgbm_log_v1",
            family="lightgbm",
            params={
                "objective": "regression",
                "n_estimators": 900,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 30,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "reg_lambda": 3.0,
                "reg_alpha": 0.2,
            },
        ),
        ModelSpec(
            name="lgbm_log_v2",
            family="lightgbm",
            params={
                "objective": "regression",
                "n_estimators": 1300,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.65,
                "reg_lambda": 5.0,
                "reg_alpha": 0.4,
            },
        ),
        ModelSpec(
            name="lgbm_poisson_v1",
            family="lightgbm_direct",
            params={
                "objective": "poisson",
                "n_estimators": 900,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "reg_lambda": 3.0,
                "reg_alpha": 0.2,
            },
        ),
        ModelSpec(
            name="lgbm_tweedie_v1",
            family="lightgbm_direct",
            params={
                "objective": "tweedie",
                "tweedie_variance_power": 1.2,
                "n_estimators": 900,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "reg_lambda": 3.0,
                "reg_alpha": 0.2,
            },
        ),
        ModelSpec(
            name="xgb_squaredlog_v1",
            family="xgboost_direct",
            params={
                "objective": "reg:squaredlogerror",
                "n_estimators": 900,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_lambda": 4.0,
                "reg_alpha": 0.2,
                "gamma": 0.05,
            },
        ),
        ModelSpec(
            name="xgb_squaredlog_v2",
            family="xgboost_direct",
            params={
                "objective": "reg:squaredlogerror",
                "n_estimators": 1300,
                "learning_rate": 0.018,
                "max_depth": 5,
                "min_child_weight": 45,
                "subsample": 0.75,
                "colsample_bytree": 0.65,
                "reg_lambda": 6.0,
                "reg_alpha": 0.4,
                "gamma": 0.08,
            },
        ),
        ModelSpec(
            name="xgb_poisson_v1",
            family="xgboost_direct",
            params={
                "objective": "count:poisson",
                "n_estimators": 1000,
                "learning_rate": 0.025,
                "max_depth": 4,
                "min_child_weight": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.65,
                "reg_lambda": 5.0,
                "reg_alpha": 0.3,
                "gamma": 0.05,
                "max_delta_step": 1.0,
            },
        ),
        ModelSpec(
            name="xgb_tweedie_v1",
            family="xgboost_direct",
            params={
                "objective": "reg:tweedie",
                "tweedie_variance_power": 1.2,
                "n_estimators": 900,
                "learning_rate": 0.025,
                "max_depth": 4,
                "min_child_weight": 35,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_lambda": 5.0,
                "reg_alpha": 0.3,
                "gamma": 0.05,
            },
        ),
        ModelSpec(
            name="catboost_conservative_v1",
            family="catboost",
            params={
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
            },
        ),
        ModelSpec(
            name="catboost_conservative_v2",
            family="catboost",
            params={
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 5,
                "l2_leaf_reg": 10.0,
                "random_strength": 2.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            },
        ),
        ModelSpec(
            name="catboost_conservative_v3",
            family="catboost",
            params={
                "iterations": 900,
                "learning_rate": 0.025,
                "depth": 5,
                "l2_leaf_reg": 12.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.75,
            },
        ),
        ModelSpec(
            name="catboost_poisson_v1",
            family="catboost_direct",
            params={
                "loss_function": "Poisson",
                "eval_metric": "RMSE",
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 10.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
            },
        ),
        ModelSpec(
            name="catboost_poisson_v2",
            family="catboost_direct",
            params={
                "loss_function": "Poisson",
                "eval_metric": "RMSE",
                "iterations": 1000,
                "learning_rate": 0.02,
                "depth": 5,
                "l2_leaf_reg": 12.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.75,
            },
        ),
        ModelSpec(
            name="catboost_tweedie_v1",
            family="catboost_direct",
            params={
                "loss_function": "Tweedie:variance_power=1.2",
                "eval_metric": "RMSE",
                "iterations": 800,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 12.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.75,
            },
        ),
        ModelSpec(
            name="poisson_glm_v1",
            family="poisson_glm",
            params={
                "alpha": 0.5,
                "max_iter": 1000,
                "tol": 1e-6,
            },
        ),
        ModelSpec(
            name="poisson_glm_v2",
            family="poisson_glm",
            params={
                "alpha": 1.0,
                "max_iter": 1000,
                "tol": 1e-6,
            },
        ),
        ModelSpec(
            name="tweedie_glm_p13_v1",
            family="tweedie_glm",
            params={
                "power": 1.3,
                "alpha": 0.5,
                "link": "log",
                "max_iter": 1000,
                "tol": 1e-6,
            },
        ),
        ModelSpec(
            name="tweedie_glm_p15_v1",
            family="tweedie_glm",
            params={
                "power": 1.5,
                "alpha": 1.0,
                "link": "log",
                "max_iter": 1000,
                "tol": 1e-6,
            },
        ),
        ModelSpec(
            name="extratrees_conservative_v1",
            family="extra_trees",
            params={
                "n_estimators": 500,
                "max_depth": 18,
                "min_samples_leaf": 8,
                "min_samples_split": 12,
                "max_features": 0.6,
                "bootstrap": False,
            },
        ),
        ModelSpec(
            name="extratrees_conservative_v2",
            family="extra_trees",
            params={
                "n_estimators": 700,
                "max_depth": 14,
                "min_samples_leaf": 12,
                "min_samples_split": 18,
                "max_features": 0.5,
                "bootstrap": False,
            },
        ),
    ]


def filter_model_registry(model_registry: list[ModelSpec], include_models_arg: str) -> list[ModelSpec]:
    if not include_models_arg.strip():
        return model_registry

    requested_names = [name.strip() for name in include_models_arg.split(",") if name.strip()]
    model_lookup = {spec.name: spec for spec in model_registry}
    missing = [name for name in requested_names if name not in model_lookup]
    if missing:
        raise ValueError(f"Requested model names not found in registry: {', '.join(missing)}")
    return [model_lookup[name] for name in requested_names]


def run_repeated_cv_for_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int,
    random_state: int,
) -> ModelResult:
    oof_sum = np.zeros(len(train_df), dtype=np.float64)
    oof_count = np.zeros(len(train_df), dtype=np.int32)
    test_sum = np.zeros(len(test_df), dtype=np.float64)
    fold_scores: list[dict[str, object]] = []

    for split_index, (train_idx, valid_idx) in enumerate(cv_splits):
        estimator = build_estimator(spec, numeric_columns, categorical_columns, random_state)
        estimator.fit(train_df.iloc[train_idx], y_train[train_idx])

        valid_pred = np.clip(estimator.predict(train_df.iloc[valid_idx]), 0.0, None)
        test_pred = np.clip(estimator.predict(test_df), 0.0, None)

        oof_sum[valid_idx] += valid_pred
        oof_count[valid_idx] += 1
        test_sum += test_pred

        fold_rmsle = rmsle(y_train[valid_idx], valid_pred)
        fold_scores.append(
            {
                "model_name": spec.name,
                "family": spec.family,
                "repeat_index": split_index // n_splits,
                "fold_index": split_index % n_splits,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "fold_rmsle": float(fold_rmsle),
            }
        )

    if np.any(oof_count == 0):
        raise ValueError(f"Model {spec.name} has rows without out-of-fold coverage.")

    oof_predictions = oof_sum / oof_count
    test_predictions = test_sum / len(cv_splits)
    return ModelResult(
        spec=spec,
        oof_rmsle=rmsle(y_train, oof_predictions),
        fold_scores=fold_scores,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
    )


def select_diverse_models(results: list[ModelResult], max_models: int) -> list[ModelResult]:
    selected: list[ModelResult] = []
    seen_families: set[str] = set()
    selected_names: set[str] = set()

    for result in sorted(results, key=lambda item: item.oof_rmsle):
        if result.spec.family not in seen_families:
            selected.append(result)
            seen_families.add(result.spec.family)
            selected_names.add(result.spec.name)
        if len(selected) >= max_models:
            break

    if len(selected) < max_models:
        for result in sorted(results, key=lambda item: item.oof_rmsle):
            if result.spec.name not in selected_names:
                selected.append(result)
                selected_names.add(result.spec.name)
            if len(selected) >= max_models:
                break

    return selected


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value.")
    return weights / total


def blend_prediction_matrix(prediction_matrix: np.ndarray, weights: np.ndarray, space: str) -> np.ndarray:
    clipped_matrix = np.clip(prediction_matrix, 0.0, None)
    if space == "raw":
        return clipped_matrix @ weights
    if space == "log":
        return np.expm1(np.log1p(clipped_matrix) @ weights)
    raise ValueError(f"Unsupported blend space: {space}")


def blend_candidate_name(base_name: str, space: str) -> str:
    if space == "raw":
        return base_name
    if space == "log":
        return base_name.replace("blend_", "blend_log_", 1)
    raise ValueError(f"Unsupported blend space: {space}")


def build_greedy_blend_candidate(
    ranked_results: list[ModelResult],
    y_train: np.ndarray,
    max_members: int,
    space: str,
) -> BlendCandidate | None:
    if len(ranked_results) < 2:
        return None

    selected_members = [ranked_results[0]]
    selected_weights = np.array([1.0], dtype=np.float64)
    if space == "raw":
        current_state = ranked_results[0].oof_predictions.copy()
        current_score = rmsle(y_train, current_state)
    elif space == "log":
        current_state = np.log1p(np.clip(ranked_results[0].oof_predictions, 0.0, None))
        current_score = rmsle(y_train, np.expm1(current_state))
    else:
        raise ValueError(f"Unsupported blend space: {space}")

    remaining = ranked_results[1:max_members]
    alpha_grid = np.arange(0.1, 0.55, 0.05)

    while remaining and len(selected_members) < max_members:
        best_candidate_result: ModelResult | None = None
        best_candidate_weights: np.ndarray | None = None
        best_candidate_state: np.ndarray | None = None
        best_candidate_score = current_score

        for candidate in remaining:
            candidate_state = (
                candidate.oof_predictions
                if space == "raw"
                else np.log1p(np.clip(candidate.oof_predictions, 0.0, None))
            )
            for alpha in alpha_grid:
                candidate_weights = np.concatenate([selected_weights * (1.0 - alpha), np.array([alpha])])
                blended_state = current_state * (1.0 - alpha) + candidate_state * alpha
                candidate_predictions = (
                    np.clip(blended_state, 0.0, None)
                    if space == "raw"
                    else np.expm1(blended_state)
                )
                candidate_score = rmsle(y_train, candidate_predictions)
                if candidate_score + 1e-9 < best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate_result = candidate
                    best_candidate_weights = candidate_weights
                    best_candidate_state = blended_state

        if best_candidate_result is None or best_candidate_weights is None or best_candidate_state is None:
            break

        selected_members.append(best_candidate_result)
        selected_weights = normalize_weights(best_candidate_weights)
        current_state = best_candidate_state
        current_score = best_candidate_score
        remaining = [candidate for candidate in remaining if candidate is not best_candidate_result]

    if len(selected_members) < 2:
        return None
    return BlendCandidate(
        name=blend_candidate_name("blend_greedy_forward", space),
        members=selected_members,
        weights=selected_weights,
        space=space,
    )


def build_equal_weight_subset_search_candidates(
    results: list[ModelResult],
    y_train: np.ndarray,
    max_models: int,
    max_subset_size: int,
) -> list[BlendCandidate]:
    ranked = sorted(results, key=lambda item: item.oof_rmsle)[: min(max_models, len(results))]
    if len(ranked) < 2:
        return []

    candidates: list[BlendCandidate] = []
    max_k = min(max_subset_size, len(ranked))
    for space in ("raw", "log"):
        for subset_size in range(2, max_k + 1):
            best_members: tuple[ModelResult, ...] | None = None
            best_score = float("inf")
            equal_weights = normalize_weights(np.ones(subset_size, dtype=np.float64))
            for subset in itertools.combinations(ranked, subset_size):
                oof_prediction_matrix = np.column_stack([result.oof_predictions for result in subset])
                oof_predictions = blend_prediction_matrix(oof_prediction_matrix, equal_weights, space)
                candidate_score = rmsle(y_train, oof_predictions)
                if candidate_score + 1e-9 < best_score:
                    best_members = subset
                    best_score = candidate_score

            if best_members is None:
                continue

            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name(f"blend_equal_search_best_k{subset_size}", space),
                    members=list(best_members),
                    weights=equal_weights,
                    space=space,
                )
            )
    return candidates


def build_blend_candidates(
    results: list[ModelResult],
    top_k_models: int,
    y_train: np.ndarray,
    equal_blend_search_models: int,
    max_equal_blend_size: int,
) -> list[BlendCandidate]:
    ranked = sorted(results, key=lambda item: item.oof_rmsle)
    top_ranked = ranked[: min(top_k_models, len(ranked))]
    diverse_ranked = select_diverse_models(ranked, min(top_k_models, len(ranked)))

    candidates: list[BlendCandidate] = []
    for space in ("raw", "log"):
        if len(top_ranked) >= 2:
            top2 = top_ranked[:2]
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top2_equal", space),
                    members=top2,
                    weights=normalize_weights(np.ones(2, dtype=np.float64)),
                    space=space,
                )
            )
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top2_inverse_rmsle", space),
                    members=top2,
                    weights=normalize_weights(np.array([1.0 / result.oof_rmsle for result in top2], dtype=np.float64)),
                    space=space,
                )
            )
        if len(top_ranked) >= 3:
            top3 = top_ranked[:3]
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top3_equal", space),
                    members=top3,
                    weights=normalize_weights(np.ones(3, dtype=np.float64)),
                    space=space,
                )
            )
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top3_inverse_rmsle", space),
                    members=top3,
                    weights=normalize_weights(np.array([1.0 / result.oof_rmsle for result in top3], dtype=np.float64)),
                    space=space,
                )
            )
        if len(top_ranked) >= 4:
            top4 = top_ranked[:4]
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top4_equal", space),
                    members=top4,
                    weights=normalize_weights(np.ones(4, dtype=np.float64)),
                    space=space,
                )
            )
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name("blend_top4_inverse_rmsle", space),
                    members=top4,
                    weights=normalize_weights(np.array([1.0 / result.oof_rmsle for result in top4], dtype=np.float64)),
                    space=space,
                )
            )
        if len(diverse_ranked) >= 2:
            candidate_name = f"blend_diverse_top{len(diverse_ranked)}_equal"
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name(candidate_name, space),
                    members=diverse_ranked,
                    weights=normalize_weights(np.ones(len(diverse_ranked), dtype=np.float64)),
                    space=space,
                )
            )
            candidates.append(
                BlendCandidate(
                    name=blend_candidate_name(f"blend_diverse_top{len(diverse_ranked)}_inverse_rmsle", space),
                    members=diverse_ranked,
                    weights=normalize_weights(
                        np.array([1.0 / result.oof_rmsle for result in diverse_ranked], dtype=np.float64)
                    ),
                    space=space,
                )
            )

        greedy_candidate = build_greedy_blend_candidate(
            ranked_results=ranked,
            y_train=y_train,
            max_members=min(max(top_k_models, 4), len(ranked)),
            space=space,
        )
        if greedy_candidate is not None:
            candidates.append(greedy_candidate)

    candidates.extend(
        build_equal_weight_subset_search_candidates(
            results=results,
            y_train=y_train,
            max_models=equal_blend_search_models,
            max_subset_size=max_equal_blend_size,
        )
    )

    deduplicated: dict[str, BlendCandidate] = {}
    for candidate in candidates:
        member_signature = ",".join(result.spec.name for result in candidate.members)
        key = f"{candidate.name}:{candidate.space}:{member_signature}"
        deduplicated[key] = candidate
    return list(deduplicated.values())


def evaluate_blend_candidates(
    candidates: list[BlendCandidate],
    y_train: np.ndarray,
) -> list[dict[str, object]]:
    blend_results: list[dict[str, object]] = []
    for candidate in candidates:
        oof_prediction_matrix = np.column_stack([result.oof_predictions for result in candidate.members])
        test_prediction_matrix = np.column_stack([result.test_predictions for result in candidate.members])
        oof_predictions = blend_prediction_matrix(oof_prediction_matrix, candidate.weights, candidate.space)
        test_predictions = blend_prediction_matrix(test_prediction_matrix, candidate.weights, candidate.space)
        blend_results.append(
            {
                "name": candidate.name,
                "family": "blend",
                "space": candidate.space,
                "members": [result.spec.name for result in candidate.members],
                "weights": [float(weight) for weight in candidate.weights],
                "oof_rmsle": float(rmsle(y_train, oof_predictions)),
                "oof_predictions": oof_predictions,
                "test_predictions": test_predictions,
            }
        )
    return sorted(blend_results, key=lambda item: item["oof_rmsle"])


def build_submission_frame(
    sample_submission_path: Path,
    test_ids: np.ndarray,
    predictions: np.ndarray,
    submission_format: str = DEFAULT_SUBMISSION_FORMAT,
) -> pl.DataFrame:
    return build_baseline_submission_frame(
        sample_submission_path=sample_submission_path,
        unique_ids=test_ids,
        predictions=predictions,
        submission_format=submission_format,
    )


def build_residual_segment_report(
    feature_table: pl.DataFrame,
    train_ids: np.ndarray,
    y_train: np.ndarray,
    predictions: np.ndarray,
) -> pl.DataFrame:
    candidate_columns = [
        "fin_missing_flag",
        "demo_customer_banking_type",
        "demo_income_category",
        "demo_low_income_flag",
        "demo_marital_status",
        "txn_active_months_total",
        "txn_months_count_le_2",
        "txn_recent_3m_count",
    ]
    available_columns = [column for column in candidate_columns if column in feature_table.columns]
    scored = (
        feature_table.filter(pl.col("__split__") == "train")
        .select(["UniqueID"] + available_columns)
        .join(
            pl.DataFrame(
                {
                    "UniqueID": pl.Series("UniqueID", train_ids),
                    "next_3m_txn_count_true": pl.Series("next_3m_txn_count_true", y_train),
                    "prediction": pl.Series("prediction", np.clip(predictions, 0.0, None)),
                }
            ),
            on="UniqueID",
            how="left",
        )
        .with_columns(
            [
                (pl.col("next_3m_txn_count_true") - pl.col("prediction")).alias("residual"),
                (
                    (pl.col("next_3m_txn_count_true") + 1.0).log()
                    - (pl.col("prediction").clip(lower_bound=0.0) + 1.0).log()
                )
                .abs()
                .alias("abs_log_err"),
                pl.when(pl.col("next_3m_txn_count_true") == 0)
                .then(pl.lit("0"))
                .when(pl.col("next_3m_txn_count_true") <= 3)
                .then(pl.lit("1_3"))
                .when(pl.col("next_3m_txn_count_true") <= 10)
                .then(pl.lit("4_10"))
                .when(pl.col("next_3m_txn_count_true") <= 30)
                .then(pl.lit("11_30"))
                .otherwise(pl.lit("31_plus"))
                .alias("segment_target_bin"),
            ]
        )
    )

    derived_segment_columns: list[tuple[str, str]] = [("target_bin", "segment_target_bin")]
    derived_exprs: list[pl.Expr] = []
    if "txn_active_months_total" in available_columns:
        derived_exprs.append(
            pl.when(pl.col("txn_active_months_total") <= 6)
            .then(pl.lit("01_06"))
            .when(pl.col("txn_active_months_total") <= 12)
            .then(pl.lit("07_12"))
            .when(pl.col("txn_active_months_total") <= 24)
            .then(pl.lit("13_24"))
            .otherwise(pl.lit("25_plus"))
            .alias("segment_active_months_bin")
        )
        derived_segment_columns.append(("active_months_bin", "segment_active_months_bin"))
    if "txn_months_count_le_2" in available_columns:
        derived_exprs.append(
            pl.when(pl.col("txn_months_count_le_2") == 0)
            .then(pl.lit("0"))
            .when(pl.col("txn_months_count_le_2") == 1)
            .then(pl.lit("1"))
            .when(pl.col("txn_months_count_le_2") <= 3)
            .then(pl.lit("2_3"))
            .otherwise(pl.lit("4_plus"))
            .alias("segment_sparse_month_bin")
        )
        derived_segment_columns.append(("sparse_month_bin", "segment_sparse_month_bin"))
    if "txn_recent_3m_count" in available_columns:
        derived_exprs.append(
            pl.when(pl.col("txn_recent_3m_count") <= 20)
            .then(pl.lit("00_20"))
            .when(pl.col("txn_recent_3m_count") <= 80)
            .then(pl.lit("21_80"))
            .when(pl.col("txn_recent_3m_count") <= 200)
            .then(pl.lit("81_200"))
            .otherwise(pl.lit("201_plus"))
            .alias("segment_recent_3m_count_bin")
        )
        derived_segment_columns.append(("recent_3m_count_bin", "segment_recent_3m_count_bin"))

    if derived_exprs:
        scored = scored.with_columns(derived_exprs)

    for column in [
        "fin_missing_flag",
        "demo_customer_banking_type",
        "demo_income_category",
        "demo_low_income_flag",
        "demo_marital_status",
    ]:
        if column in available_columns:
            derived_segment_columns.append((column, column))

    report_frames: list[pl.DataFrame] = []
    for segment_name, column_name in derived_segment_columns:
        report_frames.append(
            scored.group_by(column_name)
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("next_3m_txn_count_true").mean().alias("mean_target"),
                    pl.col("prediction").mean().alias("mean_prediction"),
                    pl.col("residual").mean().alias("mean_residual"),
                    pl.col("abs_log_err").mean().alias("mean_abs_log_err"),
                ]
            )
            .filter(pl.col("n") >= 50)
            .with_columns(
                [
                    pl.lit(segment_name).alias("segment_name"),
                    pl.col(column_name).cast(pl.String).fill_null("__NULL__").alias("segment_value"),
                ]
            )
            .select(
                [
                    "segment_name",
                    "segment_value",
                    "n",
                    "mean_target",
                    "mean_prediction",
                    "mean_residual",
                    "mean_abs_log_err",
                ]
            )
        )

    return (
        pl.concat(report_frames, how="vertical_relaxed")
        if report_frames
        else pl.DataFrame(
            schema={
                "segment_name": pl.String,
                "segment_value": pl.String,
                "n": pl.Int64,
                "mean_target": pl.Float64,
                "mean_prediction": pl.Float64,
                "mean_residual": pl.Float64,
                "mean_abs_log_err": pl.Float64,
            }
        )
    ).sort(["mean_abs_log_err", "n"], descending=[True, True])


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_artifacts(
    run_dir: Path,
    dataset_paths: DatasetPaths,
    feature_cache_path: Path,
    cache_mode: str,
    feature_table: pl.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    y_train: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_results: list[ModelResult],
    blend_results: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    write_json(
        run_dir / "config.json",
        {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "data_paths": asdict(dataset_paths),
            "feature_cache_path": str(feature_cache_path.resolve()),
            "feature_cache_mode": cache_mode,
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
            "model_registry": [asdict(result.spec) for result in model_results],
        },
    )
    write_json(
        run_dir / "feature_manifest.json",
        {
            "feature_table_rows": int(feature_table.height),
            "feature_table_columns": int(len(feature_table.columns)),
            "train_rows": int(len(train_ids)),
            "test_rows": int(len(test_ids)),
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "categorical_features": categorical_columns,
        },
    )

    fold_scores = [row for result in model_results for row in result.fold_scores]
    pd.DataFrame(fold_scores).to_csv(run_dir / "fold_scores.csv", index=False)

    model_score_rows = []
    for result in sorted(model_results, key=lambda item: item.oof_rmsle):
        fold_rmsles = [row["fold_rmsle"] for row in result.fold_scores]
        model_score_rows.append(
            {
                "model_name": result.spec.name,
                "family": result.spec.family,
                "oof_rmsle": result.oof_rmsle,
                "fold_rmsle_mean": float(np.mean(fold_rmsles)),
                "fold_rmsle_std": float(np.std(fold_rmsles)),
                "fit_count": len(result.fold_scores),
                "params_json": json.dumps(result.spec.params, sort_keys=True),
            }
        )
    pd.DataFrame(model_score_rows).to_csv(run_dir / "base_model_scores.csv", index=False)

    blend_score_rows = [
        {
            "blend_name": blend["name"],
            "space": blend["space"],
            "oof_rmsle": blend["oof_rmsle"],
            "members": ",".join(blend["members"]),
            "weights_json": json.dumps(blend["weights"]),
        }
        for blend in blend_results
    ]
    pd.DataFrame(blend_score_rows).to_csv(run_dir / "blend_scores.csv", index=False)

    oof_payload: dict[str, object] = {
        "UniqueID": train_ids,
        "next_3m_txn_count_true": y_train,
    }
    test_payload: dict[str, object] = {"UniqueID": test_ids}
    for result in model_results:
        oof_payload[f"pred_{result.spec.name}"] = result.oof_predictions
        test_payload[f"pred_{result.spec.name}"] = result.test_predictions
    for blend in blend_results:
        oof_payload[f"pred_{blend['name']}"] = blend["oof_predictions"]
        test_payload[f"pred_{blend['name']}"] = blend["test_predictions"]
    pl.DataFrame(oof_payload).write_parquet(run_dir / "oof_predictions.parquet")
    pl.DataFrame(test_payload).write_parquet(run_dir / "test_predictions.parquet")

    prediction_correlation = pd.DataFrame(
        {
            result.spec.name: result.oof_predictions
            for result in sorted(model_results, key=lambda item: item.spec.name)
        }
    ).corr()
    prediction_correlation.to_csv(run_dir / "oof_prediction_correlation.csv")

    best_base_model = min(model_results, key=lambda item: item.oof_rmsle)
    best_candidate_predictions = best_base_model.oof_predictions
    best_candidate_name = best_base_model.spec.name
    if blend_results:
        best_blend = min(blend_results, key=lambda item: item["oof_rmsle"])
        if best_blend["oof_rmsle"] < best_base_model.oof_rmsle:
            best_candidate_predictions = best_blend["oof_predictions"]
            best_candidate_name = best_blend["name"]
    residual_report = build_residual_segment_report(
        feature_table=feature_table,
        train_ids=train_ids,
        y_train=y_train,
        predictions=best_candidate_predictions,
    )
    residual_report.write_csv(run_dir / "residual_segment_report.csv")

    submissions_dir = run_dir / "submissions"
    submissions_raw_dir = run_dir / "submissions_raw"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submissions_raw_dir.mkdir(parents=True, exist_ok=True)
    for result in model_results:
        submission = build_baseline_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            result.test_predictions,
            submission_format=args.submission_format,
        )
        submission.write_csv(submissions_dir / f"{result.spec.name}.csv")
        raw_submission = build_baseline_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            result.test_predictions,
            submission_format="raw",
        )
        raw_submission.write_csv(submissions_raw_dir / f"{result.spec.name}.csv")
    for blend in blend_results:
        submission = build_baseline_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            blend["test_predictions"],
            submission_format=args.submission_format,
        )
        submission.write_csv(submissions_dir / f"{blend['name']}.csv")
        raw_submission = build_baseline_submission_frame(
            dataset_paths.sample_submission_path,
            test_ids,
            blend["test_predictions"],
            submission_format="raw",
        )
        raw_submission.write_csv(submissions_raw_dir / f"{blend['name']}.csv")

    best_blend = min(blend_results, key=lambda item: item["oof_rmsle"]) if blend_results else None
    summary_lines = [
        "# Experiment Summary",
        "",
        f"- Feature pipeline version: `{FEATURE_PIPELINE_VERSION}`",
        f"- Feature cache mode: `{cache_mode}`",
        f"- Repeated CV: `{args.n_splits}` folds x `{args.n_repeats}` repeats",
        f"- Submission artifact format: `{args.submission_format}`",
        f"- Best base model: `{best_base_model.spec.name}` with OOF RMSLE `{best_base_model.oof_rmsle:.6f}`",
    ]
    if best_blend is not None:
        summary_lines.append(
            f"- Best blend: `{best_blend['name']}` with OOF RMSLE `{best_blend['oof_rmsle']:.6f}`"
        )
        summary_lines.append(
            f"- Blend members: `{', '.join(best_blend['members'])}` in `{best_blend['space']}` space"
        )
    summary_lines.extend(
        [
            "",
            "## Base Models",
            "",
        ]
    )
    for row in model_score_rows:
        summary_lines.append(
            f"- `{row['model_name']}` ({row['family']}): OOF RMSLE `{row['oof_rmsle']:.6f}`, "
            f"fold mean `{row['fold_rmsle_mean']:.6f}`, fold std `{row['fold_rmsle_std']:.6f}`"
        )
    if blend_results:
        summary_lines.extend(["", "## Blend Candidates", ""])
        for row in blend_score_rows:
            summary_lines.append(
                f"- `{row['blend_name']}` ({row['space']}): OOF RMSLE `{float(row['oof_rmsle']):.6f}`, "
                f"members `{row['members']}`"
            )
    if residual_report.height > 0:
        summary_lines.extend(["", "## Hardest Segments", ""])
        for row in residual_report.head(8).to_dicts():
            summary_lines.append(
                f"- `{row['segment_name']}={row['segment_value']}`: n `{row['n']}`, "
                f"mean abs log error `{float(row['mean_abs_log_err']):.6f}`, "
                f"mean residual `{float(row['mean_residual']):.3f}`"
            )
        summary_lines.append("")
        summary_lines.append(
            f"- Full residual segment report: `residual_segment_report.csv` using `{best_candidate_name}` predictions"
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_paths = resolve_dataset_paths(args)
    run_dir = build_run_directory(args.output_dir, args.run_name)

    feature_table, cache_mode = load_or_build_feature_table(
        dataset_paths=dataset_paths,
        cache_path=args.feature_cache_path,
        refresh_feature_cache=args.refresh_feature_cache,
    )
    train_df, test_df, y_train, train_ids, test_ids, numeric_columns, categorical_columns = prepare_model_frames(
        feature_table
    )

    cv = RepeatedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    cv_splits = list(cv.split(train_df, y_train))

    model_registry = filter_model_registry(build_model_registry(), args.include_models)
    model_results = []
    for spec in model_registry:
        print(f"Running {spec.name}...")
        result = run_repeated_cv_for_model(
            spec=spec,
            train_df=train_df,
            test_df=test_df,
            y_train=y_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            cv_splits=cv_splits,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        model_results.append(result)
        print(f"  OOF RMSLE: {result.oof_rmsle:.6f}")

    blend_candidates = build_blend_candidates(
        model_results,
        args.top_k_blend_models,
        y_train,
        args.equal_blend_search_models,
        args.max_equal_blend_size,
    )
    blend_results = evaluate_blend_candidates(blend_candidates, y_train)
    for blend in blend_results:
        print(f"Blend {blend['name']}: {blend['oof_rmsle']:.6f}")

    write_artifacts(
        run_dir=run_dir,
        dataset_paths=dataset_paths,
        feature_cache_path=args.feature_cache_path,
        cache_mode=cache_mode,
        feature_table=feature_table,
        train_ids=train_ids,
        test_ids=test_ids,
        y_train=y_train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_results=model_results,
        blend_results=blend_results,
        args=args,
    )
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
