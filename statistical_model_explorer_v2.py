"""
Interactive Statistical Model Explorer for the ADS road-readiness pilot.

What it does
------------
* Imports a CSV file.
* Lets the analyst select and optionally transform an outcome (Y).
* Automatically recommends a model from the observed Y values, with a manual
  override.
* Lets the analyst choose any combination of predictors (X).
* Creates derived predictors, including binary AND/OR combinations.
* Previews predictor representation on the exact complete-case analysis rows.
* Fits an inference-oriented statsmodels model and reports coefficients,
  robust standard errors, confidence intervals, p-values, fit statistics,
  diagnostics, and cross-validation results.
* Keeps a comparison history for models fitted during the current session.
* Exports a ZIP package containing all results from the latest model.

Install and run
---------------
Python 3.10+ is recommended.

    python -m pip install streamlit pandas numpy scipy statsmodels scikit-learn matplotlib
    streamlit run statistical_model_explorer.py

Then open the local address shown in the terminal, usually
http://localhost:8501.

Important interpretation note
-----------------------------
This application supports exploratory statistical association analysis. A
small p-value does not establish that a roadway/image characteristic caused a
change in detection performance. Repeated model searching also makes ordinary
p-values optimistic; confirm important findings on new data when possible.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import warnings
import zipfile
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor


APP_TITLE = "ADS Road-Readiness Statistical Model Explorer"

MODEL_AUTO = "Auto-select from Y"
MODEL_OLS = "Ordinary least squares (continuous Y)"
MODEL_FRACTIONAL = "Fractional logistic regression (0–1 Y)"
MODEL_BINARY = "Binary logistic regression (binary Y)"
MODEL_POISSON = "Poisson regression (count Y)"
MODEL_NEG_BINOMIAL = "Negative-binomial regression (overdispersed count Y)"

MODEL_CHOICES = [
    MODEL_AUTO,
    MODEL_OLS,
    MODEL_FRACTIONAL,
    MODEL_BINARY,
    MODEL_POISSON,
    MODEL_NEG_BINOMIAL,
]

OUTCOME_AS_STORED = "Use values as stored"
OUTCOME_EQUALS = "Convert to binary: 1 when Y equals a selected value"
OUTCOME_AT_LEAST = "Convert to binary: 1 when Y is at or above a threshold"
OUTCOME_BELOW = "Convert to binary: 1 when Y is below a threshold"

DERIVED_AND = "Binary AND (all selected columns equal 1)"
DERIVED_OR = "Binary OR (any selected column equals 1)"
DERIVED_PRODUCT = "Product / interaction"
DERIVED_SUM = "Sum"
DERIVED_MEAN = "Mean"
DERIVED_DIFFERENCE = "Difference (first minus second)"
DERIVED_RATIO = "Ratio (first divided by second)"
DERIVED_THRESHOLD = "Binary threshold (column at or above value)"

DERIVED_OPERATIONS = [
    DERIVED_AND,
    DERIVED_OR,
    DERIVED_PRODUCT,
    DERIVED_SUM,
    DERIVED_MEAN,
    DERIVED_DIFFERENCE,
    DERIVED_RATIO,
    DERIVED_THRESHOLD,
]


# Explicit widget keys keep the analyst's selections stable when Streamlit
# reruns after a derived variable is added or removed. Widgets whose option
# lists depend on the dataframe are validated before they are rendered.
ANALYSIS_WIDGET_KEYS = [
    "analysis_outcome_column",
    "analysis_outcome_treatment",
    "analysis_event_value",
    "analysis_outcome_threshold",
    "analysis_model_family",
    "analysis_positive_value",
    "analysis_predictors",
    "analysis_categorical_predictors",
    "analysis_previous_predictors",
    "analysis_group_choice",
    "analysis_covariance",
    "analysis_standardize_numeric",
    "analysis_cv_folds",
    "analysis_model_label",
]


def keep_valid_selectbox_state(
    key: str, options: list[Any], default: Any | None = None
) -> None:
    """Keep a saved selectbox value only when it is still available."""
    if not options:
        st.session_state.pop(key, None)
        return
    fallback = options[0] if default is None or default not in options else default
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = fallback


def keep_valid_multiselect_state(
    key: str, options: list[Any], default: list[Any] | None = None
) -> None:
    """Remove unavailable saved choices without clearing the valid choices."""
    allowed = set(options)
    if key not in st.session_state:
        st.session_state[key] = [value for value in (default or []) if value in allowed]
        return
    saved = st.session_state.get(key, [])
    st.session_state[key] = [value for value in saved if value in allowed]


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read common delimited text files, trying delimiter detection first."""
    try:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(raw))


def binary_one(series: pd.Series) -> pd.Series:
    """Treat numeric 1 and common text/boolean forms of true as one."""
    numeric = pd.to_numeric(series, errors="coerce").eq(1)
    text = series.astype("string").str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "t"}
    )
    return (numeric | text).astype(float)


def apply_derived_variables(
    source_df: pd.DataFrame, specs: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply saved derived-variable specifications in creation order."""
    df = source_df.copy()
    errors: list[str] = []

    for spec in specs:
        name = spec["name"]
        operation = spec["operation"]
        columns = spec["columns"]

        missing = [column for column in columns if column not in df.columns]
        if missing:
            errors.append(f"{name}: missing source column(s): {', '.join(missing)}")
            continue

        try:
            if operation == DERIVED_AND:
                flags = pd.concat([binary_one(df[column]) for column in columns], axis=1)
                df[name] = flags.all(axis=1).astype(int)
            elif operation == DERIVED_OR:
                flags = pd.concat([binary_one(df[column]) for column in columns], axis=1)
                df[name] = flags.any(axis=1).astype(int)
            elif operation in {DERIVED_PRODUCT, DERIVED_SUM, DERIVED_MEAN}:
                values = pd.concat(
                    [pd.to_numeric(df[column], errors="coerce") for column in columns],
                    axis=1,
                )
                if operation == DERIVED_PRODUCT:
                    df[name] = values.prod(axis=1, min_count=len(columns))
                elif operation == DERIVED_SUM:
                    df[name] = values.sum(axis=1, min_count=len(columns))
                else:
                    df[name] = values.mean(axis=1, skipna=False)
            elif operation == DERIVED_DIFFERENCE:
                first = pd.to_numeric(df[columns[0]], errors="coerce")
                second = pd.to_numeric(df[columns[1]], errors="coerce")
                df[name] = first - second
            elif operation == DERIVED_RATIO:
                numerator = pd.to_numeric(df[columns[0]], errors="coerce")
                denominator = pd.to_numeric(df[columns[1]], errors="coerce").replace(0, np.nan)
                df[name] = numerator / denominator
            elif operation == DERIVED_THRESHOLD:
                values = pd.to_numeric(df[columns[0]], errors="coerce")
                df[name] = np.where(values.isna(), np.nan, (values >= spec["value"]).astype(int))
        except Exception as exc:  # keep the app usable if one definition fails
            errors.append(f"{name}: {exc}")

    return df, errors


def infer_model(y_raw: pd.Series) -> tuple[str, str]:
    """Recommend a model from the nonmissing observed outcome values."""
    observed = y_raw.dropna()
    if observed.empty:
        raise ValueError("The selected outcome contains no usable values.")

    unique_count = observed.nunique(dropna=True)
    numeric = pd.to_numeric(observed, errors="coerce")

    if unique_count == 2:
        return MODEL_BINARY, "Y has exactly two observed values."

    if numeric.isna().any():
        raise ValueError(
            "Auto-selection supports nonnumeric Y only when it has exactly two values. "
            "Recode this outcome or select a numeric outcome."
        )

    if numeric.between(0, 1, inclusive="both").all():
        return MODEL_FRACTIONAL, "All observed Y values are between 0 and 1."

    is_integer = np.isclose(numeric.to_numpy(), np.round(numeric.to_numpy())).all()
    if is_integer and (numeric >= 0).all():
        mean_y = float(numeric.mean())
        variance_y = float(numeric.var(ddof=1)) if len(numeric) > 1 else 0.0
        if mean_y > 0 and variance_y > 1.25 * mean_y:
            return (
                MODEL_NEG_BINOMIAL,
                f"Y is a nonnegative count and appears overdispersed "
                f"(variance {variance_y:.3g} > mean {mean_y:.3g}).",
            )
        return MODEL_POISSON, "Y is a nonnegative integer count."

    return MODEL_OLS, "Y is numeric and is not binary, a 0–1 fraction, or a nonnegative count."


def transform_outcome(
    series: pd.Series,
    treatment: str,
    event_value: Any | None,
    threshold: float | None,
    final_model: str,
    stored_positive_value: Any | None,
) -> pd.Series:
    """Apply the selected outcome definition and return a numeric series."""
    if treatment == OUTCOME_EQUALS:
        return pd.Series(
            np.where(series.isna(), np.nan, (series == event_value).astype(int)),
            index=series.index,
            dtype=float,
        )

    if treatment in {OUTCOME_AT_LEAST, OUTCOME_BELOW}:
        numeric = pd.to_numeric(series, errors="coerce")
        if treatment == OUTCOME_AT_LEAST:
            values = numeric >= float(threshold)
        else:
            values = numeric < float(threshold)
        return pd.Series(
            np.where(numeric.isna(), np.nan, values.astype(int)),
            index=series.index,
            dtype=float,
        )

    if final_model == MODEL_BINARY:
        return pd.Series(
            np.where(series.isna(), np.nan, (series == stored_positive_value).astype(int)),
            index=series.index,
            dtype=float,
        )

    return pd.to_numeric(series, errors="coerce").astype(float)


def validate_outcome(y: pd.Series, model_name: str) -> None:
    observed = y.dropna()
    if observed.nunique() < 2:
        raise ValueError("Y must contain at least two distinct usable values.")

    if model_name == MODEL_BINARY and not set(observed.unique()).issubset({0.0, 1.0}):
        raise ValueError("Binary logistic regression requires Y coded as 0 and 1.")
    if model_name == MODEL_FRACTIONAL and not observed.between(0, 1).all():
        raise ValueError("Fractional logistic regression requires every Y value to be from 0 to 1.")
    if model_name in {MODEL_POISSON, MODEL_NEG_BINOMIAL}:
        if (observed < 0).any() or not np.isclose(observed, np.round(observed)).all():
            raise ValueError("Count regression requires nonnegative integer Y values.")


def retained_analysis_index(
    df: pd.DataFrame,
    y: pd.Series,
    x_columns: list[str],
    categorical_columns: list[str],
    group_column: str | None,
) -> pd.Index:
    """Return rows usable for Y, selected X variables, and an optional group ID."""
    if not x_columns:
        raise ValueError("Select at least one predictor.")

    raw_x = df[x_columns].copy()
    valid = y.notna() & raw_x.notna().all(axis=1)
    if group_column:
        valid &= df[group_column].notna()

    candidate_index = df.index[valid]
    converted = raw_x.loc[candidate_index].copy()
    for column in x_columns:
        if column not in categorical_columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")

    return converted.index[converted.notna().all(axis=1)]


def prepare_design(
    df: pd.DataFrame,
    y: pd.Series,
    x_columns: list[str],
    categorical_columns: list[str],
    group_column: str | None,
    standardize_numeric: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.Series | None, list[str], pd.Index]:
    """Create a complete-case numeric design matrix with reference-coded categories."""
    retained_index = retained_analysis_index(
        df, y, x_columns, categorical_columns, group_column
    )
    y_model = y.loc[retained_index].astype(float)
    raw_x = df.loc[retained_index, x_columns].copy()
    groups = df.loc[retained_index, group_column].copy() if group_column else None
    if len(y_model) < 10:
        raise ValueError("Fewer than 10 complete rows remain after missing values are removed.")

    for column in x_columns:
        if column not in categorical_columns:
            raw_x[column] = pd.to_numeric(raw_x[column], errors="coerce")

    design = pd.get_dummies(
        raw_x,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    ).astype(float)

    warnings_list: list[str] = []
    constant_columns = [column for column in design if design[column].nunique() <= 1]
    if constant_columns:
        design = design.drop(columns=constant_columns)
        warnings_list.append(
            "Dropped constant predictor column(s): " + ", ".join(constant_columns)
        )

    if design.empty:
        raise ValueError("No varying predictor columns remain after preprocessing.")

    if standardize_numeric:
        original_numeric = [
            column
            for column in x_columns
            if column not in categorical_columns and column in design.columns
        ]
        standardized: list[str] = []
        for column in original_numeric:
            unique_values = set(design[column].dropna().unique())
            if unique_values.issubset({0.0, 1.0}):
                continue
            std = float(design[column].std(ddof=0))
            if std > 0:
                design[column] = (design[column] - design[column].mean()) / std
                standardized.append(column)
        if standardized:
            warnings_list.append(
                "Standardized numeric predictor(s): " + ", ".join(standardized)
            )

    design = sm.add_constant(design, has_constant="add")
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank < design.shape[1]:
        raise ValueError(
            "The predictor matrix is rank-deficient. Remove a redundant predictor, "
            "category, or derived variable and try again."
        )

    return y_model, design, groups, warnings_list, design.index


def estimate_nb_alpha(y: pd.Series) -> float:
    """Method-of-moments dispersion estimate for the GLM NB2 variance."""
    mean_y = float(y.mean())
    variance_y = float(y.var(ddof=1)) if len(y) > 1 else mean_y
    if mean_y <= 0:
        return 1.0
    return max((variance_y - mean_y) / (mean_y**2), 1e-8)


def fit_statsmodels(
    model_name: str,
    y: pd.Series,
    design: pd.DataFrame,
    covariance: str,
    groups: pd.Series | None,
) -> tuple[Any, dict[str, Any]]:
    """Fit the requested model with the selected covariance estimator."""
    metadata: dict[str, Any] = {}
    if model_name == MODEL_OLS:
        model = sm.OLS(y, design)
    elif model_name in {MODEL_BINARY, MODEL_FRACTIONAL}:
        model = sm.GLM(y, design, family=sm.families.Binomial())
    elif model_name == MODEL_POISSON:
        model = sm.GLM(y, design, family=sm.families.Poisson())
    elif model_name == MODEL_NEG_BINOMIAL:
        alpha = estimate_nb_alpha(y)
        metadata["negative_binomial_alpha"] = alpha
        model = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    fit_kwargs: dict[str, Any] = {}
    if covariance == "HC3 robust":
        fit_kwargs["cov_type"] = "HC3"
    elif covariance == "Cluster-robust by group":
        if groups is None:
            raise ValueError("Select a group column to use cluster-robust standard errors.")
        if groups.nunique() < 10:
            raise ValueError("Cluster-robust inference requires at least 10 groups in this app.")
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": groups}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(**fit_kwargs)
    metadata["fit_warnings"] = sorted({str(item.message) for item in caught})
    return result, metadata


def calculate_vif(design: pd.DataFrame) -> pd.DataFrame:
    predictors = design.drop(columns=["const"], errors="ignore")
    if predictors.shape[1] < 2:
        return pd.DataFrame(columns=["term", "VIF"])
    rows = []
    for index, column in enumerate(predictors.columns):
        try:
            value = float(variance_inflation_factor(predictors.to_numpy(), index))
        except Exception:
            value = math.inf
        rows.append({"term": column, "VIF": value})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def predictor_representation_tables(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_columns: list[str],
    retained_index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize selected predictors on the exact rows retained for a model.

    Values are summarized before optional standardization so that percentages
    and descriptive statistics remain interpretable in their original units.
    """
    analysis = df.loc[retained_index, predictors].copy()
    analysis_rows = len(analysis)
    summary_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []

    true_tokens = {"1", "true", "yes", "y", "t"}
    false_tokens = {"0", "false", "no", "n", "f"}

    for predictor in predictors:
        series = analysis[predictor]
        nonmissing = series.dropna()
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_nonmissing = numeric.dropna()
        numeric_complete = len(numeric_nonmissing) == len(nonmissing)
        numeric_values = set(numeric_nonmissing.unique())

        binary: pd.Series | None = None
        if numeric_complete and numeric_values.issubset({0.0, 1.0}):
            binary = numeric.astype(float)
        else:
            normalized = nonmissing.astype("string").str.strip().str.lower()
            tokens = set(normalized.unique())
            if tokens and tokens.issubset(true_tokens | false_tokens):
                token_mapping = {
                    **{value: 1.0 for value in true_tokens},
                    **{value: 0.0 for value in false_tokens},
                }
                mapped = series.astype("string").str.strip().str.lower().map(
                    token_mapping
                )
                binary = mapped.astype(float)

        base_row: dict[str, Any] = {
            "predictor": predictor,
            "predictor_type": None,
            "analysis_rows": analysis_rows,
            "unique_values": int(nonmissing.nunique()),
            "count_1": np.nan,
            "percent_1": np.nan,
            "count_0": np.nan,
            "percent_0": np.nan,
            "minority_count": np.nan,
            "minority_percent": np.nan,
            "representation_flag": None,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }

        if binary is not None:
            count_1 = int((binary == 1).sum())
            count_0 = int((binary == 0).sum())
            percent_1 = 100 * count_1 / analysis_rows if analysis_rows else np.nan
            percent_0 = 100 * count_0 / analysis_rows if analysis_rows else np.nan
            minority_count = min(count_1, count_0)
            minority_percent = min(percent_1, percent_0)
            if minority_count == 0:
                representation_flag = "Constant in analysis rows"
            elif minority_percent < 1:
                representation_flag = "Extreme imbalance (<1% minority)"
            elif minority_percent < 5:
                representation_flag = "Imbalanced (<5% minority)"
            else:
                representation_flag = "Within 5%-95%"

            base_row.update(
                {
                    "predictor_type": "Binary",
                    "count_1": count_1,
                    "percent_1": percent_1,
                    "count_0": count_0,
                    "percent_0": percent_0,
                    "minority_count": minority_count,
                    "minority_percent": minority_percent,
                    "representation_flag": representation_flag,
                }
            )
        elif predictor in categorical_columns or not numeric_complete:
            counts = nonmissing.astype("string").value_counts(dropna=False)
            base_row.update(
                {
                    "predictor_type": "Categorical",
                    "representation_flag": "See categorical-level table",
                }
            )
            for level, count in counts.items():
                level_rows.append(
                    {
                        "predictor": predictor,
                        "level": str(level),
                        "count": int(count),
                        "percent": 100 * int(count) / analysis_rows if analysis_rows else np.nan,
                    }
                )
        else:
            base_row.update(
                {
                    "predictor_type": "Continuous numeric",
                    "representation_flag": "Not applicable",
                    "mean": float(numeric_nonmissing.mean()),
                    "std": float(numeric_nonmissing.std(ddof=1)),
                    "min": float(numeric_nonmissing.min()),
                    "p25": float(numeric_nonmissing.quantile(0.25)),
                    "median": float(numeric_nonmissing.median()),
                    "p75": float(numeric_nonmissing.quantile(0.75)),
                    "max": float(numeric_nonmissing.max()),
                }
            )

        summary_rows.append(base_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(level_rows)


def coefficient_table(result: Any, model_name: str) -> pd.DataFrame:
    confidence = result.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": np.asarray(result.params),
            "standard_error": np.asarray(result.bse),
            "test_statistic": np.asarray(result.tvalues),
            "p_value": np.asarray(result.pvalues),
            "ci_95_lower": np.asarray(confidence.iloc[:, 0]),
            "ci_95_upper": np.asarray(confidence.iloc[:, 1]),
        }
    )
    table["significance"] = pd.cut(
        table["p_value"],
        bins=[-np.inf, 0.001, 0.01, 0.05, 0.10, np.inf],
        labels=["p<0.001", "p<0.01", "p<0.05", "p<0.10", "not significant"],
    ).astype(str)
    if model_name in {MODEL_BINARY, MODEL_FRACTIONAL}:
        table["exp_coefficient"] = np.exp(table["coefficient"])
    return table


def calculate_fit_statistics(
    result: Any, y: pd.Series, predicted: np.ndarray, model_name: str
) -> dict[str, float | int | str]:
    residual = y.to_numpy() - predicted
    n = len(y)
    k = len(result.params)
    stats: dict[str, float | int | str] = {
        "observations": n,
        "estimated_parameters": k,
        "AIC": float(result.aic),
        "BIC_log_likelihood": float(-2 * result.llf + k * np.log(n)),
        "RMSE_in_sample": float(np.sqrt(np.mean(residual**2))),
        "MAE_in_sample": float(np.mean(np.abs(residual))),
    }

    if model_name == MODEL_OLS:
        stats["R_squared"] = float(result.rsquared)
        stats["adjusted_R_squared"] = float(result.rsquared_adj)
    else:
        null_deviance = float(getattr(result, "null_deviance", np.nan))
        deviance = float(getattr(result, "deviance", np.nan))
        if np.isfinite(null_deviance) and null_deviance > 0:
            stats["deviance_explained_pseudo_R2"] = 1 - deviance / null_deviance
        pearson = float(getattr(result, "pearson_chi2", np.nan))
        df_resid = float(getattr(result, "df_resid", np.nan))
        if np.isfinite(pearson) and df_resid > 0:
            stats["Pearson_dispersion"] = pearson / df_resid

    if model_name == MODEL_BINARY:
        stats["classification_accuracy_at_0.5"] = float(
            np.mean((predicted >= 0.5).astype(int) == y.to_numpy())
        )
        try:
            stats["ROC_AUC_in_sample"] = float(roc_auc_score(y, predicted))
        except ValueError:
            pass

    return stats


def cross_validate_model(
    model_name: str,
    y: pd.Series,
    design: pd.DataFrame,
    groups: pd.Series | None,
    requested_folds: int,
) -> tuple[dict[str, float | int | str], pd.Series, list[str]]:
    """Generate out-of-fold predictions using ordinary or grouped folds."""
    messages: list[str] = []
    n = len(y)

    if groups is not None:
        folds = min(requested_folds, int(groups.nunique()))
        if folds < 2:
            raise ValueError("At least two unique groups are required for grouped cross-validation.")
        splitter = GroupKFold(n_splits=folds)
        splits = splitter.split(design, y, groups)
        split_description = f"{folds}-fold grouped cross-validation"
    elif model_name == MODEL_BINARY:
        smallest_class = int(y.value_counts().min())
        folds = min(requested_folds, smallest_class)
        if folds < 2:
            raise ValueError("Each binary outcome class needs at least two observations for CV.")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        splits = splitter.split(design, y)
        split_description = f"{folds}-fold stratified cross-validation"
    else:
        folds = min(requested_folds, n)
        if folds < 2:
            raise ValueError("At least two folds are required for cross-validation.")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        splits = splitter.split(design)
        split_description = f"{folds}-fold cross-validation"

    oof = pd.Series(np.nan, index=y.index, dtype=float)
    failed_folds = 0

    for train_position, test_position in splits:
        x_train = design.iloc[train_position]
        x_test = design.iloc[test_position]
        y_train = y.iloc[train_position]
        try:
            fold_result, _ = fit_statsmodels(
                model_name, y_train, x_train, "Conventional", None
            )
            prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            oof.iloc[test_position] = prediction
        except Exception as exc:
            failed_folds += 1
            messages.append(f"A cross-validation fold failed: {exc}")

    valid = oof.notna()
    if valid.sum() == 0:
        raise ValueError("Cross-validation failed for every fold.")

    y_valid = y.loc[valid].to_numpy()
    prediction = oof.loc[valid].to_numpy()
    cv_stats: dict[str, float | int | str] = {
        "CV_method": split_description,
        "CV_predictions": int(valid.sum()),
        "CV_failed_folds": failed_folds,
        "CV_MAE": float(mean_absolute_error(y_valid, prediction)),
    }

    if model_name == MODEL_OLS:
        cv_stats["CV_primary_metric"] = "RMSE"
        cv_stats["CV_primary_value"] = float(np.sqrt(np.mean((y_valid - prediction) ** 2)))
    elif model_name in {MODEL_BINARY, MODEL_FRACTIONAL}:
        clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
        log_loss = -np.mean(y_valid * np.log(clipped) + (1 - y_valid) * np.log(1 - clipped))
        cv_stats["CV_primary_metric"] = "Log loss"
        cv_stats["CV_primary_value"] = float(log_loss)
        cv_stats["CV_Brier_or_fractional_MSE"] = float(np.mean((y_valid - clipped) ** 2))
        if model_name == MODEL_BINARY:
            try:
                cv_stats["CV_ROC_AUC"] = float(roc_auc_score(y_valid, clipped))
            except ValueError:
                pass
    else:
        positive_prediction = np.clip(prediction, 1e-12, None)
        cv_stats["CV_primary_metric"] = "Mean Poisson deviance"
        cv_stats["CV_primary_value"] = float(
            mean_poisson_deviance(y_valid, positive_prediction)
        )

    return cv_stats, oof, messages


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def results_zip(result_bundle: dict[str, Any], history: list[dict[str, Any]]) -> bytes:
    """Package the latest model's tables, summary, and configuration."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("coefficients.csv", result_bundle["coefficients"].to_csv(index=False))
        archive.writestr(
            "fit_statistics.csv",
            pd.DataFrame([result_bundle["fit_statistics"]]).to_csv(index=False),
        )
        archive.writestr(
            "cross_validation.csv",
            pd.DataFrame([result_bundle["cv_statistics"]]).to_csv(index=False),
        )
        archive.writestr("predictions.csv", result_bundle["predictions"].to_csv(index=False))
        archive.writestr("vif.csv", result_bundle["vif"].to_csv(index=False))
        if "predictor_representation" in result_bundle:
            archive.writestr(
                "predictor_representation.csv",
                result_bundle["predictor_representation"].to_csv(index=False),
            )
        categorical_representation = result_bundle.get("categorical_level_representation")
        if categorical_representation is not None and not categorical_representation.empty:
            archive.writestr(
                "categorical_level_representation.csv",
                categorical_representation.to_csv(index=False),
            )
        archive.writestr("model_summary.txt", result_bundle["summary_text"])
        archive.writestr(
            "run_configuration.json",
            json.dumps(result_bundle["configuration"], indent=2, default=safe_json_value),
        )
        archive.writestr("model_comparison_history.csv", pd.DataFrame(history).to_csv(index=False))
        archive.writestr(
            "README.txt",
            "Lower AIC/BIC and lower cross-validation loss are better. Compare AIC/BIC only "
            "for models using the same outcome, model family, and analysis rows. Treat p-values "
            "as exploratory after trying multiple predictor combinations. Coefficients describe "
            "adjusted statistical associations, not causal effects. Predictor representation "
            "is calculated from the exact retained model rows before optional standardization.\n",
        )
    return buffer.getvalue()


def render_predictor_representation(
    representation_source: pd.DataFrame,
    categorical_levels_source: pd.DataFrame,
) -> None:
    """Render binary, continuous, and categorical representation summaries."""
    representation = representation_source.copy()
    binary_representation = representation[
        representation["predictor_type"] == "Binary"
    ][
        [
            "predictor",
            "analysis_rows",
            "count_1",
            "percent_1",
            "count_0",
            "percent_0",
            "minority_count",
            "minority_percent",
            "representation_flag",
        ]
    ].copy()
    if not binary_representation.empty:
        for column in ["percent_1", "percent_0", "minority_percent"]:
            binary_representation[column] = binary_representation[column].round(3)
        st.markdown("**Binary predictors**")
        st.dataframe(binary_representation, use_container_width=True, hide_index=True)

    continuous_representation = representation[
        representation["predictor_type"] == "Continuous numeric"
    ][
        [
            "predictor",
            "analysis_rows",
            "unique_values",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
        ]
    ].copy()
    if not continuous_representation.empty:
        for column in ["mean", "std", "min", "p25", "median", "p75", "max"]:
            continuous_representation[column] = continuous_representation[column].round(3)
        st.markdown("**Continuous numeric predictors**")
        st.dataframe(continuous_representation, use_container_width=True, hide_index=True)

    categorical_levels = categorical_levels_source.copy()
    if not categorical_levels.empty:
        categorical_levels["percent"] = categorical_levels["percent"].round(3)
        with st.expander("Categorical predictor level representation"):
            st.dataframe(categorical_levels, use_container_width=True, hide_index=True)


def render_result(bundle: dict[str, Any]) -> None:
    st.header("Latest model results")
    st.caption(
        f"Run: {bundle['configuration']['model_label']} | "
        f"Outcome: {bundle['configuration']['outcome_column']} | "
        f"Model: {bundle['configuration']['model_name']}"
    )

    for message in bundle["messages"]:
        st.warning(message)

    fit_stats = bundle["fit_statistics"]
    cv_stats = bundle["cv_statistics"]
    cards = st.columns(4)
    cards[0].metric("Rows used", int(fit_stats["observations"]))
    cards[1].metric("Parameters", int(fit_stats["estimated_parameters"]))
    cards[2].metric("AIC", f"{fit_stats['AIC']:.3f}")
    cards[3].metric(
        f"CV {cv_stats['CV_primary_metric']}", f"{cv_stats['CV_primary_value']:.4f}"
    )

    st.subheader("Predictor representation in analysis rows")
    if "predictor_representation" not in bundle:
        st.info("Rerun the model to generate predictor-representation statistics.")
    else:
        render_predictor_representation(
            bundle["predictor_representation"],
            bundle["categorical_level_representation"],
        )
        st.caption(
            "Statistics use the exact rows retained for this model and the predictors' original "
            "values before optional standardization. Binary balance flags are screening aids only; "
            "the app does not automatically remove a predictor based on prevalence."
        )

    st.subheader("Coefficient estimates")
    st.dataframe(bundle["coefficients"], use_container_width=True, hide_index=True)
    st.caption(
        "A positive coefficient is associated with a higher expected Y, holding the other "
        "included predictors constant; a negative coefficient is associated with a lower "
        "expected Y. Logistic-family coefficients are on the log-odds scale."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Fit statistics")
        st.dataframe(
            pd.DataFrame(
                [{"statistic": key, "value": value} for key, value in fit_stats.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Cross-validation")
        st.dataframe(
            pd.DataFrame(
                [{"statistic": key, "value": value} for key, value in cv_stats.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    predictions = bundle["predictions"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(
        predictions["observed_Y"], predictions["fitted_Y"], alpha=0.65, edgecolor="none"
    )
    observed_min = float(predictions["observed_Y"].min())
    observed_max = float(predictions["observed_Y"].max())
    axes[0].plot([observed_min, observed_max], [observed_min, observed_max], "--", color="gray")
    axes[0].set_xlabel("Observed Y")
    axes[0].set_ylabel("Fitted Y")
    axes[0].set_title("Observed versus fitted")
    axes[1].hist(predictions["residual"], bins=min(25, max(5, len(predictions) // 8)))
    axes[1].set_xlabel("Observed minus fitted")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual distribution")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("Multicollinearity check (VIF)"):
        if bundle["vif"].empty:
            st.info("VIF requires at least two nonconstant predictors.")
        else:
            st.dataframe(bundle["vif"], use_container_width=True, hide_index=True)
            st.caption(
                "Large VIF values indicate unstable separation of predictor effects. "
                "Review overlapping tags and derived variables rather than relying on a rigid cutoff."
            )

    with st.expander("Complete statsmodels summary"):
        st.code(bundle["summary_text"], language="text")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Upload one image-level CSV, define the outcome and predictors, and fit repeated "
        "candidate models. The app recommends a model family from the outcome values but "
        "allows a documented manual override."
    )

    st.info(
        "Recommended use: predefine plausible predictors using engineering judgment, compare "
        "a limited number of models, and describe results as adjusted associations. Do not "
        "interpret the selected model as proof of causation."
    )

    uploaded = st.file_uploader("1. Upload the analysis CSV", type=["csv", "txt", "tsv"])
    if uploaded is None:
        st.stop()

    raw = uploaded.getvalue()
    file_signature = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("file_signature") != file_signature:
        st.session_state.file_signature = file_signature
        st.session_state.derived_specs = []
        st.session_state.run_history = []
        st.session_state.last_result = None
        # A different file can have completely different column names and
        # category values, so only a file change clears the analysis widgets.
        for key in ANALYSIS_WIDGET_KEYS:
            st.session_state.pop(key, None)

    try:
        base_df = read_csv_bytes(raw)
    except Exception as exc:
        st.error(f"The file could not be read as a delimited text file: {exc}")
        st.stop()

    if base_df.empty or base_df.shape[1] < 2:
        st.error("The CSV must contain at least one data row and at least two columns.")
        st.stop()

    df, derived_errors = apply_derived_variables(
        base_df, st.session_state.get("derived_specs", [])
    )
    for error in derived_errors:
        st.warning(error)

    st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
    with st.expander("Preview data and missing-value counts"):
        st.dataframe(df.head(25), use_container_width=True)
        missing = (
            df.isna().sum().rename("missing_rows").to_frame().assign(dtype=df.dtypes.astype(str))
        )
        st.dataframe(missing, use_container_width=True)

    st.header("2. Create optional predictors")
    st.caption(
        "Derived variables are recreated from the uploaded CSV on every run. Binary AND, for "
        "example, equals 1 only when every selected source column is coded as 1/true/yes."
    )

    operation = st.selectbox("Operation", DERIVED_OPERATIONS)
    available_columns = list(df.columns)
    if operation == DERIVED_THRESHOLD:
        source_columns = [st.selectbox("Source column", available_columns)]
        derived_value = st.number_input("Threshold", value=0.5, format="%.6f")
    elif operation in {DERIVED_DIFFERENCE, DERIVED_RATIO}:
        first_column = st.selectbox("First column", available_columns, key="derived_first")
        second_options = [column for column in available_columns if column != first_column]
        second_column = st.selectbox("Second column", second_options, key="derived_second")
        source_columns = [first_column, second_column]
        derived_value = None
    else:
        source_columns = st.multiselect("Source columns", available_columns)
        derived_value = None

    new_variable_name = st.text_input("New variable name")
    if st.button("Add derived variable"):
        clean_name = new_variable_name.strip()
        minimum_sources = 2 if operation != DERIVED_THRESHOLD else 1
        if not clean_name:
            st.error("Enter a name for the new variable.")
        elif clean_name in base_df.columns or any(
            spec["name"] == clean_name for spec in st.session_state.derived_specs
        ):
            st.error("That name already exists. Use a unique derived-variable name.")
        elif len(source_columns) < minimum_sources:
            st.error(f"Select at least {minimum_sources} source column(s).")
        else:
            st.session_state.derived_specs.append(
                {
                    "name": clean_name,
                    "operation": operation,
                    "columns": source_columns,
                    "value": derived_value,
                }
            )
            st.rerun()

    if st.session_state.derived_specs:
        spec_table = pd.DataFrame(st.session_state.derived_specs)
        spec_table["columns"] = spec_table["columns"].apply(lambda values: ", ".join(values))
        st.dataframe(spec_table, use_container_width=True, hide_index=True)
        remove_name = st.selectbox(
            "Derived variable to remove",
            [spec["name"] for spec in st.session_state.derived_specs],
        )
        if st.button("Remove selected derived variable"):
            st.session_state.derived_specs = [
                spec for spec in st.session_state.derived_specs if spec["name"] != remove_name
            ]
            st.session_state.last_result = None
            st.rerun()

    st.header("3. Define the outcome (Y)")
    outcome_options = list(df.columns)
    keep_valid_selectbox_state("analysis_outcome_column", outcome_options)
    outcome_column = st.selectbox(
        "Outcome column",
        outcome_options,
        key="analysis_outcome_column",
    )
    outcome_treatment_options = [
        OUTCOME_AS_STORED,
        OUTCOME_EQUALS,
        OUTCOME_AT_LEAST,
        OUTCOME_BELOW,
    ]
    keep_valid_selectbox_state(
        "analysis_outcome_treatment",
        outcome_treatment_options,
        OUTCOME_AS_STORED,
    )
    outcome_treatment = st.selectbox(
        "Outcome treatment",
        outcome_treatment_options,
        key="analysis_outcome_treatment",
    )

    event_value = None
    outcome_threshold = None
    if outcome_treatment == OUTCOME_EQUALS:
        outcome_values = list(df[outcome_column].dropna().unique())
        keep_valid_selectbox_state("analysis_event_value", outcome_values)
        event_value = st.selectbox(
            "Value treated as the event (Y = 1)",
            outcome_values,
            key="analysis_event_value",
        )
        y_for_inference = pd.Series(
            np.where(
                df[outcome_column].isna(),
                np.nan,
                (df[outcome_column] == event_value).astype(int),
            ),
            index=df.index,
        )
    elif outcome_treatment in {OUTCOME_AT_LEAST, OUTCOME_BELOW}:
        numeric_outcome = pd.to_numeric(df[outcome_column], errors="coerce")
        median_value = float(numeric_outcome.median()) if numeric_outcome.notna().any() else 0.0
        outcome_threshold = st.number_input(
            "Outcome threshold",
            value=median_value,
            format="%.6f",
            key="analysis_outcome_threshold",
        )
        comparison = (
            numeric_outcome >= outcome_threshold
            if outcome_treatment == OUTCOME_AT_LEAST
            else numeric_outcome < outcome_threshold
        )
        y_for_inference = pd.Series(
            np.where(numeric_outcome.isna(), np.nan, comparison.astype(int)), index=df.index
        )
    else:
        y_for_inference = df[outcome_column]

    try:
        recommended_model, recommendation_reason = infer_model(y_for_inference)
        st.write(f"Recommended model: **{recommended_model}**")
        st.caption(recommendation_reason + " Auto-selection is a heuristic; verify the outcome definition.")
    except ValueError as exc:
        recommended_model = MODEL_OLS
        recommendation_reason = str(exc)
        st.warning(str(exc))

    keep_valid_selectbox_state(
        "analysis_model_family", MODEL_CHOICES, MODEL_AUTO
    )
    selected_model = st.selectbox(
        "Model family", MODEL_CHOICES, key="analysis_model_family"
    )
    final_model = recommended_model if selected_model == MODEL_AUTO else selected_model

    stored_positive_value = None
    if outcome_treatment == OUTCOME_AS_STORED and final_model == MODEL_BINARY:
        outcome_values = list(df[outcome_column].dropna().unique())
        if len(outcome_values) < 2:
            st.error("A binary model requires two observed outcome values.")
            st.stop()
        keep_valid_selectbox_state("analysis_positive_value", outcome_values)
        stored_positive_value = st.selectbox(
            "Value treated as the event/positive outcome (Y = 1)",
            outcome_values,
            key="analysis_positive_value",
        )

    st.header("4. Select predictors (X) and analysis settings")
    candidate_predictors = [column for column in df.columns if column != outcome_column]
    keep_valid_multiselect_state("analysis_predictors", candidate_predictors)
    predictors = st.multiselect(
        "Predictor columns",
        candidate_predictors,
        key="analysis_predictors",
    )
    default_categorical = [
        column
        for column in predictors
        if not pd.api.types.is_numeric_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
    ]

    # Preserve earlier categorical choices. If the analyst just added a new
    # nonnumeric predictor, default only that new predictor to categorical
    # instead of rebuilding and resetting the whole selection.
    previous_predictors = set(
        st.session_state.get("analysis_previous_predictors", [])
    )
    if "analysis_categorical_predictors" not in st.session_state:
        categorical_default = default_categorical
    else:
        categorical_default = [
            column
            for column in st.session_state.analysis_categorical_predictors
            if column in predictors
        ]
        newly_selected = set(predictors) - previous_predictors
        categorical_default.extend(
            column
            for column in default_categorical
            if column in newly_selected and column not in categorical_default
        )
    st.session_state.analysis_categorical_predictors = categorical_default
    st.session_state.analysis_previous_predictors = list(predictors)
    categorical_predictors = st.multiselect(
        "Treat selected predictors as categorical",
        predictors,
        key="analysis_categorical_predictors",
        help="Use this for text categories and numeric category codes. One reference level is omitted.",
    )

    group_options = ["None"] + list(df.columns)
    keep_valid_selectbox_state("analysis_group_choice", group_options, "None")
    group_choice = st.selectbox(
        "Optional sequence/site/group ID",
        group_options,
        key="analysis_group_choice",
        help=(
            "Select a video, trip, roadway, or site ID when multiple rows are related. "
            "The app will keep each group within one cross-validation fold."
        ),
    )
    group_column = None if group_choice == "None" else group_choice

    covariance_options = ["HC3 robust", "Conventional"]
    if group_column:
        covariance_options.insert(0, "Cluster-robust by group")
    keep_valid_selectbox_state(
        "analysis_covariance", covariance_options, covariance_options[0]
    )
    covariance = st.selectbox(
        "Coefficient standard errors",
        covariance_options,
        key="analysis_covariance",
    )
    standardize_numeric = st.checkbox(
        "Standardize nonbinary numeric predictors",
        value=False,
        key="analysis_standardize_numeric",
    )
    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=10,
        value=5,
        key="analysis_cv_folds",
    )
    model_label = st.text_input(
        "Model label",
        value=f"{outcome_column} candidate {len(st.session_state.run_history) + 1}",
        key="analysis_model_label",
    )

    st.warning(
        "Do not use Y itself—or a variable calculated from Y—as a predictor. I metrics used as "
        "predictors should be calculated independently of the processor output used to calculate Y."
    )

    if predictors:
        try:
            preview_y = transform_outcome(
                df[outcome_column],
                outcome_treatment,
                event_value,
                outcome_threshold,
                final_model,
                stored_positive_value,
            )
            validate_outcome(preview_y, final_model)
            preview_index = retained_analysis_index(
                df,
                preview_y,
                predictors,
                categorical_predictors,
                group_column,
            )
            preview_representation, preview_categorical_levels = (
                predictor_representation_tables(
                    df,
                    predictors,
                    categorical_predictors,
                    preview_index,
                )
            )
            with st.expander(
                "Preview predictor representation after missing-value removal",
                expanded=True,
            ):
                st.write(
                    f"{len(preview_index):,} of {len(df):,} uploaded rows would be retained "
                    "for the current Y, X, and optional group-ID selections."
                )
                render_predictor_representation(
                    preview_representation,
                    preview_categorical_levels,
                )
                st.caption(
                    "Values are shown in their original units before optional standardization. "
                    "Representation flags do not automatically exclude predictors."
                )
        except Exception as exc:
            st.warning(f"Predictor-representation preview is unavailable: {exc}")

    if st.button("Run statistical model", type="primary"):
        try:
            y_transformed = transform_outcome(
                df[outcome_column],
                outcome_treatment,
                event_value,
                outcome_threshold,
                final_model,
                stored_positive_value,
            )
            validate_outcome(y_transformed, final_model)
            y_model, design, groups, prep_messages, retained_index = prepare_design(
                df,
                y_transformed,
                predictors,
                categorical_predictors,
                group_column,
                standardize_numeric,
            )
            predictor_representation, categorical_level_representation = (
                predictor_representation_tables(
                    df,
                    predictors,
                    categorical_predictors,
                    retained_index,
                )
            )
            result, fit_metadata = fit_statsmodels(
                final_model, y_model, design, covariance, groups
            )
            fitted = np.asarray(result.predict(design), dtype=float)
            fit_statistics = calculate_fit_statistics(result, y_model, fitted, final_model)
            cv_statistics, oof_prediction, cv_messages = cross_validate_model(
                final_model, y_model, design, groups, cv_folds
            )

            messages = prep_messages + fit_metadata.get("fit_warnings", []) + cv_messages
            parameter_count = int(design.shape[1])
            if len(y_model) < 10 * parameter_count:
                messages.append(
                    f"Only {len(y_model)} rows are available for {parameter_count} estimated "
                    "parameters. Coefficients and p-values may be unstable; simplify the model."
                )
            if final_model == MODEL_BINARY:
                class_counts = y_model.value_counts()
                if int(class_counts.min()) < 10 * max(1, parameter_count - 1):
                    messages.append(
                        "The smaller outcome class has relatively few observations for the number "
                        "of predictors. Binary-logit estimates may be unstable."
                    )

            vif = calculate_vif(design)
            if not vif.empty and np.isinf(vif["VIF"]).any():
                messages.append("At least one predictor has infinite VIF; review collinearity.")

            row_set_text = ",".join(map(str, retained_index.tolist()))
            row_set_id = hashlib.sha1(row_set_text.encode("utf-8")).hexdigest()[:10]
            predictions = pd.DataFrame(
                {
                    "source_dataframe_index": retained_index,
                    "observed_Y": y_model.to_numpy(),
                    "fitted_Y": fitted,
                    "out_of_fold_Y": oof_prediction.loc[retained_index].to_numpy(),
                    "residual": y_model.to_numpy() - fitted,
                }
            )

            configuration = {
                "run_time": datetime.now().isoformat(timespec="seconds"),
                "model_label": model_label.strip() or "Unnamed model",
                "source_file": uploaded.name,
                "outcome_column": outcome_column,
                "outcome_treatment": outcome_treatment,
                "event_value": event_value,
                "outcome_threshold": outcome_threshold,
                "positive_value": stored_positive_value,
                "model_name": final_model,
                "auto_recommendation_reason": recommendation_reason,
                "predictors": predictors,
                "categorical_predictors": categorical_predictors,
                "derived_variables": st.session_state.derived_specs,
                "group_column": group_column,
                "covariance": covariance,
                "standardized_numeric_predictors": standardize_numeric,
                "row_set_id": row_set_id,
                "negative_binomial_alpha": fit_metadata.get("negative_binomial_alpha"),
            }

            bundle = {
                "configuration": configuration,
                "coefficients": coefficient_table(result, final_model),
                "fit_statistics": fit_statistics,
                "cv_statistics": cv_statistics,
                "predictions": predictions,
                "vif": vif,
                "predictor_representation": predictor_representation,
                "categorical_level_representation": categorical_level_representation,
                "summary_text": result.summary().as_text(),
                "messages": messages,
            }
            st.session_state.last_result = bundle

            history_row = {
                "run_time": configuration["run_time"],
                "model_label": configuration["model_label"],
                "outcome": outcome_column,
                "outcome_treatment": outcome_treatment,
                "model": final_model,
                "predictors": " + ".join(predictors),
                "rows": fit_statistics["observations"],
                "parameters": fit_statistics["estimated_parameters"],
                "row_set_id": row_set_id,
                "AIC": fit_statistics["AIC"],
                "BIC": fit_statistics["BIC_log_likelihood"],
                "adjusted_R_squared": fit_statistics.get("adjusted_R_squared"),
                "pseudo_R_squared": fit_statistics.get("deviance_explained_pseudo_R2"),
                "CV_metric": cv_statistics["CV_primary_metric"],
                "CV_value": cv_statistics["CV_primary_value"],
            }
            st.session_state.run_history.append(history_row)
        except Exception as exc:
            st.error(f"The model could not be fitted: {exc}")

    if st.session_state.get("last_result") is not None:
        render_result(st.session_state.last_result)

        st.header("Model comparison history")
        history_df = pd.DataFrame(st.session_state.run_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        latest = st.session_state.last_result
        current_config = latest["configuration"]
        comparable = history_df[
            (history_df["outcome"] == current_config["outcome_column"])
            & (history_df["model"] == current_config["model_name"])
            & (history_df["row_set_id"] == current_config["row_set_id"])
            & (history_df["CV_metric"] == latest["cv_statistics"]["CV_primary_metric"])
        ]
        if not comparable.empty:
            best = comparable.sort_values("CV_value", ascending=True).iloc[0]
            st.success(
                f"Best directly comparable run by cross-validated {best['CV_metric']}: "
                f"{best['model_label']} ({best['CV_value']:.4f})."
            )
        st.caption(
            "Direct comparisons require the same outcome definition, model family, and row-set ID. "
            "Lower AIC, BIC, and cross-validation loss are better. Fit statistics should be "
            "considered together with coefficient stability, engineering plausibility, and diagnostics."
        )

        download_bytes = results_zip(
            st.session_state.last_result, st.session_state.run_history
        )
        st.download_button(
            "Download latest results package (ZIP)",
            data=download_bytes,
            file_name="ads_statistical_model_results.zip",
            mime="application/zip",
        )

        if st.button("Clear model comparison history"):
            st.session_state.run_history = []
            st.session_state.last_result = None
            st.rerun()

    st.divider()
    st.caption(
        "Suggested primary outcomes: one of IoU or F1 (not both as separate primary analyses), "
        "plus any genuinely distinct D metrics. If rows come from the same video or roadway "
        "sequence, select the sequence/site ID for grouped cross-validation and consider "
        "cluster-robust standard errors."
    )


if __name__ == "__main__":
    main()
