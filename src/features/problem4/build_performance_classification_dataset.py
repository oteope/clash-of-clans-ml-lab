"""Builder for the clan classification dataset by performance.

Reuse the Problem 3 regression dataset and assign a performance class
(low/medium/high) based on war_success_rate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# Variables that should not be used as predictive features.
EXCLUDED_COLUMNS = [
    "war_wins",
    "war_losses",
    "war_ties",
    "war_win_streak",
    "war_points",
    "war_total",
    "war_success_rate",
    "win_rate",
    "loss_rate",
    "tie_rate",
]


def build_performance_classification_dataset(
    regression_df: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """
    Build the classification dataset from the regression dataset.

    Parameters
    ----------
    regression_df : pd.DataFrame
        Problem 3 regression dataset (clan_war_performance_regression.parquet).
        It must contain at least the ``clan_tag`` and ``war_success_rate`` columns.
    low_threshold : float
        Lower threshold. Clans with ``war_success_rate < low_threshold``
        are classified as ``low``.
    high_threshold : float
        Upper threshold. Clans with ``war_success_rate >= high_threshold``
        are classified as ``high``. The remainder are classified as ``medium``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per clan, the Problem 3 predictive features
        (excluding war outcome variables), and the ``performance_class`` column
        with ``low``, ``medium``, or ``high`` values.
    """
    # Initial validation: required columns.
    required_cols = {"clan_tag", "war_success_rate"}
    missing = required_cols - set(regression_df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en regression_df: {sorted(missing)}"
        )

    # Threshold validation.
    if low_threshold is None or high_threshold is None:
        raise ValueError("low_threshold y high_threshold son obligatorios.")
    if not (0.0 <= low_threshold <= 1.0):
        raise ValueError("low_threshold debe estar en [0, 1].")
    if not (0.0 <= high_threshold <= 1.0):
        raise ValueError("high_threshold debe estar en [0, 1].")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold debe ser estrictamente menor que high_threshold.")

    # Work on a copy to avoid modifying the original DataFrame.
    df = regression_df.copy()

    # 1. Validate war_success_rate (non-null and within [0, 1]).
    invalid_mask = (
        df["war_success_rate"].isna()
        | (df["war_success_rate"] < 0.0)
        | (df["war_success_rate"] > 1.0)
    )
    if invalid_mask.any():
        n_dropped = int(invalid_mask.sum())
        warnings.warn(
            f"Se descartan {n_dropped} observaciones con war_success_rate "
            "fuera del rango [0, 1] o nulo.",
            UserWarning,
        )
        df = df.loc[~invalid_mask].copy()

    if df.empty:
        raise ValueError(
            "No quedan observaciones válidas después de filtrar war_success_rate."
        )

    # 2. Ensure clan_tag is unique.
    if df["clan_tag"].duplicated().any():
        duplicated_tags = df.loc[df["clan_tag"].duplicated(), "clan_tag"].unique()
        raise ValueError(
            "clan_tag debe ser único. Se encontraron duplicados: "
            f"{list(duplicated_tags[:5])}"
        )

    # 3. Create the performance_class column according to the thresholds.
    conditions = [
        df["war_success_rate"] < low_threshold,
        (df["war_success_rate"] >= low_threshold)
        & (df["war_success_rate"] < high_threshold),
        df["war_success_rate"] >= high_threshold,
    ]
    choices = ["low", "medium", "high"]
    df["performance_class"] = np.select(conditions, choices, default="medium")

    # 4. Remove columns that should not be used as features.
    cols_to_drop = [col for col in EXCLUDED_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df
