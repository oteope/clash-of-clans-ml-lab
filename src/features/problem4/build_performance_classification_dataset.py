"""Builder para el dataset de clasificación de clanes según rendimiento.

Reutiliza el dataset de regresión del Problema 3 y asigna una clase
de rendimiento (low/medium/high) en función de war_success_rate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# Variables que no deben usarse como features predictoras.
EXCLUDED_COLUMNS = [
    "war_wins",
    "war_losses",
    "war_ties",
    "war_win_streak",
    "war_points",
    "war_total",
    "war_success_rate",
]


def build_performance_classification_dataset(
    regression_df: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """
    Construye el dataset de clasificación a partir del dataset de regresión.

    Parameters
    ----------
    regression_df : pd.DataFrame
        Dataset de regresión del Problema 3 (clan_war_performance_regression.parquet).
        Debe contener al menos las columnas ``clan_tag`` y ``war_success_rate``.
    low_threshold : float
        Umbral inferior. Los clanes con ``war_success_rate < low_threshold``
        se clasifican como ``low``.
    high_threshold : float
        Umbral superior. Los clanes con ``war_success_rate >= high_threshold``
        se clasifican como ``high``. El resto se clasifica como ``medium``.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por clan, las features predictoras del Problema 3
        (excluyendo las variables de resultado bélico) y la columna
        ``performance_class`` con valores ``low``, ``medium`` o ``high``.
    """
    # Validación inicial: columnas imprescindibles.
    required_cols = {"clan_tag", "war_success_rate"}
    missing = required_cols - set(regression_df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en regression_df: {sorted(missing)}"
        )

    # Validación de thresholds.
    if low_threshold is None or high_threshold is None:
        raise ValueError("low_threshold y high_threshold son obligatorios.")
    if not (0.0 <= low_threshold <= 1.0):
        raise ValueError("low_threshold debe estar en [0, 1].")
    if not (0.0 <= high_threshold <= 1.0):
        raise ValueError("high_threshold debe estar en [0, 1].")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold debe ser estrictamente menor que high_threshold.")

    # Trabajamos sobre una copia para no modificar el DataFrame original.
    df = regression_df.copy()

    # 1. Validar war_success_rate (no nulo y dentro de [0, 1]).
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

    # 2. Garantizar clan_tag único.
    if df["clan_tag"].duplicated().any():
        duplicated_tags = df.loc[df["clan_tag"].duplicated(), "clan_tag"].unique()
        raise ValueError(
            "clan_tag debe ser único. Se encontraron duplicados: "
            f"{list(duplicated_tags[:5])}"
        )

    # 3. Crear la columna performance_class según los thresholds.
    conditions = [
        df["war_success_rate"] < low_threshold,
        (df["war_success_rate"] >= low_threshold)
        & (df["war_success_rate"] < high_threshold),
        df["war_success_rate"] >= high_threshold,
    ]
    choices = ["low", "medium", "high"]
    df["performance_class"] = np.select(conditions, choices, default="medium")

    # 4. Eliminar columnas que no deben usarse como features.
    cols_to_drop = [col for col in EXCLUDED_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df
