"""Tests para el builder del dataset de clasificación de clanes (Problema 4)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Añadir src al path para importar features
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from features.problem4.build_performance_classification_dataset import (
    build_performance_classification_dataset,
)


LEAKAGE_COLUMNS = [
    "war_success_rate",
    "war_wins",
    "war_losses",
    "war_ties",
    "war_win_streak",
    "war_points",
    "war_total",
    "win_rate",
    "loss_rate",
    "tie_rate",
]


def make_regression_df(
    war_success_rates,
    include_features=True,
    include_excluded=True,
    include_variant_leakage=True,
):
    n = len(war_success_rates)
    data = {
        "clan_tag": [f"#CLAN{i}" for i in range(n)],
        "war_success_rate": war_success_rates,
    }
    if include_features:
        data["structural_feature_1"] = np.arange(n)
        data["composition_feature_2"] = np.linspace(0, 1, n)
    if include_excluded:
        data.update(
            {
                "war_wins": np.arange(n),
                "war_losses": np.arange(n, 2 * n),
                "war_ties": np.zeros(n, dtype=int),
                "war_win_streak": np.zeros(n, dtype=int),
                "war_points": np.full(n, 100),
                "war_total": np.arange(2 * n, 3 * n),
            }
        )
    if include_variant_leakage:
        data.update(
            {
                "win_rate": np.full(n, 0.5),
                "loss_rate": np.full(n, 0.3),
                "tie_rate": np.full(n, 0.2),
            }
        )
    return pd.DataFrame(data)


def test_granularity_one_row_per_clan():
    df = make_regression_df([0.1, 0.3, 0.5, 0.7, 0.9])
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    assert len(result) == len(df)
    assert result["clan_tag"].is_unique
    assert set(result["clan_tag"]) == set(df["clan_tag"])


def test_classification_thresholds():
    war_success = [0.20, 0.40, 0.59, 0.60, 0.90, 0.0, 1.0]
    expected = ["low", "medium", "medium", "high", "high", "low", "high"]

    df = make_regression_df(war_success, include_excluded=False, include_variant_leakage=False)
    result = build_performance_classification_dataset(df, 0.40, 0.60)

    mapping = dict(zip(result["clan_tag"], result["performance_class"]))
    for i, exp in enumerate(expected):
        tag = f"#CLAN{i}"
        assert mapping[tag] == exp


def test_target_exists_and_only_allowed_values():
    df = make_regression_df([0.1, 0.5, 0.9], include_excluded=False, include_variant_leakage=False)
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    assert "performance_class" in result.columns
    assert set(result["performance_class"]) == {"low", "medium", "high"}


def test_war_success_rate_not_left_in_features():
    df = make_regression_df([0.2, 0.5, 0.8])
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    assert "war_success_rate" not in result.columns


def test_leakage_columns_are_removed():
    df = make_regression_df([0.2, 0.5, 0.8])
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    for col in LEAKAGE_COLUMNS:
        assert col not in result.columns, f"Leakage column {col} presente"


def test_reuses_regression_features():
    df = make_regression_df(
        [0.2, 0.5, 0.8],
        include_features=True,
        include_excluded=True,
        include_variant_leakage=True,
    )
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    # Columnas que deberían conservarse
    preserved = ["structural_feature_1", "composition_feature_2"]
    for col in preserved:
        assert col in result.columns
        for tag in df["clan_tag"]:
            original_val = df.loc[df["clan_tag"] == tag, col].iloc[0]
            result_val = result.loc[result["clan_tag"] == tag, col].iloc[0]
            assert original_val == result_val

    # La única columna nueva debe ser performance_class
    assert set(result.columns) - set(df.columns) == {"performance_class"}


@pytest.mark.parametrize(
    "low, high",
    [
        (0.6, 0.4),      # low >= high
        (-0.1, 0.6),     # low fuera de rango
        (0.4, 1.1),      # high fuera de rango
        (0.2, 0.2),      # iguales
        (1.5, 2.0),      # ambos fuera de rango
    ],
)
def test_threshold_validation(low, high):
    df = make_regression_df([0.3, 0.5, 0.8])
    with pytest.raises(ValueError):
        build_performance_classification_dataset(df, low, high)


def test_war_success_rate_validation_nan_and_out_of_range():
    war_success = [float("nan"), -0.1, 0.3, 1.5, 0.7]
    df = make_regression_df(war_success)

    with pytest.warns(UserWarning):
        result = build_performance_classification_dataset(df, 0.4, 0.6)

    # Solo deben quedar los valores válidos: 0.3 y 0.7
    expected_tags = {"#CLAN2", "#CLAN4"}
    assert set(result["clan_tag"]) == expected_tags

    # No debe haber clases inválidas
    assert set(result["performance_class"]).issubset({"low", "medium", "high"})
    assert result["performance_class"].notna().all()


def test_immutability_does_not_modify_input():
    df = make_regression_df([0.2, 0.5, 0.8])
    df_before = df.copy(deep=True)

    build_performance_classification_dataset(df, 0.4, 0.6)

    pd.testing.assert_frame_equal(df, df_before)


def test_reproducibility_same_input_yields_same_output():
    df = make_regression_df([0.2, 0.5, 0.8])
    res1 = build_performance_classification_dataset(df, 0.4, 0.6)
    res2 = build_performance_classification_dataset(df, 0.4, 0.6)

    pd.testing.assert_frame_equal(res1, res2)


def test_distribution_all_classes_can_appear():
    war_success = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6]
    df = make_regression_df(war_success)
    result = build_performance_classification_dataset(df, 0.4, 0.6)

    assert set(result["performance_class"]) == {"low", "medium", "high"}
