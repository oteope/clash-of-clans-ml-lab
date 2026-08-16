from typing import Optional

import numpy as np
import pandas as pd


# Variables que jamás pueden usarse como features,
# independientemente de los resultados del análisis.
ALWAYS_EXCLUDED = {"clan_rank", "previous_clan_rank", "role"}

# Variables que queremos auditar como posibles predictores/proxies.
CANDIDATE_VARIABLES = [
    "trophies",
    "town_hall_level",
    "exp_level",
    "war_stars",
    "attack_wins",
    "defense_wins",
    "donations",
    "donations_received",
    "capital_contributions",
    "builder_base_trophies",
]


def _ensure_player_tag_column(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.DataFrame:
    """
    Garantiza que el DataFrame tenga una columna ``player_tag``.

    Si ``player_tag`` ya es columna, devuelve una copia.
    Si está como índice, lo restaura como columna, soportando índices con
    nombre o sin él.
    """
    if "player_tag" in df.columns:
        return df.copy()

    df_reset = df.reset_index()

    # Si reset_index ya creó player_tag, lo usamos directamente.
    if "player_tag" in df_reset.columns:
        return df_reset

    # Si no existe player_tag y reset_index añadió exactamente una columna,
    # la primera columna es el índice original y debe contener player_tag.
    if len(df_reset.columns) == len(df.columns) + 1:
        first_col = df_reset.columns[0]
        df_reset = df_reset.rename(columns={first_col: "player_tag"})
        return df_reset

    # Si no pudimos identificar player_tag, y el primer caso no se dio,
    # reportamos un mensaje específico para facilitar el diagnóstico.
    raise KeyError(
        f"{df_name}: no se pudo identificar la columna 'player_tag'"
    )


def _merge_for_analysis(
    clan_members_df: pd.DataFrame,
    player_features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Prepara una tabla unida para auditar la relación con clan_rank.

    Si se proporcionan player_features, hace un inner join para añadir
    columnas como war_stars, attack_wins y defense_wins. En caso de colisión,
    conserva la columna original de clan_members.
    """
    if player_features_df is None:
        return clan_members_df.copy()

    clan_members = _ensure_player_tag_column(clan_members_df, "clan_members_df")
    player_features = _ensure_player_tag_column(player_features_df, "player_features_df")

    # Normalizar explícitamente player_tag como identificador string.
    clan_members["player_tag"] = clan_members["player_tag"].astype(str)
    player_features["player_tag"] = player_features["player_tag"].astype(str)

    merged = clan_members.merge(
        player_features,
        on="player_tag",
        how="inner",
        suffixes=("", "_player"),
    )

    # Eliminar columnas duplicadas provenientes de player_features
    duplicate_cols = [
        col
        for col in merged.columns
        if col.endswith("_player") and col[:-7] in merged.columns
    ]
    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)

    return merged


def _per_clan_spearman(merged: pd.DataFrame, variable: str) -> pd.Series:
    """Calcula la correlación de Spearman entre clan_rank y variable por clan."""

    def safe_corr(group: pd.DataFrame) -> float:
        if group["clan_rank"].nunique() < 2:
            return np.nan
        if group[variable].nunique() < 2:
            return np.nan
        return group["clan_rank"].corr(group[variable], method="spearman")

    return merged.groupby("clan_tag").apply(safe_corr)


def audit_clan_rank_proxies(
    clan_members_df: pd.DataFrame,
    player_features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Audita la relación de clan_rank con variables candidatas.

    Devuelve un DataFrame con una fila por variable y columnas:
    variable, median_spearman, p90_abs_spearman, pct_clans_abs_gt_90,
    classification, reason.
    """
    merged = _merge_for_analysis(clan_members_df, player_features_df)

    rows = []
    for variable in CANDIDATE_VARIABLES:
        if variable not in merged.columns:
            continue

        if variable in ALWAYS_EXCLUDED:
            rows.append(
                {
                    "variable": variable,
                    "median_spearman": np.nan,
                    "p90_abs_spearman": np.nan,
                    "pct_clans_abs_gt_90": np.nan,
                    "classification": "EXCLUDE",
                    "reason": "Variable prohibida por reglas de leakage.",
                }
            )
            continue

        corr_series = _per_clan_spearman(merged, variable).dropna()

        if corr_series.empty:
            rows.append(
                {
                    "variable": variable,
                    "median_spearman": np.nan,
                    "p90_abs_spearman": np.nan,
                    "pct_clans_abs_gt_90": np.nan,
                    "classification": "SAFE",
                    "reason": "No hay clanes suficientes para evaluar la relación.",
                }
            )
            continue

        median = float(np.nanmedian(corr_series))
        p90_abs = float(np.nanpercentile(np.abs(corr_series), 90))
        high_pct = float((np.abs(corr_series) > 0.9).mean())

        if abs(median) >= 0.95 and high_pct >= 0.8:
            classification = "TOO_DIRECT"
            reason = (
                "Relación clan-level casi determinista con clan_rank. "
                "Puede contener la misma información que el ranking."
            )
        elif abs(median) >= 0.8 or high_pct >= 0.5:
            classification = "POTENTIAL PROXY"
            reason = (
                "Fuerte relación con clan_rank. Posible proxy del ranking; "
                "debe revisarse antes de incluirla definitivamente."
            )
        else:
            classification = "SAFE"
            reason = "Relación moderada/leve; se mantiene como predictora legítima."

        rows.append(
            {
                "variable": variable,
                "median_spearman": round(median, 4),
                "p90_abs_spearman": round(p90_abs, 4),
                "pct_clans_abs_gt_90": round(high_pct, 4),
                "classification": classification,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def print_audit_report(analysis_df: pd.DataFrame) -> None:
    """Imprime una tabla legible con el resultado de la auditoría."""
    print("\n=== Auditoría de proxies de clan_rank ===")
    for _, row in analysis_df.iterrows():
        print(f"- {row['variable']:25s}: {row['classification']:11s} | {row['reason']}")
    print("==========================================\n")


def main(
    clan_members_df: Optional[pd.DataFrame] = None,
    player_features_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Punto de entrada para ejecutar el módulo desde consola.

    Carga los datos si no se proporcionan, ejecuta la auditoría y
    muestra un informe legible por consola.
    """
    if clan_members_df is None:
        try:
            from src.features.problem2.build_clan_rank_dataset import load_inputs

            clan_members_df, player_features_df, _ = load_inputs()
        except Exception as exc:
            print(f"Error cargando datos: {exc}")
            return

    merged_for_count = _merge_for_analysis(clan_members_df, player_features_df)
    n_obs = len(merged_for_count)
    print(f"Número de observaciones analizadas: {n_obs}")

    analysis_df = audit_clan_rank_proxies(clan_members_df, player_features_df)

    print_audit_report(analysis_df)

    trophies_row = analysis_df.loc[analysis_df["variable"] == "trophies"]
    if not trophies_row.empty:
        row = trophies_row.iloc[0]
        print("\nConclusión sobre trophies:")
        print(f"- Clasificación: {row['classification']}")
        print(f"- Correlación mediana: {row['median_spearman']}")
        print(f"- p90 abs: {row['p90_abs_spearman']}")
        print(f"- % clanes con |corr| > 0.9: {row['pct_clans_abs_gt_90']}")
        print(f"- Motivo: {row['reason']}")
    else:
        print("\nConclusión sobre trophies: variable no encontrada en el análisis")


if __name__ == "__main__":
    main()
