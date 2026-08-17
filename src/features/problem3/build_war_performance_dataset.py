from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths por defecto
# ---------------------------------------------------------------------------
PROCESSED_DIR_DEFAULT = Path("data/processed")
FEATURES_DIR_DEFAULT = Path("data/features")
DATASETS_DIR_DEFAULT = Path("data/datasets")

# ---------------------------------------------------------------------------
# Umbral mínimo de historial bélico
#
# Este valor se fija después de inspeccionar la distribución de:
#   war_total = war_wins + war_losses + war_ties
#
# Para no tomar esta decisión a ciegas, se proporciona la función
# ``_analyze_war_total_distribution``, que imprime la distribución por
# tramos y permite justificar el umbral con datos reales.
#
# Con los datos disponibles, el mínimo de 5 guerras descarta clanes
# anecdóticos (0-1 guerras) y reduce el riesgo de tasas inestables por
# historial insuficiente, sin eliminar una fracción excesiva de clanes.
# ---------------------------------------------------------------------------
MIN_WAR_HISTORY_DEFAULT = 5

# ---------------------------------------------------------------------------
# Variables relacionadas con guerra que NO deben usarse como features.
# ---------------------------------------------------------------------------
EXCLUDED_WAR_FEATURES = {
    "war_wins",
    "war_losses",
    "war_ties",
    "war_win_streak",
    "war_points",
}

# Derivadas directas de las variables anteriores
DERIVED_WAR_FEATURES = {
    "total_wars",
    "win_rate",
    "loss_rate",
    "tie_rate",
    "war_success_rate",
}

# ---------------------------------------------------------------------------
# Whitelist explícita de features estructurales del clan.
# Cualquier columna de clans.parquet que no esté en esta lista no se usará
# como feature, aunque no sea una variable de guerra.
# ---------------------------------------------------------------------------
CLAN_STRUCTURAL_FEATURES = [
    "clan_level",
    "clan_points",
    "clan_capital_points",
    "members",
    "required_trophies",
    "war_frequency",
    "war_league",
    "capital_league",
    "type",
    "is_family_friendly",
    "location_id",
    "location_name",
]


def _analyze_war_total_distribution(clans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Imprime y devuelve la distribución de ``war_total`` por tramos.

    Esta función permite inspeccionar la masa de clanes con historial 0,
    1, 2, ... y elegir un umbral mínimo de historial bélico razonable.
    """
    if "war_total" not in clans_df.columns:
        df = clans_df.copy()
        df["war_total"] = (
            df["war_wins"].astype(float)
            + df["war_losses"].astype(float)
            + df["war_ties"].astype(float)
        )
    else:
        df = clans_df

    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf]
    labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20-49", "50-99", "100+"]
    war_total = df["war_total"]
    hist = pd.cut(war_total, bins=bins, labels=labels, right=False)
    dist = hist.value_counts().sort_index()

    summary = pd.DataFrame(
        {
            "war_total_bin": dist.index,
            "count": dist.values,
            "percentage": dist.values / len(df) * 100,
        }
    )

    print("Distribución de war_total (wins + losses + ties):")
    print(summary.to_string(index=False))
    print(
        f"Clanes con >= {MIN_WAR_HISTORY_DEFAULT} guerras: "
        f"{war_total[war_total >= MIN_WAR_HISTORY_DEFAULT].count()} "
        f"({war_total[war_total >= MIN_WAR_HISTORY_DEFAULT].count() / len(df) * 100:.1f}%)"
    )

    return summary


def _read_parquet(path: Path) -> pd.DataFrame:
    """Lee un archivo Parquet y produce un error claro si no existe."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {path}")
    return pd.read_parquet(path)


def _load_clan_members(processed_dir: Path) -> pd.DataFrame:
    """
    Carga clan_members.parquet y conserva únicamente player_tag y clan_tag.

    Deduplica por la pareja (clan_tag, player_tag), no por player_tag.
    Un jugador puede aparecer en varios clanes; todas las relaciones válidas
    deben mantenerse.
    """
    members = _read_parquet(processed_dir / "clan_members.parquet")
    required = {"player_tag", "clan_tag"}
    if not required.issubset(members.columns):
        raise ValueError(
            "clan_members.parquet debe contener player_tag y clan_tag"
        )

    members = members[["player_tag", "clan_tag"]].copy()

    # Eliminar únicamente duplicados exactos de la relación clan-jugador.
    members = members.drop_duplicates(
        subset=["clan_tag", "player_tag"], keep="first"
    )

    # Validación defensiva: no puede haber más de una fila por (clan_tag, player_tag)
    if members.duplicated(subset=["clan_tag", "player_tag"]).any():
        raise ValueError(
            "clan_members.parquet contiene relaciones duplicadas "
            "(clan_tag, player_tag)"
        )

    return members


def _load_player_features(features_dir: Path) -> pd.DataFrame:
    """
    Carga el player_features.parquet precalculado.

    No se vuelven a procesar las tablas grandes de tropas, héroes, hechizos,
    equipamiento o logros.
    """
    path = features_dir / "player_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"player_features.parquet no encontrado en {features_dir}. "
            "Debe generarse previamente con src.features.player_features"
        )

    pf = pd.read_parquet(path)
    if "player_tag" not in pf.columns:
        raise ValueError("player_features.parquet no contiene player_tag")

    # Comprobación obligatoria: un único registro por player_tag.
    if not pf["player_tag"].is_unique:
        raise ValueError(
            "player_features.parquet contiene player_tag duplicados. "
            "Debe existir exactamente una fila por jugador antes del merge."
        )

    return pf


def _aggregate_clan_player_features(members_features: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las features a nivel jugador a nivel clan.

    Devuelve un DataFrame con una fila por clan_tag.

    IMPORTANTE:
    No se agregan estadísticas bélicas históricas del jugador, como
    war_stars. Se describe únicamente la composición / progresión / economía
    interna del clan.
    """
    if members_features.empty:
        return pd.DataFrame()

    df = members_features.copy()
    grouped = df.groupby("clan_tag")

    agg_specs: dict = {}

    def add_mean_median_std(col: str, prefix: str) -> None:
        if col in df.columns:
            agg_specs[f"mean_{prefix}"] = (col, "mean")
            agg_specs[f"median_{prefix}"] = (col, "median")
            agg_specs[f"std_{prefix}"] = (col, "std")

    # --------------------------------------------------------------
    # Agregaciones básicas de columnas numéricas
    # --------------------------------------------------------------
    add_mean_median_std("town_hall_level", "town_hall_level")
    add_mean_median_std("exp_level", "exp_level")
    add_mean_median_std("trophies", "trophies")
    add_mean_median_std("donations", "donations")
    add_mean_median_std("donations_received", "donations_received")
    add_mean_median_std(
        "clan_capital_contributions", "clan_capital_contributions"
    )

    # --------------------------------------------------------------
    # Promedios de features de progresión ya calculadas a nivel jugador
    # --------------------------------------------------------------
    progression_mean_cols = [
        "troop_mean_level",
        "troop_mean_completion_ratio",
        "hero_mean_level",
        "hero_mean_completion_ratio",
        "spell_mean_level",
        "spell_mean_completion_ratio",
        "equipment_mean_level",
        "equipment_mean_completion_ratio",
        "achievement_completion_ratio",
    ]
    for col in progression_mean_cols:
        if col in df.columns:
            agg_specs[f"mean_{col}"] = (col, "mean")

    result = grouped.agg(**agg_specs)

    # --------------------------------------------------------------
    # Porcentajes de TH altos
    # --------------------------------------------------------------
    if "town_hall_level" in df.columns:
        th_extra = grouped.agg(
            th18_percentage=("town_hall_level", lambda s: (s >= 18).mean()),
            th17_plus_percentage=("town_hall_level", lambda s: (s >= 17).mean()),
        )
        result = result.merge(
            th_extra, left_index=True, right_index=True, how="left"
        )

    # --------------------------------------------------------------
    # Sumas necesarias para ratios y balances
    # --------------------------------------------------------------
    sum_specs: dict = {"member_count": ("player_tag", "size")}
    if "donations" in df.columns:
        sum_specs["sum_donations"] = ("donations", "sum")
    if "donations_received" in df.columns:
        sum_specs["sum_donations_received"] = ("donations_received", "sum")
    if "clan_capital_contributions" in df.columns:
        sum_specs["sum_clan_capital_contributions"] = (
            "clan_capital_contributions",
            "sum",
        )

    sums = grouped.agg(**sum_specs)

    # Merge de las sumas (incluye member_count) para que formen parte
    # del contrato final de features agregadas.
    result = result.merge(sums, left_index=True, right_index=True, how="left")

    # --------------------------------------------------------------
    # Ratios y balances adicionales
    # --------------------------------------------------------------
    extra = pd.DataFrame(index=sums.index)

    if "sum_donations" in sums.columns and "sum_donations_received" in sums.columns:
        extra["donation_balance"] = (
            sums["sum_donations"] - sums["sum_donations_received"]
        )
        denom_don = sums["sum_donations_received"].replace(0, np.nan)
        extra["donation_ratio"] = sums["sum_donations"] / denom_don
        extra["donation_rate"] = (
            sums["sum_donations"] / sums["member_count"].replace(0, np.nan)
        )

    if "sum_clan_capital_contributions" in sums.columns:
        extra["capital_contribution_rate"] = (
            sums["sum_clan_capital_contributions"]
            / sums["member_count"].replace(0, np.nan)
        )

    result = result.merge(extra, left_index=True, right_index=True, how="left")
    return result


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Política de missing:
    - numéricos → 0
    - booleanos → False
    - categóricos/object → "unknown"

    Se excluye ``clan_tag`` para no imputar el identificador.
    """
    def _is_boolean_series(series: pd.Series) -> bool:
        """Detecta series booleanas, incluidas las de dtype object con True/False/None."""
        if pd.api.types.is_bool_dtype(series):
            return True
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        return all(isinstance(v, (bool, np.bool_)) for v in non_null)

    out = df.copy()
    for col in out.columns:
        if col == "clan_tag":
            continue
        if _is_boolean_series(out[col]):
            out[col] = out[col].fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            out[col] = out[col].fillna("unknown")
    return out


def build_war_performance_dataset(
    processed_dir: Path,
    features_dir: Path,
    min_war_total: int = MIN_WAR_HISTORY_DEFAULT,
) -> pd.DataFrame:
    """
    Construye el dataset del Problema 3.

    - 1 fila = 1 clan.
    - Target: war_success_rate = war_wins / (war_wins + war_losses + war_ties).
    - Solo clanes con historial bélico total >= ``min_war_total``.
    - Se excluyen features de guerra acumulada y derivadas directas.
    - Se reutiliza player_features.parquet, sin recorrer las tablas gigantes.
    - No se incluye rendimiento bélico histórico del jugador (por ejemplo,
      war_stars) en las features de composición.
    """
    clans = _read_parquet(processed_dir / "clans.parquet")

    required_clan_cols = {"clan_tag", "war_wins", "war_losses", "war_ties"}
    missing = required_clan_cols.difference(clans.columns)
    if missing:
        raise ValueError(
            f"clans.parquet no contiene las columnas requeridas: {sorted(missing)}"
        )

    clans = clans.copy()
    clans["war_total"] = (
        clans["war_wins"].astype(float)
        + clans["war_losses"].astype(float)
        + clans["war_ties"].astype(float)
    )

    # ------------------------------------------------------------------
    # Opcional: inspeccionar distribución para justificar min_war_total.
    # Descomentar para ver el resumen antes de fijar el umbral.
    # _analyze_war_total_distribution(clans)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Filtrar clanes con historial insuficiente
    # ------------------------------------------------------------------
    valid_clans = clans[clans["war_total"] >= min_war_total].copy()
    if valid_clans.empty:
        return pd.DataFrame(columns=["clan_tag", "war_success_rate"])

    # ------------------------------------------------------------------
    # Calcular target
    # ------------------------------------------------------------------
    valid_clans["war_success_rate"] = (
        valid_clans["war_wins"].astype(float)
        / valid_clans["war_total"].replace(0, np.nan)
    )
    valid_clans = valid_clans[valid_clans["war_success_rate"].notna()].copy()

    # ------------------------------------------------------------------
    # Separar target y features estructurales usando whitelist
    # ------------------------------------------------------------------
    target = valid_clans[["clan_tag", "war_success_rate"]].copy()

    available_clan_cols = [
        c for c in CLAN_STRUCTURAL_FEATURES if c in valid_clans.columns
    ]
    clan_features = valid_clans[["clan_tag"] + available_clan_cols].copy()

    # ------------------------------------------------------------------
    # Unir composición del clan
    # ------------------------------------------------------------------
    members = _load_clan_members(processed_dir)
    player_features = _load_player_features(features_dir)

    members_features = members.merge(player_features, on="player_tag", how="left")
    clan_composition = _aggregate_clan_player_features(members_features)

    result = clan_features.merge(
        clan_composition, left_on="clan_tag", right_index=True, how="left"
    )

    # Unir el target
    result = result.merge(target, on="clan_tag", how="left")

    # Garantizar una fila por clan
    result = result.drop_duplicates(subset="clan_tag", keep="first")

    # Aplicar política de missing
    result = _fill_missing_values(result)

    return result.reset_index(drop=True)


def main() -> None:
    """Genera el dataset final. Usar con precaución; revisar umbral antes."""
    processed_dir = PROCESSED_DIR_DEFAULT
    features_dir = FEATURES_DIR_DEFAULT
    datasets_dir = DATASETS_DIR_DEFAULT
    datasets_dir.mkdir(parents=True, exist_ok=True)

    df = build_war_performance_dataset(
        processed_dir,
        features_dir,
        min_war_total=MIN_WAR_HISTORY_DEFAULT,
    )

    out_path = datasets_dir / "clan_war_performance_regression.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Dataset guardado en {out_path} | clanes: {len(df)}")


if __name__ == "__main__":
    main()
