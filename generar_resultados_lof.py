"""Entrena LOF con el alimentador y genera el ranking de priorización."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "datos" / "DATA_ALIMENTADOR.xlsx"
OUTPUT_DIR = BASE_DIR / "salidas_reales"
MODEL_DIR = BASE_DIR / "modelos"
RANDOM_STATE = 42
MONTHS = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5,
    "JUNIO": 6, "JULIO": 7, "AGOSTO": 8, "SETIEMBRE": 9,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12,
}


def normalize(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def find_column(columns: pd.Index, *terms: str) -> str | None:
    for term in terms:
        normalized_term = normalize(term)
        for column in columns:
            if normalized_term in normalize(column):
                return str(column)
    return None


def month_columns(columns: pd.Index, prefix: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for column in columns:
        normalized = normalize(column)
        if normalize(prefix) not in normalized:
            continue
        for month, order in MONTHS.items():
            if month in normalized:
                found.append((order, str(column)))
                break
    return [column for _, column in sorted(found)]


def monthly_slope(values: pd.Series) -> float:
    numeric = np.asarray(values, dtype=float)
    valid = np.isfinite(numeric)
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(numeric))[valid], numeric[valid], 1)[0])


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    consumption_columns = month_columns(raw.columns, "CONSUMO")
    days_columns = month_columns(raw.columns, "DIA")
    if not consumption_columns:
        raise ValueError("No se encontraron columnas mensuales de consumo.")

    consumption = raw[consumption_columns].apply(pd.to_numeric, errors="coerce")
    if len(days_columns) == len(consumption_columns):
        days = raw[days_columns].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
        days.columns = consumption.columns
        daily = consumption.div(days)
    else:
        daily = consumption / 30.0

    features = pd.DataFrame(index=raw.index)
    features["daily_mean"] = daily.mean(axis=1)
    features["daily_median"] = daily.median(axis=1)
    features["daily_std"] = daily.std(axis=1)
    features["daily_min"] = daily.min(axis=1)
    features["daily_max"] = daily.max(axis=1)
    features["coefficient_variation"] = features["daily_std"] / features["daily_mean"].replace(0, np.nan)
    features["monthly_trend"] = daily.apply(monthly_slope, axis=1)
    features["recent_3m_daily_mean"] = daily.iloc[:, -3:].mean(axis=1)
    features["recent_vs_annual"] = features["recent_3m_daily_mean"] / features["daily_mean"].replace(0, np.nan)
    features["zero_month_ratio"] = (consumption.fillna(0) == 0).mean(axis=1)
    features["missing_month_ratio"] = consumption.isna().mean(axis=1)

    power = find_column(raw.columns, "POTENCIA_USUARIO_CONTRATADO", "POTENCIA")
    connection = find_column(raw.columns, "ACOMETIDA", "TIPO_CONEXIONADO")
    features["contracted_power_kw"] = pd.to_numeric(raw[power], errors="coerce") if power else np.nan
    features["connection_type"] = raw[connection].fillna("NO_DISPONIBLE").astype(str) if connection else "NO_DISPONIBLE"
    return features.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    raw = pd.read_excel(INPUT_PATH)
    features = build_features(raw)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    matrix = preprocessor.fit_transform(features)
    model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1)
    model.fit(matrix)
    raw_scores = model.score_samples(matrix)
    rank_percent = pd.Series(raw_scores).rank(method="first", pct=True).mul(100)
    anomaly_score = (100 - rank_percent).round(2)

    supply = find_column(raw.columns, "CUENTA_ID", "CUENTA", "SUMINISTRO")
    sed = find_column(raw.columns, "SED_ID", "SED")
    feeder = find_column(raw.columns, "ALIMENTADOR_ID", "ALIMENTADOR")
    result = pd.DataFrame({
        "codigo_suministro": raw[supply].astype(str) if supply else raw.index.astype(str),
        "sed_id": raw[sed].astype(str) if sed else "NO_DISPONIBLE",
        "alimentador_id": raw[feeder].astype(str) if feeder else "NO_DISPONIBLE",
        "score_anomalia": anomaly_score,
        "percentil_prioridad": rank_percent.round(3),
        "consumo_diario_promedio": features["daily_mean"].round(2),
        "tendencia_mensual": features["monthly_trend"].round(4),
        "meses_cero_pct": (features["zero_month_ratio"] * 100).round(1),
    })
    result["prioridad"] = pd.cut(
        result["percentil_prioridad"], bins=[-0.01, 1, 3, 5, 10, 100],
        labels=["CRITICO", "ALTO", "MEDIO", "BAJO", "NO_PRIORIZADO"], include_lowest=True,
    ).astype(str)
    result = result.sort_values(["score_anomalia", "codigo_suministro"], ascending=[False, True]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_DIR / "alimentador_priorizado_lof.csv", index=False, encoding="utf-8-sig")
    result[result["prioridad"] != "NO_PRIORIZADO"][["codigo_suministro"]].to_csv(
        OUTPUT_DIR / "codigos_sospechosos_lof_top10.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(model, MODEL_DIR / "lof_alimentador.joblib")
    joblib.dump({"preprocessor": preprocessor, "features": features.columns.tolist()}, MODEL_DIR / "preprocesador_lof.joblib")
    print(f"Suministros procesados: {len(result):,}")
    print(result["prioridad"].value_counts().reindex(["CRITICO", "ALTO", "MEDIO", "BAJO", "NO_PRIORIZADO"]).fillna(0).astype(int).to_string())
    print(f"Ranking: {OUTPUT_DIR / 'alimentador_priorizado_lof.csv'}")
    print(f"Lista para entrega: {OUTPUT_DIR / 'codigos_sospechosos_lof_top10.csv'}")


if __name__ == "__main__":
    main()
