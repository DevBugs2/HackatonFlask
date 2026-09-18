"""Panel Flask para priorizar inspecciones con Local Outlier Factor."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
RESULT_PATH = BASE_DIR / "salidas_reales" / "alimentador_priorizado_lof.csv"
MAX_MAP_POINTS = 1_200
SAYLLA_CENTER = (-13.57025, -71.82714)
PRIORITIES = ["CRITICO", "ALTO", "MEDIO", "BAJO", "NO_PRIORIZADO"]

app = Flask(__name__)


def _stable_location(code: str) -> tuple[str, float, float]:
    """Ubicación demostrativa estable: la fuente no contiene coordenadas reales."""
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    sector_index = digest[0] % 4
    sectors = ["Norte", "Centro", "Este", "Sur"]
    centers = [(-13.5655, -71.8275), (-13.5702, -71.8271), (-13.5705, -71.8209), (-13.5750, -71.8285)]
    latitude, longitude = centers[sector_index]
    return (
        sectors[sector_index],
        latitude + ((digest[1] / 255) - 0.5) * 0.0032,
        longitude + ((digest[2] / 255) - 0.5) * 0.0036,
    )


@lru_cache(maxsize=1)
def dashboard_data() -> pd.DataFrame:
    if not RESULT_PATH.exists():
        raise FileNotFoundError("Ejecute primero: python generar_resultados_lof.py")
    data = pd.read_csv(RESULT_PATH, dtype={"codigo_suministro": str, "sed_id": str, "alimentador_id": str})
    required = {"codigo_suministro", "score_anomalia", "prioridad"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"El ranking no contiene: {', '.join(sorted(missing))}")
    locations = data["codigo_suministro"].map(_stable_location)
    data[["sector", "latitude", "longitude"]] = pd.DataFrame(locations.tolist(), index=data.index)
    data["prioridad"] = data["prioridad"].astype(str).str.upper()
    data["score_anomalia"] = pd.to_numeric(data["score_anomalia"], errors="coerce").fillna(0)
    return data


def _filtered_data() -> pd.DataFrame:
    data = dashboard_data().copy()
    priority = request.args.get("priority", "TODOS").upper()
    sector = request.args.get("sector", "TODOS").upper()
    if priority in PRIORITIES:
        data = data[data["prioridad"] == priority]
    if sector != "TODOS":
        data = data[data["sector"].str.upper() == sector]
    return data


def _map_sample(data: pd.DataFrame) -> pd.DataFrame:
    priority_rows = data[data["prioridad"] != "NO_PRIORIZADO"]
    if len(priority_rows) >= MAX_MAP_POINTS:
        return priority_rows.head(MAX_MAP_POINTS)
    remainder = data.drop(index=priority_rows.index)
    context_count = min(MAX_MAP_POINTS - len(priority_rows), len(remainder))
    context = remainder.sample(context_count, random_state=42) if context_count else remainder.head(0)
    return pd.concat([priority_rows, context])


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/documentacion")
def documentation():
    try:
        distribution = dashboard_data()["prioridad"].value_counts().reindex(PRIORITIES, fill_value=0).to_dict()
    except (FileNotFoundError, ValueError):
        distribution = {priority: 0 for priority in PRIORITIES}
    return render_template(
        "documentacion.html",
        validation={"top_1": 6.3, "top_3": 74.4, "top_5": 91.2, "top_10": 97.1},
        distribution=distribution,
    )


@app.get("/api/dashboard")
def dashboard():
    try:
        data = _filtered_data()
        all_data = dashboard_data()
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    sector_summary = (
        all_data.groupby("sector", as_index=False)
        .agg(suministros=("codigo_suministro", "size"), score_promedio=("score_anomalia", "mean"))
        .sort_values("score_promedio", ascending=False)
    )
    summary = {
        "total": int(len(data)),
        "critical": int((data["prioridad"] == "CRITICO").sum()),
        "high": int((data["prioridad"] == "ALTO").sum()),
        "average_risk": round(float(data["score_anomalia"].mean()), 1) if len(data) else 0,
        "distribution": all_data["prioridad"].value_counts().reindex(PRIORITIES, fill_value=0).to_dict(),
    }
    return jsonify({
        "summary": summary,
        "points": _map_sample(data).replace({np.nan: None}).to_dict(orient="records"),
        "sectors": sector_summary.round(1).to_dict(orient="records"),
    })


@app.get("/api/supplies")
def supplies():
    try:
        data = _filtered_data()
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    search = request.args.get("search", "").strip().lower()
    if search:
        mask = (
            data["codigo_suministro"].str.lower().str.contains(search, na=False)
            | data["sed_id"].str.lower().str.contains(search, na=False)
            | data["sector"].str.lower().str.contains(search, na=False)
        )
        data = data[mask]
    data = data.sort_values(["score_anomalia", "codigo_suministro"], ascending=[False, True])
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 12
    total = len(data)
    start = (page - 1) * per_page
    records = data.iloc[start : start + per_page].replace({np.nan: None}).to_dict(orient="records")
    return jsonify({"records": records, "page": page, "pages": max((total + per_page - 1) // per_page, 1), "total": total})


@app.get("/reporte/<codigo_suministro>")
def report(codigo_suministro: str):
    try:
        data = dashboard_data()
    except (FileNotFoundError, ValueError) as error:
        abort(503, str(error))
    match = data[data["codigo_suministro"] == codigo_suministro]
    if match.empty:
        abort(404)
    supply = match.iloc[0].to_dict()
    supply["inspection_recommendation"] = (
        "Priorizar inspección de campo" if supply["prioridad"] in {"CRITICO", "ALTO", "MEDIO", "BAJO"}
        else "Mantener en monitoreo y revisar en la siguiente programación"
    )
    return render_template("report.html", supply=supply)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
