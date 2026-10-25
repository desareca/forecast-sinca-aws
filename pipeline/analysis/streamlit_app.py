"""Dashboard Streamlit de análisis de modelos (Fase 3.5).

Uso local (contenedor o venv):
    streamlit run pipeline/analysis/streamlit_app.py --server.port 8501

Fuente de datos:
- **S3 (real)**: descarga `s3://sinca-mlflow/_mlflow/` (necesita credenciales).
- **Fixture local**: `pipeline/analysis/fixtures/state/` (sin AWS), para iterar.

El frontend es delgado: la carga vive en `load` y las figuras en `plots`.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

_HERE = pathlib.Path(__file__).resolve().parent
_PIPELINE = _HERE.parent
for _path in (str(_PIPELINE), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analysis import load, plots  # noqa: E402

FIXTURE_STATE = _HERE / "fixtures" / "state"

st.set_page_config(page_title="SINCA — Análisis de modelos", layout="wide")


@st.cache_data(show_spinner="Cargando estado de MLflow…")
def _ensure_state(data_dir: str | None) -> str:
    return load.ensure_state(data_dir=data_dir)


@st.cache_data(show_spinner=False)
def _list_runs(state_dir: str) -> list[dict]:
    return load.list_runs(state_dir)


@st.cache_data(show_spinner=False)
def _artifacts(state_dir: str, run_id: str) -> dict:
    return load.load_artifacts(state_dir, run_id)


@st.cache_data(show_spinner=False)
def _zone_icap(state_dir: str, runs: list[dict]) -> pd.DataFrame:
    return load.zone_icap_from_runs(state_dir, runs)


def _source_selector() -> str | None:
    options = ["S3 (real)"]
    if FIXTURE_STATE.is_dir():
        options.append("Fixture local")
    choice = st.sidebar.radio("Fuente de datos", options, index=len(options) - 1 if len(options) > 1 else 0)
    if choice == "Fixture local":
        return str(FIXTURE_STATE)
    return None


def _run_label(run: dict) -> str:
    params = run["params"]
    if params.get("tipo") == "persistencia":
        return "persistencia (referencia)"
    return f"{run['run_name']} | {params.get('estacion', '?')} / {params.get('pollutant', '?')}"


def main() -> None:
    st.title("SINCA — Análisis de modelos")
    st.caption("Baseline Fase 3 · series real vs predicho, ICAP_zona, feature importance y RMSE vs persistencia")

    data_dir = _source_selector()
    try:
        state_dir = _ensure_state(data_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo cargar el estado de MLflow: {exc}")
        st.stop()

    runs = _list_runs(state_dir)
    if not runs:
        st.warning("No hay runs en el estado de MLflow.")
        st.stop()

    training_runs = [run for run in runs if run["params"].get("tipo") != "persistencia"]
    if not training_runs:
        st.warning("No hay runs de entrenamiento (solo persistencia).")
        st.stop()

    # --- Sidebar: run y serie ------------------------------------------------
    labels = {_run_label(run): run for run in training_runs}
    selected_label = st.sidebar.selectbox("Run", list(labels.keys()))
    selected_run = labels[selected_label]

    series = load.series_options(training_runs)
    series_labels = [f"{station} / {pollutant}" for station, pollutant in series]
    selected_series = st.sidebar.selectbox("Serie", series_labels)
    station, pollutant = series[series_labels.index(selected_series)]

    split = st.sidebar.radio("Split (series)", ["test", "valid"], horizontal=True)

    st.sidebar.markdown("---")
    metrics = selected_run["metrics"]
    st.sidebar.metric("test RMSE", f"{metrics.get('test_rmse', float('nan')):.2f}")
    st.sidebar.metric("test weighted RMSE", f"{metrics.get('test_weighted_rmse', float('nan')):.2f}")
    st.sidebar.metric(
        "rolling weighted RMSE (mean)",
        f"{metrics.get('rolling_weighted_rmse_mean', float('nan')):.2f}",
    )

    artifacts = _artifacts(state_dir, selected_run["run_id"])
    predictions = artifacts["predictions.parquet"]

    tab_series, tab_zone, tab_importance, tab_rmse = st.tabs(
        ["Series real vs pred", "ICAP zona", "Feature importance", "RMSE vs persistencia"]
    )

    with tab_series:
        if predictions is None or predictions.empty:
            st.warning(
                "Este run no tiene `predictions.parquet` persistido (corrida previa al change "
                "`analisis-modelos`). Se muestran solo las métricas."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"métrica": key, "valor": value} for key, value in sorted(metrics.items())]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.plotly_chart(
                plots.series_figure(predictions, station, pollutant, split),
                width="stretch",
            )

    with tab_zone:
        zone = _zone_icap(state_dir, training_runs)
        if zone.empty:
            st.info("No hay predicciones persistidas para derivar el ICAP_zona.")
        else:
            st.plotly_chart(plots.zone_icap_figure(zone), width="stretch")

    with tab_importance:
        importance = artifacts["feature_importance.parquet"]
        if importance is None or importance.empty:
            st.info("Este run no tiene `feature_importance.parquet` persistido.")
        else:
            top_n = st.slider("Top N features", 5, 30, 15)
            st.plotly_chart(
                plots.feature_importance_figure(importance, station, pollutant, top_n=top_n),
                width="stretch",
            )

    with tab_rmse:
        summary = load.rmse_summary(runs)
        if summary.empty:
            st.info("No hay métricas de RMSE para comparar.")
        else:
            st.plotly_chart(plots.rmse_comparison_figure(summary), width="stretch")
            st.dataframe(summary, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
