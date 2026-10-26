"""Dashboard Streamlit de análisis de modelos (Fase 3.5).

Uso local (contenedor o venv):
    streamlit run pipeline/analysis/streamlit_app.py --server.port 8501

Flujo: se elige **primero la serie** (estación/contaminante) y luego **uno o
varios runs** de esa serie (multi-select) para graficarlos/compararlos.

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
def _zone_icap_by_model_version(state_dir: str, runs: list[dict], split: str) -> dict:
    return load.zone_icap_by_model_version(state_dir, runs, split)


@st.cache_data(show_spinner=False)
def _rmse_by_model(state_dir: str, runs: list[dict], station: str, pollutant: str, selected: tuple[str, ...]) -> list[dict]:
    return load.rmse_by_model(state_dir, runs, station, pollutant, set(selected))


@st.cache_data(show_spinner=False)
def _candidates_by_model(
    state_dir: str, runs: list[dict], station: str, pollutant: str, selected: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    return load.candidates_by_model(state_dir, runs, station, pollutant, set(selected))


def _source_selector() -> str | None:
    options = ["S3 (real)"]
    if FIXTURE_STATE.is_dir():
        options.append("Fixture local")
    choice = st.sidebar.radio(
        "Fuente de datos", options, index=len(options) - 1 if len(options) > 1 else 0
    )
    return str(FIXTURE_STATE) if choice == "Fixture local" else None


def _run_label(run: dict) -> str:
    return f"{run['run_name']} · {run['run_id'][:8]}"


def _metrics_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "run": run["run_name"],
                "run_id": run["run_id"][:8],
                "test_rmse": metrics.get("test_rmse"),
                "test_weighted_rmse": metrics.get("test_weighted_rmse"),
                "rolling_weighted_rmse_mean": metrics.get("rolling_weighted_rmse_mean"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("SINCA — Análisis de modelos")
    st.caption(
        "Baseline Fase 3 · series real vs predicho, ICAP_zona, feature importance y RMSE vs persistencia"
    )

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

    # --- 1) Serie primero ----------------------------------------------------
    series = load.series_options(training_runs)
    series_labels = [f"{station} / {pollutant}" for station, pollutant in series]
    selected_series = st.sidebar.selectbox("Serie", series_labels)
    station, pollutant = series[series_labels.index(selected_series)]

    # --- 2) Modelos (selector único: alimenta series e ICAP) ----------------
    # La unidad es `(familia, versión)`: agrupa los runs de las 6 series de esa
    # generación. Así se distingue `lightgbm v1` de `dmd v1`, y el mismo selector
    # maneja ambas pestañas.
    model_versions = load.available_model_versions(training_runs)
    model_version_labels = [f"{model} v{version}" for model, version in model_versions]
    selected_models = st.sidebar.multiselect(
        "Modelos (comparar)", model_version_labels, default=model_version_labels
    )
    if not selected_models:
        st.warning("Seleccioná al menos un modelo/versión.")
        st.stop()
    selected_model_keys = set(selected_models)

    split = st.sidebar.radio("Split", ["test", "valid"], horizontal=True)

    # Runs de la serie elegida para los modelos seleccionados.
    series_runs = [
        run
        for run in training_runs
        if run["params"].get("estacion") == station
        and run["params"].get("pollutant") == pollutant
        and f"{load.model_family(run)} v{run['params'].get('version')}" in selected_model_keys
    ]
    if not series_runs:
        st.warning("No hay runs de esta serie para los modelos seleccionados.")
        st.stop()

    # --- Diagnósticos de los runs seleccionados -----------------------------
    prediction_frames: dict[str, pd.DataFrame] = {}
    importance_frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for run in series_runs:
        artifacts = _artifacts(state_dir, run["run_id"])
        label = _run_label(run)
        predictions = artifacts["predictions.parquet"]
        importance = artifacts["feature_importance.parquet"]
        if predictions is None or predictions.empty:
            missing.append(run["run_name"])
        else:
            prediction_frames[label] = predictions
        if importance is not None and not importance.empty:
            importance_frames[label] = importance

    tab_series, tab_zone, tab_importance, tab_rmse = st.tabs(
        ["Series real vs pred", "ICAP zona", "Feature importance", "RMSE vs persistencia"]
    )

    with tab_series:
        if not prediction_frames:
            st.warning(
                "Los runs seleccionados no tienen `predictions.parquet` persistido "
                "(corridas previas al change `analisis-modelos`). Se muestran solo las métricas."
            )
            st.dataframe(_metrics_table(series_runs), hide_index=True, width="stretch")
        else:
            if missing:
                st.info(
                    "Sin diagnósticos persistidos para: " + ", ".join(sorted(set(missing)))
                )
            if len(prediction_frames) == 1:
                only = next(iter(prediction_frames.values()))
                figure = plots.series_figure(only, station, pollutant, split)
            else:
                figure = plots.series_comparison_figure(
                    prediction_frames, station, pollutant, split
                )
            st.plotly_chart(figure, width="stretch")

    with tab_zone:
        zones = _zone_icap_by_model_version(state_dir, training_runs, split)
        zones = {label: zones[label] for label in selected_models if label in zones}
        if not zones:
            st.info("No hay predicciones para los modelos seleccionados en este split.")
        else:
            st.plotly_chart(
                plots.zone_icap_comparison_figure(zones, split), width="stretch"
            )

    with tab_importance:
        if not importance_frames:
            st.info("Los runs seleccionados no tienen `feature_importance.parquet` persistido.")
        else:
            top_n = st.slider("Top N features", 5, 30, 15)
            if len(importance_frames) == 1:
                only = next(iter(importance_frames.values()))
                figure = plots.feature_importance_figure(
                    only, station, pollutant, top_n=top_n
                )
            else:
                figure = plots.feature_importance_comparison_figure(
                    importance_frames, top_n=top_n
                )
            st.plotly_chart(figure, width="stretch")

    with tab_rmse:
        rows = _rmse_by_model(
            state_dir, runs, station, pollutant, tuple(sorted(selected_models))
        )
        col_test, col_valid = st.columns(2)
        with col_test:
            st.subheader(f"Test — {station} / {pollutant}")
            if not any(row["test_rmse"] is not None for row in rows):
                st.info("Sin métricas de test para esta serie.")
            else:
                st.plotly_chart(plots.rmse_test_figure(rows), width="stretch")
        with col_valid:
            st.subheader(f"Valid (CV) - {station} / {pollutant}")
            candidates = _candidates_by_model(
                state_dir, training_runs, station, pollutant, tuple(sorted(selected_models))
            )
            if candidates:
                labels_with_candidates = [
                    label for label in selected_models if label in candidates
                ]
                chosen_model = st.selectbox(
                    "Modelo (candidatos)", labels_with_candidates, key="valid_model"
                )
                model_candidates = candidates[chosen_model]
                st.plotly_chart(
                    plots.rmse_candidates_figure(model_candidates, chosen_model),
                    width="stretch",
                )
                st.caption(
                    "Barras ordenadas de mejor a peor (score = RMSE ponderado medio sobre "
                    "las 12 ventanas del search, el criterio con que se elige el ganador); "
                    "la mejor va resaltada y el hover muestra sus hiperparámetros. La tabla "
                    "de abajo muestra la validación completa (42 ventanas) del modelo final."
                )
                with st.expander("Hiperparámetros de los candidatos"):
                    st.dataframe(
                        load.candidates_table(model_candidates),
                        hide_index=True,
                        width="stretch",
                    )
            elif not any(row["valid_mean"] is not None for row in rows):
                st.info("Sin métricas de validación (ventana deslizante).")
            else:
                st.plotly_chart(plots.rmse_valid_figure(rows), width="stretch")

        if any(row["valid_mean"] is not None for row in rows):
            st.caption(
                "La persistencia no se evalúa en la ventana deslizante (solo en test)."
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
