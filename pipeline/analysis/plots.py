"""Figuras Plotly puras para el dashboard de análisis.

Sin dependencia de Streamlit: se pueden usar en tests, notebooks o reportes HTML.
Las bandas ICAP se dibujan como `hrect`:
- en el gráfico de **concentración**, en el equivalente en µg/m³ de ICAP 100/200/300
  (`icap_to_concentration`, porque el ICAP es un índice, no una concentración);
- en el gráfico de **`ICAP_zona`**, directamente en 100/200/300.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from icap.icap import ICAP_ANCHORS  # noqa: E402

# Umbrales oficiales del ICAP (D.S. 12/2011).
ICAP_BANDS: tuple[tuple[float, str, str], ...] = (
    (100.0, "alerta", "rgba(250, 204, 21, 0.18)"),
    (200.0, "preemergencia", "rgba(249, 115, 22, 0.18)"),
    (300.0, "emergencia", "rgba(220, 38, 38, 0.18)"),
)

COLOR_TRUE = "#1f2937"
COLOR_PRED = "#2563eb"
COLOR_LGBM = "#2563eb"
COLOR_PERSISTENCE = "#9ca3af"


def icap_to_concentration(pollutant: str, icap_value: float) -> float:
    """Concentración (µg/m³) equivalente a un valor de ICAP (invierte las anclas)."""
    anchors = ICAP_ANCHORS[pollutant]
    icaps = np.array([anchor[1] for anchor in anchors], dtype=float)
    concentrations = np.array([anchor[0] for anchor in anchors], dtype=float)
    return float(
        np.interp(icap_value, icaps, concentrations, left=0.0, right=concentrations[-1])
    )


def _add_bands(fig: go.Figure, bounds: list[tuple[float, float, str, str]]) -> None:
    for low, high, label, color in bounds:
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=10,
        )


def _concentration_bands(pollutant: str) -> list[tuple[float, float, str, str]]:
    """Bandas entre umbrales ICAP, en concentración, para el gráfico de serie."""
    concentrations = [
        icap_to_concentration(pollutant, icap_value) for icap_value, _, _ in ICAP_BANDS
    ]
    return [
        (0.0, concentrations[0], "regular", "rgba(16, 185, 129, 0.10)"),
        (concentrations[0], concentrations[1], "alerta", ICAP_BANDS[0][2]),
        (concentrations[1], concentrations[2], "preemergencia", ICAP_BANDS[1][2]),
        (
            concentrations[2],
            concentrations[2] * 1.6 + 10,
            "emergencia",
            ICAP_BANDS[2][2],
        ),
    ]


def _filter_predictions(
    predictions: pd.DataFrame,
    station: str,
    pollutant: str,
    split: str | None,
) -> pd.DataFrame:
    work = predictions
    if "estacion" in work.columns:
        work = work[work["estacion"] == station]
    if "pollutant" in work.columns:
        work = work[work["pollutant"] == pollutant]
    if split and "split" in work.columns:
        work = work[work["split"] == split]
    return work.sort_values("timestamp")


def series_figure(
    predictions: pd.DataFrame,
    station: str,
    pollutant: str,
    split: str | None = "test",
) -> go.Figure:
    """Serie real vs predicha con bandas ICAP (concentración)."""
    work = _filter_predictions(predictions, station, pollutant, split)

    fig = go.Figure()
    _add_bands(fig, _concentration_bands(pollutant))
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"], y=work["y_true"], name="real",
            mode="lines", line={"color": COLOR_TRUE, "width": 1.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"], y=work["y_pred"], name="predicho",
            mode="lines", line={"color": COLOR_PRED, "width": 1.6},
        )
    )
    fig.update_layout(
        title=f"{station} / {pollutant} — real vs predicho ({split or 'todos'})",
        xaxis_title="timestamp",
        yaxis_title="µg/m³",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left"},
        margin={"l": 40, "r": 20, "t": 85, "b": 40},
        height=460,
    )
    return fig


RUN_PALETTE = (
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#4b5563",
)


def series_comparison_figure(
    frames: dict[str, pd.DataFrame],
    station: str,
    pollutant: str,
    split: str | None = "test",
) -> go.Figure:
    """Compara varios runs de la misma serie: `real` una vez + una pred por run."""
    fig = go.Figure()
    _add_bands(fig, _concentration_bands(pollutant))

    first = next(iter(frames.values()))
    real = _filter_predictions(first, station, pollutant, split)
    fig.add_trace(
        go.Scatter(
            x=real["timestamp"], y=real["y_true"], name="real",
            mode="lines", line={"color": COLOR_TRUE, "width": 2.2},
        )
    )
    for index, (label, predictions) in enumerate(frames.items()):
        work = _filter_predictions(predictions, station, pollutant, split)
        fig.add_trace(
            go.Scatter(
                x=work["timestamp"], y=work["y_pred"], name=label,
                mode="lines",
                line={"color": RUN_PALETTE[index % len(RUN_PALETTE)], "width": 1.4},
            )
        )
    fig.update_layout(
        title=f"{station} / {pollutant} — comparación de runs ({split or 'todos'})",
        xaxis_title="timestamp",
        yaxis_title="µg/m³",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left"},
        margin={"l": 40, "r": 20, "t": 85, "b": 40},
        height=460,
    )
    return fig


def zone_icap_comparison_figure(
    frames: dict[str, pd.DataFrame], split: str | None = "test"
) -> go.Figure:
    """`ICAP_zona` real vs predicho por versión (bandas 100/200/300)."""
    top = 300.0
    for zone in frames.values():
        if "icap_true" in zone.columns and not zone["icap_true"].dropna().empty:
            top = max(top, float(zone["icap_true"].max()))

    fig = go.Figure()
    _add_bands(
        fig,
        [
            (0.0, 100.0, "regular", "rgba(16, 185, 129, 0.10)"),
            (100.0, 200.0, "alerta", ICAP_BANDS[0][2]),
            (200.0, 300.0, "preemergencia", ICAP_BANDS[1][2]),
            (300.0, max(500.0, top * 1.2), "emergencia", ICAP_BANDS[2][2]),
        ],
    )

    first = next(iter(frames.values())).sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=first["timestamp"], y=first["icap_true"], name="ICAP_zona real",
            mode="lines", line={"color": COLOR_TRUE, "width": 2.2},
        )
    )
    for index, (label, zone) in enumerate(frames.items()):
        work = zone.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=work["timestamp"], y=work["icap_pred"], name=f"ICAP_zona {label}",
                mode="lines",
                line={"color": RUN_PALETTE[index % len(RUN_PALETTE)], "width": 1.4},
            )
        )
    fig.update_layout(
        title=f"ICAP_zona — real vs predicho ({split or 'todos'})",
        xaxis_title="timestamp",
        yaxis_title="ICAP (índice)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left"},
        margin={"l": 40, "r": 20, "t": 85, "b": 40},
        height=460,
    )
    return fig


def feature_importance_figure(
    importance: pd.DataFrame,
    station: str | None = None,
    pollutant: str | None = None,
    top_n: int = 20,
) -> go.Figure:
    """Importancia de features (gain) en barras horizontales."""
    work = importance
    if station and "estacion" in work.columns:
        work = work[work["estacion"] == station]
    if pollutant and "pollutant" in work.columns:
        work = work[work["pollutant"] == pollutant]
    work = work.sort_values("gain", ascending=True).tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=work["gain"], y=work["feature"], orientation="h",
            marker_color=COLOR_PRED,
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Feature importance (gain) — {station or ''} {pollutant or ''}".strip(),
        xaxis_title="gain",
        yaxis_title="feature",
        margin={"l": 120, "r": 20, "t": 60, "b": 40},
        height=460,
    )
    return fig


def feature_importance_comparison_figure(
    frames: dict[str, pd.DataFrame],
    top_n: int = 15,
) -> go.Figure:
    """Compara la importancia (gain) de varios runs: barras agrupadas por feature."""
    merged: pd.DataFrame | None = None
    for label, importance in frames.items():
        part = importance[["feature", "gain"]].rename(columns={"gain": label})
        merged = part if merged is None else merged.merge(part, on="feature", how="outer")
    if merged is None or merged.empty:
        return go.Figure()

    merged = merged.fillna(0.0)
    run_labels = list(frames.keys())
    merged["_max"] = merged[run_labels].max(axis=1)
    merged = merged.sort_values("_max", ascending=True).tail(top_n)

    fig = go.Figure()
    for index, label in enumerate(run_labels):
        fig.add_trace(
            go.Bar(
                x=merged[label], y=merged["feature"], orientation="h", name=label,
                marker_color=RUN_PALETTE[index % len(RUN_PALETTE)],
                hovertemplate="%{y}: %{x:,.0f}<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        title="Feature importance (gain) — comparación de runs",
        xaxis_title="gain",
        yaxis_title="feature",
        barmode="group",
        legend={"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left"},
        margin={"l": 120, "r": 20, "t": 85, "b": 40},
        height=520,
    )
    return fig


def rmse_test_figure(rows: list[dict]) -> go.Figure:
    """Barras de RMSE de test: persistencia + modelos seleccionados (una serie)."""
    labels = [row["label"] for row in rows]
    values = [row["test_rmse"] for row in rows]
    colors = [
        COLOR_PERSISTENCE if row["label"] == "persistencia" else COLOR_LGBM
        for row in rows
    ]
    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=colors,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="RMSE de test — persistencia vs modelos",
        xaxis_title="modelo",
        yaxis_title="RMSE (µg/m³)",
        margin={"l": 40, "r": 20, "t": 60, "b": 60},
        height=420,
    )
    return fig


def rmse_valid_figure(rows: list[dict]) -> go.Figure:
    """Barras de RMSE de valid (CV): media por ventana, sin rango min–max.

    Fallback para runs sin reporte del search (sin candidatos que graficar).
    """
    model_rows = [row for row in rows if row.get("valid_mean") is not None]
    labels = [row["label"] for row in model_rows]
    means = [row["valid_mean"] for row in model_rows]

    def _fmt(value):
        return f"{value:.2f}" if value is not None else "—"

    hover_text = [
        f"<b>{row['label']}</b><br>"
        f"media: {_fmt(row.get('valid_mean'))}<br>"
        f"sd: {_fmt(row.get('valid_sd'))}<br>"
        f"min: {_fmt(row.get('valid_min'))}<br>"
        f"max: {_fmt(row.get('valid_max'))}"
        for row in model_rows
    ]

    fig = go.Figure(
        go.Bar(
            x=labels, y=means,
            marker_color=COLOR_LGBM,
            text=hover_text,
            textposition="none",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="RMSE de valid (ventana deslizante) — media",
        xaxis_title="modelo",
        yaxis_title="RMSE (µg/m³)",
        margin={"l": 40, "r": 20, "t": 60, "b": 60},
        height=420,
    )
    return fig


COLOR_CANDIDATE = "#93b4f7"
COLOR_CANDIDATE_BEST = "#b45309"


def rmse_candidates_figure(candidates: pd.DataFrame, label: str) -> go.Figure:
    """Barras del `score` de cada candidato del search, ordenadas de mejor a peor.

    `score` = RMSE ponderado promedio sobre las ventanas del search (12 por
    defecto). El mejor candidato (el seleccionado) va primero y pintado
    distinto; el hover de cada barra muestra sus hiperparámetros.
    """
    frame = candidates.sort_values("score", ascending=True).reset_index(drop=True)
    best_trial = int(frame.loc[0, "trial"])
    labels = [f"c{int(trial) + 1}" for trial in frame["trial"]]
    colors = [
        COLOR_CANDIDATE_BEST if int(trial) == best_trial else COLOR_CANDIDATE
        for trial in frame["trial"]
    ]
    hover_text = [
        f"<b>{label} · candidato {int(row.trial) + 1}</b><br>"
        f"score: {row.score:.2f}"
        + ("<br>★ el mejor (seleccionado)" if int(row.trial) == best_trial else "")
        + "".join(f"<br>{key}: {value}" for key, value in (row.params or {}).items())
        for row in frame.itertuples()
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=frame["score"],
            marker_color=colors,
            text=[f"{value:.2f}" for value in frame["score"]],
            textposition="outside",
            cliponaxis=False,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"RMSE de valid (CV) — candidatos del search · {label}",
        xaxis_title="candidato (ordenado de mejor a peor)",
        yaxis_title="RMSE (µg/m³)",
        margin={"l": 40, "r": 20, "t": 60, "b": 60},
        height=420,
        bargap=0.25,
        showlegend=False,
    )
    return fig
