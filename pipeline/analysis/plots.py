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
            annotation_position="top left",
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


def series_figure(
    predictions: pd.DataFrame,
    station: str,
    pollutant: str,
    split: str | None = "test",
) -> go.Figure:
    """Serie real vs predicha con bandas ICAP (concentración)."""
    work = predictions
    if "estacion" in work.columns:
        work = work[work["estacion"] == station]
    if "pollutant" in work.columns:
        work = work[work["pollutant"] == pollutant]
    if split and "split" in work.columns:
        work = work[work["split"] == split]
    work = work.sort_values("timestamp")

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
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        height=460,
    )
    return fig


def zone_icap_figure(zone: pd.DataFrame) -> go.Figure:
    """`ICAP_zona` real vs predicho con bandas 100/200/300."""
    work = zone.sort_values("timestamp") if "timestamp" in zone.columns else zone

    fig = go.Figure()
    _add_bands(
        fig,
        [
            (0.0, 100.0, "regular", "rgba(16, 185, 129, 0.10)"),
            (100.0, 200.0, "alerta", ICAP_BANDS[0][2]),
            (200.0, 300.0, "preemergencia", ICAP_BANDS[1][2]),
            (300.0, max(500.0, float(work.get("icap_true", pd.Series([300])).max() or 300)), "emergencia", ICAP_BANDS[2][2]),
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"], y=work["icap_true"], name="ICAP_zona real",
            mode="lines", line={"color": COLOR_TRUE, "width": 1.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"], y=work["icap_pred"], name="ICAP_zona predicho",
            mode="lines", line={"color": COLOR_PRED, "width": 1.6},
        )
    )
    fig.update_layout(
        title="ICAP_zona — real vs predicho (test held-out)",
        xaxis_title="timestamp",
        yaxis_title="ICAP (índice)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
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


def rmse_comparison_figure(summary: pd.DataFrame) -> go.Figure:
    """Barras agrupadas de RMSE por serie: LightGBM vs persistencia."""
    work = summary.copy()
    if "serie" not in work.columns:
        work["serie"] = work["estacion"].astype(str) + " / " + work["pollutant"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=work["serie"], y=work["rmse_lightgbm"], name="LightGBM",
            marker_color=COLOR_LGBM,
            hovertemplate="%{x}<br>LightGBM: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=work["serie"], y=work["rmse_persistence"], name="persistencia",
            marker_color=COLOR_PERSISTENCE,
            hovertemplate="%{x}<br>persistencia: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="RMSE de test por serie — LightGBM vs persistencia",
        xaxis_title="serie",
        yaxis_title="RMSE (µg/m³)",
        barmode="group",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 60, "b": 90},
        height=460,
    )
    return fig
