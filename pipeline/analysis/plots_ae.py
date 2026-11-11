"""Figuras Plotly de los autoencoders (change `ae-reconstruccion`, tarea 7.2).

Sin dependencia de Streamlit, igual que `plots.py`. Consumen los artefactos
`reconstruction.parquet` y `diagnostics.parquet` de un run conjunto de AE, y las
métricas `dmd_*` del mismo run.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLOR_TRUE = "#1f2937"
COLOR_RECON = "#dc2626"
COLOR_REF = "#9ca3af"
COLOR_PCA = "#059669"


def _ts_like(value, tz) -> pd.Timestamp:
    """`value` como Timestamp con la misma zona que la serie (o sin zona si `tz` es None)."""
    stamp = pd.Timestamp(value)
    if tz is None:
        return stamp.tz_localize(None) if stamp.tz is not None else stamp
    return stamp.tz_convert(tz) if stamp.tz is not None else stamp.tz_localize(tz)


def reconstruction_figure(recon: pd.DataFrame, serie: str, start=None, end=None) -> go.Figure:
    """Real vs reconstruido de una serie, con zoom opcional a un tramo (`start`, `end`)."""
    frame = recon[recon["serie"] == serie].sort_values("timestamp")
    tz = frame["timestamp"].dt.tz
    if start is not None:
        frame = frame[frame["timestamp"] >= _ts_like(start, tz)]
    if end is not None:
        frame = frame[frame["timestamp"] <= _ts_like(end, tz)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["y_true"], mode="lines",
                             name="real", line=dict(color=COLOR_TRUE, width=1.2)))
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["y_recon"], mode="lines",
                             name="reconstruido", line=dict(color=COLOR_RECON, width=1.2)))
    fig.update_layout(title=f"Reconstrucción de MP — {serie} (µg/m³)", template="plotly_white",
                      height=380, margin=dict(l=40, r=20, t=50, b=40), yaxis_title="µg/m³")
    return fig


def flattening_figure(diagnostics: pd.DataFrame, split: str = "test") -> go.Figure:
    """Diagnósticos de aplanamiento por serie: razón de varianza y pendiente (1 = ideal)."""
    frame = diagnostics[(diagnostics["split"] == split) & diagnostics["metric"].isin(["var_ratio", "slope"])]
    wide = frame.pivot_table(index="serie", columns="metric", values="value", aggfunc="first").reset_index()
    fig = go.Figure()
    for metric, color in (("var_ratio", COLOR_RECON), ("slope", COLOR_TRUE)):
        if metric in wide:
            fig.add_trace(go.Bar(x=wide["serie"], y=wide[metric], name=metric, marker_color=color))
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLOR_REF)
    fig.update_layout(title=f"Aplanamiento ({split}): razón de varianza y pendiente", template="plotly_white",
                      barmode="group", height=360, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def error_by_level_figure(diagnostics: pd.DataFrame, split: str = "test") -> go.Figure:
    """RMSE y sesgo por tramo de concentración (< 50, 50–100, > 100 µg/m³)."""
    bins = ("lt50", "50_100", "gt100")
    rows = []
    for serie, group in diagnostics[diagnostics["split"] == split].groupby("serie"):
        values = dict(zip(group["metric"], group["value"]))
        for b in bins:
            if f"rmse_{b}" in values:
                rows.append({"serie": serie, "tramo": b, "rmse": values[f"rmse_{b}"], "sesgo": values.get(f"bias_{b}")})
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    if not frame.empty:
        for b in bins:
            sub = frame[frame["tramo"] == b]
            fig.add_trace(go.Bar(x=sub["serie"], y=sub["rmse"], name=f"RMSE {b}"))
    fig.update_layout(title=f"Error por nivel de concentración ({split})", template="plotly_white",
                      barmode="group", height=360, margin=dict(l=40, r=20, t=50, b=40), yaxis_title="µg/m³")
    return fig


def linearizability_figure(metrics: dict[str, float]) -> go.Figure:
    """Error de `z` a 1 y 24 pasos con `K` y `K^24` (rango completo vs truncado)."""
    groups = {
        "rango completo": "dmd_latent_rmse_",
        "truncado": "dmd_trunc_latent_rmse_",
        "valid (completo)": "dmd_valid_latent_rmse_",
    }
    fig = go.Figure()
    for label, prefix in groups.items():
        xs, ys = [], []
        for horizon in (1, 24):
            key = f"{prefix}{horizon}"
            if key in metrics:
                xs.append(f"{horizon} paso(s)")
                ys.append(metrics[key])
        if ys:
            fig.add_trace(go.Bar(x=xs, y=ys, name=label))
    fig.update_layout(title="Linealizabilidad del latente (RMSE de z)", template="plotly_white",
                      barmode="group", height=340, margin=dict(l=40, r=20, t=50, b=40))
    return fig
