"""Tests del resumen de `sel` (ae-dmd-prep, tarea 7.2). Sin AWS: frames sintéticos y un sqlite de fixture."""
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from analysis import sel_summary as ss

FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "pipeline" / "analysis" / "fixtures" / "state"
SERIES = [
    ("Nielol", "pm25"), ("Nielol", "pm10"),
    ("Las Encinas", "pm25"), ("Las Encinas", "pm10"),
    ("Padre Las Casas II", "pm25"), ("Padre Las Casas II", "pm10"),
]


def _frame(hours: int, offset: float = 0.0, drop_first: bool = False) -> pd.DataFrame:
    """Predicciones sintéticas de las 6 series; `drop_first` quita la primera hora de todas."""
    stamps = pd.date_range("2026-04-01", periods=hours, freq="h", tz="America/Santiago")
    rows = []
    for station, pollutant in SERIES:
        y = np.linspace(10, 60, hours) + (5.0 if pollutant == "pm10" else 0.0)
        frame = pd.DataFrame({
            "timestamp": stamps, "estacion": station, "pollutant": pollutant,
            "y_true": y, "y_pred": y + offset,
        })
        rows.append(frame.iloc[1:] if drop_first else frame)
    return pd.concat(rows, ignore_index=True)


def test_filas_comunes_excluye_la_fila_faltante_para_todas():
    full = _frame(48)
    missing_one = _frame(48, drop_first=True)
    common = ss.common_rows({"a": full, "persistencia": missing_one})
    assert len(common["a"]) == len(common["persistencia"]) == len(missing_one)
    assert common["a"]["timestamp"].min() > full["timestamp"].min()
    assert set(map(tuple, common["a"][ss.KEYS].itertuples(index=False, name=None))) == set(
        map(tuple, common["persistencia"][ss.KEYS].itertuples(index=False, name=None))
    )


def test_fila_con_y_pred_nan_en_una_etiqueta_sale_para_todas():
    """Persistencia tiene NaN en el arranque: esa fila no puede quedar en las otras etiquetas."""
    base = _frame(48)
    with_nan = base.copy()
    with_nan.loc[with_nan.index[0], "y_pred"] = np.nan
    common = ss.common_rows({"persistencia": with_nan, "lightgbm": base})
    assert len(common["persistencia"]) == len(common["lightgbm"]) == len(base) - 1
    _, means = ss.summary_tables({"persistencia": with_nan, "lightgbm": base})
    assert means.set_index("modelo").loc["lightgbm", "n"] == means.set_index("modelo").loc["persistencia", "n"]


def test_media_es_promedio_simple_de_las_seis_series():
    frames = {"lightgbm": _frame(72, offset=1.0), "persistencia": _frame(72, offset=3.0)}
    per_series, means = ss.summary_tables(frames, alpha=1.0)
    assert set(means["modelo"]) == {"lightgbm", "persistencia"}
    lgbm = per_series[per_series["modelo"] == "lightgbm"]
    assert len(lgbm) == 6
    assert means.set_index("modelo").loc["lightgbm", "rmse"] == pytest.approx(lgbm["rmse"].mean())
    assert means.set_index("modelo").loc["lightgbm", "rmse"] == pytest.approx(1.0)
    assert means.set_index("modelo").loc["persistencia", "rmse"] == pytest.approx(3.0)


def test_n_comun_es_el_numero_de_filas_tras_interseccion():
    frames = {"a": _frame(40), "b": _frame(40, drop_first=True)}
    _, means = ss.summary_tables(frames)
    by_label = means.set_index("modelo")
    # 40 horas por serie en "a" y 39 en "b": la intersección deja 39 por serie en ambas.
    assert int(by_label.loc["a", "n"]) == 39
    assert int(by_label.loc["b", "n"]) == 39


def test_label_for_reconoce_las_familias():
    assert ss.label_for({"tags": {"modelo": "persistence-aedmd"}, "params": {"tipo": "referencia"}}) == "persistencia"
    assert ss.label_for({"tags": {"modelo": "lightgbm-aedmd"}, "params": {"tipo": "entrenamiento"}}) == "lightgbm"
    assert ss.label_for({"tags": {"modelo": "ae-mlp-dmd", "seed": "1"}, "params": {}}) == "dmd-mlp-s1"
    assert ss.label_for({"tags": {"modelo": "ae-mlp"}, "params": {}}) is None


def test_ae_table_tiene_las_columnas_pedidas():
    runs = [{"run_name": "sinca-zona-ae_mlp-aedmd-v1-s0", "metrics": {"gate_pass": 0.0, "sel_recon_rmse_mean": 2.5}}]
    table = ss.ae_table(runs)
    assert list(table.columns) == ["run", *ss.AE_METRICS]
    assert table.loc[0, "sel_recon_rmse_mean"] == pytest.approx(2.5)
    assert np.isnan(table.loc[0, "dmd_pydmd_ok"])


def test_sin_runs_de_la_particion_sale_con_cero(tmp_path, monkeypatch, capsys):
    """Sin runs `particion=ae-dmd`: mensaje y salida 0 (no descarga nada de S3 en el test)."""
    state = tmp_path / "state"
    shutil.copytree(FIXTURE, state, ignore=shutil.ignore_patterns("mlruns", "__pycache__"))
    monkeypatch.setattr(ss.analysis_load, "ensure_state", lambda data_dir=None, run_ids=None, **_: str(state))
    monkeypatch.setattr(ss, "get_client", lambda profile: object())
    argv = ["--partition", "ae-dmd", "--state-dir", str(state), "--out", str(tmp_path / "out.md"), "--profile", "x"]
    assert ss.main(argv) == 0
    assert "Sin runs" in capsys.readouterr().out
    assert not (tmp_path / "out.md").exists()
