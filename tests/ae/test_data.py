"""Tests de preparación de datos del autoencoder (tarea 1.8)."""
import numpy as np
import pandas as pd
import pytest

from ae import data as aed
from conftest import synthetic_validated


def test_channel_counts():
    assert len(aed.mp_channels()) == 6
    assert len(aed.aux_channels()) == 12
    assert aed.CAL_DIM == 10
    assert aed.AUX_INPUT_DIM == 34


def test_aux_excluye_rhum_glob_y_temp_plc2():
    names = [name for name, _, _ in aed.aux_channels()]
    assert not any(n.startswith(("rhum", "glob")) for n in names)
    assert "temp_padre_las_casas_ii" not in names
    assert "temp_nielol" in names and "temp_las_encinas" in names


def test_interpolacion_solo_huecos_cortos_y_mascara_real():
    values = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])  # hueco de 3 h
    filled, real = aed.fill_short_gaps(values)
    assert np.allclose(filled.to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert real.tolist() == [True, False, False, False, True]


def test_hueco_largo_no_se_interpola():
    values = pd.Series([1.0] + [np.nan] * 7 + [9.0])  # hueco de 7 h > 6
    filled, real = aed.fill_short_gaps(values)
    assert filled.iloc[1:8].isna().all()
    assert real.sum() == 2


def test_hueco_en_borde_no_se_interpola():
    values = pd.Series([np.nan, np.nan, 3.0, 4.0])
    filled, _ = aed.fill_short_gaps(values)
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[1])


def test_grilla_marca_huecos_largos_como_nan(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    col = "pm25_nielol"
    assert grid.mp[col].iloc[100:103].notna().all()  # corto: interpolado
    assert grid.mp[col].iloc[200:210].isna().all()   # largo: NaN
    assert not grid.mp_real[col].iloc[101]           # interpolado no es real
    assert grid.mp_real[col].iloc[99]


def test_ventana_con_hueco_largo_se_descarta(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    elig = aed.window_eligibility(grid, 24)
    # Ventanas que contienen 200..209 no son elegibles; la que termina en 235 sí.
    assert not elig[209] and not elig[220]
    assert elig[240]


def test_muestra_solo_si_ventana_completa_cae_en_particion():
    end = np.arange(0, 100)
    mask = aed.assign_samples(end, length=10, lo=20, hi=60)
    assert not mask[24]   # ventana [15..24] empieza antes de la partición (lo=20)
    assert mask[29]       # ventana [20..29] cae completa dentro
    assert mask[59]       # ventana [50..59] termina antes de hi=60
    assert not mask[60]   # ventana [51..60] termina fuera


def test_estadisticas_solo_de_train(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    # Un corrimiento grande en la mitad de test no debe cambiar la media de train.
    n = len(grid.index)
    train = np.zeros(n, dtype=bool)
    train[: n // 2] = True
    norm_a = aed.fit_normalizer(grid, train)
    shifted_mp = grid.mp.copy()
    shifted_mp.iloc[n // 2:] = shifted_mp.iloc[n // 2:] + 1000.0
    grid_shift = aed.HourlyGrid(
        index=grid.index, mp=shifted_mp, mp_real=grid.mp_real,
        aux=grid.aux, aux_real=grid.aux_real, cal=grid.cal,
    )
    norm_b = aed.fit_normalizer(grid_shift, train)
    assert np.allclose(norm_a.mean.to_numpy(), norm_b.mean.to_numpy())
    assert not np.allclose(norm_a.mean.to_numpy(), 1000.0)


def test_mp_no_se_recorta_aux_si(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    n = len(grid.index)
    train = np.ones(n, dtype=bool)
    norm = aed.fit_normalizer(grid, train)
    ng = aed.apply_normalizer(grid, norm)
    # Un pico extremo en MP debe quedar intacto (z > 5) pero recortado en aux.
    grid.mp.iloc[300, 0] = grid.mp.iloc[300, 0] * 50.0
    grid.aux.iloc[300, 0] = grid.aux.iloc[300, 0] * 50.0
    ng = aed.apply_normalizer(grid, norm)
    assert ng.mp_z.iloc[300, 0] > 5.0
    assert ng.aux_z.iloc[300, 0] == pytest.approx(5.0)


def test_inversa_recupera_micrograms(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    norm = aed.fit_normalizer(grid, np.ones(len(grid.index), dtype=bool))
    ng = aed.apply_normalizer(grid, norm)
    recovered = aed.mp_to_ugm3(ng.mp_z.to_numpy()[:50], norm)
    expected = grid.mp.to_numpy()[:50]
    ok = np.isfinite(expected)
    assert np.allclose(recovered[ok], expected[ok], rtol=1e-6)


def test_ventanas_mascaras_y_forma(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    norm = aed.fit_normalizer(grid, np.ones(len(grid.index), dtype=bool))
    ng = aed.apply_normalizer(grid, norm)
    lo, hi = 0, len(grid.index)
    win = aed.select_samples(grid, ng, 24, lo, hi)
    assert win.x_mp.shape[1:] == (24, 6)
    assert win.x_aux.shape[1:] == (24, aed.AUX_INPUT_DIM)
    assert win.y_aux.shape[1:] == (24, 12)
    assert np.isfinite(win.x_mp).all()          # sin huecos largos en la muestra
    assert win.mp_real.dtype == bool


def test_entrada_aux_sin_meteo_es_solo_calendario():
    sl = aed.aux_input_columns(False)
    assert sl.stop - sl.start == aed.CAL_DIM
    assert aed.aux_input_columns(True).stop == aed.AUX_INPUT_DIM


def test_rango_reporta_fraccion_recortada(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    norm = aed.fit_normalizer(grid, np.ones(len(grid.index), dtype=bool))
    grid.aux.iloc[400, 0] = grid.aux.iloc[400, 0] + 1e4
    ng = aed.apply_normalizer(grid, norm)
    report = aed.range_report(ng)
    assert {"canal", "min", "max", "frac_recortada"}.issubset(report.columns)
    assert (report.loc[report["grupo"] == "mp", "frac_recortada"] == 0).all()
    assert (report.loc[report["grupo"] == "aux", "frac_recortada"] > 0).any()


def test_sobrevivencia_por_L(validated):
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    spans = {"todo": (0, len(grid.index))}
    counts = aed.survival_counts(grid, (24, 72, 168), spans)
    assert list(counts["L"]) == [24, 72, 168]
    assert counts["muestras"].is_monotonic_decreasing
