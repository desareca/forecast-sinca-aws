"""Tests de métricas, compuerta (tarea 3.6) y DMD (tarea 4.5), solo numpy."""
import numpy as np
import pytest

from ae import dmd as dmdm
from ae import metrics as aem


# --- Métricas (3.6) ----------------------------------------------------------


def test_senal_suavizada_tiene_razon_de_varianza_menor_a_uno():
    rng = np.random.default_rng(0)
    real = 50 + 30 * rng.standard_normal(2000)
    kernel = np.ones(12) / 12
    smooth = np.convolve(real, kernel, mode="same")
    m = aem.recon_metrics(real, smooth)
    assert m["var_ratio"] < 1.0


def test_reconstruccion_perfecta_tiene_pendiente_uno_y_rmse_cero():
    rng = np.random.default_rng(1)
    real = rng.uniform(5, 150, 500)
    m = aem.recon_metrics(real, real.copy())
    assert m["rmse"] == pytest.approx(0.0)
    assert m["slope"] == pytest.approx(1.0)
    assert m["var_ratio"] == pytest.approx(1.0)


def test_metricas_por_tramo_de_concentracion():
    real = np.array([10.0, 60.0, 120.0, 130.0])
    recon = np.array([12.0, 58.0, 110.0, 100.0])
    m = aem.recon_metrics(real, recon)
    assert m["n_lt50"] == 1 and m["n_50_100"] == 1 and m["n_gt100"] == 2
    assert m["bias_gt100"] == pytest.approx(((110 - 120) + (100 - 130)) / 2)


def test_mascara_excluye_valores_no_reales():
    real = np.array([10.0, 20.0, 30.0, 999.0])
    recon = np.array([10.0, 20.0, 30.0, 0.0])
    mask = np.array([True, True, True, False])
    m = aem.recon_metrics(real, recon, mask)
    assert m["rmse"] == pytest.approx(0.0)


def test_espectro_atenua_altas_frecuencias_al_suavizar():
    rng = np.random.default_rng(2)
    real = rng.standard_normal(1024)
    smooth = np.convolve(real, np.ones(9) / 9, mode="same")
    s = aem.spectral_ratio(real, smooth)
    assert s["spec_ratio_lt24h"] < s["spec_ratio_24_168h"] + 1e-9
    assert s["spec_ratio_lt24h"] < 1.0


def test_pca_con_rango_completo_reconstruye_exacto():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 6))
    mean, comp = aem.pca_fit(x, d=6)
    assert np.allclose(aem.pca_reconstruct(x, mean, comp), x, atol=1e-9)


def test_diagnosticos_latente_detectan_colapso():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((400, 3))
    z = np.column_stack([base[:, 0], base[:, 0] * 1.0000001, base[:, 1], np.zeros(400)])
    diag = aem.latent_diagnostics(z, d_aux=1)
    assert diag["dead_dims"] == 1
    assert diag["eff_dim_pr"] < 3.5


def test_compuerta_solo_reporta_sin_umbrales():
    ae = {"s": {"rmse": 1.0, "var_ratio": 0.9, "slope": 0.95}}
    pca = {"s": {"rmse": 2.0}}
    gate = aem.gate_decision(ae, pca, None)
    assert gate["gate_pass"] is None
    assert gate["beats_pca"] is True


def test_compuerta_con_umbrales_decide():
    ae = {"s": {"rmse": 1.0, "var_ratio": 0.5, "slope": 0.95}}
    pca = {"s": {"rmse": 2.0}}
    gate = aem.gate_decision(ae, pca, {"min_var_ratio": 0.9, "slope_tol": 0.1})
    assert gate["gate_pass"] is False
    gate_ok = aem.gate_decision({"s": {"rmse": 1.0, "var_ratio": 0.95, "slope": 1.0}}, pca,
                                {"min_var_ratio": 0.9, "slope_tol": 0.1})
    assert gate_ok["gate_pass"] is True


# --- DMD (4.5) ---------------------------------------------------------------


def _linear_system(T=400, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.0], [0.0, 0.0, 0.5]])
    z = np.zeros((T, 3))
    z[0] = rng.standard_normal(3)
    for t in range(T - 1):
        z[t + 1] = a @ z[t]
    return a, z


def test_dmd_recupera_sistema_lineal_y_linearizabilidad_cero():
    a, z = _linear_system()
    fit = dmdm.fit_dmd([z], rank=None)
    assert np.allclose(fit["K"], a, atol=1e-8)
    lin = dmdm.linearizability(fit["K"], [z])
    assert lin["dmd_latent_rmse_1"] < 1e-8
    assert lin["dmd_latent_rmse_24"] < 1e-8


def test_dmd_truncado_tiene_rango_pedido():
    _, z = _linear_system()
    fit = dmdm.fit_dmd([z], rank=2)
    assert fit["rank"] == 2
    assert np.linalg.matrix_rank(fit["K"], tol=1e-8) <= 2


def test_dmd_segmentos_no_mezclan_pares():
    a, z = _linear_system(T=200)
    fit = dmdm.fit_dmd([z[:100], z[100:]], rank=None)
    assert np.allclose(fit["K"], a, atol=1e-8)


def test_split_segments_parte_en_huecos():
    z = np.arange(10, dtype=float).reshape(10, 1)
    pos = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11])
    segs = dmdm.split_segments(z, pos)
    assert [len(s) for s in segs] == [3, 7]


def test_pronostico_promedia_ultimas_24h_del_decodificado():
    # Serie conocida: el decodificado de la ventana que termina en t+24 coincide con la serie.
    series = np.arange(200, dtype=float)[:, None].repeat(6, axis=1)  # [T, 6]
    t = 80
    window = series[t + 24 - 71 : t + 25][None, :, :]  # L=72, termina en t+24
    forecast = dmdm.forecast_from_decoded(window)
    expected = series[t + 1 : t + 25].mean(axis=0)  # mean(MP(t+1 … t+24h))
    assert np.allclose(forecast[0], expected)


def test_pronostico_requiere_L_mayor_igual_24():
    with pytest.raises(ValueError):
        dmdm.forecast_from_decoded(np.zeros((1, 12, 6)))


def test_operador_persistible():
    _, z = _linear_system()
    fit = dmdm.fit_dmd([z])
    arrays = dmdm.dmd_operator_arrays(fit)
    assert {"K", "rank", "eigvals_real", "eigvals_imag"}.issubset(arrays)
    assert arrays["K"].shape == (3, 3)
