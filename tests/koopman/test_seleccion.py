"""Tarea 6.5: selección preregistrada (D14)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sint_koopman import FakeS3
from koopman import seleccion as S

_RNG = np.random.default_rng(1)


def P(media, ruido=0.3):
    """Puntaje sintético con ruido propio (dos candidatas no comparten remuestreos idénticos)."""
    return SimpleNamespace(media=media, media_boot=media + ruido * _RNG.normal(size=2000))


def C(fase, enc="gru", var="principal", media=30.0, ok=False):
    return {S.nombre_config(fase, enc, var): {"fase": fase, "encoder": enc, "variante": var, "puntaje": P(media),
                                               "identificable": ok, "runs": {0: f"{fase}{enc}{var}0"},
                                               "bloques": {0: [0, 1, 2]}}}


def test_identificable_gana_aunque_otra_sea_mejor():
    cand = {**C("f1", media=30.0), **C("f2", media=25.0), **C("b3", media=31.0, ok=True)}
    s = S.seleccionar(cand)
    assert s["configuracion"] == "b3-gru" and s["interpretabilidad"] == "cumple"
    assert s["identificables"] == ["b3-gru"] and s["runs"] == {"0": "b3gruprincipal0"}


def test_sin_identificables_gana_por_precision():
    cand = {**C("f1", media=30.0), **C("f2", media=27.0), **C("f3", media=24.0), **C("f3", var="icap", media=24.1)}
    s = S.seleccionar(cand)
    assert s["configuracion"] == "f3-gru" and s["interpretabilidad"] == "no cumple"
    assert [p["reemplaza"] for p in s["pasos"]] == [True, True, False]
    assert s["orden"] == ["f1-gru", "f2-gru", "f3-gru", "f3-gru-icap"]


def test_empate_queda_la_mas_simple():
    cand = {**C("f1", media=30.0), **C("f1", enc="conv1ds", media=29.9), **C("b1", var="ll01", media=30.05)}
    s = S.seleccionar(cand)
    assert s["configuracion"] == "f1-gru"
    assert s["orden"] == ["f1-gru", "f1-conv1ds", "b1-gru-ll01"]  # GRU < conv1d dentro de F1; F1 < B1


def test_decgru_y_controles_se_reportan_y_no_se_eligen():
    cand = {**C("f1", media=30.0), **C("f1", var="decgru", media=20.0), **C("f1", var="kden", media=19.0, ok=True)}
    s = S.seleccionar(cand)
    assert s["configuracion"] == "f1-gru" and s["interpretabilidad"] == "no cumple"
    assert s["decoder_gru_vs_lineal"]["significativamente_mejor"]
    assert {f["configuracion"] for f in s["tabla"] if not f["candidata"]} == {"f1-gru-decgru", "f1-gru-kden"}
    assert "decoder GRU" in s["resumen"]


def test_no_finito_o_sin_candidatas_falla():
    with pytest.raises(ValueError):
        S.seleccionar({**C("f1", media=float("nan"))})
    with pytest.raises(ValueError):
        S.seleccionar({**C("f1", var="kden")})


def test_escribir_json():
    s3 = FakeS3()
    key = S.escribir(S.seleccionar({**C("f1")}), s3, "b", "camp", "seleccion")
    assert key == "model-artifacts/orquestacion/deep-koopman/camp/seleccion.json"
    assert json.loads(s3.objects[key])["configuracion"] == "f1-gru"
