"""Tarea 6.2: reglas de la compuerta -> rama."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sint_koopman import FakeS3
from koopman import compuerta as C

G = np.random.default_rng(0).normal(size=2000)
_RNG = np.random.default_rng(1)


def P(media, ruido=0.3):
    """Puntaje sintético con ruido propio (dos candidatos no comparten remuestreos idénticos)."""
    return SimpleNamespace(media=media, media_boot=media + ruido * _RNG.normal(size=2000))


def cand(nombre="gru", media=30.0, ok=True, lin_rec=1.0, enc="gru", var="principal"):
    return {nombre: {"puntaje": P(media), "identificable": ok, "ganadores": {"0": "r0", "1": "r1", "2": "r2"},
                     "lin_rec": lin_rec, "encoder": enc, "variante": var}}


def test_aprobada_pasa():
    d = C.decidir("f1", cand())
    assert d["rama"] == "pasa" and d["ganadores"]["0"] == "r0" and d["encoders"] == ["gru"]


def test_fallo_con_y_sin_desbalance():
    assert C.decidir("f1", cand(ok=False, lin_rec=25.0))["rama"] == "b1"
    assert C.decidir("f1", cand(ok=False, lin_rec=3.0))["rama"] == "b2"
    assert C.decidir("b2", cand(ok=False))["rama"] == "b3"
    assert C.decidir("b3", cand(ok=False))["rama"] == "b4"
    assert C.decidir("b4", cand(ok=False))["rama"] == "c"
    assert C.decidir("b4", cand(ok=True))["rama"] == "pasa"


def test_b1():
    c = {**cand("gru-ll01", ok=False, var="ll01"), **cand("gru-ll10", ok=False, var="ll10")}
    assert C.decidir("b1", c)["rama"] == "b2"
    c = {**cand("gru-ll01", ok=True, var="ll01"), **cand("gru-ll10", ok=False, var="ll10")}
    assert C.decidir("b1", c)["rama"] == "pasa"
    c = {**cand("gru-ll01", 30.0, ok=True, var="ll01"), **cand("gru-ll10", 30.05, ok=True, var="ll10")}
    assert C.decidir("b1", c)["rama"] == "pausa"  # empate ambiguo


def test_conv1d_solo_si_iguala_o_supera():
    c = {**cand("gru", ok=True), **cand("conv1ds", ok=False, enc="conv1ds")}
    assert C.decidir("f1", c)["encoders"] == ["gru"]
    c = {**cand("gru", 30.0, ok=True), **cand("conv1ds", 30.1, ok=True, enc="conv1ds")}
    assert C.decidir("f1", c)["encoders"] == ["gru", "conv1ds"]
    c = {**cand("gru", 30.0, ok=True), **cand("conv1ds", 35.0, ok=True, enc="conv1ds")}
    assert C.decidir("f1", c)["encoders"] == ["gru"]


def test_decoder_gru_mejor_pausa_y_faltante_pausa():
    c = {**cand("gru", 30.0), **cand("gru-decgru", 25.0, var="decgru")}
    assert C.decidir("f1", c)["rama"] == "pausa"
    c = cand()
    c["gru"]["puntaje"] = SimpleNamespace(media=float("nan"), media_boot=G)
    assert C.decidir("f1", c)["rama"] == "pausa"


def test_f2_y_f3():
    assert C.decidir("f2", cand())["rama"] == "continuar"
    c = {**cand("gru", 30.0), **cand("gru-icap", 29.95, var="icap")}
    d = C.decidir("f3", c)
    assert d["rama"] == "continuar" and d["final"] == "gru"  # empate -> sin icap
    c = {**cand("gru", 30.0), **cand("gru-icap", 25.0, var="icap")}
    assert C.decidir("f3", c)["final"] == "gru-icap"


def test_escribir_valida_rama():
    s3 = FakeS3()
    key = C.escribir(C.decidir("f1", cand()), s3, "b", "camp1", "f1")
    assert key == "model-artifacts/orquestacion/deep-koopman/camp1/f1.json"
    assert json.loads(s3.objects[key])["rama"] == "pasa"
    with pytest.raises(ValueError):
        C.escribir({"rama": "otra"}, s3, "b", "camp1", "f1")
