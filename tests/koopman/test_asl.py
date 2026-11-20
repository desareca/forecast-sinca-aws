"""Tareas 7.1 y 7.8: la máquina de estados (ASL) cumple D18 (campaña fija, sin ramas ni pausas)."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASL = ROOT / "infra" / "stepfunctions" / "deep-koopman.asl.json"
ERRORES_D18 = {"SageMaker.AmazonSageMakerException", "SageMaker.ResourceLimitExceededException"}
FASES = ["F1", "B", "F2", "F3", "Seleccion", "Cierre", "Test"]


def _render():
    text = ASL.read_text(encoding="utf-8")
    assert set(re.findall(r"\$\{(\w+)\}", text)) == {"lambda_arn", "topic_arn"}
    text = text.replace("${lambda_arn}", "arn:aws:lambda:us-east-1:111111111111:function:x").replace(
        "${topic_arn}", "arn:aws:sns:us-east-1:111111111111:t")
    return json.loads(text)


def _todos(states, prefijo=""):
    for nombre, st in states.items():
        yield prefijo + nombre, st
        if st["Type"] == "Map":
            yield from _todos(st["ItemProcessor"]["States"], prefijo + nombre + "/")


def _siguientes(st):
    out = []
    if "Next" in st:
        out.append(st["Next"])
    if st["Type"] == "Choice":
        out += [c["Next"] for c in st["Choices"]] + ([st["Default"]] if "Default" in st else [])
    for c in st.get("Catch", []):
        out.append(c["Next"])
    return out


def _alcanzables(s, inicio, evitar=()):
    vistos, pila = set(), [inicio]
    while pila:
        n = pila.pop()
        if n in vistos or n in evitar:
            continue
        vistos.add(n)
        pila += _siguientes(s[n])
    return vistos


def test_referencias_validas_y_nombres_unicos():
    asl = _render()
    s = asl["States"]
    assert asl["StartAt"] in s
    for nombre, st in s.items():
        for n in _siguientes(st):
            assert n in s, f"{nombre} -> {n}"
        if st["Type"] == "Map":
            inner = st["ItemProcessor"]["States"]
            for st2 in inner.values():
                for n in _siguientes(st2):
                    assert n in inner
    nombres = [n.split("/")[-1] for n, _ in _todos(s)]
    assert len(nombres) == len(set(nombres))
    assert _alcanzables(s, asl["StartAt"]) == set(s)  # sin estados huérfanos


def test_maps_en_serie_y_jobs_con_marcador_y_checkpoint():
    asl = _render()
    maps = {n: st for n, st in asl["States"].items() if st["Type"] == "Map"}
    assert list(maps) == FASES and all(st["MaxConcurrency"] == 1 for st in maps.values())
    for nombre, st in maps.items():
        assert st["ItemsPath"] == f"$.fases.{nombre.lower() if nombre != 'Seleccion' else 'seleccion'}"
        inner = st["ItemProcessor"]
        ini = inner["States"][inner["StartAt"]]
        assert ini["Parameters"]["Payload"]["modo"] == "estado_job"
        hecho = inner["States"][ini["Next"]]
        job = inner["States"][hecho["Default"]]
        assert job["Resource"].endswith("createTrainingJob.sync") and "CheckpointConfig" in job["Parameters"]


def test_sin_pausas_ni_ramas():
    todos = list(_todos(_render()["States"]))
    assert not any("waitForTaskToken" in st.get("Resource", "") for _, st in todos)
    choices = [(n, st) for n, st in todos if st["Type"] == "Choice"]
    assert len(choices) == len(FASES)  # el único Choice es el salto por marcador, uno por Map
    for n, st in choices:
        assert "/" in n and [c["Variable"] for c in st["Choices"]] == ["$.marcador.existe"]
    lambdas = [st["Parameters"]["Payload"]["modo"] for _, st in todos if st.get("Resource", "").endswith("lambda:invoke")]
    assert set(lambdas) == {"estado_job", "resumen"}


def test_retry_solo_transitorios():
    for nombre, st in _todos(_render()["States"]):
        for r in st.get("Retry", []):
            assert set(r["ErrorEquals"]) <= ERRORES_D18, nombre
            assert r["MaxAttempts"] <= 2


def test_orden_y_test_solo_tras_seleccion_y_cierre():
    asl = _render()
    s = asl["States"]
    assert "Test" not in _alcanzables(s, asl["StartAt"], evitar={"Seleccion"})
    assert "Test" not in _alcanzables(s, asl["StartAt"], evitar={"Cierre"})
    assert "Cierre" not in _alcanzables(s, asl["StartAt"], evitar={"Seleccion"})
    orden, n = [], asl["StartAt"]
    while s[n]["Type"] != "Succeed":
        if s[n]["Type"] == "Map":
            orden.append(n)
        n = s[n]["Next"]
    assert orden == FASES


def test_avisos():
    s = _render()["States"]
    avisos = {n for n, st in s.items() if st.get("Resource", "").endswith("sns:publish")}
    assert {"AvisoInicio", "AvisoF1", "AvisoB", "AvisoF2", "AvisoF3", "AvisoSeleccion", "AvisoCierre", "AvisoFinal"} <= avisos
    for f in ("F1", "B", "F2", "F3"):
        assert f"$.res_{f.lower()}" in s[f"Aviso{f}"]["Parameters"]["Message.$"]  # jobs lanzados y saltados
    for cual, aviso in (("seleccion", "AvisoSeleccion"), ("cierre", "AvisoCierre"), ("test", "AvisoFinal")):
        assert s[f"Resumen{cual.capitalize()}"]["Parameters"]["Payload"]["key.$"] == f"$.resumenes.{cual}"
        assert f"$.txt_{cual}.texto" in s[aviso]["Parameters"]["Message.$"]
