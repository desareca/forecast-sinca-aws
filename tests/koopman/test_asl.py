"""Tarea 7.1: la máquina de estados (ASL) cumple D18."""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASL = ROOT / "infra" / "stepfunctions" / "deep-koopman.asl.json"
ERRORES_D18 = {"SageMaker.AmazonSageMakerException", "SageMaker.ResourceLimitExceededException"}


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


def test_referencias_validas():
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


def test_maps_en_serie_y_jobs_con_marcador_y_checkpoint():
    asl = _render()
    maps = [st for _, st in _todos(asl["States"]) if st["Type"] == "Map"]
    assert maps and all(st.get("MaxConcurrency") == 1 for st in maps)
    for st in maps:
        inner = st["ItemProcessor"]
        ini = inner["States"][inner["StartAt"]]
        assert inner["StartAt"].endswith("Marcador") and ini["Parameters"]["Payload"]["modo"] == "estado_job"
        hecho = inner["States"][ini["Next"]]
        job = inner["States"][hecho["Default"]]
        assert job["Resource"].endswith("createTrainingJob.sync")  # todo job precedido por el chequeo del marcador
    nombres = [n.split("/")[-1] for n, _ in _todos(asl["States"])]
    assert len(nombres) == len(set(nombres))  # nombres únicos en toda la máquina
    jobs = [st for _, st in _todos(asl["States"]) if st.get("Resource", "").endswith("createTrainingJob.sync")]
    assert jobs and all("CheckpointConfig" in st["Parameters"] for st in jobs)


def test_retry_solo_transitorios():
    for nombre, st in _todos(_render()["States"]):
        for r in st.get("Retry", []):
            assert set(r["ErrorEquals"]) <= ERRORES_D18, nombre
            assert r["MaxAttempts"] <= 2


def test_test_solo_tras_la_pausa():
    s = _render()["States"]
    assert s["PausaTest"]["Resource"].endswith("waitForTaskToken") and s["PausaTest"]["TimeoutSeconds"] == 604800
    # alcanzabilidad sin pasar por PausaTest
    vistos, pila = set(), [_render()["StartAt"]]
    while pila:
        n = pila.pop()
        if n in vistos or n == "PausaTest":
            continue
        vistos.add(n)
        pila += _siguientes(s[n])
    assert "Test" not in vistos and "Cierre" in vistos


def test_pausas_y_avisos():
    s = _render()["States"]
    pausas = [n for n, st in s.items() if st.get("Resource", "").endswith("sns:publish.waitForTaskToken")]
    assert {"PausaF1", "PausaC", "PausaTest"} <= set(pausas)
    for n in pausas:
        assert "$$.Task.Token" in s[n]["Parameters"]["Message.$"]
    avisos = [n for n, st in s.items() if st.get("Resource", "").endswith("sns:publish")]
    assert {"AvisoF1", "AvisoF3", "AvisoCierre", "AvisoFinal"} <= set(avisos)
    assert s["RamaF1"]["Default"] == "RamaInvalida" and s["RamaInvalida"]["Type"] == "Fail"
