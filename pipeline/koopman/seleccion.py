"""Selección preregistrada de la configuración final (`deep-koopman`, D14). La máquina de estados no
decide: corre la campaña completa y este módulo elige antes del cierre.

- Candidatas: configuraciones con decoder lineal que no son de control: F1, B1 (λ_lin 0.1 y 10), B2, B3,
  B4, F2, F3 y F3 `icap`, cada una con GRU y conv1d si existen. Controles (`kden`, `perlib`, `sinbu`,
  `decgru`) se reportan, no se eligen.
- Orden de simplicidad fijo: F1 < B1 λ0.1 < B1 λ10 < B2 < B3 < B4 < F2 < F3 < F3 `icap`; dentro, GRU < conv1d.
- Si alguna candidata es identificable (D11) se elige solo entre esas; si no, entre todas y la
  interpretabilidad queda "no cumple". Se parte de la más simple; una más compleja la reemplaza solo
  si la diferencia emparejada de RMSE ponderado (más compleja − actual) tiene intervalo < 0.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

PREFIJO = "model-artifacts/orquestacion/deep-koopman"
ORDEN = (("f1", "principal"), ("b1", "ll01"), ("b1", "ll10"), ("b2", "principal"), ("b3", "principal"),
         ("b4", "principal"), ("f2", "principal"), ("f3", "principal"), ("f3", "icap"))
ENCODERS = ("gru", "conv1ds")
CONTROLES = ("kden", "perlib", "sinbu", "decgru")


def nombre_config(fase: str, encoder: str, variante: str) -> str:
    return f"{fase}-{encoder}" + ("" if variante == "principal" else f"-{variante}")


def es_candidata(fase: str, encoder: str, variante: str) -> bool:
    return (fase, variante) in ORDEN and encoder in ENCODERS


def rango(c: dict) -> tuple[int, int]:
    return ORDEN.index((c["fase"], c["variante"])), ENCODERS.index(c["encoder"])


def diferencia(a, b) -> dict:
    """`a − b` (puntajes con `media` y `media_boot` emparejados): diferencia e intervalo (`operador`, D9 de 7.1)."""
    from operador import evaluacion as evm

    return {k: float(v) for k, v in evm.diferencia(a, b).items()}


def seleccionar(cand: dict, ref71=None) -> dict:
    """`cand = {nombre: {"fase", "encoder", "variante", "puntaje", "identificable": bool,
    "runs": {seed: run_id}, "bloques": {seed: [...]}}}` → selección (D14)."""
    for n, c in cand.items():
        if c.get("puntaje") is None or not np.isfinite(c["puntaje"].media):
            raise ValueError(f"Resultado faltante o no finito: {n}")
    candidatas = sorted((n for n, c in cand.items() if es_candidata(c["fase"], c["encoder"], c["variante"])),
                        key=lambda n: rango(cand[n]))
    if not candidatas:
        raise ValueError("Sin candidatas para la selección")
    identificables = [n for n in candidatas if cand[n].get("identificable")]
    pool = identificables or candidatas
    actual, pasos = pool[0], []
    for n in pool[1:]:
        d = diferencia(cand[n]["puntaje"], cand[actual]["puntaje"])
        reemplaza = d["hi"] < 0
        pasos.append({"candidata": n, "contra": actual, **d, "reemplaza": reemplaza})
        if reemplaza:
            actual = n
    tabla = []
    for n, c in sorted(cand.items(), key=lambda kv: (not es_candidata(kv[1]["fase"], kv[1]["encoder"], kv[1]["variante"]),
                                                      kv[0])):
        fila = {"configuracion": n, "fase": c["fase"], "encoder": c["encoder"], "variante": c["variante"],
                "wrmse_media": float(c["puntaje"].media), "identificable": bool(c.get("identificable")),
                "candidata": es_candidata(c["fase"], c["encoder"], c["variante"])}
        if ref71 is not None:
            fila["diff_71"] = diferencia(c["puntaje"], ref71)
        tabla.append(fila)
    decoder_gru = None
    if "f1-gru-decgru" in cand and "f1-gru" in cand:
        d = diferencia(cand["f1-gru-decgru"]["puntaje"], cand["f1-gru"]["puntaje"])
        decoder_gru = {**d, "significativamente_mejor": d["hi"] < 0}
    c = cand[actual]
    interp = "cumple" if identificables else "no cumple"
    return {"configuracion": actual, "fase": c["fase"], "encoder": c["encoder"], "variante": c["variante"],
            "runs": {str(s): r for s, r in c["runs"].items()},
            "bloques_finales": {str(s): b for s, b in (c.get("bloques") or {}).items()},
            "interpretabilidad": interp, "identificables": identificables, "orden": candidatas, "pasos": pasos,
            "tabla": tabla, "decoder_gru_vs_lineal": decoder_gru,
            "resumen": (f"Seleccionada {actual} (wRMSE sel {c['puntaje'].media:.2f}); interpretabilidad {interp}; "
                        f"identificables: {', '.join(identificables) or 'ninguna'}"
                        + ("; decoder GRU significativamente mejor que el lineal (se reporta, no se elige)"
                           if decoder_gru and decoder_gru["significativamente_mejor"] else "")),
            "fecha": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def clave_json(campana: str, nombre: str) -> str:
    """`<campaña>/<nombre>.json` (`seleccion`, `cierre`, `test`)."""
    return f"{PREFIJO}/{campana}/{nombre}.json"


def clave_marcador(campana: str, clave: str) -> str:
    return f"{PREFIJO}/{campana}/jobs/{clave}.json"


def escribir(contenido: dict, s3, bucket: str, campana: str, nombre: str) -> str:
    key = clave_json(campana, nombre)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(contenido, ensure_ascii=False, default=str).encode(),
                  ContentType="application/json")
    return key
