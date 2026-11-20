"""Reglas de decisión por fase -> JSON de rama (`deep-koopman`, D9 de 7.1, D11, D12, D14, D18).

La máquina de estados no decide: lee `rama` de este JSON. Ramas permitidas: `pasa`, `b1`–`b4`, `c`,
`pausa`, `continuar`. Precisión: una diferencia cuenta solo si su intervalo excluye 0; empate → la
opción más simple (decoder lineal < GRU; estructurado < denso; GRU < conv1d; sin `icap` < con `icap`).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

RAMAS = ("pasa", "b1", "b2", "b3", "b4", "c", "pausa", "continuar")
REMEDIOS = ("b1", "b2", "b3", "b4")
UMBRAL_DESBALANCE = 10.0
PREFIJO = "model-artifacts/orquestacion/deep-koopman"


def _diff(a, b) -> dict:
    d = np.asarray(a.media_boot) - np.asarray(b.media_boot)
    return {"diff": float(a.media - b.media), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}


def _mejor_sig(d: dict) -> bool:
    return d["hi"] < 0


def _peor_sig(d: dict) -> bool:
    return d["lo"] > 0


def siguiente_remedio(fase: str, lin_rec: float) -> str:
    """Tras un fallo de la compuerta: F1 -> `b1` si `L_lin/L_rec > 10`, si no `b2`; B_k -> B_{k+1}; B4 -> `c`."""
    if fase == "f1":
        return "b1" if lin_rec > UMBRAL_DESBALANCE else "b2"
    if fase in ("b1", "b2", "b3"):
        return {"b1": "b2", "b2": "b3", "b3": "b4"}[fase]
    return "c"


def decidir(fase: str, cand: dict, ref71=None) -> dict:
    """`cand = {nombre: {"puntaje", "identificable": bool, "ganadores": {seed: run_id}, "lin_rec": float,
    "encoder": str, "variante": str}}`. Principal = `gru` (variante `principal`).

    - F1 y B1–B4: aprueba la compuerta del principal (o, en B1, de alguna variante λ_lin) → `pasa`;
      si no, el remedio siguiente. Decoder GRU significativamente mejor que el lineal → `pausa`.
      conv1d sigue a F2 solo si su compuerta iguala o supera a la del GRU y no es significativamente peor.
    - F2: `continuar`. F3: `icap` reemplaza al principal solo si es significativamente mejor.
    - Datos faltantes o no finitos → `pausa`.
    """
    out = {"fase": fase, "rama": None, "motivo": "", "ganadores": {}, "encoders": [], "cifras": {},
           "fecha": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    for nombre, c in cand.items():
        if c.get("puntaje") is None or not np.isfinite(c["puntaje"].media):
            out.update(rama="pausa", motivo=f"resultado faltante o no finito: {nombre}")
            return out
        out["cifras"][nombre] = {"wrmse_media": float(c["puntaje"].media), "identificable": bool(c.get("identificable"))}
        if ref71 is not None:
            out["cifras"][nombre]["diff_71"] = _diff(c["puntaje"], ref71)
    principal = "gru" if fase in ("f1", "f2", "f3") or "gru" in cand else next(iter(cand))
    if fase in ("f1",) + REMEDIOS:
        if fase == "b1":
            ok = [n for n in cand if cand[n].get("identificable")]
            if len(ok) > 1:
                mejor = min(ok, key=lambda n: cand[n]["puntaje"].media)
                otro = [n for n in ok if n != mejor][0]
                if not _mejor_sig(_diff(cand[mejor]["puntaje"], cand[otro]["puntaje"])):
                    out.update(rama="pausa", motivo=f"B1: {mejor} y {otro} aprueban sin diferencia significativa")
                    return out
                ok = [mejor]
            if ok:
                out.update(rama="pasa", motivo=f"B1 aprueba con {ok[0]}", ganadores=cand[ok[0]]["ganadores"],
                           encoders=["gru"])
            else:
                out.update(rama="b2", motivo="B1 no aprueba: B2 con λ_lin = 1")
            return out
        if principal not in cand:
            out.update(rama="pausa", motivo="falta el encoder principal")
            return out
        p = cand[principal]
        if "gru-decgru" in cand and _mejor_sig(_diff(cand["gru-decgru"]["puntaje"], p["puntaje"])):
            out.update(rama="pausa", motivo="decoder GRU significativamente mejor que el lineal (decide el operador)")
            return out
        encoders = ["gru"]
        out["ganadores_por_encoder"] = {"gru": p["ganadores"]}
        if fase == "f1" and "conv1ds" in cand:
            c1 = cand["conv1ds"]
            iguala = bool(c1.get("identificable")) >= bool(p.get("identificable"))
            if iguala and not _peor_sig(_diff(c1["puntaje"], p["puntaje"])):
                encoders.append("conv1ds")
                out["ganadores_por_encoder"]["conv1ds"] = c1["ganadores"]
            out["cifras"]["conv1ds_vs_gru"] = _diff(c1["puntaje"], p["puntaje"])
        out["encoders"] = encoders
        if p.get("identificable"):
            out.update(rama="pasa", motivo=f"compuerta de identificabilidad aprobada en {fase}", ganadores=p["ganadores"])
        else:
            rama = siguiente_remedio(fase, float(p.get("lin_rec", 0.0)))
            out.update(rama=rama, motivo=f"compuerta no aprobada en {fase} (L_lin/L_rec = {p.get('lin_rec', 0.0):.2f})",
                       ganadores=p["ganadores"])
        return out
    if fase == "f2":
        out.update(rama="continuar", motivo="F2 terminada", ganadores=cand[principal]["ganadores"],
                   encoders=sorted({c.get("encoder", "gru") for c in cand.values()}),
                   ganadores_por_encoder={c.get("encoder", n): c["ganadores"] for n, c in cand.items()
                                          if c.get("variante", "principal") == "principal"})
        return out
    if fase == "f3":
        elegido = principal
        if "gru-icap" in cand:
            d = _diff(cand["gru-icap"]["puntaje"], cand[principal]["puntaje"])
            out["cifras"]["icap_vs_principal"] = d
            if _mejor_sig(d):
                elegido = "gru-icap"
        if "conv1ds" in cand:
            d = _diff(cand["conv1ds"]["puntaje"], cand[elegido]["puntaje"])
            out["cifras"]["conv1ds_vs_final"] = d
            if _mejor_sig(d):
                elegido = "conv1ds"
        out.update(rama="continuar", motivo=f"F3: final = {elegido}", ganadores=cand[elegido]["ganadores"],
                   encoders=[cand[elegido].get("encoder", "gru")], final=elegido)
        return out
    out.update(rama="pausa", motivo=f"fase sin reglas: {fase}")
    return out


def validar(decision: dict) -> dict:
    if decision.get("rama") not in RAMAS:
        raise ValueError(f"Rama no permitida: {decision.get('rama')!r}")
    return decision


def clave_decision(campana: str, fase: str) -> str:
    return f"{PREFIJO}/{campana}/{fase}.json"


def clave_arranque_f2(campana: str) -> str:
    """Decisión que aprobó la compuerta (F1 o B_k): padre de F2. La escribe la compuerta con rama `pasa`."""
    return f"{PREFIJO}/{campana}/arranque_f2.json"


def clave_marcador(campana: str, clave: str) -> str:
    return f"{PREFIJO}/{campana}/jobs/{clave}.json"


def escribir(decision: dict, s3, bucket: str, campana: str, fase: str) -> str:
    """Escribe `<campaña>/<fase>.json` (y `arranque_f2.json` si la rama es `pasa`)."""
    key = clave_decision(campana, fase)
    body = json.dumps(validar(decision), ensure_ascii=False, default=str).encode()
    if decision["rama"] == "pasa":
        s3.put_object(Bucket=bucket, Key=clave_arranque_f2(campana), Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key
