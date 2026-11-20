"""Datos, fases y bucle de entrenamiento del Koopman (`deep-koopman`, D8, D9, D12, D16).

- Presupuesto fijo por fase (sin early stopping): F1 60 épocas (encoder GRU en caliente congelado las
  primeras 5), F2 30 (currículo H = 1 → 6 → 24, 10 épocas cada uno), F3 20 (H = 24 + `L_tgt`),
  `cierre` 20 (solo `K` y `B`). B1–B4 = F1 con su cambio. `smoke` = 2 épocas de F1.
- Adam, lr 1e-3 con calentamiento lineal de 2 épocas, `clipnorm = 1.0`, batch 128.
- Las variables congeladas reciben gradiente 0 (el optimizador se construye una vez con todas):
  con Adam y momentos en 0 su actualización es exactamente 0.
- Checkpoint por época en `<ckpt_dir>/<variante>/` (sobrescribe): pesos, optimizador, época,
  escalas de pérdida, RNG de numpy/Python y del barajado, historia. Retoma en la época siguiente.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pathlib
import pickle
import random

import numpy as np
import pandas as pd

from ae import data as aed

from . import perdidas as L
from .modelo import HORIZONTE, N_MP, U_IDX, KoopmanModelo, bloques, estructura

BATCH = 128
LR = 1e-3
CLIPNORM = 1.0
WARMUP_EPOCAS = 2
N_ESCALA = 2048
SEMILLA_ESCALA = 12345

FASES: dict[str, dict] = {
    "smoke": {"epocas": 2, "congeladas": 1},
    "f1": {"epocas": 60, "congeladas": 5},
    "b1": {"epocas": 60, "congeladas": 5},
    "b2": {"epocas": 60, "congeladas": 5, "cons": True},
    "b3": {"epocas": 60, "congeladas": 5, "l1": True},
    "b4": {"epocas": 60, "congeladas": 5},
    "f2": {"epocas": 30, "curriculo": ((10, 1), (10, 6), (10, 24))},
    "f3": {"epocas": 20, "H": 24, "tgt": True},
    "cierre": {"epocas": 20, "solo_op": True},
}


def h_de_epoca(cfg: dict, epoca: int) -> int:
    """Horizonte de `L_pred` en la época (currículo de F2; fijo en el resto)."""
    if "curriculo" in cfg:
        acc = 0
        for n, h in cfg["curriculo"]:
            acc += n
            if epoca < acc:
                return h
        return cfg["curriculo"][-1][1]
    return int(cfg.get("H", 0))


def lr_de_paso(paso: int, pasos_por_epoca: int, lr: float = LR) -> float:
    return lr * min(1.0, (paso + 1) / float(max(1, WARMUP_EPOCAS * pasos_por_epoca)))


# --- Datos ----------------------------------------------------------------------------


def targets_24h(grid: aed.HourlyGrid) -> np.ndarray:
    """`mean(MP(t+1…t+24))` con 24 valores reales (como `features.build`), `[T, 6]` (NaN si falta)."""
    raw = grid.mp.where(grid.mp_real)
    roll = raw.rolling(HORIZONTE, min_periods=HORIZONTE).mean().to_numpy(dtype=float)
    out = np.full_like(roll, np.nan)
    out[:-HORIZONTE] = roll[HORIZONTE:]
    return out


class Datos:
    """Ventanas de ajuste. `positions`: posiciones finales elegibles; `fin`: posición exclusiva del
    tramo de ajuste (toda ventana usada en pérdidas multi-paso o target termina antes)."""

    def __init__(self, grid: aed.HourlyGrid, ng: aed.NormalizedGrid, positions, fin: int):
        positions = np.asarray(positions, dtype=int)
        self.pos = positions[positions + HORIZONTE < fin]
        mp = ng.mp_z.fillna(0.0).to_numpy(dtype=np.float32)
        self.enc_all = np.concatenate([mp, ng.aux_z.to_numpy(dtype=np.float32),
                                       ng.aux_avail.to_numpy(dtype=np.float32)], axis=1)
        self.mp_all = mp
        self.real_all = ng.mp_real.to_numpy().astype(np.float32)
        self.u_all = ng.cal.to_numpy(dtype=np.float32)[:, U_IDX]
        self.v1 = np.isin(self.pos + 1, positions).astype(np.float32)
        y = targets_24h(grid)[self.pos]
        self.y = y.astype(np.float32)
        self.ym = np.isfinite(y).astype(np.float32)
        self.w = L.pesos_metrica(y).astype(np.float32)
        self.iz, self.izw = L.icap_zona_real(y)

    def __len__(self) -> int:
        return len(self.pos)

    def lote(self, idx) -> dict:
        p = self.pos[np.asarray(idx, dtype=int)]
        w0 = (p - HORIZONTE + 1)[:, None] + np.arange(HORIZONTE)[None, :]
        wk = w0[:, None, :] + np.arange(HORIZONTE + 1)[None, :, None]  # [N, 25, 24]
        uk = p[:, None] + np.arange(1, HORIZONTE + 1)[None, :]
        return {"x0": self.enc_all[w0], "x1": self.enc_all[w0 + 1], "v1": self.v1[idx],
                "mp": self.mp_all[wk], "real": self.real_all[wk], "U": self.u_all[uk],
                "y": np.nan_to_num(self.y[idx]), "ym": self.ym[idx], "w": self.w[idx],
                "iz": self.iz[idx], "izw": self.izw[idx]}


# --- Checkpoints ------------------------------------------------------------------------


class Checkpoint:
    """Un checkpoint por variante (sobrescribe): `<dir>/pesos.npz` + `<dir>/estado.json`."""

    def __init__(self, directorio: str | os.PathLike | None):
        self.dir = pathlib.Path(directorio) if directorio else None

    def existe(self) -> bool:
        return bool(self.dir and (self.dir / "estado.json").exists())

    def guardar(self, modelo: KoopmanModelo, opt, estado: dict) -> None:
        if not self.dir:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"m_{k}": v for k, v in modelo.pesos().items()}
        arrays.update({f"opt_{i}": np.asarray(v) for i, v in enumerate(opt.variables)})
        tmp = self.dir / "pesos.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self.dir / "pesos.npz")
        tmpj = self.dir / "estado.tmp.json"
        tmpj.write_text(json.dumps(estado, default=str), encoding="utf-8")
        os.replace(tmpj, self.dir / "estado.json")

    def cargar(self, modelo: KoopmanModelo, opt) -> dict:
        estado = json.loads((self.dir / "estado.json").read_text(encoding="utf-8"))
        npz = np.load(self.dir / "pesos.npz")
        modelo.cargar_pesos({k[2:]: npz[k] for k in npz.files if k.startswith("m_")})
        opt.build(modelo.entrenables)
        for i, v in enumerate(opt.variables):
            v.assign(npz[f"opt_{i}"])
        return estado


def _rng_estado(rng: np.random.Generator) -> dict:
    return {"np_shuffle": rng.bit_generator.state, "np_global": base64.b64encode(pickle.dumps(np.random.get_state())).decode(),
            "py": base64.b64encode(pickle.dumps(random.getstate())).decode()}


def _rng_restaurar(estado: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = estado["np_shuffle"]
    np.random.set_state(pickle.loads(base64.b64decode(estado["np_global"])))
    random.setstate(pickle.loads(base64.b64decode(estado["py"])))
    return rng


# --- Bucle ------------------------------------------------------------------------------


def _escalas(modelo, datos: Datos, cfg: dict, escala_mp, lambdas) -> dict:
    """Valor inicial de cada término en una muestra fija de 2048 ventanas de ajuste."""
    rng = np.random.default_rng(SEMILLA_ESCALA)
    idx = np.sort(rng.choice(len(datos), size=min(N_ESCALA, len(datos)), replace=False))
    cfg_e = dict(cfg)
    if "curriculo" in cfg or cfg.get("H"):
        cfg_e["H"] = HORIZONTE
    acc: dict[str, list] = {}
    for i in range(0, len(idx), 512):
        t = L.terminos(modelo, datos.lote(idx[i:i + 512]), cfg_e, escala_mp, training=False)
        for k, v in t.items():
            acc.setdefault(k, []).append(float(v) * len(idx[i:i + 512]))
    return {k: max(sum(v) / len(idx), 1e-12) for k, v in acc.items()}


def entrenar(modelo: KoopmanModelo, datos: Datos, fase: str, escala_mp, *, en_caliente: bool = False,
             lambdas: dict | None = None, epocas: int | None = None, ckpt_dir=None, seed: int = 0,
             cfg_perdida: dict | None = None, fallar_tras_epoca: int | None = None, log=print) -> dict:
    """Entrena `modelo` en la fase. `cfg_perdida`: en `cierre`, la configuración de pérdida de la
    fase final. Devuelve la historia por época y las escalas."""
    import keras
    import tensorflow as tf

    cfg = {**FASES[fase], **(cfg_perdida or {})}
    if fase == "cierre":
        cfg["solo_op"] = True
    n_ep = int(epocas if epocas is not None else cfg["epocas"])
    congeladas = int(cfg.get("congeladas", 0)) if en_caliente else 0
    lambdas = {**L.LAMBDAS, **(lambdas or {})}
    escala_mp = tuple(tf.constant(np.asarray(a, dtype="float32")) for a in escala_mp)
    opt = keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    todas = modelo.entrenables
    ckpt = Checkpoint(ckpt_dir)
    pasos_ep = int(math.ceil(len(datos) / BATCH))
    if ckpt.existe():
        estado = ckpt.cargar(modelo, opt)
        rng = _rng_restaurar(estado["rng"])
        escalas, historia, inicio = estado["escalas"], estado["historia"], int(estado["epoca"]) + 1
        if estado.get("terminado"):
            log(f"Checkpoint terminado ({fase}): no se repite")
            return {"historia": historia, "escalas": escalas, "retomado_en": None}
        log(f"Retoma {fase} en la época {inicio + 1} de {n_ep}")
    else:
        opt.build(todas)
        rng = np.random.default_rng(seed)
        escalas = _escalas(modelo, datos, {**cfg, "H": h_de_epoca(cfg, n_ep - 1)} if "curriculo" in cfg else cfg,
                           escala_mp, lambdas)
        historia, inicio = [], 0
    retomado = inicio if inicio > 0 else None
    ids_enc = {id(v) for v in modelo.vars_enc}
    ids_dec = {id(v) for v in modelo.vars_dec}
    paso_fn = PasoCompilado(modelo, opt, cfg, escala_mp, escalas, lambdas)
    z_ini = normas_z(modelo, datos, "ini" if inicio == 0 else "ret")
    for ep in range(inicio, n_ep):
        H = h_de_epoca(cfg, ep)
        congelado = ep < congeladas
        if cfg.get("solo_op"):
            activas = {id(v) for v in todas} - ids_enc - ids_dec
        elif congelado:
            activas = {id(v) for v in todas} - ids_enc
        else:
            activas = {id(v) for v in todas}
        act = tf.constant([1.0 if id(v) in activas else 0.0 for v in todas], dtype=tf.float32)
        perm = rng.permutation(len(datos))
        orden = hashlib.sha1(perm.tobytes()).hexdigest()[:12]
        suma: dict[str, float] = {}
        for s in range(pasos_ep):
            paso = ep * pasos_ep + s
            opt.learning_rate = lr_de_paso(paso, pasos_ep)
            terms, loss = paso_fn(datos.lote(perm[s * BATCH:(s + 1) * BATCH]), H, act)
            for k, v in {**terms, "total": loss}.items():
                suma[k] = suma.get(k, 0.0) + float(v)
        fila = {"epoca": ep + 1, "H": H, "congelado": congelado, "lr": float(lr_de_paso((ep + 1) * pasos_ep - 1, pasos_ep)),
                "orden": orden, **{f"loss_{k}": v / pasos_ep for k, v in suma.items()},
                **{f"norm_{k}": (v / pasos_ep) / (1.0 if k in L.SIN_NORMALIZAR else escalas.get(k, 1.0))
                   for k, v in suma.items() if k in escalas},
                **({"trazados": paso_fn.trazados} if paso_fn.trazados else {})}
        if ep == inicio:
            fila.update(z_ini)
        if ep == n_ep - 1:
            fila.update(normas_z(modelo, datos, "fin"))
        historia.append(fila)
        log(json.dumps(fila))
        terminado = ep == n_ep - 1
        ckpt.guardar(modelo, opt, {"fase": fase, "epoca": ep, "epocas": n_ep, "escalas": escalas, "historia": historia,
                                   "rng": _rng_estado(rng), "terminado": terminado})
        if fallar_tras_epoca is not None and ep + 1 == fallar_tras_epoca:
            raise RuntimeError(f"Fallo forzado tras guardar la época {ep + 1} (--dry-fallar-en)")
    return {"historia": historia, "escalas": escalas, "retomado_en": retomado, "trazados": paso_fn.trazados}


# --- Paso compilado (3.5) ---------------------------------------------------------------


def h_max(cfg: dict) -> int:
    """Horizonte máximo de la fase (fija el desenrollado del paso compilado)."""
    if "curriculo" in cfg:
        return max(h for _, h in cfg["curriculo"])
    return int(cfg.get("H_max", cfg.get("H", 0)))


def perdida_y_gradientes(modelo: KoopmanModelo, lote: dict, cfg: dict, H, escala_mp, escalas: dict, lambdas: dict,
                         training: bool = True):
    """Términos, pérdida total y gradientes (0 donde no hay) sobre `modelo.entrenables`."""
    import tensorflow as tf

    todas = modelo.entrenables
    with tf.GradientTape() as tape:
        terms = L.terminos(modelo, lote, {**cfg, "H": H}, escala_mp, training=training)
        loss = L.total(terms, escalas, lambdas)
    grads = tape.gradient(loss, todas)
    return terms, loss, [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, todas)]


class PasoCompilado:
    """Paso de optimización en `tf.function` con un solo trazado por fase: `H` (currículo) y la máscara de
    variables activas (congelado, `cierre`) entran como tensores; el lr es la variable del optimizador;
    el lote tiene dimensión de batch libre. `trazados` cuenta los trazados."""

    def __init__(self, modelo: KoopmanModelo, opt, cfg: dict, escala_mp, escalas: dict, lambdas: dict,
                 training: bool = True):
        self.modelo, self.opt, self.escala_mp, self.training = modelo, opt, escala_mp, training
        self.cfg = {k: v for k, v in cfg.items() if k not in ("H", "curriculo", "epocas", "congeladas")}
        self.cfg["H_max"] = h_max(cfg)
        self.escalas, self.lambdas = dict(escalas), dict(lambdas)
        self.trazados = 0
        self._fn = None

    def _cuerpo(self, lote, H, act):
        self.trazados += 1
        todas = self.modelo.entrenables
        terms, loss, grads = perdida_y_gradientes(self.modelo, lote, self.cfg, H, self.escala_mp, self.escalas,
                                                  self.lambdas, self.training)
        self.opt.apply_gradients([(g * act[i], v) for i, (g, v) in enumerate(zip(grads, todas))])
        return terms, loss

    def __call__(self, lote_np: dict, H: int, act):
        import tensorflow as tf

        lote = {k: tf.constant(v) for k, v in lote_np.items()}
        if self._fn is None:
            sig = ({k: tf.TensorSpec((None,) + tuple(v.shape[1:]), v.dtype) for k, v in lote.items()},
                   tf.TensorSpec([], tf.int32), tf.TensorSpec([len(self.modelo.entrenables)], tf.float32))
            self._fn = tf.function(self._cuerpo, input_signature=sig)
        return self._fn(lote, tf.constant(int(H), tf.int32), act)


def normas_z(modelo: KoopmanModelo, datos: Datos, momento: str, n: int = N_ESCALA) -> dict:
    """Norma media de `z` por bloque (`z_<momento>_b<j>`) en la muestra fija de ventanas de las escalas."""
    import tensorflow as tf

    rng = np.random.default_rng(SEMILLA_ESCALA)
    idx = np.sort(rng.choice(len(datos), size=min(n, len(datos)), replace=False))
    p = datos.pos[idx]
    w0 = (p - HORIZONTE + 1)[:, None] + np.arange(HORIZONTE)[None, :]
    z = np.concatenate([np.asarray(modelo.encode(tf.constant(datos.enc_all[w0[i:i + 512]]), training=False))
                        for i in range(0, len(w0), 512)])
    return {f"z_{momento}_b{j}": float(np.linalg.norm(z[:, b], axis=1).mean()) for j, b in enumerate(bloques(modelo.d))}


def medir_paso(modelo: KoopmanModelo, datos: Datos, cfg: dict, escala_mp, *, n: int = 10, previos: int = 2,
               seed: int = 0) -> dict:
    """Mide segundos por paso de optimización (humo, 8.1/8.2) con el paso compilado de la pérdida `cfg`
    (`H`, `H_max`, `tgt`…), sin checkpoint ni escalas (1.0); los `previos` pasos (con el trazado) no se
    cronometran. Devuelve `s_paso`, `pasos_epoca`, `s_epoca` estimados y `s_trazado`."""
    import time

    import keras
    import tensorflow as tf

    escala_mp = tuple(tf.constant(np.asarray(a, dtype="float32")) for a in escala_mp)
    opt = keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    todas = modelo.entrenables
    opt.build(todas)
    paso_fn = PasoCompilado(modelo, opt, cfg, escala_mp, {}, L.LAMBDAS)
    act = tf.ones([len(todas)], tf.float32)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    t_traz = None
    for s in range(previos + n):
        if s == previos:
            t_traz = time.perf_counter() - t0
            t0 = time.perf_counter()
        _, loss = paso_fn(datos.lote(rng.choice(len(datos), size=min(BATCH, len(datos)), replace=False)),
                          int(cfg.get("H", 0)), act)
        float(loss)
    s_paso = (time.perf_counter() - t0) / n
    pasos = int(math.ceil(len(datos) / BATCH))
    return {"s_paso": s_paso, "pasos_epoca": pasos, "s_epoca": s_paso * pasos, "s_trazado": t_traz,
            "trazados": paso_fn.trazados}


# --- Orden canónico (D9) ---------------------------------------------------------------


def orden_canonico(modelo: KoopmanModelo) -> list[int]:
    """Libres y reales por `ρ` descendente (permuta `z`: Dense final del encoder, entrada del decoder,
    filas de `B` y parámetros); signo de cada real con el mayor peso del decoder positivo.
    Devuelve la permutación de coordenadas aplicada. Sin efecto en `kden`."""
    if modelo.op.kden is not None:
        return list(range(modelo.d))
    n_lib, n_real = estructura(modelo.d)
    rho = np.asarray(modelo.op.rho(), dtype=float)
    lib_ord = list(np.argsort(-rho[1:1 + n_lib], kind="stable"))
    real_ord = list(np.argsort(-rho[1 + n_lib:], kind="stable"))
    bl = bloques(modelo.d)
    nuevo = [bl[0]] + [bl[1 + i] for i in lib_ord] + [bl[1 + n_lib + j] for j in real_ord]
    perm = np.concatenate(nuevo)
    signo = np.ones(modelo.d, dtype="float32")
    kd = modelo.kernel_decoder()  # d × ·
    for j in range(n_real):
        c = int(perm[2 + 2 * n_lib + j])
        fila = kd[c]
        if fila[np.argmax(np.abs(fila))] < 0:
            signo[c] = -1.0
    # encoder: última Dense
    dz = modelo.enc.get_layer("enc_z")
    k, b = dz.get_weights()
    dz.set_weights([(k * signo[None, :])[:, perm], (b * signo)[perm]])
    capa = modelo.dec.get_layer("dec_lin" if modelo.config["decoder"] == "lineal" else "dec_gru")
    w = capa.get_weights()
    w[0] = (w[0] * signo[:, None])[perm]
    capa.set_weights(w)
    if modelo.op.B is not None:
        modelo.op.B.assign((np.asarray(modelo.op.B) * signo[:, None])[perm])
    a = np.asarray(modelo.op.a)
    a_new = np.concatenate([a[:1], a[1:1 + n_lib][lib_ord], a[1 + n_lib:][real_ord]])
    modelo.op.a.assign(a_new.astype("float32"))
    modelo.op.b.assign(np.asarray(modelo.op.b)[lib_ord].astype("float32"))
    return [int(x) for x in perm]
