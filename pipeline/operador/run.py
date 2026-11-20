"""CLI del operador global por etapas (`dmd-operador-global`, D1–D12).

    python pipeline/operador/run.py --etapa <id> --state-dir .\\mlflow-op --local-validated-dir .\\validated-op --no-download

Solo `base` descarga (con `--profile`); el resto corre sobre el estado local. Cada etapa con
decisión exige la decisión del operador de la etapa anterior en `decisiones.json`.
Ver `pipeline/operador/README.md`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_PIPELINE = pathlib.Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ae import data as aed  # noqa: E402
from analysis import load as analysis_load  # noqa: E402
from features import partition as pt  # noqa: E402
from features.build import load_validated  # noqa: E402
from operador import estado as est_mod  # noqa: E402
from operador import estimadores as estim  # noqa: E402
from operador import evaluacion as evm  # noqa: E402
from operador import modos  # noqa: E402
from operador import registro as reg  # noqa: E402
from operador import rollout as ro  # noqa: E402
from train import mlflow_log  # noqa: E402

log = logging.getLogger("operador")

PARTICION = "ae-dmd"
VERSION_V2 = "2"
ARCHS = ("conv1d", "gru")
EVAL_ROWS = "eval_rows.parquet"
SIMPLICIDAD = ("hankel", "hankelmet", "hankelest", "hankel48", "pca", "conv1d", "gru")


# --- Selección de runs v2 ----------------------------------------------------------


def clasificar_v2(runs: list[dict]) -> dict:
    """Runs de `ae-dmd-prep` v2: AE (6), DMD por serie (36), LightGBM v2 (6), persistencia v2 (1)."""
    out = {"ae": {}, "dmd": {}, "lgbm": [], "pers": None}
    for run in runs:
        tags, params = run.get("tags", {}), run.get("params", {})
        if tags.get("particion") != PARTICION or str(params.get("version")) != VERSION_V2:
            continue
        modelo = tags.get("modelo", "")
        seed = tags.get("seed")
        if modelo in (f"ae-{a}" for a in ARCHS):
            out["ae"][(modelo[3:], int(seed))] = run
        elif modelo in (f"ae-{a}-dmd" for a in ARCHS):
            out["dmd"].setdefault((modelo[3:-4], int(seed)), []).append(run)
        elif modelo == "lightgbm-aedmd":
            out["lgbm"].append(run)
        elif modelo == "persistence-aedmd":
            out["pers"] = run
    return out


def ids_descarga(v2: dict) -> list[str]:
    ids = [r["run_id"] for r in v2["ae"].values()]
    ids += [r["run_id"] for rs in v2["dmd"].values() for r in rs]
    ids += [r["run_id"] for r in v2["lgbm"]]
    ids += [r["params"]["source_run_id"] for r in v2["lgbm"] if r["params"].get("source_run_id")]
    if v2["pers"] is not None:
        ids.append(v2["pers"]["run_id"])
    return ids


def descargar(state_dir: str, profile: str) -> dict:
    """Descarga selectiva (D1): sqlite + artefactos de los runs v2 y `lightgbm-v1` fuente."""
    from persist.s3 import get_client

    if pathlib.Path(state_dir, mlflow_log.DB_FILENAME).exists():
        raise SystemExit(f"Ya existe {state_dir}/{mlflow_log.DB_FILENAME}: base no sobrescribe un estado local.")
    client = get_client(profile)
    db_key = f"{mlflow_log.MLFLOW_PREFIX_DEFAULT}/{mlflow_log.DB_FILENAME}"
    head = client.head_object(Bucket=mlflow_log.MLFLOW_BUCKET_DEFAULT, Key=db_key)
    analysis_load.ensure_state(data_dir=state_dir, run_ids=[], client=client)
    v2 = clasificar_v2(analysis_load.list_runs(state_dir))
    ids = ids_descarga(v2)
    objs = mlflow_log.download_state(mlflow_log.MLFLOW_BUCKET_DEFAULT, mlflow_log.MLFLOW_PREFIX_DEFAULT, state_dir,
                                     client, run_ids=ids)
    meta = {"last_modified": head["LastModified"].isoformat(), "objetos": len(objs), "runs": len(ids),
            "descargado": dt.datetime.now().isoformat(timespec="seconds"), "published_runs": []}
    pathlib.Path(state_dir, ".download.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --- Recursos compartidos ----------------------------------------------------------


class Recursos:
    def __init__(self, args):
        self.args = args
        self.state = args.state_dir
        self.out = pathlib.Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.part = pt.get_partition(PARTICION)
        self.runs = analysis_load.list_runs(self.state)
        self.v2 = clasificar_v2(self.runs)
        self._data = None
        self._grid = None
        self._ctx = None
        self._ae = {}
        self._ev = None
        self._rows = None
        self._features = None

    # Datos
    @property
    def data(self):
        if self._data is None:
            self._data = load_validated(bucket=None, local_dir=self.args.local_validated_dir)
        return self._data

    @property
    def grid(self) -> aed.HourlyGrid:
        if self._grid is None:
            d = self.data
            self._grid = aed.build_grid(d["sinca"], d["open_meteo"], d["feriados"])
        return self._grid

    @property
    def ctx(self) -> est_mod.Contexto:
        if self._ctx is None:
            self._ctx = est_mod.contexto(self.grid, self.part, ("train",))
        return self._ctx

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            from features.build import build_features

            d = self.data
            self._features = build_features(d["sinca"], d["open_meteo"], d["feriados"])
        return self._features

    def artifact(self, run_id: str, name: str) -> str:
        path = analysis_load.artifact_path(self.state, run_id, name)
        if not path:
            raise FileNotFoundError(f"Run {run_id} sin {name} en {self.state}")
        return path

    def preds(self, run: dict, split: str = "sel") -> pd.DataFrame:
        frame = pd.read_parquet(self.artifact(run["run_id"], "predictions.parquet"))
        return frame[frame["split"] == split][["timestamp", "estacion", "pollutant", "y_true", "y_pred"]]

    # Filas y evaluador
    def dmd_frames(self) -> dict[str, pd.DataFrame]:
        out = {}
        for (arch, seed), runs in sorted(self.v2["dmd"].items()):
            out[f"dmd-{arch}-s{seed}"] = pd.concat([self.preds(r) for r in runs], ignore_index=True)
        return out

    @property
    def rows(self) -> pd.DataFrame:
        if self._rows is None:
            path = self.out / EVAL_ROWS
            if not path.exists():
                raise SystemExit(f"Falta {path}: correr antes --etapa base.")
            self._rows = pd.read_parquet(path)
        return self._rows

    @property
    def ev(self) -> evm.Evaluador:
        if self._ev is None:
            self._ev = evm.Evaluador(self.rows, B=self.args.bootstrap, seed=0)
        return self._ev

    def eval_positions(self, rows: pd.DataFrame | None = None) -> np.ndarray:
        rows = self.rows if rows is None else rows
        stamps = pd.DatetimeIndex(pd.to_datetime(rows["timestamp"]).unique())
        index = self.grid.index
        if index.tz is not None and stamps.tz is None:
            stamps = stamps.tz_localize(index.tz)
        elif index.tz is not None:
            stamps = stamps.tz_convert(index.tz)
        pos = index.get_indexer(stamps)
        if (pos < 0).any():
            raise ValueError("Timestamps de eval_rows fuera de la grilla")
        return np.sort(pos)

    # AE
    def ae(self, arch: str, seed: int) -> dict:
        key = (arch, seed)
        if key not in self._ae:
            import keras

            run = self.v2["ae"][key]
            enc = keras.saving.load_model(self.artifact(run["run_id"], "encoder.keras"), compile=False)
            dec = keras.saving.load_model(self.artifact(run["run_id"], "decoder_mp.keras"), compile=False)
            latent = pd.read_parquet(self.artifact(run["run_id"], "latent.parquet"))
            aux_input = str(run["params"].get("aux_input", "True")).lower() in ("true", "1")
            estado = est_mod.ae_state(self.ctx, latent, dec, enc, aed.aux_input_columns(aux_input), nombre=f"{arch}-s{seed}")
            npz = np.load(self.artifact(run["run_id"], "dmd_operator.npz"))
            self._ae[key] = {"run": run, "estado": estado, "K_v2": npz["K"]}
        return self._ae[key]

    def estado(self, rep: str, seed: int = 0, ctx: est_mod.Contexto | None = None, union_fit: bool = False) -> est_mod.Estado:
        ctx = ctx or self.ctx
        if rep == "hankel":
            return est_mod.hankel(ctx, 24, False, union_fit=union_fit)
        if rep == "hankelmet":
            return est_mod.hankel(ctx, 24, True, union_fit=union_fit)
        if rep == "hankel48":
            return est_mod.hankel(ctx, 48, False, union_fit=union_fit)
        if rep == "hankelest":
            e = est_mod.hankel(ctx, 24, False, union_fit=union_fit, nombre="hankelest")
            return e
        if rep == "pca":
            return est_mod.pca_state(ctx, union_fit=union_fit)
        if rep in ARCHS:
            return self.ae(rep, seed)["estado"]
        raise ValueError(f"Representación desconocida: {rep}")


# --- Evaluación y registro de una variante --------------------------------------


def evaluar(rec: Recursos, nombre: str, stamps, forecast, decoded, positions, op: estim.Operador | None = None,
            mask: np.ndarray | None = None) -> dict:
    ev = rec.ev
    yhat = ev.alinear(nombre, stamps, forecast, mask)
    p = ev.puntaje(yhat, mask)
    tabla = ev.tabla(p, mask)
    frames = {
        "predictions.parquet": ev.predicciones(yhat, mask),
        "horizon.parquet": evm.horizonte(decoded, positions, rec.grid),
        "episodes.parquet": evm.episodios(ev, yhat, mask),
        "tramos.parquet": evm.tramos(ev, yhat, mask),
        "bootstrap.parquet": tabla,
        "spectrum.parquet": modos.espectro(op.eigvals(), op.pasos) if op is not None else modos.espectro(np.zeros(0, complex)),
    }
    return {"nombre": nombre, "yhat": yhat, "puntaje": p, "tabla": tabla, "frames": frames, "op": op}


def metricas_de(tabla: pd.DataFrame) -> dict[str, float]:
    out = {}
    for _, row in tabla.iterrows():
        key = "media" if row["serie"] == "media" else row["serie"].replace("|", "_").replace(" ", "_").lower()
        for col in ("wrmse", "razon_mejor", "razon_lgbm", "razon_pers", "skill_pers"):
            out[f"sel_{col}_{key}"] = row[col]
            out[f"sel_{col}_{key}_lo"] = row[f"{col}_lo"]
            out[f"sel_{col}_{key}_hi"] = row[f"{col}_hi"]
    return out


def registrar(rec: Recursos, res: dict, rep: str, exp: str, etapa: str, seed: int, tipo: str = "operador",
              estimador: str = "exacto", rollout: str = "autonomo", extra_params: dict | None = None,
              extra_frames: dict | None = None, ae_run_id: str | None = None) -> str:
    op = res["op"]
    name = reg.run_name(rep, exp, rec.part.token, rec.args.version, seed)
    params = {"tipo": tipo, "etapa": etapa, "estimador": estimador if op is not None else "ninguno",
              "rollout": rollout, "rank": op.rank if op is not None else "na", "version": rec.args.version,
              "representacion": rep, "bootstrap_B": rec.args.bootstrap, "bootstrap_dias": evm.DIAS_DEFAULT,
              **(extra_params or {})}
    tags = reg.tags_base(rep, etapa, PARTICION, seed, ae_run_id)
    frames = dict(res["frames"])
    frames.update(extra_frames or {})
    arrays = {"operator.npz": op.arrays()} if op is not None else None
    run_id = reg.log_variante(rec.state, name, tags, params, metricas_de(res["tabla"]), frames, arrays)
    log.info("Run %s (%s)", name, run_id)
    return run_id


# --- Etapas ------------------------------------------------------------------------


def etapa_base(rec: Recursos) -> None:
    """Fija `eval_rows` y comprueba continuidad con el DMD v2 (sin runs, D10)."""
    lgbm = pd.concat([rec.preds(r) for r in rec.v2["lgbm"]], ignore_index=True)
    pers = rec.preds(rec.v2["pers"])
    dmd = rec.dmd_frames()
    if not dmd or len(dmd) != len(rec.v2["ae"]) or len(rec.v2["lgbm"]) != 6 or rec.v2["pers"] is None:
        raise SystemExit(f"Runs v2 incompletos: AE {len(rec.v2['ae'])}, DMD {len(dmd)} semillas, "
                         f"LightGBM {len(rec.v2['lgbm'])} series, persistencia {rec.v2['pers'] is not None}")
    rows = evm.construir_eval_rows(dmd, lgbm, pers)
    rows.to_parquet(rec.out / EVAL_ROWS, index=False)
    rec._rows = rows
    n = rows.groupby(["estacion", "pollutant"]).size().rename("n").reset_index()
    pos = rec.eval_positions()
    stamps = rec.grid.index[pos]
    cont = []
    for (arch, seed) in sorted(rec.v2["ae"]):
        a = rec.ae(arch, seed)
        e = a["estado"]
        op = estim.exacto(e.segments(), None)
        k_rel = float(np.linalg.norm(op.K - a["K_v2"]) / np.linalg.norm(a["K_v2"]))
        forecast, _ = ro.autonomo(op, e, pos)
        yhat = rec.ev.alinear(f"{arch}-s{seed}", stamps, forecast)
        v2 = pd.concat([rec.preds(r) for r in rec.v2["dmd"][(arch, seed)]], ignore_index=True)
        yv = rows[evm.KEYS].merge(v2[evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
        rel = np.abs(yhat - yv) / np.maximum(np.abs(yv), 1e-12)
        p = rec.ev.puntaje(yhat)
        cont.append({"ae": f"{arch}-s{seed}", "K_rel": k_rel, "ypred_max_rel": float(rel.max()),
                     "ypred_mediana_rel": float(np.median(rel)), "wrmse_media": p.media,
                     "ok": bool(k_rel <= 1e-8 and rel.max() <= 1e-4)})
    cont = pd.DataFrame(cont)
    meta = {}
    meta_path = pathlib.Path(rec.state, ".download.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    refs = pd.DataFrame([{"ref": name, "wrmse_media": rec.ev.ref(name).media} for name in evm.REFS])
    reg.resumen_md("base", [
        ("Filas de evaluación", f"`eval_rows`: {len(rows)} filas; n mínimo por serie = {int(n['n'].min())}.\n\n" + reg.md_table(n)),
        ("Referencias (sel)", reg.md_table(refs)),
        ("Continuidad con DMD v2", reg.md_table(cont, 10)),
        ("Descarga", f"objetos = {meta.get('objetos')}, runs = {meta.get('runs')}, LastModified = {meta.get('last_modified')}"),
    ], rec.out, {"continuidad": cont.to_dict("records"), "n": n.to_dict("records")})
    print(reg.md_table(n))
    print(reg.md_table(cont, 10))


def lectura_oraculo(t: pd.DataFrame) -> str:
    hi = float(t.loc[t["serie"] == "media", "razon_mejor_hi"].iloc[0])
    return "techo en la representación" if hi >= 0.95 else "el problema es la dinámica"


def etapa_10(rec: Recursos) -> None:
    pos = rec.eval_positions()
    stamps = rec.grid.index[pos]
    fam: dict[str, list] = {}
    filas = []
    for arch in ARCHS:
        for seed in rec.args.seeds:
            a = rec.ae(arch, seed)
            forecast, dec = ro.oraculo(a["estado"], pos)
            res = evaluar(rec, f"oraculo-{arch}-s{seed}", stamps, forecast, dec, pos)
            registrar(rec, res, arch, "e10", "1.0", seed, tipo="diagnostico", rollout="oraculo",
                      ae_run_id=a["run"]["run_id"])
            fam.setdefault(arch, []).append(res["puntaje"])
            filas.append(_fila(res["tabla"], f"{arch}-s{seed}"))
    e = rec.estado("pca")
    forecast, dec = ro.oraculo(e, pos)
    res = evaluar(rec, "oraculo-pca", stamps, forecast, dec, pos)
    registrar(rec, res, "pca", "e10", "1.0", 0, tipo="diagnostico", rollout="oraculo")
    fam["pca"] = [res["puntaje"]]
    filas.append(_fila(res["tabla"], "pca"))
    lect = []
    tablas = {}
    for name, items in fam.items():
        t = rec.ev.tabla(evm.Puntaje.promedio(items))
        tablas[name] = t
        lect.append({"representacion": name, **_fila(t, name), "lectura": lectura_oraculo(t)})
    secciones = [("Lectura por familia (media de semillas; regla D9: techo si el intervalo de la razón vs mejor referencia alcanza 0.95)",
                  reg.md_table(pd.DataFrame(lect))),
                 ("Por variante (media de 6 series)", reg.md_table(pd.DataFrame(filas)))]
    for name, t in tablas.items():
        secciones.append((f"Por serie — {name}", reg.md_table(_cols(t))))
    reg.resumen_md("1.0", secciones, rec.out, {"lectura": lect})
    print(reg.md_table(pd.DataFrame(lect)))


def _fila(t: pd.DataFrame, nombre: str) -> dict:
    m = t[t["serie"] == "media"].iloc[0]
    return {"variante": nombre, "wrmse": m["wrmse"], "wrmse_lo": m["wrmse_lo"], "wrmse_hi": m["wrmse_hi"],
            "razon_mejor": m["razon_mejor"], "lo": m["razon_mejor_lo"], "hi": m["razon_mejor_hi"],
            "razon_lgbm": m["razon_lgbm"], "razon_pers": m["razon_pers"]}


def _cols(t: pd.DataFrame) -> pd.DataFrame:
    return t[["serie", "n", "mejor_ref", "wrmse", "wrmse_lo", "wrmse_hi", "razon_mejor", "razon_mejor_lo",
              "razon_mejor_hi", "razon_lgbm", "razon_pers", "skill_pers", "skill_pers_lo", "skill_pers_hi"]]



# --- Configuración vigente -> operador -> pronóstico -------------------------------

FAMILIAS_AE = ARCHS


def rank_defecto(rep: str):
    """Hankel: Gavish–Donoho desde 1.1; PCA y AE: rango completo (continuidad con ae-dmd-prep)."""
    return "gd" if rep.startswith("hankel") else None


def miembros(rec: Recursos, rep: str) -> list[int]:
    return list(rec.args.seeds) if rep in FAMILIAS_AE else [0]


def pesos_por_fila(rec: Recursos, est: est_mod.Estado, mask: np.ndarray | None = None) -> np.ndarray:
    """Peso del par `(t, t+1)` = `max_s w_s(target_s(t))` solo si `t` está en `row_mask(train)`; si no, 1 (D5).

    Target faltante -> peso 1 en esa serie.
    """
    from evaluate.evaluate import sample_weights

    mask = est.fit_mask() if mask is None else mask
    stamps = rec.grid.index[est.positions[mask]]
    feats = rec.features
    w = np.ones((len(stamps), len(evm.SERIES)))
    for c, (station, pollutant) in enumerate(evm.SERIES):
        f = feats[(feats["estacion"] == station) & (feats["pollutant"] == pollutant)]
        tgt = pd.Series(f["target"].to_numpy(float), index=pd.DatetimeIndex(f["timestamp"])).reindex(stamps).to_numpy()
        ws = sample_weights(np.nan_to_num(tgt, nan=-np.inf), pollutant)
        w[:, c] = np.where(np.isfinite(tgt), ws, 1.0)
    wmax = w.max(axis=1)
    en_train = pt.row_mask(stamps, rec.part, "train")
    return np.where(en_train, wmax, 1.0)


def pesos_segmentos(rec: Recursos, est: est_mod.Estado) -> list[np.ndarray]:
    w = pesos_por_fila(rec, est)
    lens = [len(seg) for seg in est.segment_positions()]
    return list(np.split(w, np.cumsum(lens)[:-1]))


def mascara_tramos(stamps, part: pt.Partition, tramos: tuple[str, ...]) -> np.ndarray:
    """`inicio(primer tramo) <= t < fin(último tramo) − 24 h` (como `row_mask`, sobre tramos contiguos)."""
    ts = pd.DatetimeIndex(pd.to_datetime(stamps))
    if ts.tz is not None:
        ts = ts.tz_convert(pt.TZ).tz_localize(None)
    start = pt.tramo_bounds(part, tramos[0])[0].tz_localize(None)
    stop = pt.tramo_bounds(part, tramos[-1])[1].tz_localize(None) - pd.Timedelta(hours=pt.EMBARGO_HOURS)
    return np.asarray((ts >= start) & (ts < stop), dtype=bool)


def filas_duan(rec: Recursos, est: est_mod.Estado) -> np.ndarray:
    """Posiciones de ajuste con `t+24` dentro de los tramos de ajuste: fuente de los factores de Duan."""
    pos = est.positions[est.fit_mask()]
    tramos = tuple(est.extra.get("fit_tramos") or est.fit_split)
    if tramos == ("fit",):
        tramos = ("train",)
    return pos[mascara_tramos(rec.grid.index[pos], rec.part, tramos)]


def aux_fn_de(est: est_mod.Estado):
    """Bloque auxiliar del paso `k` del rollout exógeno (ventana persistida; Hankel: entrada DMDc)."""
    ng = est.extra["ng"]
    if est.nombre.startswith("hankel"):
        return lambda pos, k: ro.entrada_hankel(ng, pos, k)
    if "proj_aux" in est.extra:
        proj = est.extra["proj_aux"]
        return lambda pos, k: proj(ro.ventana_aux_persistida(ng, pos, k).reshape(len(pos), -1).astype(float))
    enc, sl, d_mp = est.extra["encoder"], est.extra["aux_cols"], est.d_mp

    def fn(pos, k):
        xa = ro.ventana_aux_persistida(ng, pos, k)
        xm = np.zeros((len(pos), est_mod.HORIZONTE, est_mod.N_MP), dtype=np.float32)
        return est_mod._predict(enc, [xm, xa[..., sl]])[:, d_mp:]
    return fn


def aux0_de(est: est_mod.Estado, pos: np.ndarray):
    return ro.entrada_hankel(est.extra["ng"], pos, 0) if est.nombre.startswith("hankel") else None


def ajustar(rec: Recursos, est: est_mod.Estado, rep: str, cfg: dict) -> estim.Operador:
    """Operador según la configuración vigente (`cfg`: pesos, duan, rollout, estimador, horizonte, rango, modos)."""
    ranks = cfg.get("rank", {})
    rank = ranks.get(rep_key(rep, est), ranks.get(rep, rank_defecto(rep)))
    segs = est.segments()
    w = pesos_segmentos(rec, est) if cfg.get("pesos") else None
    if cfg.get("rollout") == "exogeno":
        if est.nombre.startswith("hankel"):
            xs = [seg[:, :est.d_mp] for seg in segs]
            us = [ro.entrada_hankel(est.extra["ng"], p, 0) for p in est.segment_positions()]
            op = estim.dmdc(xs, us, w, rank or "gd")
        else:
            op = estim.dmdc([seg[:, :est.d_mp] for seg in segs], [seg[:, est.d_mp:] for seg in segs], w, rank)
    elif cfg.get("horizonte") == "directo24":
        op = estim.directo24(segs, rank, w)
    elif cfg.get("estimador") == "fbdmd":
        op = estim.fbdmd(segs, w, rank)
    elif rep == "hankelest":
        op = estim.por_bloques(segs, est_mod.bloques_por_estacion(24), rank or "gd", weights=w)
    elif w is not None:
        op = estim.ponderado(segs, w, rank)
    else:
        op = estim.exacto(segs, rank)
    if cfg.get("estimador") == "recorte":
        op = estim.recorte(op)
    if cfg.get("modos", {}).get(rep_key(rep, est)) is not None:
        op = modos_mod_aplicar(op, cfg["modos"][rep_key(rep, est)])
    if cfg.get("duan"):
        op.meta["duan_S"] = factores_duan(rec, est, op, cfg)
    return op


def rep_key(rep: str, est: est_mod.Estado) -> str:
    return est.nombre if rep in FAMILIAS_AE else rep


def modos_mod_aplicar(op: estim.Operador, conservar) -> estim.Operador:
    return modos.operador_con_modos(op, conservar)


def factores_duan(rec: Recursos, est: est_mod.Estado, op: estim.Operador, cfg: dict) -> np.ndarray:
    pos = filas_duan(rec, est)
    log_pred = ventanas_log(rec, est, op, cfg, pos)
    log_real, real = ro.ventanas_reales_log(rec.grid, pos)
    return ro.duan_factores(log_real, log_pred, real)


def estado_final(est: est_mod.Estado, op: estim.Operador, cfg: dict, pos: np.ndarray) -> np.ndarray:
    if cfg.get("rollout") == "exogeno":
        return ro.exogeno_estado_final(op, est, pos, aux_fn_de(est), aux0_de(est, pos))
    return ro.estado_final(op, est.at(pos))


def ventanas_log(rec: Recursos, est: est_mod.Estado, op: estim.Operador, cfg: dict, pos: np.ndarray,
                 lote: int = 20000) -> np.ndarray:
    """Ventana pronosticada `[N, 24, 6]` en log1p (rollout de la configuración, sin Duan)."""
    pos = np.asarray(pos, int)
    return np.concatenate([est.decode_log(estado_final(est, op, cfg, pos[i:i + lote])) for i in range(0, len(pos), lote)])


def rank_gd(est: est_mod.Estado) -> int:
    z0, _, _ = estim.pares(est.segments())
    return estim.rango_gd(z0)


def pronostico(rec: Recursos, est: est_mod.Estado, op: estim.Operador, cfg: dict, pos: np.ndarray):
    """Pronóstico `[N, 6]` y ventana `[N, 24, 6]` (µg/m³) con la configuración vigente."""
    if "duan_S" in op.meta:
        lp = ventanas_log(rec, est, op, cfg, pos)
        dec = np.clip(np.exp(lp) * op.meta["duan_S"][None] - 1.0, 0.0, None)
    else:
        dec = est.decode(estado_final(est, op, cfg, pos))
    return dec.mean(axis=1), dec


def correr_variante(rec: Recursos, rep: str, seed: int, cfg: dict, nombre: str, mask_fn=None):
    """Ajusta y evalúa una variante en `sel`. Devuelve el resultado de `evaluar` (+ estado)."""
    est = rec.estado(rep, seed)
    op = ajustar(rec, est, rep, cfg)
    pos = rec.eval_positions()
    mask = None
    if mask_fn is not None:
        pos, mask = mask_fn(rec, est, pos)
    forecast, dec = pronostico(rec, est, op, cfg, pos)
    res = evaluar(rec, nombre, rec.grid.index[pos], forecast, dec, pos, op, mask)
    res["estado"] = est
    res["mask"] = mask
    return res


def mascara_elegible(rec: Recursos, est: est_mod.Estado, pos: np.ndarray):
    """Solo posiciones con estado elegible propio (h = 48): filas de eval_rows en esos timestamps."""
    keep = est.lookup(pos) >= 0
    pos = pos[keep]
    stamps = rec.grid.index[pos]
    rows_ts = pd.DatetimeIndex(pd.to_datetime(rec.rows["timestamp"]))
    mask = np.asarray(rows_ts.isin(stamps))
    return pos, mask


def _resumen_familia(rec: Recursos, items: list[evm.Puntaje]) -> evm.Puntaje:
    return evm.Puntaje.promedio(items) if len(items) > 1 else items[0]


def seleccionar(punt: dict[str, evm.Puntaje], orden: list[str], maximo: int = 2) -> dict:
    """D9: ganador = la más simple que ninguna otra supera significativamente; pasa la segunda
    (mejor puntaje del resto) solo si no es significativamente peor que el ganador."""
    cands = [c for c in orden if c in punt]
    batida = {c: [o for o in cands if o != c and evm.significativa(evm.diferencia(punt[o], punt[c]))] for c in cands}
    ganador = next(c for c in cands if not batida[c])
    resto = sorted((c for c in cands if c != ganador), key=lambda c: punt[c].media)
    pasan = [ganador]
    segunda = None
    if resto and maximo > 1:
        segunda = resto[0]
        d = evm.diferencia(punt[segunda], punt[ganador])
        if d["lo"] <= 0:
            pasan.append(segunda)
    return {"ganador": ganador, "pasan": pasan, "segunda": segunda, "batida_por": batida}


def tabla_diferencias(punt: dict[str, evm.Puntaje], base: str) -> pd.DataFrame:
    rows = []
    for name, p in punt.items():
        d = evm.diferencia(p, punt[base])
        rows.append({"variante": name, "wrmse_media": p.media, "diff_vs_" + base: d["diff"], "lo": d["lo"], "hi": d["hi"],
                     "significativa": evm.significativa(d)})
    return pd.DataFrame(rows)


def etapa_11(rec: Recursos) -> None:
    cfg = dict(rec.args.config)
    reps = [r for r in ("hankel", "hankelmet", "pca") + FAMILIAS_AE if r in cfg.get("representaciones", SIMPLICIDAD)]
    exp_code = {"hankel": "e11a", "hankelmet": "e11b", "pca": "e11c", "conv1d": "e11d", "gru": "e11e",
                "hankelest": "e11f", "hankel48": "e11g"}
    punt: dict[str, evm.Puntaje] = {}
    seeds_p: dict[str, list[evm.Puntaje]] = {}
    filas, gd = [], []
    for rep in reps + ["hankelest"]:
        for seed in miembros(rec, rep):
            name = f"{rep}-s{seed}" if rep in FAMILIAS_AE else rep
            res = correr_variante(rec, rep, seed, cfg, name)
            ae_id = rec.ae(rep, seed)["run"]["run_id"] if rep in FAMILIAS_AE else None
            r_gd = rank_gd(res["estado"])
            registrar(rec, res, rep, exp_code[rep], "1.1", seed, ae_run_id=ae_id, extra_params={"rank_gd": r_gd})
            seeds_p.setdefault(rep, []).append(res["puntaje"])
            filas.append({**_fila(res["tabla"], name), "rank": res["op"].rank, "rank_gd": r_gd})
            gd.append({"variante": name, "d": res["estado"].d, "rank": res["op"].rank, "rank_gd": r_gd})
        punt[rep] = _resumen_familia(rec, seeds_p[rep])
    # 1.1g: h = 48 sobre su intersección con eval_rows (comparación emparejada con hankel en esas filas).
    res48 = correr_variante(rec, "hankel48", 0, cfg, "hankel48", mascara_elegible)
    registrar(rec, res48, "hankel48", "e11g", "1.1", 0, extra_params={"rank_gd": rank_gd(res48["estado"]),
                                                                     "n_interseccion": int(res48["mask"].sum())})
    m48 = res48["mask"]
    hk = rec.estado("hankel")
    op_hk = ajustar(rec, hk, "hankel", cfg)
    pos_all = rec.eval_positions()
    f_hk, _ = ro.autonomo(op_hk, hk, pos_all)
    p_hk48 = rec.ev.puntaje(rec.ev.alinear("hankel", rec.grid.index[pos_all], f_hk), m48)
    d48 = evm.diferencia(res48["puntaje"], p_hk48)
    filas.append({**_fila(res48["tabla"], "hankel48 (intersección)"), "rank": res48["op"].rank, "rank_gd": rank_gd(res48["estado"])})

    # Regla GRU (D9) y estabilidad entre semillas.
    estab = []
    for rep in FAMILIAS_AE:
        if rep in seeds_p:
            medias = [p.media for p in seeds_p[rep]]
            lo, hi = evm.intervalo(punt[rep].media_boot)
            estab.append({"familia": rep, "rango_semillas": max(medias) - min(medias), "semiancho_ic": (hi - lo) / 2,
                          "descartar": (max(medias) - min(medias)) > (hi - lo) / 2})
    estab = pd.DataFrame(estab)
    candidatos = {k: v for k, v in punt.items() if k != "hankelest"}
    if not estab.empty and bool(estab.loc[estab["familia"] == "gru", "descartar"].any()):
        candidatos.pop("gru", None)
    sel = seleccionar(candidatos, [r for r in SIMPLICIDAD if r in candidatos])
    diffs = tabla_diferencias(punt, "hankel")
    pares_tab = pd.DataFrame([{"a": a, "b": b, **evm.diferencia(punt[a], punt[b]),
                               "a_mejor_sig": evm.significativa(evm.diferencia(punt[a], punt[b]))}
                              for a in candidatos for b in candidatos if a != b])
    d_met = evm.diferencia(punt["hankelmet"], punt["hankel"]) if "hankelmet" in punt else None
    d_est = evm.diferencia(punt["hankelest"], punt["hankel"])
    abl = pd.DataFrame([
        {"comparacion": "1.1b hankelmet − hankel (meteo)", **(d_met or {})},
        {"comparacion": "1.1f hankelest − hankel (por estación)", **d_est},
        {"comparacion": f"1.1g hankel48 − hankel (intersección, n mín {int(res48['puntaje'].n.min())})", **d48},
    ])
    fam_tab = pd.DataFrame([{"variante": k, **{c: v for c, v in _fila(rec.ev.tabla(p), k).items() if c != "variante"}}
                            for k, p in punt.items()])
    propuesta = {"ganador": sel["ganador"], "pasan": sel["pasan"], "segunda": sel["segunda"],
                 "gru_descartado": "gru" not in candidatos, "d48": d48, "d_est": d_est, "d_met": d_met}
    reg.resumen_md("1.1", [
        ("Propuesta (D9)", f"Pasan: {', '.join(sel['pasan'])} (ganador {sel['ganador']}; segunda evaluada: {sel['segunda']})."),
        ("Puntaje por representación (AE = media de semillas)", reg.md_table(fam_tab)),
        ("Diferencias emparejadas vs hankel (media de 6 series)", reg.md_table(diffs)),
        ("Ablaciones", reg.md_table(abl)),
        ("Diferencias por pares entre candidatos (a − b)", reg.md_table(pares_tab)),
        ("Estabilidad entre semillas (regla GRU, D9)", reg.md_table(estab)),
        ("Por variante", reg.md_table(pd.DataFrame(filas))),
        ("Rango", reg.md_table(pd.DataFrame(gd))),
        *[(f"Por serie — {k}", reg.md_table(_cols(rec.ev.tabla(p)))) for k, p in punt.items()],
    ], rec.out, {"propuesta": propuesta, "estabilidad": estab.to_dict("records"),
                 "diferencias": diffs.to_dict("records")})
    print(reg.md_table(fam_tab))
    print(reg.md_table(diffs))
    print(reg.md_table(abl))
    print(reg.md_table(estab))
    print(reg.md_table(pares_tab))
    print(propuesta)



def _diag_resumen(res: dict) -> dict:
    ep = res["frames"]["episodes.parquet"]
    tr = res["frames"]["tramos.parquet"]
    g = tr[tr["tramo"] == "gt100"]
    return {"episodios_wrmse": float(ep["wrmse"].mean()) if "wrmse" in ep else float("nan"),
            "episodios_sesgo": float(ep["sesgo"].mean()) if "sesgo" in ep else float("nan"),
            "sesgo_gt100": float(g["sesgo"].mean()) if "sesgo" in g else float("nan"),
            "pendiente_todo": float(tr.loc[tr["tramo"] == "todo", "pendiente"].mean())}


def params_cfg(cfg: dict) -> dict:
    return {"pesos": bool(cfg.get("pesos")), "duan": bool(cfg.get("duan")),
            "rollout": cfg.get("rollout", "autonomo"), "horizonte": cfg.get("horizonte", "horario"),
            "estimador": cfg.get("estimador", "exacto")}


def etapa_variantes(rec: Recursos, etapa: str, variantes: dict[str, tuple[dict, str]], orden: list[str],
                    titulo: str, tipo_por_variante: dict[str, str] | None = None, candidatas: list[str] | None = None,
                    extra_frames_fn=None) -> dict:
    """Evalúa variantes (`nombre -> (delta de config, exp)`) sobre las representaciones vigentes.

    Selección D9 por representación (orden de simplicidad `orden`); devuelve la propuesta.
    """
    cfg0 = dict(rec.args.config)
    reps = cfg0["representaciones"]
    resultados: dict[str, dict] = {}
    filas = []
    run_ids: dict = {}
    ops: dict = {}
    for rep in reps:
        punt: dict[str, evm.Puntaje] = {}
        diag = {}
        for var, (delta, exp) in variantes.items():
            cfg = {**cfg0, **delta}
            items, ds = [], []
            for seed in miembros(rec, rep):
                res = correr_variante(rec, rep, seed, cfg, f"{rep}-{var}-s{seed}")
                ae_id = rec.ae(rep, seed)["run"]["run_id"] if rep in FAMILIAS_AE else None
                pc = params_cfg(cfg)
                extra = {"pesos": pc["pesos"], "duan": pc["duan"], "horizonte": pc["horizonte"], "variante": var}
                frames = extra_frames_fn(rec, res, cfg) if extra_frames_fn else None
                run_id = registrar(rec, res, rep, exp, etapa, seed, tipo=(tipo_por_variante or {}).get(var, "operador"),
                                   estimador=res["op"].estimador, rollout=pc["rollout"], extra_params=extra,
                                   extra_frames=frames, ae_run_id=ae_id)
                run_ids[(rep, var, seed)] = run_id
                ops[(rep, var, seed)] = res["op"]
                items.append(res["puntaje"])
                ds.append(_diag_resumen(res))
                filas.append({"rep": rep, **_fila(res["tabla"], f"{var}-s{seed}"), **ds[-1]})
            punt[var] = _resumen_familia(rec, items)
            diag[var] = {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}
        cands = {k: v for k, v in punt.items() if candidatas is None or k in candidatas}
        sel = seleccionar(cands, [o for o in orden if o in cands], maximo=1)
        base = orden[0]
        pares_tab = pd.DataFrame([{"a": a, "b": b, **evm.diferencia(punt[a], punt[b])} for i, a in enumerate(punt)
                                  for b in list(punt)[i + 1:]])
        resultados[rep] = {"punt": punt, "diag": diag, "sel": sel, "pares": pares_tab,
                           "diffs": tabla_diferencias(punt, base), "fam": pd.DataFrame(
                               [{"variante": k, **{c: v for c, v in _fila(rec.ev.tabla(p), k).items() if c != "variante"},
                                 **diag[k]} for k, p in punt.items()])}
    ganadores = {rep: r["sel"]["ganador"] for rep, r in resultados.items()}
    resultados["_runs"] = run_ids
    resultados["_ops"] = ops
    secciones = [("Propuesta (D9, por representación)", ", ".join(f"{k}: {v}" for k, v in ganadores.items()))]
    for rep, r in ((k, v) for k, v in resultados.items() if not k.startswith("_")):
        secciones += [(f"{rep} — media de 6 series (AE = media de semillas) y diagnósticos", reg.md_table(r["fam"])),
                      (f"{rep} — diferencias emparejadas vs {orden[0]}", reg.md_table(r["diffs"])),
                      (f"{rep} — diferencias por pares (a − b)", reg.md_table(r["pares"]))]
    secciones.append(("Por variante y semilla", reg.md_table(pd.DataFrame(filas))))
    for rep, r in ((k, v) for k, v in resultados.items() if not k.startswith("_")):
        for k, p in r["punt"].items():
            secciones.append((f"Por serie — {rep} {k}", reg.md_table(_cols(rec.ev.tabla(p)))))
    reg.resumen_md(etapa, secciones, rec.out, {
        "ganadores": ganadores,
        "diferencias": {rep: r["diffs"].to_dict("records") for rep, r in resultados.items() if not rep.startswith("_")},
        "pares": {rep: r["pares"].to_dict("records") for rep, r in resultados.items() if not rep.startswith("_")},
        "familias": {rep: r["fam"].to_dict("records") for rep, r in resultados.items() if not rep.startswith("_")}},
        )
    for rep, r in ((k, v) for k, v in resultados.items() if not k.startswith("_")):
        print(rep)
        print(reg.md_table(r["fam"]))
        print(reg.md_table(r["diffs"]))
        print(reg.md_table(r["pares"]))
    print(ganadores)
    return resultados


def etapa_12(rec: Recursos) -> None:
    variantes = {"base": ({"pesos": False, "duan": False}, "e12base"), "duan": ({"pesos": False, "duan": True}, "e12b"),
                 "pesos": ({"pesos": True, "duan": False}, "e12a"), "pesos+duan": ({"pesos": True, "duan": True}, "e12ab")}
    etapa_variantes(rec, "1.2", variantes, ["base", "duan", "pesos", "pesos+duan"], "Picos")


def frames_kma(rec: Recursos, res: dict, cfg: dict) -> dict:
    op = res["op"]
    if "K_ma" not in op.meta:
        return {}
    kma = pd.DataFrame(op.meta["K_ma"], columns=[f"aux_{j}" for j in range(op.meta["K_ma"].shape[1])])
    kma.insert(0, "fila_mp", np.arange(len(kma)))
    return {"k_ma.parquet": kma}


def etapa_13(rec: Recursos) -> None:
    variantes = {"autonoma": ({"rollout": "autonomo"}, "e13a"), "exogena": ({"rollout": "exogeno"}, "e13b")}
    etapa_variantes(rec, "1.3", variantes, ["autonoma", "exogena"], "Exógenas", extra_frames_fn=frames_kma)


def etapa_14(rec: Recursos) -> None:
    """K horario (principal) vs A₂₄ directo (solo referencia de precisión, `tipo=referencia`)."""
    variantes = {"horario": ({"horizonte": "horario"}, "e14a"), "directo24": ({"horizonte": "directo24"}, "e14b")}
    etapa_variantes(rec, "1.4", variantes, ["horario", "directo24"], "Horizonte",
                    tipo_por_variante={"directo24": "referencia"}, candidatas=["horario"])


def fit_eig_fn(rec: Recursos, est: est_mod.Estado, rep: str, cfg: dict):
    """Autovalores del operador re-estimado con pesos por fila (conteos del bootstrap de días)."""
    segs = est.segments()
    lens = np.cumsum([len(sg) for sg in segs])[:-1]
    rank = cfg.get("rank", {}).get(rep, rank_defecto(rep))

    def fn(w_rows):
        w = list(np.split(np.asarray(w_rows, float), lens))
        if cfg.get("estimador") == "fbdmd":
            op = estim.fbdmd(segs, w, rank)
        else:
            op = estim.ponderado(segs, w, rank)
            if cfg.get("estimador") == "recorte":
                op = estim.recorte(op)
        return op.eigvals()
    return fn


def dias_de_filas(rec: Recursos, est: est_mod.Estado) -> np.ndarray:
    stamps = rec.grid.index[est.positions[est.fit_mask()]]
    local = stamps.tz_convert(pt.TZ).tz_localize(None) if stamps.tz is not None else stamps
    return local.normalize().to_numpy()


def estabilidad_semillas(ops: dict[int, estim.Operador]) -> pd.DataFrame:
    """Modos de la semilla de referencia (la primera) emparejados (húngaro) con los de las demás."""
    seeds = sorted(ops)
    ref = ops[seeds[0]].eigvals()
    ref = ref[modos.representantes(ref)]
    ref = ref[np.argsort(-np.abs(ref), kind="stable")]
    out = modos.espectro(ref)[["modo", "abs", "periodo_h", "vida_media_h", "tipo"]].copy()
    for s in seeds[1:]:
        lam = ops[s].eigvals()
        lam = lam[modos.representantes(lam)]
        i, j = modos.emparejar(ref, lam)
        per = np.full(len(ref), np.nan)
        hl = np.full(len(ref), np.nan)
        ok = np.zeros(len(ref), bool)
        per[i], hl[i], ok[i] = modos.periodo_h(lam[j]), modos.vida_media_h(lam[j]), modos.aparece(ref[i], lam[j])
        out[f"periodo_s{s}"], out[f"vida_media_s{s}"], out[f"aparece_s{s}"] = per, hl, ok
    return out


def etapa_15(rec: Recursos) -> None:
    variantes = {"exacto": ({"estimador": "exacto"}, "e15a"), "recorte": ({"estimador": "recorte"}, "e15b"),
                 "fbdmd": ({"estimador": "fbdmd"}, "e15c")}
    res = etapa_variantes(rec, "1.5", variantes, ["exacto", "recorte", "fbdmd"], "Estimador")
    cfg0 = dict(rec.args.config)
    secciones = []
    for rep in cfg0["representaciones"]:
        ganador = res[rep]["sel"]["ganador"]
        cfg = {**cfg0, "estimador": ganador}
        ops = {}
        for seed in miembros(rec, rep):
            est = rec.estado(rep, seed)
            op = res["_ops"][(rep, ganador, seed)]
            ops[seed] = op
            boot = modos.bootstrap_modos(op.eigvals(), fit_eig_fn(rec, est, rep, cfg), dias_de_filas(rec, est),
                                         B=rec.args.bootstrap_modos, dias=evm.DIAS_DEFAULT, seed=0)
            run_id = res["_runs"][(rep, ganador, seed)]
            boot.to_parquet(reg.artifacts_dir(rec.state, run_id) / "bootstrap_modos.parquet", index=False)
            secciones.append((f"Bootstrap por modo — {rep} s{seed} ({ganador}, B = {rec.args.bootstrap_modos})",
                              reg.md_table(boot)))
        if len(ops) > 1:
            secciones.append((f"Estabilidad entre semillas — {rep} ({ganador})", reg.md_table(estabilidad_semillas(ops))))
    path = rec.out / "1.5.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n" + "\n".join(f"## {t}\n\n{x}\n" for t, x in secciones),
                    encoding="utf-8")
    for t, x in secciones:
        print(t)
        print(x)


def puntaje_op(rec: Recursos, est: est_mod.Estado, op: estim.Operador, cfg: dict, pos: np.ndarray, nombre: str):
    forecast, dec = pronostico(rec, est, op, cfg, pos)
    yhat = rec.ev.alinear(nombre, rec.grid.index[pos], forecast)
    return rec.ev.puntaje(yhat), forecast, dec


ARMONICOS_H = (12.0, 8.0)
APARICION_MIN = 0.8


def protegidos_de(rec: Recursos, est: est_mod.Estado, rep: str, cfg: dict, op: estim.Operador, reps_l) -> tuple[list, pd.DataFrame]:
    """Modo diario (≈ 24 h) siempre; armónicos (≈ 12 h, ≈ 8 h) solo si aparecen en ≥ 80 % del bootstrap."""
    prot = []
    diario = modos.modo_diario(reps_l)
    if diario is not None:
        prot.append(diario)
    boot = modos.bootstrap_modos(op.eigvals(), fit_eig_fn(rec, est, rep, cfg), dias_de_filas(rec, est),
                                 B=rec.args.bootstrap_modos, dias=evm.DIAS_DEFAULT, seed=0)
    for h in ARMONICOS_H:
        arm = modos.modo_diario(reps_l, h)
        if arm is None:
            continue
        j = int(np.argmin(np.abs(boot["lam_re"].to_numpy() + 1j * boot["lam_im"].to_numpy() - arm)))
        if float(boot["aparicion"].iloc[j]) >= APARICION_MIN:
            prot.append(arm)
    return prot, boot


def etapa_16(rec: Recursos) -> None:
    """Rango y modos (enmienda del operador, pausa 8.3): PCA/AE desde rango completo (GD solo se
    reporta), eliminación contra la configuración inicial, modo diario protegido. `d = 16` fijo."""
    cfg0 = dict(rec.args.config)
    pos = rec.eval_positions()
    secciones, resumen = [], []
    fam: dict[str, list] = {}
    fam_full: dict[str, list] = {}
    modos_sel = {}
    for rep in cfg0["representaciones"]:
        for seed in miembros(rec, rep):
            est = rec.estado(rep, seed)
            key = rep_key(rep, est)
            r_gd = rank_gd(est)
            cfg_nd = {**cfg0, "duan": False}
            op0 = ajustar(rec, est, rep, cfg_nd)  # rango por defecto: completo en PCA/AE, GD en Hankel
            lam = op0.eigvals()
            reps_l = list(lam[modos.representantes(lam)])
            prot, boot = protegidos_de(rec, est, rep, cfg0, op0, reps_l)

            def evaluar_fn(conservar, est=est, op0=op0, key=key):
                op_m = modos.operador_con_modos(op0, conservar)
                if cfg0.get("duan"):
                    op_m.meta["duan_S"] = factores_duan(rec, est, op_m, cfg0)
                return puntaje_op(rec, est, op_m, cfg0, pos, f"{key}-modos")[0]

            kept, curva = modos.eliminacion_atras(reps_l, evaluar_fn, protegidos=prot)
            p_full = evaluar_fn(reps_l)
            op_fin = modos.operador_con_modos(op0, kept)
            if cfg0.get("duan"):
                op_fin.meta["duan_S"] = factores_duan(rec, est, op_fin, cfg0)
            forecast, dec = pronostico(rec, est, op_fin, cfg0, pos)
            res = evaluar(rec, f"{key}-e16", rec.grid.index[pos], forecast, dec, pos, op_fin)
            d_full = evm.diferencia(res["puntaje"], p_full)
            ae_id = rec.ae(rep, seed)["run"]["run_id"] if rep in FAMILIAS_AE else None
            registrar(rec, res, rep, "e16", "1.6", seed, estimador=op_fin.estimador, rollout=cfg0.get("rollout", "autonomo"),
                      extra_params={"rank_gd": r_gd, "n_modos": len(kept), "duan": bool(cfg0.get("duan")),
                                    "regla_16": "vs-inicial+diario-protegido", "rank_inicial": op0.rank},
                      extra_frames={"curva_modos.parquet": curva, "bootstrap_modos.parquet": boot}, ae_run_id=ae_id)
            fam.setdefault(rep, []).append(res["puntaje"])
            fam_full.setdefault(rep, []).append(p_full)
            modos_sel[key] = modos.a_lista(kept)
            esp = modos.espectro(np.asarray(kept))
            esp["protegido"] = [any(abs(l - p) < 1e-8 for p in prot) for l in esp["lam_re"] + 1j * esp["lam_im"]]
            resumen.append({"variante": key, "rank_inicial": op0.rank, "rank_gd": r_gd, "modos_inicial": len(reps_l),
                            "protegidos": len(prot), "modos_final": len(kept), "wrmse_inicial": p_full.media,
                            "wrmse_final": res["puntaje"].media, "dif_final_menos_inicial": d_full["diff"],
                            "lo": d_full["lo"], "hi": d_full["hi"]})
            secciones += [(f"{key} — curva de eliminación", reg.md_table(curva)),
                          (f"{key} — modos conservados", reg.md_table(esp))]
    punt = {rep: _resumen_familia(rec, items) for rep, items in fam.items()}
    punt_full = {rep: _resumen_familia(rec, items) for rep, items in fam_full.items()}
    fam_tab = pd.DataFrame([{"variante": k, **{c: v for c, v in _fila(rec.ev.tabla(p), k).items() if c != "variante"},
                             **{f"dif_vs_inicial{s_}": v for s_, v in zip(("", "_lo", "_hi"),
                                                                        evm.diferencia(p, punt_full[k]).values())}}
                            for k, p in punt.items()])
    pares_tab = pd.DataFrame([{"a": a, "b": b, **evm.diferencia(punt[a], punt[b])} for a in punt for b in punt if a < b])
    reg.resumen_md("1.6", [
        ("Regla", "Enmienda del operador (pausa 8.3): rango completo en PCA/AE, comparación contra la "
                  "configuración inicial, modo diario protegido (armónicos si aparición ≥ 0.8)."),
        ("Por representación (final; AE = media de semillas)", reg.md_table(fam_tab)),
        ("Diferencias entre representaciones finales", reg.md_table(pares_tab)),
        ("Por variante", reg.md_table(pd.DataFrame(resumen))),
        *secciones,
        *[(f"Por serie — {k}", reg.md_table(_cols(rec.ev.tabla(p)))) for k, p in punt.items()],
    ], rec.out, {"modos": modos_sel, "resumen": resumen})
    print(reg.md_table(fam_tab))
    print(reg.md_table(pares_tab))
    print(reg.md_table(pd.DataFrame(resumen)))
    for t, x in secciones:
        if "conservados" in t:
            print(t)
            print(x)




TRAMOS_CIERRE = ("train", "es", "sel")
CIERRE_JSON = "cierre.json"


def estado_cierre(rec: Recursos, rep: str, seed: int) -> est_mod.Estado:
    """Estado del reajuste hasta 2026-06-30 (D11): PCA reajusta normalización y PCA con train+es+sel;
    el AE conserva normalización y encoder (hasta 2026-02) y solo cambia el tramo de ajuste de K."""
    import dataclasses

    if rep in FAMILIAS_AE:
        e = rec.estado(rep, seed)
        return dataclasses.replace(e, fit_split=TRAMOS_CIERRE, extra={**e.extra, "fit_tramos": TRAMOS_CIERRE})
    if not hasattr(rec, "_ctx_cierre"):
        rec._ctx_cierre = est_mod.contexto(rec.grid, rec.part, TRAMOS_CIERRE)
    return rec.estado(rep, seed, ctx=rec._ctx_cierre, union_fit=True)


def interpretabilidad_semillas(kept: dict[int, np.ndarray], boots: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """§9: por modo de la semilla de referencia, presencia en las demás semillas (húngaro), aparición en el
    bootstrap (≥ 80 %) e intervalos de período/vida media que se solapan entre semillas."""
    seeds = sorted(kept)
    ref = np.asarray(kept[seeds[0]], complex)
    rows = []
    for m, lam in enumerate(ref):
        row = {"modo": m, "periodo_h": float(modos.periodo_h([lam])[0]), "vida_media_h": float(modos.vida_media_h([lam])[0])}
        presente, boot_ok, solapa = True, True, True
        b0 = boots[seeds[0]].iloc[int(np.argmin(np.abs(boots[seeds[0]]["lam_re"] + 1j * boots[seeds[0]]["lam_im"] - lam)))]
        boot_ok &= b0["aparicion"] >= APARICION_MIN
        for s in seeds[1:]:
            other = np.asarray(kept[s], complex)
            i, j = modos.emparejar(np.array([lam]), other)
            ok = bool(len(j) and modos.aparece(np.array([lam]), other[j])[0])
            presente &= ok
            if len(j):
                bs = boots[s]
                b1 = bs.iloc[int(np.argmin(np.abs(bs["lam_re"] + 1j * bs["lam_im"] - other[j[0]])))]
                boot_ok &= b1["aparicion"] >= APARICION_MIN
                for col in ("vida_media_h", "periodo_h"):
                    lo0, hi0, lo1, hi1 = b0[f"{col}_lo"], b0[f"{col}_hi"], b1[f"{col}_lo"], b1[f"{col}_hi"]
                    if np.isfinite([lo0, hi0, lo1, hi1]).all() and (col == "vida_media_h" or abs(lam.imag) > 1e-10):
                        solapa &= bool(lo0 <= hi1 and lo1 <= hi0)
            row[f"s{s}"] = ok
        row.update({"presente_semillas": presente, "bootstrap_80": bool(boot_ok), "intervalos_solapan": solapa})
        rows.append(row)
    return pd.DataFrame(rows)


def etapa_cierre(rec: Recursos) -> None:
    """Reajuste hasta 2026-06-30 con rango y modos de 1.6 fijados por emparejamiento húngaro; sin test."""
    cfg = dict(rec.args.config)
    reps = [cfg["oficial"]] + [r for r in cfg.get("secundarias", [])]
    out = {"oficial": cfg["oficial"], "runs": {}, "interpretabilidad": {}}
    secciones = []
    for rep in reps:
        kept_s, boots = {}, {}
        for seed in miembros(rec, rep):
            est = estado_cierre(rec, rep, seed)
            key = rep_key(rep, est)
            cfg_nd = {k: v for k, v in cfg.items() if k not in ("modos", "duan")}
            op_r = ajustar(rec, est, rep, cfg_nd)
            op, empar = modos.fijar_modos(op_r, cfg["modos"][key])
            if cfg.get("duan"):
                op.meta["duan_S"] = factores_duan(rec, est, op, cfg)
            lam_k = np.asarray([complex(*x) for x in op.meta["modos_conservados"]])
            boot = modos.bootstrap_modos(lam_k, fit_eig_fn(rec, est, rep, cfg_nd), dias_de_filas(rec, est),
                                         B=rec.args.bootstrap_modos, dias=evm.DIAS_DEFAULT, seed=0)
            name = reg.run_name(rep, "cierre", rec.part.token, rec.args.version, seed)
            ae_id = rec.ae(rep, seed)["run"]["run_id"] if rep in FAMILIAS_AE else None
            params = {"tipo": "operador", "etapa": "cierre", "estimador": op.estimador, "rollout": cfg.get("rollout"),
                      "rank": op.rank, "version": rec.args.version, "representacion": rep, "duan": bool(cfg.get("duan")),
                      "n_modos": len(lam_k), "tramos_ajuste": "+".join(TRAMOS_CIERRE),
                      "rol": "oficial" if rep == cfg["oficial"] else "secundario"}
            run_id = reg.log_variante(rec.state, name, reg.tags_base(rep, "cierre", PARTICION, seed, ae_id), params,
                                      {"n_modos": len(lam_k), "n_filas_ajuste": int(est.fit_mask().sum())},
                                      {"spectrum.parquet": modos.espectro(lam_k), "bootstrap_modos.parquet": boot,
                                       "emparejamiento.parquet": empar}, {"operator.npz": op.arrays()})
            out["runs"][key] = run_id
            kept_s[seed], boots[seed] = lam_k, boot
            secciones += [(f"{key} — emparejamiento sel → cierre", reg.md_table(empar)),
                          (f"{key} — bootstrap por modo (cierre, B = {rec.args.bootstrap_modos})", reg.md_table(boot))]
        it = interpretabilidad_semillas(kept_s, boots)
        ok = bool(len(it) <= 10 and it["bootstrap_80"].all() and (it["presente_semillas"].all() if len(kept_s) > 1 else True)
                  and it["intervalos_solapan"].all())
        out["interpretabilidad"][rep] = {"n_modos": int(len(it)), "cumple": ok, "tabla": it.to_dict("records")}
        secciones.insert(0, (f"Interpretabilidad (§9) — {rep}: {'cumple' if ok else 'no cumple'}", reg.md_table(it)))
    (rec.out / CIERRE_JSON).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    reg.resumen_md("cierre", secciones, rec.out)
    for t, x in secciones[:3]:
        print(t)
        print(x)
    print(json.dumps({k: v for k, v in out.items() if k != "interpretabilidad"}, indent=1))
    print({k: (v["n_modos"], v["cumple"]) for k, v in out["interpretabilidad"].items()})


# --- Test (evaluación única, D11/D12) -------------------------------------------


def cargar_operador(rec: Recursos, run_id: str) -> estim.Operador:
    npz = np.load(rec.artifact(run_id, "operator.npz"))
    meta = {k: npz[k] for k in npz.files if k in ("duan_S", "modos_conservados")}
    return estim.Operador(A=npz["A"], U=npz["U"] if "U" in npz.files else None, rank=int(npz["rank"]),
                          estimador="cierre", pasos=int(npz["pasos"]), meta=meta)


def referencias_test(rec: Recursos) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`lightgbm-v1` holdout (test) y persistencia calculada en el mismo proceso."""
    from baseline.persistence import predict_frame

    by_id = {r["run_id"]: r for r in rec.runs}
    lg = []
    for r in rec.v2["lgbm"]:
        src = by_id[r["params"]["source_run_id"]]
        lg.append(rec.preds(src, "test"))
    lgbm = pd.concat(lg, ignore_index=True)
    pers = predict_frame(rec.data["sinca"], rec.features)
    pers = pers[pt.row_mask(pers["timestamp"], rec.part, "test")]
    return lgbm, pers


def veredicto(t: pd.DataFrame) -> dict:
    """D12: razón vs mejor ≤ 1.05 en ≥ 4/6 (falla solo si todo el intervalo > 1.05); PLC II (ambas series)
    skill puntual > 0; piso: en las 6, el intervalo del skill vs persistencia no queda entero bajo 0."""
    s = t[t["serie"] != "media"]
    cumple_razon = s["razon_mejor_lo"] <= 1.05
    plc = s[s["serie"].str.startswith("Padre Las Casas II")]
    piso = s["skill_pers_hi"] >= 0
    precision = bool(cumple_razon.sum() >= 4 and (plc["skill_pers"] > 0).all() and piso.all())
    return {"series_razon_ok": int(cumple_razon.sum()), "plc_skill_pos": bool((plc["skill_pers"] > 0).all()),
            "piso_ok": bool(piso.all()), "precision": precision}


def etapa_test(rec: Recursos) -> None:
    from evaluate.evaluate import evaluate_predictions, zone_icap

    previos = [r for r in reg.runs_de_etapa(rec.state, "test", rec.args.version)]
    if previos and not rec.args.force_test:
        raise SystemExit(f"Ya hay {len(previos)} runs de test (versión {rec.args.version}): repetir exige --force-test.")
    path = rec.out / CIERRE_JSON
    if not path.exists():
        raise SystemExit(f"Falta {path}: correr antes --etapa cierre.")
    cierre = json.loads(path.read_text(encoding="utf-8"))
    cfg = dict(rec.args.config)
    reps = [cierre["oficial"]] + [r for r in cfg.get("secundarias", [])]
    preds: dict[str, pd.DataFrame] = {}
    for rep in reps:
        for seed in miembros(rec, rep):
            est = estado_cierre(rec, rep, seed)
            key = rep_key(rep, est)
            op = cargar_operador(rec, cierre["runs"][key])
            pos = est.positions[(est.split == "test")]
            pos = pos[pt.row_mask(rec.grid.index[pos], rec.part, "test")]
            forecast, _ = pronostico(rec, est, op, cfg, pos)
            stamps = rec.grid.index[pos]
            pieces = [pd.DataFrame({"timestamp": stamps, "estacion": st, "pollutant": po, "y_pred": forecast[:, c]})
                      for c, (st, po) in enumerate(evm.SERIES)]
            tg = rec.features[["timestamp", "estacion", "pollutant", "target"]].rename(columns={"target": "y_true"})
            preds[key] = pd.concat(pieces, ignore_index=True).merge(tg, on=evm.KEYS)
    lgbm, pers = referencias_test(rec)
    rows = evm.construir_eval_rows(preds, lgbm, pers)
    ev = evm.Evaluador(rows, B=rec.args.bootstrap, seed=0)
    stamps_rows = rows
    tablas, resultados = {}, {}
    for rep in reps:
        items = []
        for seed in miembros(rec, rep):
            key = f"{rep}-s{seed}" if rep in FAMILIAS_AE else rep
            f = rows[evm.KEYS].merge(preds[key][evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
            items.append(ev.puntaje(f))
            frame_all = ev.predicciones(f, split="test")
            for (st, po), g in frame_all.groupby(["estacion", "pollutant"]):
                name = reg.run_name_test(st, po, rep, rec.part.token, rec.args.version, seed)
                tags = {**reg.tags_base(rep, "test", PARTICION, seed), "evaluacion": "test", "serie": f"{st}|{po}"}
                params = {"estacion": st, "pollutant": po, "version": rec.args.version, "seed": seed, "tipo": "operador",
                          "rol": "oficial" if rep == cierre["oficial"] else "secundario", "cierre_run_id": cierre["runs"][key]}
                metrics = {f"test_{k}": v for k, v in evaluate_predictions(g).items()}
                reg.log_variante(rec.state, name, tags, params, metrics,
                                 {"predictions.parquet": g, "zone_icap.parquet": zone_icap(g),
                                  "rolling_scores.parquet": pd.DataFrame({"window_index": pd.Series(dtype=int),
                                                                          "rmse": pd.Series(dtype=float)})})
        p = evm.Puntaje.promedio(items) if len(items) > 1 else items[0]
        tablas[rep] = ev.tabla(p)
        resultados[rep] = veredicto(tablas[rep])
    interp = cierre["interpretabilidad"]
    ofi = cierre["oficial"]
    final = {"oficial": ofi, "precision": resultados[ofi], "interpretabilidad": interp[ofi]["cumple"],
             "veredicto": "cumple" if resultados[ofi]["precision"] and interp[ofi]["cumple"] else "no cumple",
             "secundarias": {r: resultados[r] for r in reps if r != ofi},
             "n_filas_min": int(stamps_rows.groupby(["estacion", "pollutant"]).size().min())}
    reg.resumen_md("test", [("Veredicto D12 (oficial)", json.dumps(final, ensure_ascii=False, default=str)),
                            *[(f"Test por serie — {r}", reg.md_table(_cols(t))) for r, t in tablas.items()]],
                   rec.out, final)
    for r, t in tablas.items():
        print(r)
        print(reg.md_table(_cols(t)))
    print(final)


ETAPA_FN = {"base": etapa_base, "1.0": etapa_10, "1.1": etapa_11, "1.2": etapa_12, "1.3": etapa_13,
            "1.4": etapa_14, "1.5": etapa_15, "1.6": etapa_16,
            "cierre": etapa_cierre, "test": etapa_test}


# --- CLI ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operador lineal global por etapas (dmd-operador-global).")
    p.add_argument("--etapa", required=True, choices=reg.ETAPAS)
    p.add_argument("--state-dir", required=True, help="Estado MLflow local (mlflow.sqlite + mlruns/).")
    p.add_argument("--local-validated-dir", required=True, help="Copia local de sinca-data/validated/.")
    p.add_argument("--no-download", action="store_true", help="No descarga (todas las etapas salvo base).")
    p.add_argument("--profile", default=None, help="Profile AWS de la cuenta personal (solo base con descarga).")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2", type=lambda s: [int(x) for x in s.split(",") if x != ""])
    p.add_argument("--output-dir", default=str(reg.OUTPUT_DIR))
    p.add_argument("--decisiones", default=str(reg.DECISIONES_PATH))
    p.add_argument("--bootstrap", type=int, default=evm.B_DEFAULT, help="Remuestreos B (D8: 2000).")
    p.add_argument("--bootstrap-modos", type=int, default=200, help="Re-estimaciones por modo (D8: 200).")
    p.add_argument("--force-test", action="store_true", help="Repite la evaluación en test (solo por orden del operador).")
    return p.parse_args(argv)


def run(args) -> int:
    t0 = time.perf_counter()
    if args.etapa == "base" and not args.no_download:
        if not args.profile:
            raise SystemExit("--etapa base sin --no-download requiere --profile (cuenta personal).")
        meta = descargar(args.state_dir, args.profile)
        log.info("Descarga: %s", meta)
    elif args.profile and args.etapa != "base":
        raise SystemExit("--profile solo se usa en la etapa base.")
    decisiones = reg.leer_decisiones(args.decisiones)
    args.config = reg.exigir_decision("cierre" if args.etapa == "test" else args.etapa, decisiones)
    if args.etapa not in ETAPA_FN:
        raise SystemExit(f"Etapa {args.etapa} aún no implementada.")
    rec = Recursos(args)
    ETAPA_FN[args.etapa](rec)
    log.info("Etapa %s en %.1f s", args.etapa, time.perf_counter() - t0)
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
