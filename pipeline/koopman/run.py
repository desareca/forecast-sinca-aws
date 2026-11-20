"""Entry point del Koopman entrenado (`deep-koopman`): un Training Job por (encoder, semilla, fase).

    python koopman/run.py --fase f1 --encoder gru --seed 0 --variantes principal --campana <id> --clave <clave>

Fases: `smoke`, `f1`, `b1`–`b4`, `f2`, `f3`, `cierre`, `test`, `compuerta` (con `--de <fase>`).
Al terminar con éxito (tras publicar) escribe el marcador `<campaña>/jobs/<clave>.json` (D2); la
compuerta escribe su decisión `<campaña>/<fase>.json`, que es también su marcador. `--dry`: sin
MLflow ni entrenamiento (la compuerta escribe `--rama-forzada`); `--dry-ckpt`: entrenamiento
sintético de 4 épocas con checkpoints; `--dry-fallar-en N`: falla tras guardar la época N.
Ver `pipeline/koopman/README.md`.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ae import data as aed  # noqa: E402
from features import partition as pt  # noqa: E402
from features.build import load_validated  # noqa: E402
from koopman import compuerta as CP  # noqa: E402
from koopman import entrenar as T  # noqa: E402
from koopman import evaluar as EV  # noqa: E402
from koopman import identificabilidad as ID  # noqa: E402
from koopman import inicio as IN  # noqa: E402
from koopman.modelo import KoopmanModelo, bloques  # noqa: E402
from operador import estado as E  # noqa: E402
from operador import evaluacion as evm  # noqa: E402
from operador import registro as G  # noqa: E402
from train import mlflow_log  # noqa: E402

log = logging.getLogger("koopman")

STAGE = "deep-koopman"
PARTICION = "ae-dmd"
FASES = ("smoke", "f1", "b1", "b2", "b3", "b4", "f2", "f3", "cierre", "test", "compuerta")
VARIANTES = ("principal", "kden", "perlib", "sinbu", "decgru", "ll01", "ll10", "icap")
LAMBDA_LIN = {"ll01": 0.1, "ll10": 10.0}
TRAMOS_CIERRE = ("train", "es", "sel")
AE_V2 = "sinca-zona-ae_gru-aedmd-v2-s{seed}"
E15A = "sinca-zona-dmdop-gru-e15a-aedmd-v1-s{seed}"
TEST_71 = "sinca-{slug}-{pollutant}-dmdop-gru-aedmd-v1-s{seed}"


def _flag(name: str) -> list[str]:
    return [f"--{name}", f"--{name.replace('-', '_')}"]


def _bool(s) -> bool:
    return str(s).lower() in ("1", "true", "yes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Koopman entrenado (deep-koopman): una fase por job.")
    p.add_argument(*_flag("fase"), choices=FASES, required=True)
    p.add_argument(*_flag("de"), choices=FASES[:-2], default=None, help="Fase evaluada por la compuerta.")
    p.add_argument(*_flag("encoder"), choices=("gru", "conv1ds"), default="gru")
    p.add_argument(*_flag("variantes"), default="principal", help=f"Separadas por coma: {', '.join(VARIANTES)}.")
    p.add_argument(*_flag("seed"), type=int, default=0)
    p.add_argument(*_flag("seeds"), default="0,1,2", help="Semillas (compuerta, cierre y test).")
    p.add_argument(*_flag("parent-from"), default=None, help="JSON de decisión (s3://… o ruta local) con los ganadores.")
    p.add_argument(*_flag("parent-run-id"), default=None)
    p.add_argument(*_flag("mlflow-runs"), default="auto", help="auto (referencias + arranque + padre) | none | IDs.")
    p.add_argument(*_flag("experiment"), default=mlflow_log.EXPERIMENT_NAME)
    p.add_argument(*_flag("version"), type=int, default=1)
    p.add_argument(*_flag("force-test"), type=_bool, nargs="?", const=True, default=False)
    p.add_argument(*_flag("no-upload"), type=_bool, nargs="?", const=True, default=False)
    p.add_argument(*_flag("dry"), type=_bool, nargs="?", const=True, default=False)
    p.add_argument(*_flag("dry-ckpt"), type=_bool, nargs="?", const=True, default=False)
    p.add_argument(*_flag("dry-fallar-en"), type=int, default=None)
    p.add_argument(*_flag("rama-forzada"), default=None)
    p.add_argument(*_flag("campana"), default="local")
    p.add_argument(*_flag("clave"), default=None)
    p.add_argument(*_flag("hidden"), type=int, default=64)
    p.add_argument(*_flag("epocas"), type=int, default=None, help="Solo pruebas: sobrescribe el presupuesto.")
    p.add_argument(*_flag("max-filas"), type=int, default=None, help="Solo pruebas: recorta las ventanas de ajuste.")
    p.add_argument(*_flag("bootstrap"), type=int, default=evm.B_DEFAULT)
    p.add_argument(*_flag("bootstrap-modos"), type=int, default=200)
    p.add_argument(*_flag("data-bucket"), default=os.environ.get("DATA_BUCKET", "sinca-data"))
    p.add_argument(*_flag("mlflow-bucket"), default=os.environ.get("MLFLOW_BUCKET", mlflow_log.MLFLOW_BUCKET_DEFAULT))
    p.add_argument(*_flag("mlflow-prefix"), default=os.environ.get("MLFLOW_PREFIX", mlflow_log.MLFLOW_PREFIX_DEFAULT))
    p.add_argument(*_flag("mlflow-dir"), default=os.environ.get("MLFLOW_DIR", "/opt/ml/mlflow"))
    p.add_argument(*_flag("local-validated-dir"), default=os.environ.get("LOCAL_VALIDATED_DIR"))
    p.add_argument(*_flag("model-dir"), default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument(*_flag("checkpoint-dir"), default=os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    args = p.parse_args(argv)
    args.variantes = [v for v in str(args.variantes).split(",") if v]
    bad = [v for v in args.variantes if v not in VARIANTES]
    if bad:
        p.error(f"Variantes desconocidas: {bad}")
    args.seeds = [int(x) for x in str(args.seeds).split(",") if x != ""]
    if args.fase == "compuerta" and not args.de:
        p.error("--fase compuerta requiere --de")
    args.clave = args.clave or (f"compuerta-{args.de}" if args.fase == "compuerta" else
                                f"{args.fase}-{args.encoder}{''.join(v for v in args.variantes if v != 'principal')}-s{args.seed}")
    return args


# --- Nombres ------------------------------------------------------------------------------


def run_name(enc: str, fase: str, var: str, version: int, seed: int) -> str:
    var = "" if var == "principal" else var
    return f"sinca-zona-koop-{enc}-{fase}{var}-aedmd-v{version}-s{seed}"


def run_name_test(station: str, pollutant: str, enc: str, version: int, seed: int) -> str:
    return f"sinca-{mlflow_log.station_slug(station)}-{pollutant}-koop-{enc}-aedmd-v{version}-s{seed}"


def marcador(s3, bucket: str, campana: str, clave: str, contenido: dict) -> str:
    key = CP.clave_marcador(campana, clave)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(contenido, default=str).encode(), ContentType="application/json")
    return key


def _s3(args):
    import boto3

    return boto3.client("s3")


def _leer_json(ruta: str, s3=None) -> dict:
    if ruta.startswith("s3://"):
        bucket, key = ruta[5:].split("/", 1)
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    return json.loads(pathlib.Path(ruta).read_text(encoding="utf-8"))


# --- Datos sintéticos (modo seco con checkpoints) ---------------------------------------------


def datos_sinteticos(dias: int = 70, seed: int = 0) -> dict:
    """Datos validados sintéticos mínimos (sin AWS) para `--dry-ckpt`."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=24 * dias, freq="h", tz=pt.TZ)
    h = np.arange(len(ts))
    frames = []
    for station in ("Padre Las Casas II", "Nielol", "Las Encinas"):
        pm25 = np.clip(40 + 15 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 4, len(ts)), 1, None)
        frames.append(pd.DataFrame({"estacion": station, "timestamp": ts, "pm25": pm25, "pm10": pm25 * 1.8,
                                    "pres": 1010 + rng.normal(0, 2, len(ts)), "wspd": np.abs(rng.normal(2, 1, len(ts))),
                                    "rain": np.abs(rng.normal(0, 0.2, len(ts))), "temp": 10 + rng.normal(0, 2, len(ts))}))
    return {"sinca": pd.concat(frames, ignore_index=True),
            "open_meteo": pd.DataFrame({"timestamp": ts, "boundary_layer_height": 300 + rng.normal(0, 20, len(ts))}),
            "feriados": pd.DataFrame({"fecha": [pd.Timestamp("2026-01-01").date()]})}


# --- Modo seco ------------------------------------------------------------------------------


def ejecutar_seco(args, s3) -> dict:
    if args.fase == "compuerta":
        rama = args.rama_forzada or ("continuar" if args.de in ("f2", "f3") else "pasa")
        decision = {"fase": args.de, "rama": rama, "motivo": "modo seco (rama forzada)", "ganadores": {},
                    "encoders": ["gru"], "cifras": {}, "seco": True,
                    "fecha": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        key = CP.escribir(decision, s3, args.mlflow_bucket, args.campana, args.de)
        log.info("Compuerta seca %s -> %s (%s)", args.de, rama, key)
        return {"decision": decision}
    info = {"clave": args.clave, "seco": True, "fase": args.fase}
    if args.dry_ckpt:
        data = datos_sinteticos()
        grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
        ctx = E.contexto(grid, pt.get_partition(PARTICION))
        pos, _ = E.posiciones(ctx, 24, ("train",))
        datos = T.Datos(grid, ctx.ng, pos, ctx.spans["train"][1])
        modelo = KoopmanModelo(encoder="gru", hidden=8, seed=args.seed)
        out = T.entrenar(modelo, datos, "f1", _escala(ctx.norm), epocas=4,
                         ckpt_dir=pathlib.Path(args.checkpoint_dir) / "principal", seed=args.seed,
                         fallar_tras_epoca=args.dry_fallar_en, log=log.info)
        info.update(epochs_run=len(out["historia"]), retomado_en=out["retomado_en"])
        log.info("epochs_run=%d retomado_en=%s", len(out["historia"]), out["retomado_en"])
    marcador(s3, args.mlflow_bucket, args.campana, args.clave, info)
    return info


# --- Recursos del job real --------------------------------------------------------------------


def _escala(norm: aed.Normalizer) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c, _, _ in aed.mp_channels()]
    return norm.mean[cols].to_numpy(dtype=float), norm.std[cols].to_numpy(dtype=float)


class Job:
    def __init__(self, args, s3=None, descargar: bool = True):
        self.args, self.s3 = args, s3
        self.state = args.mlflow_dir
        self.part = pt.get_partition(PARTICION)
        self.last_modified = None
        if descargar and not args.no_upload:
            self.descargar_estado()
        else:
            meta = pathlib.Path(self.state, ".download.json")
            self.last_modified = json.loads(meta.read_text())["last_modified"] if meta.exists() else None
        from analysis import load as analysis_load

        self.runs = analysis_load.list_runs(self.state, experiment_name=None)
        data = load_validated(args.data_bucket, local_dir=args.local_validated_dir)
        self.grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
        self.data = data
        self.ctx = E.contexto(self.grid, self.part)
        self._ev = None

    def descargar_estado(self) -> None:
        a = self.args
        db_key = f"{a.mlflow_prefix}/{mlflow_log.DB_FILENAME}"
        self.last_modified = mlflow_log.iso_ts(self.s3.head_object(Bucket=a.mlflow_bucket, Key=db_key)["LastModified"])
        mlflow_log.download_state(a.mlflow_bucket, a.mlflow_prefix, self.state, self.s3, run_ids=[])
        from analysis import load as analysis_load

        runs = analysis_load.list_runs(self.state, experiment_name=None)
        if a.mlflow_runs == "none":
            ids = []
        elif a.mlflow_runs == "auto":
            ids = EV.run_ids_referencia(runs) + self._ids_arranque(runs) + self._ids_padres(runs)
        else:
            ids = mlflow_log.parse_mlflow_runs(a.mlflow_runs) or []
        mlflow_log.download_state(a.mlflow_bucket, a.mlflow_prefix, self.state, self.s3, run_ids=sorted(set(ids)))

    def _ids_arranque(self, runs) -> list[str]:
        nombres = {n.format(seed=s) for n in (AE_V2, E15A) for s in self.args.seeds + [self.args.seed]}
        nombres |= {TEST_71.format(slug=mlflow_log.station_slug(st), pollutant=po, seed=s)
                    for _, st, po in aed.mp_channels() for s in self.args.seeds}
        return [r["run_id"] for r in runs if r["run_name"] in nombres and not r["tags"].get("descartado")]

    def _ids_padres(self, runs) -> list[str]:
        ids = [self.args.parent_run_id] if self.args.parent_run_id else []
        if self.args.parent_from:
            ids += [str(v) for v in _leer_json(self.args.parent_from, self.s3).get("ganadores", {}).values()]
        ids += [r["run_id"] for r in runs if r["tags"].get("stage") == STAGE and not r["tags"].get("descartado")]
        return ids

    # --- referencias y evaluación
    @property
    def ev(self) -> evm.Evaluador:
        if self._ev is None:
            self.rows = EV.filas_referencia(self.state, self.runs)
            self._ev = evm.Evaluador(self.rows, B=self.args.bootstrap, seed=0)
        return self._ev

    def run_por_nombre(self, nombre: str) -> dict | None:
        return next((r for r in self.runs if r["run_name"] == nombre and not r["tags"].get("descartado")), None)

    def artefacto(self, run_id: str, nombre: str) -> str:
        from analysis import load as analysis_load

        path = analysis_load.artifact_path(self.state, run_id, nombre)
        if not path:
            raise FileNotFoundError(f"Run {run_id} sin {nombre}")
        return path

    def datos_ajuste(self, tramos=("train",)) -> tuple[T.Datos, np.ndarray]:
        if tramos == ("train",):
            pos, _ = E.posiciones(self.ctx, 24, ("train",))
            fin = self.ctx.spans["train"][1]
        else:
            ctx_u = E.Contexto(self.grid, self.ctx.norm, self.ctx.ng, self.ctx.spans, tuple(tramos), self.part)
            pos, split = E.posiciones(ctx_u, 24, tuple(tramos), union_fit=True)
            pos = pos[split == "fit"]
            fin = self.ctx.spans[tramos[-1]][1]
        if self.args.max_filas:
            pos = pos[: self.args.max_filas]
            fin = min(fin, int(pos[-1]) + 30)
        return T.Datos(self.grid, self.ctx.ng, pos, fin), pos

    # --- modelos
    def modelo_nuevo(self, var: str, seed: int, d: int = 10) -> tuple[KoopmanModelo, dict]:
        import keras

        enc = self.args.encoder
        modelo = KoopmanModelo(encoder=enc, decoder="gru" if var == "decgru" else "lineal", d=d,
                               variante=var if var in ("kden", "perlib", "sinbu") else "",
                               hidden=self.args.hidden, seed=seed)
        info = {"arranque": "frio"}
        lam = None
        if enc == "gru":
            ae = self.run_por_nombre(AE_V2.format(seed=seed))
            if ae is not None:
                encoder_v2 = keras.saving.load_model(self.artefacto(ae["run_id"], "encoder.keras"), compile=False)
                IN.encoder_desde_v2(modelo, encoder_v2)
                info = {"arranque": "caliente", "ae_run_id": ae["run_id"]}
            op = self.run_por_nombre(E15A.format(seed=seed))
            if op is not None:
                npz = np.load(self.artefacto(op["run_id"], "operator.npz"))
                lam = npz["eigvals_real"] + 1j * npz["eigvals_imag"]
                info["e15a_run_id"] = op["run_id"]
        info["k_init"] = IN.init_k(modelo, lam)
        if modelo.config["decoder"] == "lineal":
            datos, pos = self.datos_ajuste()
            IN.decoder_lineal_ls(modelo, self.ctx.ng, pos[: 20000])
        return modelo, info

    def modelo_de_run(self, run_id: str) -> tuple[KoopmanModelo, dict]:
        cfg = json.loads(pathlib.Path(self.artefacto(run_id, "koopman_config.json")).read_text())
        modelo = KoopmanModelo(**cfg["modelo"])
        npz = np.load(self.artefacto(run_id, "koopman_pesos.npz"))
        modelo.cargar_pesos({k: npz[k] for k in npz.files})
        if cfg.get("bloques_finales") is not None:
            modelo.fijar_bloques(cfg["bloques_finales"])
        return modelo, cfg


# --- Evaluación completa de un modelo (sel) ---------------------------------------------------


def evaluar_completo(job: Job, modelo: KoopmanModelo, nombre: str, con_duan: bool) -> dict:
    ev = job.ev
    ref71 = EV.puntaje_71(job.state, job.runs, ev)
    res = EV.evaluar_modelo(modelo, job.ctx, ev, nombre, ref71=ref71)
    out = {"base": res, "ref71": ref71}
    pos_tr, _ = E.posiciones(job.ctx, 24, ("train",))
    pos_tr = pos_tr[pt.row_mask(job.grid.index[pos_tr], job.part, "train")]
    if job.args.max_filas:
        pos_tr = pos_tr[: job.args.max_filas]
    if con_duan:
        S = EV.factores_duan(modelo, job.ctx, pos_tr)
        out["duan"] = EV.evaluar_modelo(modelo, job.ctx, ev, nombre + "-duan", duan_S=S, ref71=ref71)
        out["duan_S"] = S
    # identificabilidad (D11) y coherencia
    zH = ID.z_horizonte(modelo, job.ctx.ng, res["positions"])
    w = EV.pesos_por_timestamp(ev, res["positions"], job.grid)
    a = ID.aportes(ID.contribuciones(modelo, zH), w)
    dom = ID.dominantes(a)
    pos_fit, _ = E.posiciones(job.ctx, 24, ("train",))
    if job.args.max_filas:
        pos_fit = pos_fit[: job.args.max_filas]
    zl = ID.latente(modelo, job.ctx.ng, pos_fit)
    local = job.grid.index[pos_fit].tz_convert(pt.TZ).tz_localize(None) if job.grid.index.tz is not None else job.grid.index[pos_fit]
    boot = ID.bootstrap_latente(modelo, zl, pos_fit, local.normalize().to_numpy(), B=job.args.bootstrap_modos)
    out.update(aportes=a, dominantes=dom, tabla_modos=ID.tabla_modos(modelo, a, dom), boot=boot,
               coherencia=ID.coherencia(modelo, zl, pos_fit), latente=zl, pos_fit=pos_fit)
    # eliminación de bloques (D13), reportada; no se repite si los bloques ya están fijados (cierre)
    def puntaje_bloques(conservar):
        modelo.fijar_bloques(conservar)
        try:
            return EV.evaluar_modelo(modelo, job.ctx, ev, nombre + "-bloques")["puntaje"]
        finally:
            modelo.fijar_bloques(None)

    if modelo.op.kden is None and modelo.mascara is None:
        out["bloques_finales"], out["curva_bloques"] = ID.eliminacion_bloques(modelo, puntaje_bloques)
    return out


def metricas(ev_out: dict, historia: list[dict]) -> dict:
    from operador.run import metricas_de, veredicto

    res = ev_out["base"]
    m = metricas_de(res["tabla"])
    if "duan" in ev_out:
        m.update({f"duan_{k}": v for k, v in metricas_de(ev_out["duan"]["tabla"]).items() if k.startswith("sel_wrmse_media")
                  or k.startswith("sel_razon_mejor_media")})
    if "diff_71" in res:
        m.update({"diff_71": res["diff_71"]["diff"], "diff_71_lo": res["diff_71"]["lo"], "diff_71_hi": res["diff_71"]["hi"]})
    m["d12_precision"] = float(veredicto(res["tabla"])["precision"])
    m["n_dominantes"] = len(ev_out["dominantes"])
    if historia:
        last = historia[-1]
        m["epochs_run"] = len(historia)
        rec, lin = last.get("norm_rec"), last.get("norm_lin")
        if rec and lin is not None:
            m["lin_rec"] = float(lin) / max(float(rec), 1e-12)
        m.update({k: v for k, v in last.items() if k.startswith("loss_") or k.startswith("norm_")})
    return m


def registrar(job: Job, modelo: KoopmanModelo, fase: str, var: str, seed: int, ev_out: dict, entreno: dict,
              info: dict, parent_run_id: str | None, segundos: float) -> str:
    import tempfile

    a = job.args
    enc = modelo.config["encoder"]
    name = run_name(enc, fase, var, a.version, seed)
    tags = {"modelo": f"koop-{enc}", "stage": STAGE, "particion": PARTICION, "fase": fase, "variante": var,
            "seed": str(seed), "campana": a.campana}
    if parent_run_id:
        tags["parent_run_id"] = parent_run_id
    params = {"tipo": "koopman", "fase": fase, "variante": var, "encoder": enc, "decoder": modelo.config["decoder"],
              "d": modelo.d, "version": a.version, "arranque": info.get("arranque"), "clave": a.clave,
              "bloques_finales": json.dumps(ev_out.get("bloques_finales")), "retomado_en": entreno.get("retomado_en")}
    mets = metricas(ev_out, entreno.get("historia", []))
    mets["segundos"] = segundos
    if entreno.get("historia"):
        mets["segundos_por_epoca"] = segundos / max(1, len(entreno["historia"]) - (entreno.get("retomado_en") or 0))
    frames = dict(ev_out["base"]["frames"])
    frames.update({"modos.parquet": ev_out["tabla_modos"], "bootstrap_modos.parquet": ev_out["boot"],
                   "coherencia.parquet": ev_out["coherencia"], "curvas.parquet": pd.DataFrame(entreno.get("historia", [])),
                   "latent.parquet": pd.DataFrame(ev_out["latente"], columns=[f"z_{i}" for i in range(modelo.d)]).assign(
                       timestamp=job.grid.index[ev_out["pos_fit"]], split="train")})
    if "curva_bloques" in ev_out:
        frames["curva_bloques.parquet"] = ev_out["curva_bloques"]
    if "duan" in ev_out:
        frames["predictions_duan.parquet"] = ev_out["duan"]["frames"]["predictions.parquet"]
    k = modelo.op.matriz_np()
    arrays = {"koopman.npz": {"K": k, "B": np.asarray(modelo.op.B) if modelo.op.B is not None else np.zeros((modelo.d, 0)),
                              "a": np.asarray(modelo.op.a), "b": np.asarray(modelo.op.b),
                              "eig_re": modelo.op.autovalores().real, "eig_im": modelo.op.autovalores().imag,
                              **({"duan_S": ev_out["duan_S"]} if "duan_S" in ev_out else {})},
              "koopman_pesos.npz": modelo.pesos()}
    cfg = {"modelo": modelo.config, "fase": fase, "variante": var, "info": info,
           "bloques_finales": ev_out.get("bloques_finales"), "parent_run_id": parent_run_id,
           "cfg_perdida": {**T.FASES.get(fase, {}), **({"icap": True} if var == "icap" else {})}}
    run_id = G.log_variante(job.state, name, tags, params, mets, frames, arrays,
                            {"koopman_config.json": json.dumps(cfg, default=str)}, experiment=a.experiment)
    out_dir = G.artifacts_dir(job.state, run_id)
    with tempfile.TemporaryDirectory() as tmp:
        modelo.enc.save(os.path.join(tmp, "encoder.keras"))
        modelo.dec.save(os.path.join(tmp, "decoder.keras"))
        import shutil

        for f in ("encoder.keras", "decoder.keras"):
            shutil.move(os.path.join(tmp, f), out_dir / f)
    log.info("Run %s (%s)", name, run_id)
    return run_id


# --- Fases ------------------------------------------------------------------------------------


def _cfg_variante(fase: str, var: str) -> tuple[dict, dict, int]:
    cfg, lambdas, d = {}, {}, 8 if fase == "b4" else 10
    if var in LAMBDA_LIN:
        lambdas["lin"] = LAMBDA_LIN[var]
    if var == "icap":
        cfg["icap"] = True
    return cfg, lambdas, d


def fase_entrenamiento(job: Job) -> list[str]:
    a = job.args
    fase = a.fase
    ids = []
    for var in a.variantes:
        t0 = time.perf_counter()
        cfg_p, lambdas, d = _cfg_variante(fase, var)
        parent = None
        if fase in ("f2", "f3"):
            parent = _padre(job, a.seed)
            modelo, cfg_parent = job.modelo_de_run(parent)
            modelo.fijar_bloques(None)
            info = {"arranque": "padre", "parent_run_id": parent}
            en_caliente = False
        else:
            modelo, info = job.modelo_nuevo(var, a.seed, d)
            en_caliente = info.get("arranque") == "caliente"
        datos, _ = job.datos_ajuste()
        entreno = T.entrenar(modelo, datos, fase, _escala(job.ctx.norm),
                             en_caliente=en_caliente, lambdas=lambdas, epocas=a.epocas,
                             ckpt_dir=pathlib.Path(a.checkpoint_dir) / var, seed=a.seed, cfg_perdida=cfg_p,
                             log=log.info)
        T.orden_canonico(modelo)
        ev_out = evaluar_completo(job, modelo, f"koop-{fase}{var}-s{a.seed}", con_duan=fase in ("smoke", "f1", "b1",
                                                                                                   "b2", "b3", "b4", "f2"))
        ids.append(registrar(job, modelo, fase, var, a.seed, ev_out, entreno, info, parent, time.perf_counter() - t0))
    return ids


def _padre(job: Job, seed: int) -> str:
    a = job.args
    if a.parent_run_id:
        return a.parent_run_id
    if not a.parent_from:
        raise SystemExit(f"--fase {a.fase} requiere --parent-from o --parent-run-id")
    dec = _leer_json(a.parent_from, job.s3)
    gan = dec.get("ganadores_por_encoder", {}).get(a.encoder) or dec.get("ganadores", {})
    if str(seed) not in gan:
        raise SystemExit(f"La decisión {a.parent_from} no trae ganador para la semilla {seed}")
    return str(gan[str(seed)])


def fase_cierre(job: Job) -> list[str]:
    """D16: encoder y decoder congelados; `K`, `B` (y Duan si el final es F1/F2) reajustados 20 épocas con la
    pérdida de la fase final sobre train+es+sel, con los bloques fijados en D13."""
    a = job.args
    t0 = time.perf_counter()
    parent = _padre(job, a.seed)
    modelo, cfg = job.modelo_de_run(parent)
    datos, pos = job.datos_ajuste(TRAMOS_CIERRE)
    cfg_p = {k: v for k, v in cfg.get("cfg_perdida", {}).items() if k in ("H", "tgt", "icap", "cons", "l1")}
    if "curriculo" in cfg.get("cfg_perdida", {}):
        cfg_p["H"] = 24
    entreno = T.entrenar(modelo, datos, "cierre", _escala(job.ctx.norm), epocas=a.epocas,
                         ckpt_dir=pathlib.Path(a.checkpoint_dir) / "cierre", seed=a.seed, cfg_perdida=cfg_p, log=log.info)
    ev_out = evaluar_completo(job, modelo, f"koop-cierre-s{a.seed}", con_duan=cfg.get("fase") in ("f1", "f2", "b1", "b2",
                                                                                                    "b3", "b4"))
    ev_out["bloques_finales"] = cfg.get("bloques_finales")
    ev_out.pop("curva_bloques", None)
    return [registrar(job, modelo, "cierre", "principal", a.seed, ev_out, entreno,
                      {"arranque": "padre", "fase_final": cfg.get("fase")}, parent, time.perf_counter() - t0)]


def fase_test(job: Job) -> list[str]:
    """Una sola evaluación en test (D16): runs por serie con el contrato de LightGBM y veredicto D12."""
    from evaluate.evaluate import evaluate_predictions, zone_icap
    from koopman.modelo import pronostico
    from operador.run import veredicto

    a = job.args
    previos = [r for r in job.runs if r["tags"].get("stage") == STAGE and r["tags"].get("fase") == "test"
               and str(r["params"].get("version")) == str(a.version) and not r["tags"].get("descartado")]
    if previos and not a.force_test:
        raise SystemExit(f"Ya hay {len(previos)} runs de test del Koopman (versión {a.version}): repetir exige --force-test.")
    from features.build import build_features

    feats = build_features(job.data["sinca"], job.data["open_meteo"], job.data["feriados"])
    tg = feats[["timestamp", "estacion", "pollutant", "target"]].rename(columns={"target": "y_true"})
    preds, cierres = {}, {}
    for seed in a.seeds:
        run = job.run_por_nombre(run_name(a.encoder, "cierre", "principal", a.version, seed))
        if run is None:
            raise SystemExit(f"Falta el run de cierre de la semilla {seed}")
        modelo, cfg = job.modelo_de_run(run["run_id"])
        pos, _ = E.posiciones(job.ctx, 24, ("test",))
        pos = pos[pt.row_mask(job.grid.index[pos], job.part, "test")]
        fc, lp = pronostico(modelo, job.ctx.ng, job.ctx.norm, pos)
        npz = np.load(job.artefacto(run["run_id"], "koopman.npz"))
        if "duan_S" in npz.files:
            fc = np.clip(np.exp(lp) * npz["duan_S"][None] - 1.0, 0.0, None).mean(axis=1)
        st = job.grid.index[pos]
        pieces = [pd.DataFrame({"timestamp": st, "estacion": s, "pollutant": p, "y_pred": fc[:, c]})
                  for c, (s, p) in enumerate(evm.SERIES)]
        preds[seed] = pd.concat(pieces, ignore_index=True).merge(tg, on=evm.KEYS)
        cierres[seed] = run["run_id"]
    by_id = {r["run_id"]: r for r in job.runs}
    from operador.run import clasificar_v2

    v2 = clasificar_v2(job.runs)
    lgbm = pd.concat([EV._preds(job.state, by_id[r["params"]["source_run_id"]], "test") for r in v2["lgbm"]], ignore_index=True)
    from baseline.persistence import predict_frame

    pers = predict_frame(job.data["sinca"], feats)
    pers = pers[pt.row_mask(pers["timestamp"], job.part, "test")]
    rows = evm.construir_eval_rows({f"k{s}": f for s, f in preds.items()}, lgbm, pers)
    ev = evm.Evaluador(rows, B=a.bootstrap, seed=0)
    items, ids = [], []
    for seed, f in preds.items():
        y = rows[evm.KEYS].merge(f[evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
        items.append(ev.puntaje(y))
        for (s, p), g in ev.predicciones(y, split="test").groupby(["estacion", "pollutant"]):
            tags = {"modelo": f"koop-{a.encoder}", "stage": STAGE, "particion": PARTICION, "fase": "test",
                    "seed": str(seed), "evaluacion": "test", "serie": f"{s}|{p}", "campana": a.campana}
            params = {"estacion": s, "pollutant": p, "version": a.version, "seed": seed, "tipo": "koopman",
                      "cierre_run_id": cierres[seed]}
            ids.append(G.log_variante(job.state, run_name_test(s, p, a.encoder, a.version, seed), tags, params,
                                      {f"test_{k}": v for k, v in evaluate_predictions(g).items()},
                                      {"predictions.parquet": g, "zone_icap.parquet": zone_icap(g),
                                       "rolling_scores.parquet": pd.DataFrame({"window_index": pd.Series(dtype=int),
                                                                               "rmse": pd.Series(dtype=float)})},
                                      experiment=a.experiment))
    p = evm.Puntaje.promedio(items)
    tabla = ev.tabla(p)
    # GRU+DMD+Duan de 7.1 en test (media de semillas), sobre las mismas filas
    t71 = []
    for seed in (0, 1, 2):
        fr = [EV._preds(job.state, r, "test") for r in job.runs
              if r["run_name"] in {TEST_71.format(slug=mlflow_log.station_slug(s), pollutant=po, seed=seed)
                                   for s, po in evm.SERIES} and not r["tags"].get("descartado")]
        if len(fr) == 6:
            f = pd.concat(fr, ignore_index=True)
            y = rows[evm.KEYS].merge(f[evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
            if np.isfinite(y).all():
                t71.append(ev.puntaje(y))
    resumen = {"veredicto_precision": veredicto(tabla), "wrmse_media": p.media,
               "diff_71": evm.diferencia(p, evm.Puntaje.promedio(t71)) if t71 else None}
    pathlib.Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(pathlib.Path(a.model_dir) / "test_tabla.parquet", index=False)
    pathlib.Path(a.model_dir, "test_resumen.json").write_text(json.dumps(resumen, default=str), encoding="utf-8")
    log.info("Test: %s", resumen)
    return ids


def fase_compuerta(job: Job) -> dict:
    """Aplica las reglas de `compuerta.decidir` a los runs de la fase `--de` de la campaña."""
    a = job.args
    runs = [r for r in job.runs if r["tags"].get("stage") == STAGE and r["tags"].get("fase") == a.de
            and r["tags"].get("campana") == a.campana and not r["tags"].get("descartado")]
    ev = job.ev
    grupos: dict[str, dict] = {}
    for r in runs:
        enc, var = r["params"].get("encoder", "gru"), r["tags"].get("variante", "principal")
        nombre = enc if var == "principal" else f"{enc}-{var}"
        grupos.setdefault(nombre, {"runs": {}, "encoder": enc, "variante": var})["runs"][int(r["tags"]["seed"])] = r
    cand = {}
    for nombre, g in grupos.items():
        items, sem = [], {}
        for seed, r in sorted(g["runs"].items()):
            f = pd.read_parquet(job.artefacto(r["run_id"], "predictions.parquet"))
            y = ev.rows[evm.KEYS].merge(f[evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
            items.append(ev.puntaje(y))
            tm = pd.read_parquet(job.artefacto(r["run_id"], "modos.parquet"))
            boot = pd.read_parquet(job.artefacto(r["run_id"], "bootstrap_modos.parquet"))
            lam = tm["lam_re"].to_numpy() + 1j * tm["lam_im"].to_numpy()
            sem[seed] = {"dominantes": lam[tm["dominante"].to_numpy(bool)], "modos": lam, "boot": boot}
        ok, tabla = ID.compuerta(sem) if len(sem) > 1 else (False, pd.DataFrame())
        lin_rec = float(np.nanmean([r["metrics"].get("lin_rec", np.nan) for r in g["runs"].values()]))
        cand[nombre] = {"puntaje": evm.Puntaje.promedio(items) if len(items) > 1 else items[0], "identificable": ok,
                        "ganadores": {str(s): r["run_id"] for s, r in g["runs"].items()}, "lin_rec": lin_rec,
                        "encoder": g["encoder"], "variante": g["variante"]}
    decision = CP.decidir(a.de, cand, EV.puntaje_71(job.state, job.runs, ev))
    key = CP.escribir(decision, job.s3, a.mlflow_bucket, a.campana, a.de)
    log.info("Compuerta %s -> %s (%s): %s", a.de, decision["rama"], key, decision["motivo"])
    return decision


# --- Principal --------------------------------------------------------------------------------


def ejecutar(args, s3=None, descargar: bool = True) -> dict:
    s3 = s3 if s3 is not None else (_s3(args) if not args.no_upload or args.dry else None)
    if args.dry:
        return ejecutar_seco(args, s3)
    if args.fase in ("f2", "f3") and args.parent_from and args.encoder != "gru":
        dec = _leer_json(args.parent_from, s3)
        if args.encoder not in dec.get("encoders", ["gru"]):
            info = {"clave": args.clave, "omitido": True, "motivo": f"{args.encoder} no sigue tras {dec.get('fase')}"}
            log.info("Job omitido: %s", info["motivo"])
            if not args.no_upload:
                marcador(s3, args.mlflow_bucket, args.campana, args.clave, info)
            return info
    job = Job(args, s3, descargar)
    if args.fase == "compuerta":
        return {"decision": fase_compuerta(job)}
    if args.fase == "cierre":
        ids = fase_cierre(job)
    elif args.fase == "test":
        ids = fase_test(job)
    else:
        ids = fase_entrenamiento(job)
    out = {"run_ids": ids}
    if args.no_upload:
        return out
    res = mlflow_log.publish_new_runs(job.state, STAGE, s3, job.last_modified, args.mlflow_bucket, args.mlflow_prefix,
                                      on_conflict_dir=str(pathlib.Path(args.model_dir) / "conflicto"))
    verificados = {r: [p.name for p in mlflow_log.run_files(job.state, r)] for r in ids}
    marcador(s3, args.mlflow_bucket, args.campana, args.clave,
             {"clave": args.clave, "run_ids": ids, "artefactos": verificados, "publicado": res["runs"],
              "fecha": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return {**out, "publicacion": res}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    ejecutar(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
