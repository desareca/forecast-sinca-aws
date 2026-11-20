"""Runs MLflow del operador y resúmenes de etapa (`dmd-operador-global`, D1, D10).

Los runs se escriben en el `mlflow.sqlite` del estado local (`--state-dir`). Los artefactos se
escriben directo en `<state-dir>/mlruns/<run_id>/artifacts/` (el experimento descargado apunta a
`/opt/ml/...`, que no existe en local); `analysis.load.artifact_path` los resuelve y
`scripts/mlflow/publish_state.py` los sube con el run.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np
import pandas as pd

from train import mlflow_log

STAGE = "dmd-operador-global"
EXPERIMENT = mlflow_log.EXPERIMENT_NAME
OUTPUT_DIR = pathlib.Path("output") / STAGE


def run_name(rep: str, exp: str, token: str, version: int, seed: int) -> str:
    """`sinca-zona-dmdop-{rep}-{exp}-{part}-v{n}-s{seed}` (run conjunto de las 6 series)."""
    return f"sinca-zona-dmdop-{rep}-{exp}-{token}-v{version}-s{seed}"


def run_name_test(station: str, pollutant: str, rep: str, token: str, version: int, seed: int) -> str:
    """`sinca-{slug}-{pollutant}-dmdop-{rep}-{part}-v{n}-s{seed}` (test por serie, contrato LightGBM)."""
    return f"sinca-{mlflow_log.station_slug(station)}-{pollutant}-dmdop-{rep}-{token}-v{version}-s{seed}"


def tags_base(rep: str, etapa: str, particion: str, seed: int, ae_run_id: str | None = None) -> dict[str, str]:
    tags = {"modelo": f"dmdop-{rep}", "stage": STAGE, "particion": particion, "etapa": etapa, "seed": str(seed)}
    if ae_run_id:
        tags["ae_run_id"] = ae_run_id
    return tags


def _client(state_dir: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    db = pathlib.Path(state_dir, mlflow_log.DB_FILENAME).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    return MlflowClient()


def _experiment_id(client, state_dir: str, experiment: str) -> str:
    exp = client.get_experiment_by_name(experiment)
    if exp is not None:
        return exp.experiment_id
    location = pathlib.Path(state_dir, mlflow_log.ARTIFACTS_DIRNAME).resolve().as_uri()
    return client.create_experiment(experiment, artifact_location=location)


def artifacts_dir(state_dir: str, run_id: str) -> pathlib.Path:
    return pathlib.Path(state_dir, mlflow_log.ARTIFACTS_DIRNAME, run_id, "artifacts")


def log_variante(state_dir: str, name: str, tags: dict, params: dict, metrics: dict,
                 frames: dict[str, pd.DataFrame] | None = None, arrays: dict[str, dict] | None = None,
                 texts: dict[str, str] | None = None, experiment: str = EXPERIMENT) -> str:
    """Crea un run con tags/params/métricas y escribe sus artefactos en el estado local."""
    from mlflow.entities import Metric, Param, RunTag

    client = _client(state_dir)
    exp_id = _experiment_id(client, state_dir, experiment)
    run = client.create_run(exp_id, run_name=name, tags={k: str(v) for k, v in tags.items()})
    run_id = run.info.run_id
    now = int(time.time() * 1000)
    ms = [Metric(k, float(v), now, 0) for k, v in metrics.items() if v is not None and np.isfinite(float(v))]
    ps = [Param(k, str(v)) for k, v in params.items() if v is not None]
    for i in range(0, max(len(ms), len(ps), 1), 100):
        client.log_batch(run_id, metrics=ms[i:i + 100], params=ps[i:i + 100], tags=[RunTag("mlflow.source.type", "LOCAL")] if i == 0 else [])
    out = artifacts_dir(state_dir, run_id)
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in (frames or {}).items():
        frame.to_parquet(out / filename, index=False)
    for filename, payload in (arrays or {}).items():
        np.savez(out / filename, **payload)
    for filename, text in (texts or {}).items():
        (out / filename).write_text(text, encoding="utf-8")
    client.set_terminated(run_id, "FINISHED")
    return run_id


def runs_de_etapa(state_dir: str, etapa: str | None = None, version: int | None = None) -> list[dict]:
    from analysis import load as analysis_load

    out = []
    for run in analysis_load.list_runs(state_dir, experiment_name=None):
        tags = run.get("tags", {})
        if tags.get("stage") != STAGE:
            continue
        if etapa is not None and tags.get("etapa") != etapa:
            continue
        if version is not None and str(run["params"].get("version")) != str(version):
            continue
        out.append(run)
    return out


# --- Resúmenes -----------------------------------------------------------------


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame is None or frame.empty:
        return "_(sin filas)_"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            v = row[col]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def resumen_md(etapa: str, secciones: list[tuple[str, str]], out_dir: str | os.PathLike = OUTPUT_DIR,
               extra_json: dict | None = None) -> pathlib.Path:
    """Escribe `output/dmd-operador-global/<etapa>.md` (y `<etapa>.json` si se da)."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = [f"# dmd-operador-global — etapa {etapa}", ""]
    for title, text in secciones:
        body += [f"## {title}", "", text, ""]
    path = out / f"{etapa}.md"
    path.write_text("\n".join(body), encoding="utf-8")
    if extra_json is not None:
        (out / f"{etapa}.json").write_text(json.dumps(extra_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# --- Decisiones del operador (arrastre entre etapas, D9) ------------------------

DECISIONES_PATH = pathlib.Path(__file__).resolve().parent / "decisiones.json"
ETAPAS: tuple[str, ...] = ("base", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "cierre", "test")
CON_DECISION: tuple[str, ...] = ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6")


def leer_decisiones(path: str | os.PathLike = DECISIONES_PATH) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def etapa_previa(etapa: str) -> str | None:
    """Etapa con decisión que debe existir antes de correr `etapa` (None si no aplica)."""
    i = ETAPAS.index(etapa)
    if etapa in ("base", "1.0"):
        return None
    if etapa == "test":
        return None  # exige el cierre (artefacto), no una decisión
    return ETAPAS[i - 1]


def exigir_decision(etapa: str, decisiones: dict) -> dict:
    """Configuración vigente (decisión de la etapa previa); error si falta."""
    previa = etapa_previa(etapa)
    if previa is None:
        return {}
    if previa not in decisiones or not decisiones[previa].get("ganador"):
        raise SystemExit(f"La etapa {etapa} requiere la decisión del operador de la etapa {previa} "
                         f"en {DECISIONES_PATH.name}.")
    return dict(decisiones[previa].get("config", {}))


def registrar_decision(etapa: str, ganador: str, config: dict, nota: str, fecha: str,
                       path: str | os.PathLike = DECISIONES_PATH) -> dict:
    if etapa not in CON_DECISION:
        raise ValueError(f"La etapa {etapa} no lleva decisión")
    data = leer_decisiones(path)
    data[etapa] = {"ganador": ganador, "config": config, "fecha": fecha, "nota": nota}
    pathlib.Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
