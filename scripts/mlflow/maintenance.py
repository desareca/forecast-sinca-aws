"""Mantenimiento del estado MLflow (change `ae-dmd-prep`, D7).

- `plan`: solo lectura (sqlite en modo `ro`). Propone (a) borrar los runs `lightgbm` con
  `version = 99` activos, con sus prefijos S3, y (b) renombrar los grupos de nombres duplicados
  de las familias `ref-*` y `sinca-zona-ae_*` a nombres únicos. No toca `lightgbm-v1` (2 por
  serie es estructural). Escribe `plan.json` e imprime la tabla.
- `apply`: sobre el sqlite **local** (una copia), ejecuta `delete_run` y `update_run(name=...)`
  según el plan. Sin `--yes` no escribe. Al final imprime, sin ejecutarlos, los comandos del
  operador: respaldo, `mlflow gc`, `aws s3 rm` (dry-run y real) y subida. Todos con `<AWS_PROFILE>`.

No llama a AWS. Uso desde la raíz del repo:

    python scripts/mlflow/maintenance.py plan --state-dir .\\mlflow-maint --out .\\mlflow-maint\\plan.json
    python scripts/mlflow/maintenance.py apply --state-dir .\\mlflow-maint --plan .\\mlflow-maint\\plan.json --yes
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sqlite3
import sys
from collections import Counter, defaultdict

# `pipeline/` en el path, por consistencia con el resto de scripts (no se importa nada de él aquí).
_PIPELINE = pathlib.Path(__file__).resolve().parents[2] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

log = logging.getLogger("mlflow-maintenance")

DB_FILENAME = "mlflow.sqlite"
S3_PREFIX = "_mlflow"
BUCKET_PLACEHOLDER = "sinca-mlflow"
EXPERIMENT_DEFAULT = "sinca-baseline"
V99 = "99"
PROFILE = "<AWS_PROFILE>"


# --- Lectura ----------------------------------------------------------------


def _connect_ro(state_dir: str) -> sqlite3.Connection:
    path = pathlib.Path(state_dir, DB_FILENAME).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def active_runs(state_dir: str, experiment: str) -> list[dict]:
    """Runs activos del experimento con params y tags (solo lectura)."""
    con = _connect_ro(state_dir)
    try:
        exp = con.execute("select experiment_id from experiments where name = ?", (experiment,)).fetchone()
        if not exp:
            return []
        rows = con.execute(
            "select run_uuid, name, experiment_id from runs where experiment_id = ? and lifecycle_stage = 'active'",
            (exp[0],),
        ).fetchall()
        out: list[dict] = []
        for run_id, name, experiment_id in rows:
            params = dict(con.execute("select key, value from params where run_uuid = ?", (run_id,)).fetchall())
            tags = dict(con.execute("select key, value from tags where run_uuid = ?", (run_id,)).fetchall())
            out.append({"run_id": run_id, "name": name, "experiment_id": str(experiment_id),
                        "params": params, "tags": tags})
        return out
    finally:
        con.close()


# --- Reglas D7 ----------------------------------------------------------------


def _family(run: dict) -> str | None:
    name = run["name"]
    if name.startswith("ref-mean"):
        return "ref-mean"
    if name.startswith("ref-pca-d"):
        return "ref-pca"
    if name.startswith("sinca-zona-ae_"):
        return "ae"
    return None


def _name_at(run: dict, level: int) -> str:
    """Nombre propuesto en el nivel D7 (1: semilla o L; 2: + d/ai/wa; 3: + id corto)."""
    params = run["params"]
    family = _family(run)
    version = params.get("version", "?")
    if family == "ref-mean":
        base = f"ref-mean-L{params.get('L', '?')}-v{version}"
    elif family == "ref-pca":
        base = f"ref-pca-d{params.get('d', '?')}-L{params.get('L', '?')}-v{version}"
    else:  # ae
        seed = params.get("seed") or run["tags"].get("seed", "?")
        base = f"{run['name']}-s{seed}"
        if level >= 2:
            aux = "1" if str(params.get("aux_input", "")).lower() in ("true", "1") else "0"
            base += f"-d{params.get('d', '?')}-ai{aux}-wa{params.get('w_aux', '?')}"
    if level >= 3:
        base += f"-r{run['run_id'][:6]}"
    return base


def build_renames(runs: list[dict]) -> list[dict]:
    """Renombres de los grupos duplicados de familias `ref-*` y `sinca-zona-ae_*`."""
    current = Counter(run["name"] for run in runs)
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["name"]].append(run)

    renames: list[dict] = []
    for name, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        families = {_family(member) for member in members}
        if len(families) != 1 or None in families:
            continue  # lightgbm-v1 y demás: no se tocan
        chosen = None
        for level in (1, 2, 3):
            proposed = {member["run_id"]: _name_at(member, level) for member in members}
            values = list(proposed.values())
            if len(set(values)) == len(values) and not any(value in current for value in values):
                chosen = proposed
                break
        if chosen is None:
            log.warning("No hay nombre único para el grupo %r; se omite.", name)
            continue
        for member in members:
            renames.append({"run_id": member["run_id"], "old": name, "new": chosen[member["run_id"]]})
    return renames


def build_deletes(runs: list[dict]) -> list[dict]:
    """Runs `lightgbm` con `version = 99` activos, con su prefijo S3 de artefactos."""
    out = []
    for run in runs:
        if run["tags"].get("modelo") == "lightgbm" and run["params"].get("version") == V99:
            out.append({
                "run_id": run["run_id"],
                "name": run["name"],
                # Ambos layouts: el que tenga objetos en S3 se borra; el otro no existe.
                "s3_prefixes": [
                    f"{S3_PREFIX}/mlruns/{run['run_id']}/",
                    f"{S3_PREFIX}/mlruns/{run['experiment_id']}/{run['run_id']}/",
                ],
            })
    return out


def make_plan(state_dir: str, experiment: str) -> dict:
    runs = active_runs(state_dir, experiment)
    return {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "deletes": build_deletes(runs),
        "renames": build_renames(runs),
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([line, sep, *body])


def render_plan(plan: dict) -> str:
    if not plan["deletes"] and not plan["renames"]:
        return "Sin acciones: no hay runs v99 activos ni duplicados de ref-* / sinca-zona-ae_*."
    parts = []
    if plan["deletes"]:
        parts.append("Borrados (lifecycle deleted + prefijo S3):")
        parts.append(_table(["run_id", "nombre", "prefijo S3"],
                            [[d["run_id"], d["name"], ", ".join(d["s3_prefixes"])] for d in plan["deletes"]]))
    if plan["renames"]:
        parts.append("Renombres:")
        parts.append(_table(["run_id", "nombre actual", "nombre nuevo"],
                            [[r["run_id"], r["old"], r["new"]] for r in plan["renames"]]))
    return "\n\n".join(parts)


# --- Aplicación (sobre copia local) ------------------------------------------


def _tracking_uri(state_dir: str) -> str:
    return f"sqlite:///{pathlib.Path(state_dir).resolve().as_posix()}/{DB_FILENAME}"


def apply_plan(state_dir: str, plan: dict, skip_renames: bool, skip_deletes: bool) -> dict:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=_tracking_uri(state_dir))
    counts = {"renamed": 0, "deleted": 0}
    if not skip_renames:
        for rename in plan["renames"]:
            client.update_run(rename["run_id"], name=rename["new"])
            counts["renamed"] += 1
    if not skip_deletes:
        for delete in plan["deletes"]:
            client.delete_run(delete["run_id"])
            counts["deleted"] += 1
    return counts


def operator_commands(state_dir: str, plan: dict) -> str:
    stamp = dt.date.today().isoformat()
    db = pathlib.Path(state_dir, DB_FILENAME)
    run_ids = ",".join(d["run_id"] for d in plan["deletes"])
    lines = [
        "Comandos del operador (no se ejecutan aquí; reemplazar <AWS_PROFILE> por el profile real):",
        "",
        "1) Respaldo del sqlite de S3 antes de subir nada:",
        f"   aws s3 cp s3://{BUCKET_PLACEHOLDER}/{S3_PREFIX}/{DB_FILENAME} "
        f"s3://{BUCKET_PLACEHOLDER}/{S3_PREFIX}/backups/mlflow-{stamp}.sqlite --profile {PROFILE}",
    ]
    if run_ids:
        lines += [
            "",
            "2) Limpieza de artefactos locales de los runs borrados:",
            f"   mlflow gc --backend-store-uri {_tracking_uri(state_dir)} --run-ids {run_ids}",
            "",
            "3) Prefijos S3 de los runs borrados (dry-run primero, luego real):",
        ]
        for delete in plan["deletes"]:
            for prefix in delete["s3_prefixes"]:
                s3 = f"s3://{BUCKET_PLACEHOLDER}/{prefix}"
                lines.append(f"   aws s3 rm {s3} --recursive --dryrun --profile {PROFILE}")
                lines.append(f"   aws s3 rm {s3} --recursive --profile {PROFILE}")
    lines += [
        "",
        "4) Subida del sqlite modificado (después de revisar los pasos anteriores):",
        f"   aws s3 cp {db} s3://{BUCKET_PLACEHOLDER}/{S3_PREFIX}/{DB_FILENAME} --profile {PROFILE}",
    ]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Mantenimiento del estado MLflow (ae-dmd-prep, D7).")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan", help="Plan de solo lectura (sqlite en modo ro).")
    plan_p.add_argument("--state-dir", required=True)
    plan_p.add_argument("--out", required=True, help="Archivo JSON del plan.")
    plan_p.add_argument("--experiment", default=EXPERIMENT_DEFAULT)

    apply_p = sub.add_parser("apply", help="Aplica el plan sobre el sqlite local (copia).")
    apply_p.add_argument("--state-dir", required=True)
    apply_p.add_argument("--plan", required=True)
    apply_p.add_argument("--yes", action="store_true", help="Sin esta bandera no se escribe nada.")
    apply_p.add_argument("--skip-renames", action="store_true")
    apply_p.add_argument("--skip-deletes", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = make_plan(args.state_dir, args.experiment)
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(render_plan(plan))
        print(f"\nPlan escrito en {out}")
        return 0

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    print(render_plan(plan))
    if not args.yes:
        print("\nNo se escribió nada: falta --yes.")
        return 0
    counts = apply_plan(args.state_dir, plan, args.skip_renames, args.skip_deletes)
    print(f"\nAplicado: {counts}")
    print("\n" + operator_commands(args.state_dir, plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
