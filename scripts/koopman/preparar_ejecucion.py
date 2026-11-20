"""Arma la entrada de la máquina `sinca-deep-koopman` (D17, D18, D20) e imprime el `start-execution`.

    python scripts/koopman/preparar_ejecucion.py --modo real --campana dk-1 --plan            # sin AWS
    python scripts/koopman/preparar_ejecucion.py --modo seco --campana dk-seco-1 --dry-ckpt --fallar-en f1:3:2 \\
        --profile <AWS_PROFILE> --image-uri <registro>/sinca-train:tf-cpu-v2 --role-arn <training_role_arn> \\
        --state-machine-arn <arn> --salida output/deep-koopman/entrada.json

Lista fija de D17 en orden (48 jobs en modo real; modo seco con `seeds = [0]`: 18 jobs). Sin `--plan`:
empaqueta `pipeline/` en `sourcedir.tar.gz` y lo sube a `_mlflow/code/deep-koopman/<sha>/`, y resuelve
el digest de la imagen (ECR). Con `--plan` no toca AWS (placeholders). Claves de job estables por
campaña (`<fase>-<enc><var>-s<seed>`; `seleccion`, `cierre`, `test`); nombres únicos por ejecución
(`sinca-koop-<clave>-<id>`, ≤ 63); checkpoints en `model-artifacts/checkpoints/deep-koopman/<campaña>/<clave>/`.
F2/F3 toman su padre del marcador del job de F1/F2 con el mismo encoder y semilla.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUCKET = "sinca-mlflow"
ORQ = "model-artifacts/orquestacion/deep-koopman"
CKPT = "model-artifacts/checkpoints/deep-koopman"
CODE = "_mlflow/code/deep-koopman"
PREFIJO_JOB = "sinca-koop-"
SEEDS = (0, 1, 2)
H = 3600
# Re-humo 8.2 (paso compilado), estimado por job ≤ 0.46 h salvo variantes 0.79 h: margen ≥ 2×.
MAX_RUN = {"smoke": H, "f1": H, "decgru": H, "variantes": 2 * H, "b": H, "f2": H, "f3": H,
           "seleccion": H, "cierre": H, "test": H}
FASES = ("f1", "b", "f2", "f3", "seleccion", "cierre", "test")  # orden de la máquina (D18)
RESUMENES = ("seleccion", "cierre", "test")  # JSON que la Lambda `resumen` lee para los avisos


def _job(campana: str, exec_id: str, clave: str, hp: dict, max_run: int) -> dict:
    nombre = f"{PREFIJO_JOB}{clave}-{exec_id}"
    if len(nombre) > 63:
        raise ValueError(f"Nombre de job > 63: {nombre}")
    return {"clave": clave, "nombre": nombre, "hp": {k: str(v) for k, v in hp.items()}, "max_run": int(max_run),
            "ckpt_uri": f"s3://{BUCKET}/{CKPT}/{campana}/{clave}/",
            "marcador": f"{ORQ}/{campana}/jobs/{clave}.json"}


def marcador_uri(campana: str, clave: str) -> str:
    return f"s3://{BUCKET}/{ORQ}/{campana}/jobs/{clave}.json"


def construir_entrada(campana: str, modo: str, submit_dir: str, imagen: str, rol: str, exec_id: str,
                      fallar_en: tuple | None = None, dry_ckpt: bool = False, seeds=None) -> dict:
    seco = modo == "seco"
    seeds = tuple(seeds) if seeds is not None else ((0,) if seco else SEEDS)
    sem = ",".join(str(s) for s in seeds)
    base = {"sagemaker_program": "koopman/run.py", "sagemaker_submit_directory": submit_dir,
            "sagemaker_region": "us-east-1", "campana": campana,
            "experiment": "sinca-smoke" if seco else "sinca-baseline"}
    if seco:
        base["dry"] = "true"
    fases: dict[str, list] = {f: [] for f in FASES}

    def add(fase, clave, hp, max_run):
        fases[fase].append(_job(campana, exec_id, clave, {**base, **hp, "clave": clave}, H // 2 if seco else max_run))

    def ent(fase, enc, s, v="principal", **extra):
        return {"fase": fase, "encoder": enc, "seed": s, "variantes": v, **extra}

    # F1: GRU, conv1d, GRU + decoder GRU y las 3 variantes de K (un job por semilla)
    for s in seeds:
        add("f1", f"f1-gru-s{s}", ent("f1", "gru", s), MAX_RUN["f1"])
    for s in seeds:
        add("f1", f"f1-conv1ds-s{s}", ent("f1", "conv1ds", s), MAX_RUN["f1"])
    for s in seeds:
        add("f1", f"f1-grudecgru-s{s}", ent("f1", "gru", s, "decgru"), MAX_RUN["decgru"])
    for s in seeds:
        add("f1", f"f1-grukdenperlibsinbu-s{s}", ent("f1", "gru", s, "kden,perlib,sinbu"), MAX_RUN["variantes"])
    # B1–B4 (siempre, D12)
    for s in seeds:
        for v in ("ll01", "ll10"):
            add("b", f"b1-gru{v}-s{s}", ent("b1", "gru", s, v), MAX_RUN["b"])
    for b in ("b2", "b3", "b4"):
        for s in seeds:
            add("b", f"{b}-gru-s{s}", ent(b, "gru", s), MAX_RUN["b"])
    # F2 y F3 por encoder; padre = marcador del job previo con el mismo encoder y semilla
    for enc in ("gru", "conv1ds"):
        for s in seeds:
            add("f2", f"f2-{enc}-s{s}", ent("f2", enc, s, **{"parent-from": marcador_uri(campana, f"f1-{enc}-s{s}")}),
                MAX_RUN["f2"])
    for enc in ("gru", "conv1ds"):
        for v in ("principal", "icap"):
            for s in seeds:
                add("f3", f"f3-{enc}{'' if v == 'principal' else v}-s{s}",
                    ent("f3", enc, s, v, **{"parent-from": marcador_uri(campana, f"f2-{enc}-s{s}")}), MAX_RUN["f3"])
    for f in ("seleccion", "cierre", "test"):
        add(f, f, {"fase": f, "seeds": sem}, MAX_RUN[f])
    if seco and dry_ckpt:
        for j in fases["f1"]:
            j["hp"]["dry-ckpt"] = "true"
    if fallar_en:
        fase, n, ep = fallar_en
        fases[fase][int(n) - 1]["hp"]["dry-fallar-en"] = str(ep)
    comun = {"imagen": imagen, "rol": rol, "bucket": BUCKET, "instancia": "ml.m5.large",
             "salida": f"s3://{BUCKET}/model-artifacts/deep-koopman/{campana}"}
    return {"campana": campana, "modo": modo, "seeds": list(seeds), "comun": comun, "fases": fases,
            "resumenes": {r: f"{ORQ}/{campana}/{r}.json" for r in RESUMENES}}


def empaquetar(pipeline_dir: pathlib.Path = ROOT / "pipeline") -> tuple[bytes, str]:
    """`sourcedir.tar.gz` reproducible de `pipeline/` (sin `__pycache__`) y su sha corto."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for path in sorted(p for p in pipeline_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            info = tar.gettarinfo(str(path), arcname=path.relative_to(pipeline_dir).as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()[:12]


def resolver_imagen(image_uri: str, session) -> str:
    """`<repo>:<tag>` -> `<repo>@sha256:...` (ECR `describe_images`)."""
    repo_uri, tag = image_uri.rsplit(":", 1)
    repo = repo_uri.split("/", 1)[1]
    det = session.client("ecr").describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    return f"{repo_uri}@{det['imageDigest']}"


def parse_fallar(text: str | None):
    if not text:
        return None
    fase, n, ep = text.split(":")
    return fase, int(n), int(ep)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Entrada de la campaña deep-koopman para Step Functions.")
    p.add_argument("--modo", choices=("real", "seco"), required=True)
    p.add_argument("--campana", required=True)
    p.add_argument("--fallar-en", default=None, help="Modo seco: <fase>:<n.º de job>:<época> (p. ej. f1:3:2).")
    p.add_argument("--dry-ckpt", action="store_true")
    p.add_argument("--seeds", default=None, help="Semillas separadas por coma (por defecto: real 0,1,2; seco 0).")
    p.add_argument("--plan", action="store_true", help="Sin AWS: placeholders de imagen, rol y código.")
    p.add_argument("--profile", default=None)
    p.add_argument("--image-uri", default=None, help="<registro>/sinca-train:tf-cpu-v2")
    p.add_argument("--role-arn", default=None)
    p.add_argument("--state-machine-arn", default="<state_machine_arn>")
    p.add_argument("--exec-id", default=None)
    p.add_argument("--salida", default=None)
    a = p.parse_args(argv)
    exec_id = a.exec_id or uuid.uuid4().hex[:8]
    data, sha = empaquetar()
    key = f"{CODE}/{sha}/sourcedir.tar.gz"
    if a.plan:
        imagen, rol = a.image_uri or "<registro>/sinca-train@sha256:<digest>", a.role_arn or "<training_role_arn>"
    else:
        if not (a.profile and a.image_uri and a.role_arn):
            p.error("Sin --plan se requieren --profile, --image-uri y --role-arn")
        import boto3

        session = boto3.Session(profile_name=a.profile, region_name="us-east-1")
        session.client("s3").put_object(Bucket=BUCKET, Key=key, Body=data)
        imagen, rol = resolver_imagen(a.image_uri, session), a.role_arn
    entrada = construir_entrada(a.campana, a.modo, f"s3://{BUCKET}/{key}", imagen, rol, exec_id,
                                parse_fallar(a.fallar_en), a.dry_ckpt,
                                [int(s) for s in a.seeds.split(",")] if a.seeds else None)
    texto = json.dumps(entrada, indent=2)
    salida = pathlib.Path(a.salida or f"output/deep-koopman/entrada-{a.campana}-{exec_id}.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    n = {f: len(v) for f, v in entrada["fases"].items()}
    print(f"Entrada: {salida} ({sum(n.values())} jobs: {n})")
    print("Comando (WSL):")
    print(f"  aws stepfunctions start-execution --state-machine-arn {a.state_machine_arn} "
          f"--name {a.campana}-{exec_id} --input file://{salida.as_posix()} --profile <AWS_PROFILE>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
