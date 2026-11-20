"""Arma la entrada de la máquina `sinca-deep-koopman` (D17, D18, D20) e imprime el `start-execution`.

    python scripts/koopman/preparar_ejecucion.py --modo real --campana dk-1 --plan            # sin AWS
    python scripts/koopman/preparar_ejecucion.py --modo seco --campana dk-seco-1 \\
        --ramas-forzadas "f1=b1,b1=pasa,f2=pausa" --profile <AWS_PROFILE> \\
        --image-uri <registro>/sinca-train:tf-cpu-v2 --role-arn <training_role_arn> \\
        --state-machine-arn <arn> --salida output/deep-koopman/entrada.json

Sin `--plan`: empaqueta `pipeline/` en `sourcedir.tar.gz` y lo sube a `_mlflow/code/deep-koopman/<sha>/`,
y resuelve el digest de la imagen (ECR). Con `--plan` no toca AWS (placeholders).
Claves de job estables por campaña (`<fase>-<enc><var>-s<seed>`); nombres únicos por ejecución
(`sinca-koop-<clave>-<id>`, ≤ 63); checkpoints en `model-artifacts/checkpoints/deep-koopman/<campaña>/<clave>/`.
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
MAX_RUN = {"smoke": H, "f1": 3 * H, "decgru": 4 * H, "variantes": 6 * H, "b": 3 * H, "f2": 3 * H, "f3": 3 * H,
           "compuerta": H, "cierre": 3 * H, "test": H}
GATES = ("f1", "b1", "b2", "b3", "b4", "f2", "f3")


def _job(campana: str, exec_id: str, clave: str, hp: dict, max_run: int, marcador: str | None = None) -> dict:
    nombre = f"{PREFIJO_JOB}{clave}-{exec_id}"
    if len(nombre) > 63:
        raise ValueError(f"Nombre de job > 63: {nombre}")
    return {"clave": clave, "nombre": nombre, "hp": {k: str(v) for k, v in hp.items()}, "max_run": int(max_run),
            "ckpt_uri": f"s3://{BUCKET}/{CKPT}/{campana}/{clave}/",
            "marcador": marcador or f"{ORQ}/{campana}/jobs/{clave}.json"}


def construir_entrada(campana: str, modo: str, submit_dir: str, imagen: str, rol: str, exec_id: str,
                      ramas_forzadas: dict | None = None, fallar_en: tuple | None = None, dry_ckpt: bool = False,
                      desde: str = "f1") -> dict:
    seco = modo == "seco"
    base = {"sagemaker_program": "koopman/run.py", "sagemaker_submit_directory": submit_dir,
            "sagemaker_region": "us-east-1", "campana": campana,
            "experiment": "sinca-smoke" if seco else "sinca-baseline"}
    if seco:
        base["dry"] = "true"
    fases: dict[str, list] = {}

    def add(fase, clave, hp, max_run, marcador=None):
        fases.setdefault(fase, []).append(_job(campana, exec_id, clave, {**base, **hp, "clave": clave},
                                               H // 2 if seco else max_run, marcador))

    for s in SEEDS:
        add("f1", f"f1-gru-s{s}", {"fase": "f1", "encoder": "gru", "seed": s, "variantes": "principal"}, MAX_RUN["f1"])
    for s in SEEDS:
        add("f1", f"f1-conv1ds-s{s}", {"fase": "f1", "encoder": "conv1ds", "seed": s, "variantes": "principal"}, MAX_RUN["f1"])
    for s in SEEDS:
        add("f1", f"f1-grudecgru-s{s}", {"fase": "f1", "encoder": "gru", "seed": s, "variantes": "decgru"}, MAX_RUN["decgru"])
    for s in SEEDS:
        add("f1", f"f1-grukdenperlibsinbu-s{s}", {"fase": "f1", "encoder": "gru", "seed": s, "variantes": "kden,perlib,sinbu"},
            MAX_RUN["variantes"])
    for s in SEEDS:
        for v in ("ll01", "ll10"):
            add("b1", f"b1-gru{v}-s{s}", {"fase": "b1", "encoder": "gru", "seed": s, "variantes": v}, MAX_RUN["b"])
    for b in ("b2", "b3", "b4"):
        for s in SEEDS:
            add(b, f"{b}-gru-s{s}", {"fase": b, "encoder": "gru", "seed": s, "variantes": "principal"}, MAX_RUN["b"])
    arranque = f"s3://{BUCKET}/{ORQ}/{campana}/arranque_f2.json"
    for enc in ("gru", "conv1ds"):  # conv1d solo corre si la compuerta lo deja seguir (si no, el job se omite)
        for s in SEEDS:
            add("f2", f"f2-{enc}-s{s}", {"fase": "f2", "encoder": enc, "seed": s, "variantes": "principal",
                                         "parent-from": arranque}, MAX_RUN["f2"])
    p_f3 = f"s3://{BUCKET}/{ORQ}/{campana}/f2.json"
    for s in SEEDS:
        for v in ("principal", "icap"):
            add("f3", f"f3-gru{'' if v == 'principal' else v}-s{s}",
                {"fase": "f3", "encoder": "gru", "seed": s, "variantes": v, "parent-from": p_f3}, MAX_RUN["f3"])
    for s in SEEDS:
        add("f3", f"f3-conv1ds-s{s}", {"fase": "f3", "encoder": "conv1ds", "seed": s, "variantes": "principal",
                                       "parent-from": p_f3}, MAX_RUN["f3"])
    for s in SEEDS:
        add("cierre", f"cierre-gru-s{s}", {"fase": "cierre", "encoder": "gru", "seed": s,
                                          "parent-from": f"s3://{BUCKET}/{ORQ}/{campana}/f3.json"}, MAX_RUN["cierre"])
    add("test", "test-gru", {"fase": "test", "encoder": "gru", "seeds": "0,1,2"}, MAX_RUN["test"])
    decisiones = {}
    for g in GATES:
        hp = {"fase": "compuerta", "de": g, "seeds": "0,1,2"}
        if seco:
            rf = (ramas_forzadas or {}).get(g)
            if rf:
                hp["rama-forzada"] = rf
        key = f"{ORQ}/{campana}/{g}.json"
        add(f"compuerta_{g}", f"compuerta-{g}", hp, MAX_RUN["compuerta"], marcador=key)
        decisiones[g] = key
    if seco and dry_ckpt:
        for j in fases["f1"]:
            j["hp"]["dry-ckpt"] = "true"
    if fallar_en:
        fase, n, ep = fallar_en
        fases[fase][int(n) - 1]["hp"]["dry-fallar-en"] = str(ep)
    comun = {"imagen": imagen, "rol": rol, "bucket": BUCKET, "instancia": "ml.m5.large",
             "salida": f"s3://{BUCKET}/model-artifacts/deep-koopman/{campana}"}
    return {"campana": campana, "modo": modo, "desde": desde, "comun": comun, "fases": fases, "decisiones": decisiones}


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


def parse_ramas(text: str | None) -> dict:
    out = {}
    for par in (text or "").split(","):
        if par.strip():
            k, v = par.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_fallar(text: str | None):
    if not text:
        return None
    fase, n, ep = text.split(":")
    return fase, int(n), int(ep)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Entrada de la campaña deep-koopman para Step Functions.")
    p.add_argument("--modo", choices=("real", "seco"), required=True)
    p.add_argument("--campana", required=True)
    p.add_argument("--ramas-forzadas", default=None, help='Modo seco: "f1=b1,b1=pasa,f2=pausa".')
    p.add_argument("--fallar-en", default=None, help="Modo seco: <fase>:<n.º de job>:<época> (p. ej. f1:8:2).")
    p.add_argument("--dry-ckpt", action="store_true")
    p.add_argument("--desde", default="f1", choices=("f1", "f2", "f3", "cierre", "test"))
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
                                parse_ramas(a.ramas_forzadas), parse_fallar(a.fallar_en), a.dry_ckpt, a.desde)
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
