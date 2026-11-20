"""Lambda de la orquestación de `deep-koopman` (D18). Sin reglas de decisión.

- `modo = "rama"`: lee el JSON de decisión que escribió el job de compuerta, valida `rama` contra la
  lista permitida y devuelve `{rama, resumen}`.
- `modo = "estado_job"`: `HeadObject` del marcador de éxito del job: existe -> `{"existe": true}`,
  404 -> `{"existe": false}`; cualquier otro error falla la tarea.
"""
import json

RAMAS = ("pasa", "b1", "b2", "b3", "b4", "c", "pausa", "continuar")

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.client("s3")
    return _S3


class RamaInvalida(Exception):
    pass


def leer_rama(evento, s3):
    obj = s3.get_object(Bucket=evento["bucket"], Key=evento["key"])
    decision = json.loads(obj["Body"].read())
    rama = decision.get("rama")
    if rama not in RAMAS:
        raise RamaInvalida(f"Rama no permitida en {evento['key']}: {rama!r}")
    resumen = decision.get("motivo", "")
    cifras = decision.get("cifras")
    if cifras:
        resumen = f"{resumen} | cifras: {json.dumps(cifras, ensure_ascii=False)[:800]}"
    return {"rama": rama, "resumen": resumen}


def estado_job(evento, s3):
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=evento["bucket"], Key=evento["key"])
    except ClientError as exc:
        codigo = str(exc.response.get("Error", {}).get("Code", ""))
        if codigo in ("404", "NoSuchKey", "NotFound"):
            return {"existe": False}
        raise
    return {"existe": True}


def handler(evento, contexto=None, s3=None):
    s3 = s3 or _s3()
    modo = evento.get("modo")
    if modo == "rama":
        return leer_rama(evento, s3)
    if modo == "estado_job":
        return estado_job(evento, s3)
    raise ValueError(f"Modo desconocido: {modo!r}")
