"""Lambda de la orquestación de `deep-koopman` (D18). Sin reglas de decisión (viven en `seleccion.py`).

- `modo = "estado_job"`: `HeadObject` del marcador de éxito del job: existe -> `{"existe": true}`,
  404 -> `{"existe": false}`; cualquier otro error falla la tarea.
- `modo = "resumen"`: lee un JSON del prefijo de orquestación (`seleccion.json`, `cierre.json`, `test.json`)
  y devuelve `{"texto": ...}` para el aviso SNS: su campo `resumen` si lo trae, si no el JSON; ≤ 800 caracteres.
"""
import json

MAX_TEXTO = 800

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.client("s3")
    return _S3


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


def resumen(evento, s3):
    obj = s3.get_object(Bucket=evento["bucket"], Key=evento["key"])
    contenido = json.loads(obj["Body"].read())
    texto = contenido.get("resumen") if isinstance(contenido, dict) else None
    if not isinstance(texto, str) or not texto:
        texto = json.dumps(contenido, ensure_ascii=False, default=str)
    return {"texto": texto[:MAX_TEXTO]}


def handler(evento, contexto=None, s3=None):
    s3 = s3 or _s3()
    modo = evento.get("modo")
    if modo == "estado_job":
        return estado_job(evento, s3)
    if modo == "resumen":
        return resumen(evento, s3)
    raise ValueError(f"Modo desconocido: {modo!r}")
