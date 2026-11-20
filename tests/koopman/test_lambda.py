"""Tareas 7.2 y 7.9: Lambda reducida a `estado_job` y `resumen` (S3 falso)."""
import importlib.util
import json
import pathlib

import pytest
from botocore.exceptions import ClientError

from sint_koopman import FakeS3

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("handler_decision_rama", ROOT / "infra" / "lambda" / "decision_rama" / "handler.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def test_resumen_de_seleccion_y_sin_modo_rama():
    from types import SimpleNamespace

    import numpy as np

    from koopman import seleccion as S

    rng = np.random.default_rng(0)

    def c(fase, media):
        return {f"{fase}-gru": {"fase": fase, "encoder": "gru", "variante": "principal", "identificable": False,
                                "puntaje": SimpleNamespace(media=media, media_boot=media + 0.3 * rng.normal(size=2000)),
                                "runs": {0: "r"}}}

    s3 = FakeS3()
    key = S.escribir(S.seleccionar({**c("f1", 30.0), **c("f2", 20.0)}), s3, "b", "camp", "seleccion")
    out = H.handler({"modo": "resumen", "bucket": "b", "key": key}, s3=s3)
    assert out["texto"].startswith("Seleccionada f2-gru") and len(out["texto"]) <= H.MAX_TEXTO
    s3.objects["d/x.json"] = json.dumps({"tabla": ["y" * 50] * 40}).encode()  # sin `resumen`: el JSON recortado
    assert len(H.handler({"modo": "resumen", "bucket": "b", "key": "d/x.json"}, s3=s3)["texto"]) == H.MAX_TEXTO
    with pytest.raises(ValueError):
        H.handler({"modo": "rama", "bucket": "b", "key": key}, s3=s3)


def test_marcador_presente_ausente_y_403():
    s3 = FakeS3()
    s3.objects["c/jobs/f1-gru-s0.json"] = b"{}"
    assert H.handler({"modo": "estado_job", "bucket": "b", "key": "c/jobs/f1-gru-s0.json"}, s3=s3) == {"existe": True}
    assert H.handler({"modo": "estado_job", "bucket": "b", "key": "c/jobs/otro.json"}, s3=s3) == {"existe": False}

    class S403(FakeS3):
        def head_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")

    with pytest.raises(ClientError):
        H.handler({"modo": "estado_job", "bucket": "b", "key": "x"}, s3=S403())
    with pytest.raises(ValueError):
        H.handler({"modo": "otro"}, s3=s3)
