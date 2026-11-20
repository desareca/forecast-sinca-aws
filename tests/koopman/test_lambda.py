"""Tarea 7.2: Lambda de decisión (S3 falso)."""
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


def test_rama_valida_y_desconocida():
    s3 = FakeS3()
    s3.objects["d/f1.json"] = json.dumps({"rama": "b1", "motivo": "falla", "cifras": {"gru": {"wrmse": 30}}}).encode()
    out = H.handler({"modo": "rama", "bucket": "b", "key": "d/f1.json"}, s3=s3)
    assert out["rama"] == "b1" and "falla" in out["resumen"]
    s3.objects["d/f2.json"] = json.dumps({"rama": "otra"}).encode()
    with pytest.raises(H.RamaInvalida):
        H.handler({"modo": "rama", "bucket": "b", "key": "d/f2.json"}, s3=s3)


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
