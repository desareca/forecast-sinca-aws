"""Higiene de profile del lanzador (tarea 13.2a): `--profile` obligatorio, sin AWS_PROFILE implícito."""
import pathlib
import sys

import pytest

PIPELINE = pathlib.Path(__file__).resolve().parents[2] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from train import launch  # noqa: E402


def test_profile_no_toma_aws_profile_del_entorno(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "perfil-ajeno")
    args = launch.parse_args([])
    assert args.profile is None


def test_main_sin_profile_falla_con_mensaje_claro(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "perfil-ajeno")
    with pytest.raises(SystemExit, match="Falta --profile"):
        launch.main(["--role-arn", "arn:aws:iam::123456789012:role/x", "--image-uri", "repo:tf-cpu-v1"])


def test_profile_explicito_se_respeta(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "perfil-ajeno")
    args = launch.parse_args(["--profile", "personal"])
    assert args.profile == "personal"


def test_entry_point_koopman():
    args = launch.parse_args(["--entry-point", "koopman/run.py"])
    assert args.entry_point in launch.ENTRY_POINTS
    hp = launch.base_hyperparameters("koopman/run.py", args)
    assert set(hp) == {"data-bucket", "mlflow-bucket", "mlflow-prefix"}
    assert args.checkpoint_s3_uri is None
    uri = "s3://b/model-artifacts/checkpoints/deep-koopman/humo/smoke-gru-s0/"
    assert launch.parse_args(["--checkpoint-s3-uri", uri]).checkpoint_s3_uri == uri
