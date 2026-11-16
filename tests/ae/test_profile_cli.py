"""Profile explícito en los CLI que tocan S3 (tarea 7.3). Sin AWS: get_client y ensure_state simulados.

Regla del proyecto: nunca el profile default implícito. Sin --profile, el CLI sale con error;
con --profile, get_client recibe ese valor y se pasa como client a ensure_state.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Reemplazo de get_client: guarda el profile recibido y devuelve un cliente de prueba."""

    def __init__(self):
        self.profiles: list[str] = []
        self.client = object()

    def __call__(self, profile=None):
        self.profiles.append(profile)
        return self.client


# --- sel_summary ---------------------------------------------------------------


def test_sel_summary_sin_profile_sale_con_error(tmp_path, capsys):
    from analysis import sel_summary as ss

    with pytest.raises(SystemExit) as exc:
        ss.main(["--partition", "ae-dmd", "--out", str(tmp_path / "x.md")])
    assert exc.value.code == 2
    assert "--profile es obligatorio" in capsys.readouterr().err


def test_sel_summary_pasa_profile_y_client_a_ensure_state(tmp_path, monkeypatch):
    from analysis import sel_summary as ss

    recorder = _Recorder()
    calls = []

    def fake_ensure_state(data_dir=None, run_ids=None, client=None, **_):
        calls.append(client)
        return str(tmp_path)

    monkeypatch.setattr(ss, "get_client", recorder)
    monkeypatch.setattr(ss.analysis_load, "ensure_state", fake_ensure_state)
    assert ss.main(["--partition", "ae-dmd", "--profile", "perfil-personal", "--out", str(tmp_path / "x.md")]) == 0
    assert recorder.profiles == ["perfil-personal"]
    assert calls == [recorder.client]  # el cliente del profile llega a ensure_state


# --- pydmd_check ---------------------------------------------------------------


def test_pydmd_check_state_dir_sin_profile_sale_con_error(capsys):
    mod = _load(ROOT / "scripts" / "ae" / "pydmd_check.py", "pydmd_check")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--state-dir", "algun-dir", "--run-ids", "abc"])
    assert exc.value.code == 2
    assert "--profile es obligatorio" in capsys.readouterr().err


def test_pydmd_check_state_dir_pasa_profile(monkeypatch, tmp_path):
    mod = _load(ROOT / "scripts" / "ae" / "pydmd_check.py", "pydmd_check")
    pytest.importorskip("pydmd")
    recorder = _Recorder()
    import analysis.load as analysis_load
    import persist.s3 as persist_s3

    class Stop(Exception):
        pass

    def fake_ensure_state(data_dir=None, run_ids=None, client=None, **_):
        assert client is recorder.client
        raise Stop

    monkeypatch.setattr(persist_s3, "get_client", recorder)
    monkeypatch.setattr(analysis_load, "ensure_state", fake_ensure_state)
    with pytest.raises(Stop):
        mod.run_rows(str(tmp_path), ["abc"], "perfil-personal")
    assert recorder.profiles == ["perfil-personal"]


def test_pydmd_check_synthetic_no_exige_profile():
    """--synthetic no toca S3: no pide --profile (sale 0 o 1 según pydmd, nunca por el profile)."""
    mod = _load(ROOT / "scripts" / "ae" / "pydmd_check.py", "pydmd_check_synth")
    code = mod.main(["--synthetic"])
    assert code in (0, 1)


# --- lgbm_sources --------------------------------------------------------------


def test_lgbm_sources_sin_profile_sale_con_error(capsys):
    mod = _load(ROOT / "scripts" / "mlflow" / "lgbm_sources.py", "lgbm_sources_cli")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--state-dir", "algun-dir"])
    assert exc.value.code == 2
    assert "--profile es obligatorio" in capsys.readouterr().err


def test_lgbm_sources_usa_el_profile_en_los_comandos(monkeypatch, capsys):
    mod = _load(ROOT / "scripts" / "mlflow" / "lgbm_sources.py", "lgbm_sources_cmds")
    run = {"run_id": "abc123", "experiment_id": "1", "estacion": "Nielol", "pollutant": "pm25",
           "start_ms": 0, "start": "?", "hp": {k: "0" for k in mod.HP_SHOWN}}
    monkeypatch.setattr(mod, "lgbm_v1_runs", lambda state_dir, experiment: [run])
    assert mod.main(["--state-dir", "algun-dir", "--profile", "perfil-personal"]) == 0
    out = capsys.readouterr().out
    assert "--profile perfil-personal" in out
    assert "<AWS_PROFILE>" not in out
