"""Tarea 7.3: entrada de la ejecución."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("preparar_ejecucion", ROOT / "scripts" / "koopman" / "preparar_ejecucion.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def _entrada(exec_id="abc12345", modo="real", **kw):
    return P.construir_entrada("dk-1", modo, "s3://b/code.tar.gz", "img@sha256:x", "arn:rol", exec_id, **kw)


def test_conteos_modo_real():
    e = _entrada()
    f = e["fases"]
    assert len(f["f1"]) == 12 and len(f["b1"]) == 6
    assert len([j for j in f["f3"] if j["hp"]["encoder"] == "gru"]) == 6  # principal + icap
    assert {j["hp"]["variantes"] for j in f["f3"] if j["hp"]["encoder"] == "gru"} == {"principal", "icap"}
    assert len(f["compuerta_f1"]) == 1 and f["compuerta_f1"][0]["marcador"] == e["decisiones"]["f1"]
    assert all("dry" not in j["hp"] for jobs in f.values() for j in jobs)


def test_nombres_y_claves():
    e1, e2 = _entrada("aaaa1111"), _entrada("bbbb2222")
    for jobs1, jobs2 in zip(e1["fases"].values(), e2["fases"].values()):
        for j1, j2 in zip(jobs1, jobs2):
            assert j1["nombre"].startswith("sinca-koop-") and len(j1["nombre"]) <= 63
            assert j1["clave"] == j2["clave"] and j1["ckpt_uri"] == j2["ckpt_uri"] and j1["marcador"] == j2["marcador"]
            assert j1["nombre"] != j2["nombre"]
            assert j1["ckpt_uri"].endswith(f"/dk-1/{j1['clave']}/")
            assert j1["hp"]["sagemaker_program"] == "koopman/run.py"


def test_modo_seco_ramas_y_fallo():
    e = _entrada(modo="seco", ramas_forzadas=P.parse_ramas("f1=b1,b1=pasa,f2=pausa"),
                 fallar_en=P.parse_fallar("f1:8:2"), dry_ckpt=True)
    f = e["fases"]
    assert all(j["hp"]["dry"] == "true" and j["hp"]["experiment"] == "sinca-smoke" for jobs in f.values() for j in jobs)
    assert f["compuerta_f1"][0]["hp"]["rama-forzada"] == "b1" and f["compuerta_f2"][0]["hp"]["rama-forzada"] == "pausa"
    assert "rama-forzada" not in f["compuerta_f3"][0]["hp"]
    assert f["f1"][7]["hp"]["dry-fallar-en"] == "2" and "dry-fallar-en" not in f["f1"][6]["hp"]
    assert all(j["hp"].get("dry-ckpt") == "true" for j in f["f1"])


def test_plan_sin_aws(tmp_path, capsys):
    out = tmp_path / "e.json"
    assert P.main(["--modo", "real", "--campana", "dk-1", "--plan", "--exec-id", "x1", "--salida", str(out)]) == 0
    assert out.exists() and "start-execution" in capsys.readouterr().out
    data, sha = P.empaquetar()
    assert P.empaquetar()[1] == sha and len(data) > 1000
