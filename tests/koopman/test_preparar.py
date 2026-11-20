"""Tareas 7.3 y 7.10: entrada de la ejecución (lista fija de D17)."""
import importlib.util
import json
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
    assert list(f) == ["f1", "b", "f2", "f3", "seleccion", "cierre", "test"]
    assert [len(f[k]) for k in f] == [12, 15, 6, 12, 1, 1, 1] and sum(len(v) for v in f.values()) == 48
    assert {(j["hp"]["encoder"], j["hp"]["variantes"]) for j in f["f3"]} == {
        (e_, v) for e_ in ("gru", "conv1ds") for v in ("principal", "icap")}
    assert all("dry" not in j["hp"] for jobs in f.values() for j in jobs)
    assert f["seleccion"][0]["hp"]["seeds"] == "0,1,2" and f["test"][0]["clave"] == "test"
    assert e["resumenes"]["seleccion"] == "model-artifacts/orquestacion/deep-koopman/dk-1/seleccion.json"
    assert "desde" not in e and "decisiones" not in e
    assert not any("rama" in k for jobs in f.values() for j in jobs for k in j["hp"])


def test_padres_desde_el_marcador_de_la_fase_previa():
    f = _entrada()["fases"]
    claves = {j["clave"]: j for jobs in f.values() for j in jobs}
    for j in f["f2"] + f["f3"]:
        padre = j["hp"]["parent-from"].rsplit("/", 1)[-1][:-5]
        previa = "f1" if j["hp"]["fase"] == "f2" else "f2"
        assert padre == f"{previa}-{j['hp']['encoder']}-s{j['hp']['seed']}" and padre in claves
        assert j["hp"]["parent-from"] == f"s3://sinca-mlflow/{claves[padre]['marcador']}"


def test_nombres_y_claves():
    e1, e2 = _entrada("aaaa1111"), _entrada("bbbb2222")
    for jobs1, jobs2 in zip(e1["fases"].values(), e2["fases"].values()):
        for j1, j2 in zip(jobs1, jobs2):
            assert j1["nombre"].startswith("sinca-koop-") and len(j1["nombre"]) <= 63
            assert j1["clave"] == j2["clave"] and j1["ckpt_uri"] == j2["ckpt_uri"] and j1["marcador"] == j2["marcador"]
            assert j1["nombre"] != j2["nombre"]
            assert j1["ckpt_uri"].endswith(f"/dk-1/{j1['clave']}/")
            assert j1["hp"]["sagemaker_program"] == "koopman/run.py"
    claves = [j["clave"] for jobs in e1["fases"].values() for j in jobs]
    assert len(claves) == len(set(claves))


def test_modo_seco_una_semilla_y_fallo():
    e = _entrada(modo="seco", fallar_en=P.parse_fallar("f1:3:2"), dry_ckpt=True)
    f = e["fases"]
    assert e["seeds"] == [0] and [len(f[k]) for k in f] == [4, 5, 2, 4, 1, 1, 1]  # 18 jobs
    assert all(j["hp"]["dry"] == "true" and j["hp"]["experiment"] == "sinca-smoke" for jobs in f.values() for j in jobs)
    assert all(j["hp"]["seed"] == "0" for k in ("f1", "b", "f2", "f3") for j in f[k])
    assert f["f1"][2]["hp"]["dry-fallar-en"] == "2" and "dry-fallar-en" not in f["f1"][1]["hp"]
    assert all(j["hp"].get("dry-ckpt") == "true" for j in f["f1"])
    assert all(j["max_run"] == P.H // 2 for jobs in f.values() for j in jobs)


def test_plan_sin_aws(tmp_path, capsys):
    out = tmp_path / "e.json"
    assert P.main(["--modo", "seco", "--campana", "dk-s", "--plan", "--exec-id", "x1", "--salida", str(out),
                   "--dry-ckpt", "--fallar-en", "f1:3:2"]) == 0
    assert json.loads(out.read_text())["seeds"] == [0] and "start-execution" in capsys.readouterr().out
    data, sha = P.empaquetar()
    assert P.empaquetar()[1] == sha and len(data) > 1000
