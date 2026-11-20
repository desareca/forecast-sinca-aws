"""Tarea 1.9 (y 2.2): CLI de punta a punta sobre un estado v2 sintético (`--no-download`, sin AWS).

Patrón de `tests/ae/test_run_partition_e2e.py`: 280 días sintéticos; AE conv1d y GRU sin entrenar
(pesos aleatorios, `d = 8`), sus latentes y un pase DMD v2 armado con la misma cadena.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from sint_operador import ARTIFACT_LOCATION, EXPERIMENT, synthetic_validated  # noqa: E402


def _log(state, name, tags, params, frames, files=None, arrays=None):
    import shutil

    from operador import registro as G

    run_id = G.log_variante(str(state), name, tags, params, {}, frames, arrays)
    for dest, src in (files or {}).items():
        shutil.copy(src, G.artifacts_dir(str(state), run_id) / dest)
    return run_id


@pytest.fixture(scope="module")
def estado_v2(tmp_path_factory):
    import mlflow

    from ae import data as aed
    from ae import dmd as dmdm
    from ae import models as aeml
    from baseline.persistence import predict_frame
    from features import partition as pt
    from features.build import build_features
    from operador import estado as E

    tmp = tmp_path_factory.mktemp("operador_e2e")
    data = synthetic_validated(start="2025-11-01", hours=24 * 280, gaps=True)
    validated = tmp / "validated"
    for key in ("sinca", "open_meteo", "feriados"):
        (validated / key).mkdir(parents=True)
        data[key].to_parquet(validated / key / "part-0.parquet", index=False)
    state = tmp / "mlflow-op"
    state.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{(state / 'mlflow.sqlite').as_posix()}")
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)

    grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
    part = pt.get_partition("ae-dmd")
    ctx = E.contexto(grid, part)
    features = build_features(data["sinca"], data["open_meteo"], data["feriados"])
    targets = features[["timestamp", "estacion", "pollutant", "target"]]
    pers = predict_frame(data["sinca"], features)
    pers = pers[pt.row_mask(pers["timestamp"], part, "sel")].assign(split="sel")
    common = {"particion": "ae-dmd", "stage": "ae-dmd-prep"}
    _log(state, "sinca-zona-persistence-aedmd-v2", {**common, "modelo": "persistence-aedmd"}, {"version": 2, "tipo": "referencia"},
         {"predictions.parquet": pers})
    pers_test = predict_frame(data["sinca"], features)
    pers_test = pers_test[pt.row_mask(pers_test["timestamp"], part, "test")].assign(split="test")
    for (station, pollutant), g in pers.groupby(["estacion", "pollutant"]):
        gt = pers_test[(pers_test["estacion"] == station) & (pers_test["pollutant"] == pollutant)]
        src = _log(state, f"sinca-{station}-lightgbm-v1", {"modelo": "lightgbm"}, {"version": 1, "estacion": station,
                   "pollutant": pollutant}, {"predictions.parquet": gt.assign(y_pred=gt["y_pred"] * 1.05)})
        _log(state, f"lgbm-{station}-{pollutant}", {**common, "modelo": "lightgbm-aedmd"},
             {"version": 2, "estacion": station, "pollutant": pollutant, "source_run_id": src},
             {"predictions.parquet": g.assign(y_pred=g["y_pred"] * 1.1)})

    pos, split = E.posiciones(ctx, 24)
    win = aed.build_windows(ctx.ng, 24, pos)
    for arch in ("conv1d", "gru"):
        cfg = aeml.AEConfig(arch=arch, length=24, d=8, hidden=8, seed=0)
        models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
        z = aeml.encode(models, cfg, win.x_mp, win.x_aux)
        latent = pd.DataFrame(z, columns=[f"z_mp_{i}" for i in range(6)] + [f"z_aux_{j}" for j in range(2)])
        latent.insert(0, "timestamp", grid.index[pos])
        latent.insert(1, "split", split)
        tr = split == "train"
        fit = dmdm.fit_dmd(dmdm.split_segments(z[tr], pos[tr]), None)
        se = (split == "sel") & pt.row_mask(grid.index[pos], part, "sel")
        dec = aed.mp_to_ugm3(aeml.decode_mp(models, dmdm.predict_latent(fit["K"], z[se], 24)), ctx.norm)
        fc = dmdm.forecast_from_decoded(dec)
        pieces = [pd.DataFrame({"timestamp": grid.index[pos[se]], "estacion": s, "pollutant": p, "y_pred": fc[:, c]})
                  for c, (_, s, p) in enumerate(aed.mp_channels())]
        preds = pd.concat(pieces).merge(targets, on=["timestamp", "estacion", "pollutant"]).rename(columns={"target": "y_true"})
        preds["split"] = "sel"
        models.encoder.save(tmp / f"enc-{arch}.keras")
        models.decoder_mp.save(tmp / f"dec-{arch}.keras")
        _log(state, f"sinca-zona-ae_{arch}-aedmd-v2-s0", {**common, "modelo": f"ae-{arch}", "seed": "0"},
             {"version": 2, "aux_input": True}, {"latent.parquet": latent},
             files={"encoder.keras": tmp / f"enc-{arch}.keras", "decoder_mp.keras": tmp / f"dec-{arch}.keras"},
             arrays={"dmd_operator.npz": dmdm.dmd_operator_arrays(fit)})
        for (station, pollutant), g in preds.groupby(["estacion", "pollutant"]):
            _log(state, f"dmd-{arch}-{station}-{pollutant}", {**common, "modelo": f"ae-{arch}-dmd", "seed": "0"},
                 {"version": 2, "estacion": station, "pollutant": pollutant}, {"predictions.parquet": g})
    return tmp, state, validated


def _argv(tmp, state, validated, etapa, *extra):
    return ["--etapa", etapa, "--state-dir", str(state), "--local-validated-dir", str(validated), "--no-download",
            "--output-dir", str(tmp / "out"), "--decisiones", str(tmp / "decisiones.json"), "--bootstrap", "50", *extra]


def test_base_y_oraculo(estado_v2):
    from analysis import load as analysis_load
    from operador import registro as G
    from operador import run as R

    tmp, state, validated = estado_v2
    (tmp / "decisiones.json").write_text("{}")
    assert R.run(R.parse_args(_argv(tmp, state, validated, "base"))) == 0
    rows = pd.read_parquet(tmp / "out" / R.EVAL_ROWS)
    assert rows.groupby(["estacion", "pollutant"]).ngroups == 6
    cont = json.loads((tmp / "out" / "base.json").read_text())["continuidad"]
    assert len(cont) == 2
    for row in cont:
        assert row["K_rel"] <= 1e-8 and row["ypred_max_rel"] <= 1e-4
    assert not G.runs_de_etapa(str(state))  # base no registra runs

    assert R.run(R.parse_args(_argv(tmp, state, validated, "1.0", "--seeds", "0"))) == 0
    runs = G.runs_de_etapa(str(state), "1.0")
    assert {r["run_name"] for r in runs} == {"sinca-zona-dmdop-conv1d-e10-aedmd-v1-s0", "sinca-zona-dmdop-gru-e10-aedmd-v1-s0",
                                             "sinca-zona-dmdop-pca-e10-aedmd-v1-s0"}
    for run in runs:
        assert run["params"]["tipo"] == "diagnostico"
        for name in ("predictions.parquet", "horizon.parquet", "episodes.parquet", "tramos.parquet", "bootstrap.parquet",
                     "spectrum.parquet"):
            assert analysis_load.artifact_path(str(state), run["run_id"], name)
    assert (tmp / "out" / "1.0.md").exists()

    with pytest.raises(SystemExit, match="1.0"):
        R.run(R.parse_args(_argv(tmp, state, validated, "1.1")))


def test_help(capsys):
    from operador import run as R

    with pytest.raises(SystemExit):
        R.parse_args(["--help"])
    assert "--etapa" in capsys.readouterr().out


def test_modos_cierre_y_test_con_guardia(estado_v2):
    from analysis import load as analysis_load
    from operador import registro as G
    from operador import run as R

    tmp, state, validated = estado_v2
    out = tmp / "out"
    if not (out / R.EVAL_ROWS).exists():
        (tmp / "decisiones.json").write_text("{}")
        R.run(R.parse_args(_argv(tmp, state, validated, "base")))
    base = {"representaciones": ["gru", "pca"], "pesos": False, "duan": True, "rollout": "autonomo",
            "horizonte": "horario", "estimador": "exacto"}
    G.registrar_decision("1.5", "exacto", base, "test", "2026-11-19", tmp / "decisiones.json")
    extra = ("--seeds", "0", "--bootstrap-modos", "10")
    assert R.run(R.parse_args(_argv(tmp, state, validated, "1.6", "--version", "2", *extra))) == 0
    modos16 = json.loads((out / "1.6.json").read_text())["modos"]
    assert set(modos16) == {"gru-s0", "pca"}
    G.registrar_decision("1.6", "gru", {**base, "oficial": "gru", "secundarias": ["pca"], "modos": modos16, "rank": {}},
                         "test", "2026-11-19", tmp / "decisiones.json")
    assert R.run(R.parse_args(_argv(tmp, state, validated, "cierre", *extra))) == 0
    cierre = json.loads((out / R.CIERRE_JSON).read_text())
    for key, run_id in cierre["runs"].items():
        run = analysis_load.run_by_id(str(state), run_id)
        assert not any(k.startswith("test_") for k in run["metrics"])
        assert int(run["metrics"]["n_modos"]) == len(modos16[key])
    assert R.run(R.parse_args(_argv(tmp, state, validated, "test", *extra))) == 0
    runs = [r for r in G.runs_de_etapa(str(state), "test") if r["tags"].get("evaluacion") == "test"]
    assert len(runs) == 12
    assert all(analysis_load.is_series_run(r) for r in runs)
    assert json.loads((out / "test.json").read_text())["veredicto"] in ("cumple", "no cumple")
    with pytest.raises(SystemExit, match="force-test"):
        R.run(R.parse_args(_argv(tmp, state, validated, "test", *extra)))
