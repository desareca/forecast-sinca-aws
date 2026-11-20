"""Tarea 5.1: evaluación con el arnés de `operador` (e2e sintético con estado MLflow temporal)."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from sint_koopman import estado_referencias, modelo_chico


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    return estado_referencias(tmp_path_factory.mktemp("koop_refs"))


def test_evaluacion_con_filas_de_71(refs):
    from analysis import load as analysis_load
    from features import partition as pt
    from koopman import evaluar as K
    from operador import estado as E
    from operador import evaluacion as evm
    from operador.run import veredicto

    state, _, grid, _ = refs
    runs = analysis_load.list_runs(str(state))
    rows = K.filas_referencia(str(state), runs)
    assert rows.groupby(["estacion", "pollutant"]).ngroups == 6
    ev = evm.Evaluador(rows, B=50)
    ref71 = K.puntaje_71(str(state), runs, ev)
    assert ref71 is not None
    ctx = E.contexto(grid, pt.get_partition("ae-dmd"))
    m = modelo_chico()
    res = K.evaluar_modelo(m, ctx, ev, "koop", ref71=ref71)
    assert set(res["frames"]) >= {"predictions.parquet", "horizon.parquet", "episodes.parquet", "tramos.parquet",
                                  "bootstrap.parquet", "spectrum.parquet"}
    assert np.isfinite(res["diff_71"]["diff"])
    pos_tr, _ = E.posiciones(ctx, 24, ("train",))
    pos_tr = pos_tr[pt.row_mask(grid.index[pos_tr], ctx.part, "train")][:500]
    S = K.factores_duan(m, ctx, pos_tr)
    assert S.shape == (24, 6) and np.all(S > 0)
    res_d = K.evaluar_modelo(m, ctx, ev, "koop-duan", duan_S=S)
    assert not np.allclose(res_d["yhat"], res["yhat"])
    assert veredicto(res["tabla"])["precision"] in (True, False)
    ids = K.run_ids_referencia(runs)
    assert len(ids) >= 6 + 6 + 6 + 1 + 3
