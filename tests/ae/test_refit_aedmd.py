"""Reajuste de LightGBM y persistencia en la partición `ae-dmd` (ae-dmd-prep, tarea 6.5).

Crea 6 runs fuente sintéticos en un MLflow temporal y corre `train.run --mode refit`. Sin AWS.
"""
import pathlib

import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("mlflow")

from conftest import synthetic_validated  # noqa: E402

EXPERIMENT = "test-refit-aedmd"


def _write_validated(root: pathlib.Path) -> pathlib.Path:
    data = synthetic_validated(start="2026-01-01", hours=24 * 280, gaps=True)
    for key, frame in (("sinca", data["sinca"]), ("open_meteo", data["open_meteo"]), ("feriados", data["feriados"])):
        target = root / key
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "part-0.parquet", index=False)
    return root


def _features(validated: pathlib.Path):
    from features.build import build_features, load_validated

    data = load_validated("validated", client=None, local_dir=str(validated))
    return build_features(data["sinca"], data["open_meteo"], data["feriados"])


SOURCE_ROUNDS = 7  # árboles del booster v1 sintético de cada run fuente (13.2)


def _source_booster(path: pathlib.Path, rounds: int = SOURCE_ROUNDS) -> None:
    """Booster LightGBM mínimo con `rounds` árboles, guardado como lo hace `save_booster` del v1."""
    import lightgbm as lgb
    import numpy as np

    from train.train import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(x, label=x[:, 0]),
                        num_boost_round=rounds)
    booster.save_model(str(path))


def _source_runs(mlflow_dir: pathlib.Path, series, with_booster: bool = True) -> dict[tuple[str, str], str]:
    """Crea un run `lightgbm` fuente por serie con `DEFAULT_PARAMS` y devuelve `{(slug, pollutant): run_id}`.

    Con `with_booster`, cada run lleva su `<run_name>.txt` (como el v1): el refit toma de ahí las rondas.
    """
    import mlflow

    from train import mlflow_log
    from train.train import DEFAULT_PARAMS

    mlflow_log.configure(str(mlflow_dir), experiment=EXPERIMENT)
    out: dict[tuple[str, str], str] = {}
    for station, pollutant in series:
        slug = mlflow_log.station_slug(station)
        name = f"fuente-{slug}-{pollutant}"
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"modelo": "lightgbm", "tipo": "entrenamiento"})
            mlflow.log_params({**DEFAULT_PARAMS, "estacion": station, "pollutant": pollutant, "version": 1})
            if with_booster:
                path = mlflow_dir.parent / f"{name}.txt"
                _source_booster(path)
                mlflow.log_artifact(str(path))
        out[(slug, pollutant)] = run.info.run_id
    return out


def _sources_text(sources: dict[tuple[str, str], str], skip: tuple[str, str] | None = None) -> str:
    return ",".join(f"{slug}:{pollutant}={run_id}" for (slug, pollutant), run_id in sources.items()
                    if (slug, pollutant) != skip)


def _argv(tmp: pathlib.Path, validated: pathlib.Path, sources_text: str, *extra: str) -> list[str]:
    return [
        "--local-validated-dir", str(validated),
        "--mlflow-dir", str(tmp / "mlflow"),
        "--model-dir", str(tmp / "model"),
        "--output-dir", str(tmp / "output"),
        "--experiment", EXPERIMENT,
        "--no-upload",
        "--mode", "refit",
        "--partition", "ae-dmd",
        "--source-run-ids", sources_text,
        *extra,
    ]


def _all_runs(tmp: pathlib.Path):
    import mlflow

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return []
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")


def _artifact(tmp: pathlib.Path, run_id: str, name: str) -> pd.DataFrame:
    from mlflow.tracking import MlflowClient

    local = MlflowClient().download_artifacts(run_id, name, dst_path=str(tmp / "dl"))
    return pd.read_parquet(local)


@pytest.fixture(scope="module")
def refit_env(tmp_path_factory):
    from features.build import series_list

    tmp = tmp_path_factory.mktemp("refit_aedmd")
    validated = _write_validated(tmp / "validated")
    features = _features(validated)
    series = series_list(features)
    sources = _source_runs(tmp / "mlflow", series)
    return {"tmp": tmp, "validated": validated, "features": features, "series": series, "sources": sources}


@pytest.fixture(scope="module")
def refit_done(refit_env):
    from train import train as tr

    tmp = refit_env["tmp"]
    assert tr.run(tr.parse_args(_argv(tmp, refit_env["validated"], _sources_text(refit_env["sources"])))) == 0
    return refit_env


def test_seis_runs_lightgbm_mas_persistencia(refit_done):
    runs = _all_runs(refit_done["tmp"])
    names = [r.info.run_name for r in runs]
    lgbm = [n for n in names if n.endswith("-lightgbm-aedmd-v1")]
    assert len(lgbm) == 6
    assert len(set(lgbm)) == 6  # 5.7: un nombre por serie, con contaminante
    assert all(("-pm25-" in n) or ("-pm10-" in n) for n in lgbm)
    assert "sinca-zona-persistence-aedmd-v1" in names


def test_predicciones_solo_en_sel_con_embargo(refit_done):
    tmp = refit_done["tmp"]
    runs = [r for r in _all_runs(tmp) if r.info.run_name.endswith("-lightgbm-aedmd-v1")]
    for run in runs:
        preds = _artifact(tmp, run.info.run_id, "predictions.parquet")
        assert set(preds["split"].unique()) == {"sel"}
        stamps = pd.to_datetime(preds["timestamp"])
        if stamps.dt.tz is not None:
            stamps = stamps.dt.tz_convert("America/Santiago").dt.tz_localize(None)
        assert stamps.max() < pd.Timestamp("2026-06-30 00:00")  # 24 h antes del fin de sel
        assert stamps.min() >= pd.Timestamp("2026-04-01 00:00")


def test_train_usa_solo_filas_antes_del_embargo_de_es(refit_done):
    from features import partition as pt

    part = pt.get_partition("ae-dmd")
    features = refit_done["features"]
    train_rows = features[pt.row_mask(features["timestamp"], part, "train")]
    assert train_rows["timestamp"].max() < pd.Timestamp("2026-02-28 00:00", tz="America/Santiago")


def test_run_lightgbm_tiene_tags_params_y_metricas(refit_done):
    tmp = refit_done["tmp"]
    run = next(r for r in _all_runs(tmp) if r.info.run_name.endswith("-lightgbm-aedmd-v1"))
    assert run.data.tags["particion"] == "ae-dmd"
    assert run.data.tags["stage"] == "ae-dmd-prep"
    assert run.data.tags["modelo"] == "lightgbm-aedmd"
    assert run.data.params["source_run_id"] in refit_done["sources"].values()
    assert "sel_weighted_rmse" in run.data.metrics or "sel_rmse" in run.data.metrics
    assert "train_loss" in run.data.metrics


def test_persistencia_de_referencia_en_sel(refit_done):
    tmp = refit_done["tmp"]
    run = next(r for r in _all_runs(tmp) if r.info.run_name == "sinca-zona-persistence-aedmd-v1")
    assert run.data.params["tipo"] == "referencia"
    assert run.data.tags["modelo"] == "persistence-aedmd"
    preds = _artifact(tmp, run.info.run_id, "predictions.parquet")
    assert set(preds["split"].unique()) == {"sel"}


def test_falta_una_serie_aborta_sin_runs_nuevos(refit_env, tmp_path):
    from train import train as tr

    before = len(_all_runs(refit_env["tmp"]))
    first = next(iter(refit_env["sources"]))
    text = _sources_text(refit_env["sources"], skip=first)
    with pytest.raises(SystemExit):
        tr.run(tr.parse_args(_argv(refit_env["tmp"], refit_env["validated"], text)))
    assert len(_all_runs(refit_env["tmp"])) == before


def test_run_fuente_de_otra_serie_es_error(refit_env):
    from train import train as tr

    keys = list(refit_env["sources"])
    (slug_a, pol_a), (slug_b, pol_b) = keys[0], keys[1]
    swapped = dict(refit_env["sources"])
    swapped[(slug_a, pol_a)] = refit_env["sources"][(slug_b, pol_b)]  # run de otra serie
    before = len(_all_runs(refit_env["tmp"]))
    with pytest.raises(SystemExit):
        tr.run(tr.parse_args(_argv(refit_env["tmp"], refit_env["validated"], _sources_text(swapped))))
    assert len(_all_runs(refit_env["tmp"])) == before


def test_refit_usa_rondas_del_booster_fuente_sin_validacion(tmp_path, monkeypatch):
    """13.2 (D11): num_boost_round = árboles del booster v1 y fit_model sin valid_df (no mira `es`)."""
    from features.build import series_list
    from train import train as tr

    validated = _write_validated(tmp_path / "validated")
    features = _features(validated)
    series = series_list(features)
    sources = _source_runs(tmp_path / "mlflow", series)

    calls: list[tuple[bool, int | None]] = []
    real = tr.fit_model

    def spy(*args, **kwargs):
        valid = args[4] if len(args) > 4 else kwargs.get("valid_df")
        calls.append((valid is None, kwargs.get("num_boost_round")))
        return real(*args, **kwargs)

    monkeypatch.setattr(tr, "fit_model", spy)
    assert tr.run(tr.parse_args(_argv(tmp_path, validated, _sources_text(sources)))) == 0
    assert len(calls) == 6
    assert all(sin_valid for sin_valid, _ in calls)
    assert all(rounds == SOURCE_ROUNDS for _, rounds in calls)

    run = next(r for r in _all_runs(tmp_path) if r.info.run_name.endswith("-lightgbm-aedmd-v1"))
    assert run.data.params["num_boost_round"] == str(SOURCE_ROUNDS)
    assert run.data.params["early_stopping"] == "off"
    assert run.data.params["rounds_source"] == "v1-booster"
    from mlflow.tracking import MlflowClient

    import lightgbm as lgb
    local = MlflowClient().download_artifacts(run.info.run_id, f"{run.info.run_name}.txt", dst_path=str(tmp_path / "dl"))
    assert lgb.Booster(model_file=local).num_trees() == SOURCE_ROUNDS


def test_run_fuente_sin_booster_es_error_claro(tmp_path):
    """13.2: un run fuente sin `.txt` no se usa con rondas del booster (error claro, antes de entrenar)."""
    from train import train as tr

    validated = _write_validated(tmp_path / "validated")
    features = _features(validated)
    from features.build import series_list

    series = series_list(features)
    sources = _source_runs(tmp_path / "mlflow", series, with_booster=False)
    before = len(_all_runs(tmp_path))
    with pytest.raises(SystemExit, match="booster"):
        tr.run(tr.parse_args(_argv(tmp_path, validated, _sources_text(sources))))
    assert len(_all_runs(tmp_path)) == before
