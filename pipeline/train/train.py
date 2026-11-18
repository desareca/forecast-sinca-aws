"""Entrenamiento del baseline LightGBM (Fase 3, design.md §4/§7/§9).

Un modelo **independiente por serie** (MP2.5/MP10 x 3 estaciones = 6 modelos),
cada uno minimizando su **RMSE ponderado** (`w_i = 1 + α·max(0, y_i − umbral_regular)`).

Estrategia de hiperparámetros (design.md §9): **random search + early stopping**
sobre las ventanas de validación de la ventana deslizante (~20 trials por serie,
espacio acotado para regularizar fuerte en un dataset chico). La búsqueda se
evalúa promediando el RMSE ponderado sobre una muestra de ventanas de validación
(configurable con `--search-windows`).

Evaluación: ventana deslizante completa (train 2 años / valid 1 mes / step 1 mes)
+ test final held-out de 3 meses (`features.build`).

Entrada: datos validados de `sinca-data/validated/`.
Salida: modelos en `SM_MODEL_DIR`, métricas/parámetros/artefactos en MLflow
(logging directo) y un reporte JSON de la búsqueda.

Uso (SageMaker Training Job, código vía `source_dir`/`entry_point`):

    python train/train.py --data-bucket sinca-data --mlflow-bucket sinca-mlflow

Uso local (sin escribir en S3):

    python train/train.py --local-validated-dir ./validated --no-upload
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

# El pipeline es la raíz de importación (`config`, `features`, ...). En SageMaker
# el toolkit ya pone `/opt/ml/code` en el PYTHONPATH; esto permite además correr
# `python pipeline/train/train.py` desde la raíz del repo.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluate.evaluate import (
    ALPHA_DEFAULT,
    evaluate_predictions,
    regular_threshold,
    sample_weights,
    weighted_rmse,
    zone_icap,
)
from features.build import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_window,
    build_features,
    load_validated,
    make_rolling_windows,
    final_holdout_split,
    series_list,
)
from features import partition as pt
from train import mlflow_log

log = logging.getLogger("train")

# --- Espacio de hiperparámetros (design.md §9) -------------------------------

HYPERPARAM_SPACE: dict[str, list] = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0],
}

DEFAULT_PARAMS: dict[str, float] = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

NUM_BOOST_ROUND = 1000  # cap; early stopping corta antes
EARLY_STOPPING_ROUNDS = 50
MIN_TRAIN_ROWS = 168  # una semana de horas: no entrenar con menos

SEARCH_TRIALS_DEFAULT = 20
SEARCH_WINDOWS_DEFAULT = 12
MODEL_VERSION_DEFAULT = 1
MODEL_NAME = "lightgbm"


# --- Entrenamiento -----------------------------------------------------------


def _xyw(df: pd.DataFrame, pollutant: str, alpha: float):
    features = df[list(FEATURE_COLUMNS)].astype("float64")
    target = df[TARGET_COLUMN].astype(float).to_numpy()
    weights = sample_weights(target, pollutant, alpha)
    return features, target, weights


def fit_model(
    train_df: pd.DataFrame,
    pollutant: str,
    params: dict,
    alpha: float = ALPHA_DEFAULT,
    valid_df: pd.DataFrame | None = None,
    seed: int = 0,
    num_boost_round: int = NUM_BOOST_ROUND,
):
    """Entrena un booster LightGBM con early stopping sobre `valid_df` (si se pasa).

    Sin `valid_df` no hay early stopping: se entrenan exactamente `num_boost_round` rondas (D11).
    """
    import lightgbm as lgb

    features, target, weights = _xyw(train_df, pollutant, alpha)
    train_set = lgb.Dataset(
        features, label=target, weight=weights,
        feature_name=list(FEATURE_COLUMNS), free_raw_data=False,
    )

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_threads": os.cpu_count() or 1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        **params,
    }

    callbacks = [lgb.log_evaluation(0)]
    valid_sets = []
    if valid_df is not None and len(valid_df):
        valid_features, valid_target, valid_weights = _xyw(valid_df, pollutant, alpha)
        valid_sets = [
            lgb.Dataset(
                valid_features, label=valid_target, weight=valid_weights,
                reference=train_set, free_raw_data=False,
            )
        ]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))

    booster = lgb.train(
        lgb_params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets, callbacks=callbacks,
    )
    return booster


def _sample_params(rng: np.random.Generator) -> dict:
    return {name: values[int(rng.integers(len(values)))] for name, values in HYPERPARAM_SPACE.items()}


def _select_windows(windows: list[dict], max_windows: int | None) -> list[dict]:
    if not max_windows or max_windows <= 0 or max_windows >= len(windows):
        return windows
    # Muestra equiespaciada de las ventanas (mantiene la ventana más reciente).
    positions = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
    return [windows[i] for i in dict.fromkeys(int(p) for p in positions)]


def random_search(
    series_df: pd.DataFrame,
    windows: list[dict],
    pollutant: str,
    n_trials: int = SEARCH_TRIALS_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 0,
    max_windows: int | None = SEARCH_WINDOWS_DEFAULT,
) -> tuple[dict, list[dict]]:
    """Random search por serie; selección por RMSE ponderado promedio sobre ventanas."""
    rng = np.random.default_rng(seed)
    splits = []
    for window in _select_windows(windows, max_windows):
        train, valid = apply_window(series_df, window)
        if len(train) >= MIN_TRAIN_ROWS and not valid.empty:
            splits.append((train, valid))

    history: list[dict] = []
    best: dict | None = None
    for trial in range(n_trials):
        params = _sample_params(rng)
        scores: list[float] = []
        for train, valid in splits:
            booster = fit_model(train, pollutant, params, alpha, valid, seed=seed + trial)
            prediction = booster.predict(
                valid[list(FEATURE_COLUMNS)].astype("float64"),
                num_iteration=booster.best_iteration or NUM_BOOST_ROUND,
            )
            scores.append(weighted_rmse(valid[TARGET_COLUMN], prediction, sample_weights(valid[TARGET_COLUMN], pollutant, alpha)))
        score = float(np.mean(scores)) if scores else float("nan")
        history.append({"trial": trial, "params": params, "score": score})
        log.info("  trial %d/%d score=%.4f params=%s", trial + 1, n_trials, score, params)
        if np.isfinite(score) and (best is None or score < best["score"]):
            best = {"params": params, "score": score}

    if best is None:  # sin ventanas evaluables: usar defaults
        best = {"params": dict(DEFAULT_PARAMS), "score": float("nan")}
    return best, history


def evaluate_rolling(
    series_df: pd.DataFrame,
    pollutant: str,
    params: dict,
    windows: list[dict],
    alpha: float = ALPHA_DEFAULT,
    seed: int = 0,
) -> tuple[list[float], pd.DataFrame]:
    """RMSE ponderado por ventana (train -> valid) y marco de predicciones agregado."""
    scores: list[float] = []
    frames: list[pd.DataFrame] = []
    for window in windows:
        train, valid = apply_window(series_df, window)
        if len(train) < MIN_TRAIN_ROWS or valid.empty:
            continue
        booster = fit_model(train, pollutant, params, alpha, valid, seed=seed)
        prediction = booster.predict(
            valid[list(FEATURE_COLUMNS)].astype("float64"),
            num_iteration=booster.best_iteration or NUM_BOOST_ROUND,
        )
        scores.append(weighted_rmse(valid[TARGET_COLUMN], prediction, sample_weights(valid[TARGET_COLUMN], pollutant, alpha)))
        frames.append(_prediction_frame(valid, prediction))

    if not frames:
        return scores, pd.DataFrame(columns=["timestamp", "estacion", "pollutant", "y_true", "y_pred"])
    return scores, pd.concat(frames, ignore_index=True)


def _prediction_frame(df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    out = df[["timestamp", "estacion", "pollutant", TARGET_COLUMN]].copy()
    out = out.rename(columns={TARGET_COLUMN: "y_true"})
    out["y_pred"] = prediction
    return out


PREDICTIONS_COLUMNS = ["timestamp", "estacion", "pollutant", "y_true", "y_pred", "split"]


def _diagnostics_predictions(rolling_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Predicciones del rolling (valid) + test held-out, con columna `split`."""
    frames: list[pd.DataFrame] = []
    if not rolling_frame.empty:
        valid = rolling_frame.copy()
        valid["split"] = "valid"
        frames.append(valid)
    if not test_frame.empty:
        test = test_frame.copy()
        test["split"] = "test"
        frames.append(test)
    if not frames:
        return pd.DataFrame(columns=PREDICTIONS_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[PREDICTIONS_COLUMNS]


def _feature_importance_frame(booster, station: str, pollutant: str) -> pd.DataFrame:
    """Gain por feature para una serie (`booster.feature_importance("gain")`)."""
    names = list(booster.feature_name())
    gains = booster.feature_importance("gain")
    return pd.DataFrame(
        {
            "estacion": station,
            "pollutant": pollutant,
            "feature": names,
            "gain": [float(gain) for gain in gains],
        }
    )


def log_diagnostics(
    rolling_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    booster,
    station: str,
    pollutant: str,
    rolling_scores: list[float] | None = None,
) -> dict[str, str]:
    """Loguea como artefactos MLflow los diagnósticos del run.

    Escribe `predictions.parquet`, `feature_importance.parquet`,
    `zone_icap.parquet` y `rolling_scores.parquet` (RMSE ponderado por ventana,
    para el rango min–max del CV) en un dir temporal con los nombres canónicos y
    los loguea al directorio de artefactos del run (`mlruns/<exp>/<run>/artifacts/`),
    no a `model-artifacts/` (que `sinca-dev-role` no lee).
    """
    import mlflow

    diag_dir = tempfile.mkdtemp(prefix="sinca-diag-")
    paths: dict[str, str] = {}

    predictions = _diagnostics_predictions(rolling_frame, test_frame)
    paths["predictions"] = os.path.join(diag_dir, "predictions.parquet")
    predictions.to_parquet(paths["predictions"], index=False)

    importance = _feature_importance_frame(booster, station, pollutant)
    paths["feature_importance"] = os.path.join(diag_dir, "feature_importance.parquet")
    importance.to_parquet(paths["feature_importance"], index=False)

    zone = zone_icap(test_frame)
    paths["zone_icap"] = os.path.join(diag_dir, "zone_icap.parquet")
    zone.to_parquet(paths["zone_icap"], index=False)

    scores = list(rolling_scores or [])
    scores_frame = pd.DataFrame({"window_index": range(len(scores)), "rmse": scores})
    paths["rolling_scores"] = os.path.join(diag_dir, "rolling_scores.parquet")
    scores_frame.to_parquet(paths["rolling_scores"], index=False)

    mlflow.log_artifacts(diag_dir)
    log.info(
        "Diagnósticos logueados (%s/%s): predictions=%d, features=%d, zone_icap=%d, ventanas=%d",
        station, pollutant, len(predictions), len(importance), len(zone), len(scores),
    )
    return paths


def save_booster(booster, model_dir: str, name: str, num_iteration: int | None = None) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.txt")
    rounds = num_iteration or booster.best_iteration or NUM_BOOST_ROUND
    booster.save_model(path, num_iteration=rounds)
    return path


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if value is not None and np.isfinite(value)}


# --- Orquestación ------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    client = None
    model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "model")
    output_dir = args.output_dir or os.environ.get("SM_OUTPUT_DATA_DIR", "output")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    data = load_validated(args.data_bucket, client=client, local_dir=args.local_validated_dir)
    log.info(
        "Validados: SINCA=%d Open-Meteo=%d feriados=%d",
        len(data["sinca"]), len(data["open_meteo"]), len(data["feriados"]),
    )

    features = build_features(
        data["sinca"], data["open_meteo"], data["feriados"],
        stations=args.stations, pollutants=args.pollutants,
    )
    log.info("Dataset de features: %d filas, %d series", len(features), len(series_list(features)))

    windows = make_rolling_windows()
    log.info("Ventanas deslizantes: %d", len(windows))

    if args.mode == "refit":
        return run_refit(args, data, features, pt.get_partition(args.partition))

    if not args.no_upload:
        downloaded = mlflow_log.download_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, client,
                                               run_ids=mlflow_log.parse_mlflow_runs(args.mlflow_runs))
        log.info("Estado MLflow previo: %d objetos", len(downloaded))

    import mlflow

    mlflow_log.configure(args.mlflow_dir, experiment=args.experiment)
    mlflow.autolog(disable=True)

    results: dict[str, dict] = {}
    for station, pollutant in series_list(features):
        log.info("=== Serie %s / %s ===", station, pollutant)
        series_df = features[(features["estacion"] == station) & (features["pollutant"] == pollutant)].sort_values("timestamp")
        series_df = series_df.reset_index(drop=True)

        best, history = random_search(
            series_df, windows, pollutant,
            n_trials=args.search_trials, alpha=args.alpha,
            seed=args.seed, max_windows=args.search_windows,
        )
        log.info("Mejores hiperparámetros (%s/%s): %s (score=%.4f)", station, pollutant, best["params"], best["score"])

        rolling_scores, rolling_frame = evaluate_rolling(series_df, pollutant, best["params"], windows, args.alpha, args.seed)
        rolling_metrics = evaluate_predictions(rolling_frame, args.alpha) if not rolling_frame.empty else {}

        train_df, test_df = final_holdout_split(series_df)
        last_valid = None
        for window in reversed(windows):
            _, valid = apply_window(series_df, window)
            if not valid.empty and len(valid) >= MIN_TRAIN_ROWS:
                last_valid = valid
                break
        final_booster = fit_model(train_df, pollutant, best["params"], args.alpha, last_valid, seed=args.seed)

        train_prediction = final_booster.predict(train_df[list(FEATURE_COLUMNS)].astype("float64"), num_iteration=final_booster.best_iteration or NUM_BOOST_ROUND)
        train_loss = weighted_rmse(train_df[TARGET_COLUMN], train_prediction, sample_weights(train_df[TARGET_COLUMN], pollutant, args.alpha))

        test_prediction = final_booster.predict(test_df[list(FEATURE_COLUMNS)].astype("float64"), num_iteration=final_booster.best_iteration or NUM_BOOST_ROUND)
        test_frame = _prediction_frame(test_df, test_prediction)
        test_metrics = evaluate_predictions(test_frame, args.alpha) if not test_frame.empty else {}

        name = mlflow_log.run_name(station, MODEL_NAME, args.version)
        model_path = save_booster(final_booster, model_dir, name)

        report = {
            "estacion": station,
            "pollutant": pollutant,
            "model": MODEL_NAME,
            "version": args.version,
            "alpha": args.alpha,
            "umbral_regular": regular_threshold(pollutant),
            "best_params": best["params"],
            "search_score": best["score"],
            "n_windows": len(rolling_scores),
            "search_history": history,
        }
        report_path = os.path.join(output_dir, f"{name}.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"serie": f"{station}|{pollutant}", "modelo": MODEL_NAME, "stage": "baseline", "tipo": "entrenamiento"})
            mlflow.log_params({
                **{k: v for k, v in best["params"].items()},
                "alpha": args.alpha,
                "umbral_regular": regular_threshold(pollutant),
                "estacion": station,
                "pollutant": pollutant,
                "n_windows_search": args.search_windows,
                "n_windows_rolling": len(rolling_scores),
                "version": args.version,
            })
            metrics = {
                "train_loss": train_loss,
                "test_loss": test_metrics.get("weighted_rmse"),
                "rolling_weighted_rmse_mean": float(np.mean(rolling_scores)) if rolling_scores else float("nan"),
                "rolling_weighted_rmse_std": float(np.std(rolling_scores)) if rolling_scores else float("nan"),
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"rolling_{k}": v for k, v in rolling_metrics.items()},
            }
            mlflow.log_metrics(_finite_metrics(metrics))
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(report_path)
            log_diagnostics(rolling_frame, test_frame, final_booster, station, pollutant, rolling_scores)

        if not args.no_upload:
            mlflow_log.upload_model_artifact(model_path, args.mlflow_bucket, name, client=client)

        results[f"{station}|{pollutant}"] = {
            "best_params": best["params"],
            "train_loss": train_loss,
            "test": test_metrics,
            "rolling_mean": float(np.mean(rolling_scores)) if rolling_scores else None,
        }
        log.info(
            "Serie %s/%s -> train_loss=%.4f test_loss=%s rolling_mean=%s",
            station, pollutant, train_loss,
            test_metrics.get("weighted_rmse"),
            results[f"{station}|{pollutant}"]["rolling_mean"],
        )

    # --- Baseline de persistencia (referencia, no es un run de entrenamiento) --
    if not args.skip_persistence:
        from baseline.persistence import predict_frame as persistence_frame

        persistence = persistence_frame(data["sinca"], features)
        train_df, test_df = final_holdout_split(features)
        test_keys = test_df[["timestamp", "estacion", "pollutant"]]
        test_persistence = persistence.merge(test_keys, on=["timestamp", "estacion", "pollutant"], how="inner")
        persist_metrics = evaluate_predictions(test_persistence, args.alpha) if not test_persistence.empty else {}

        name = "sinca-persistence-baseline"
        report_path = os.path.join(output_dir, f"{name}.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump({"model": "persistence", "test": persist_metrics}, handle, ensure_ascii=False, indent=2)

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"modelo": "persistence", "stage": "baseline", "tipo": "referencia"})
            mlflow.log_params({"alpha": args.alpha, "tipo": "persistencia"})
            mlflow.log_metrics(_finite_metrics({f"test_{k}": v for k, v in persist_metrics.items()}))
            mlflow.log_artifact(report_path)
        results["persistence"] = {"test": persist_metrics}
        log.info("Persistencia -> test_loss=%s", persist_metrics.get("weighted_rmse"))

    if not args.no_upload:
        mlflow_log.upload_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, client)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log.info("Resumen escrito en %s", summary_path)
    return 0


# --- Reajuste en la partición ae-dmd (ae-dmd-prep, D5) -----------------------

INT_HYPERPARAMS = ("num_leaves", "min_child_samples")


def parse_source_run_ids(text: str | None) -> dict[tuple[str, str], str]:
    """`slug:pollutant=run_id,...` -> `{(slug, pollutant): run_id}`."""
    out: dict[tuple[str, str], str] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, eq, run_id = item.partition("=")
        slug, colon, pollutant = left.partition(":")
        if not (eq and colon and slug.strip() and pollutant.strip() and run_id.strip()):
            raise ValueError(f"--source-run-ids mal formado: {item!r} (formato slug:pollutant=run_id)")
        out[(slug.strip(), pollutant.strip())] = run_id.strip()
    return out


def source_params(state_dir: str, run_id: str, station: str, pollutant: str) -> dict:
    """Hiperparámetros de un run `lightgbm` fuente, con su tipo (`HYPERPARAM_SPACE`)."""
    from analysis import load as analysis_load

    run = analysis_load.run_by_id(state_dir, run_id)
    if run is None:
        raise SystemExit(f"El run fuente {run_id} no existe en MLflow ({station}/{pollutant}).")
    stored = run["params"]
    if stored.get("estacion") != station or stored.get("pollutant") != pollutant:
        raise SystemExit(
            f"El run fuente {run_id} es de {stored.get('estacion')}/{stored.get('pollutant')}, "
            f"no de {station}/{pollutant}."
        )
    out: dict = {}
    for name in HYPERPARAM_SPACE:
        if name not in stored:
            raise SystemExit(f"El run fuente {run_id} no tiene el hiperparámetro {name}.")
        out[name] = int(float(stored[name])) if name in INT_HYPERPARAMS else float(stored[name])
    return out


def source_rounds(state_dir: str, run_id: str, station: str, pollutant: str) -> int:
    """Rondas del booster `.txt` del run fuente v1 (D11): su `num_trees()` tras guardarlo con `best_iteration`."""
    import lightgbm as lgb

    from analysis import load as analysis_load

    run = analysis_load.run_by_id(state_dir, run_id)
    path = analysis_load.artifact_path(state_dir, run_id, f"{run['run_name']}.txt") if run else None
    if path is None:
        raise SystemExit(f"El run fuente {run_id} ({station}/{pollutant}) no tiene el booster .txt: "
                         "no se pueden tomar sus rondas (--rounds-from-source). Descargar sus artefactos o usar false.")
    return int(lgb.Booster(model_file=path).num_trees())


def _log_frames(frames: dict[str, pd.DataFrame]) -> None:
    """Loguea parquet con nombres canónicos al directorio de artefactos del run activo."""
    import mlflow

    tmp = tempfile.mkdtemp(prefix="sinca-refit-")
    for filename, frame in frames.items():
        frame.to_parquet(os.path.join(tmp, filename), index=False)
    mlflow.log_artifacts(tmp)


def run_refit(args: argparse.Namespace, data: dict, features: pd.DataFrame, part) -> int:
    """Reajusta LightGBM y persistencia en la partición `part` (D5, ae-dmd-prep).

    Hiperparámetros de los runs fuente (sin búsqueda). Ajuste: `train` con embargo, early
    stopping sobre `es` con embargo, predicción en `sel` con embargo. Sin `upload_model_artifact`.
    """
    import mlflow

    from baseline.persistence import predict_frame as persistence_frame

    model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "model")
    output_dir = args.output_dir or os.environ.get("SM_OUTPUT_DATA_DIR", "output")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    series = series_list(features)
    sources = parse_source_run_ids(args.source_run_ids)
    missing = [f"{mlflow_log.station_slug(s)}:{p}" for s, p in series
               if (mlflow_log.station_slug(s), p) not in sources]
    if missing:
        raise SystemExit(f"Faltan run fuente para: {', '.join(missing)} (--source-run-ids).")

    rounds_mode = bool(args.rounds_from_source)
    if not args.no_upload:
        run_ids = mlflow_log.parse_mlflow_runs(args.mlflow_runs)
        if rounds_mode and run_ids is not None:
            # Las rondas salen del booster de cada run fuente: se bajan aunque --mlflow-runs no los liste (D11).
            source_ids = [sources[(mlflow_log.station_slug(s), p)] for s, p in series]
            run_ids = list(dict.fromkeys([*run_ids, *source_ids]))
        mlflow_log.download_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, None, run_ids=run_ids)
    mlflow_log.configure(args.mlflow_dir, experiment=args.experiment)
    mlflow.autolog(disable=True)

    # Validar todos los runs fuente antes de entrenar nada.
    chosen: dict[tuple[str, str], tuple[str, dict, int | None]] = {}
    for station, pollutant in series:
        run_id = sources[(mlflow_log.station_slug(station), pollutant)]
        params = source_params(args.mlflow_dir, run_id, station, pollutant)
        rounds = source_rounds(args.mlflow_dir, run_id, station, pollutant) if rounds_mode else None
        chosen[(station, pollutant)] = (run_id, params, rounds)

    token = part.token
    results: dict[str, dict] = {}
    for station, pollutant in series:
        run_id, params, rounds = chosen[(station, pollutant)]
        log.info("=== Refit %s / %s (%s) desde run %s (rondas=%s) ===", station, pollutant, part.name, run_id,
                 rounds if rounds_mode else "early stopping")
        series_df = features[(features["estacion"] == station) & (features["pollutant"] == pollutant)]
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)
        ts = series_df["timestamp"]
        train_df = series_df[pt.row_mask(ts, part, "train")]
        es_df = series_df[pt.row_mask(ts, part, "es")]
        sel_df = series_df[pt.row_mask(ts, part, "sel")]
        if len(train_df) < MIN_TRAIN_ROWS or sel_df.empty:
            raise SystemExit(f"Sin filas suficientes en train/sel para {station}/{pollutant} (partición {part.name}).")

        if rounds_mode:
            # D11: presupuesto fijo = rondas del booster v1; sin `es` como validación.
            booster = fit_model(train_df, pollutant, params, args.alpha, None, seed=args.seed, num_boost_round=rounds)
            iterations = rounds
        else:
            booster = fit_model(train_df, pollutant, params, args.alpha, es_df if len(es_df) else None, seed=args.seed)
            iterations = booster.best_iteration or NUM_BOOST_ROUND
        train_pred = booster.predict(train_df[list(FEATURE_COLUMNS)].astype("float64"), num_iteration=iterations)
        train_loss = weighted_rmse(train_df[TARGET_COLUMN], train_pred,
                                   sample_weights(train_df[TARGET_COLUMN], pollutant, args.alpha))
        sel_pred = booster.predict(sel_df[list(FEATURE_COLUMNS)].astype("float64"), num_iteration=iterations)
        sel_frame = _prediction_frame(sel_df, sel_pred)
        sel_frame["split"] = "sel"
        sel_metrics = evaluate_predictions(sel_frame, args.alpha) if not sel_frame.empty else {}

        # Con contaminante en el nombre (5.7): pm10 y pm25 de una estación no comparten nombre.
        name = f"sinca-{mlflow_log.station_slug(station)}-{pollutant}-lightgbm-{token}-v{args.version}"
        model_path = save_booster(booster, model_dir, name, iterations)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "serie": f"{station}|{pollutant}", "modelo": f"lightgbm-{token}", "stage": "ae-dmd-prep",
                "particion": part.name, "tipo": "entrenamiento",
            })
            mlflow.log_params({
                **params,
                "alpha": args.alpha, "estacion": station, "pollutant": pollutant, "version": args.version,
                "source_run_id": run_id, "particion": part.name,
                "early_stopping": "off" if rounds_mode else "on",
                **({"num_boost_round": rounds, "rounds_source": "v1-booster"} if rounds_mode else {}),
            })
            metrics = {
                "train_loss": train_loss,
                "best_iteration": float(iterations),
                **{f"sel_{k}": v for k, v in sel_metrics.items()},
            }
            mlflow.log_metrics(_finite_metrics(metrics))
            _log_frames({
                "predictions.parquet": sel_frame[PREDICTIONS_COLUMNS],
                "feature_importance.parquet": _feature_importance_frame(booster, station, pollutant),
                "zone_icap.parquet": zone_icap(sel_frame) if not sel_frame.empty else pd.DataFrame(
                    columns=["timestamp", "icap_true", "icap_pred"]),
                "rolling_scores.parquet": pd.DataFrame({"window_index": pd.Series(dtype=int),
                                                        "rmse": pd.Series(dtype=float)}),
            })
            mlflow.log_artifact(model_path)
        results[f"{station}|{pollutant}"] = {"source_run_id": run_id, "sel": sel_metrics}
        log.info("Refit %s/%s -> sel_weighted_rmse=%s", station, pollutant, sel_metrics.get("weighted_rmse"))

    # --- Persistencia en sel (referencia, D5) ----------------------------------
    persistence = persistence_frame(data["sinca"], features)
    sel_persistence = persistence[pt.row_mask(persistence["timestamp"], part, "sel")].copy()
    sel_persistence["split"] = "sel"
    persist_metrics = evaluate_predictions(sel_persistence, args.alpha) if not sel_persistence.empty else {}
    name = f"sinca-zona-persistence-{token}-v{args.version}"
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"modelo": f"persistence-{token}", "stage": "ae-dmd-prep", "particion": part.name})
        mlflow.log_params({"tipo": "referencia", "alpha": args.alpha, "particion": part.name, "version": args.version})
        mlflow.log_metrics(_finite_metrics({f"sel_{k}": v for k, v in persist_metrics.items()}))
        _log_frames({"predictions.parquet": sel_persistence[PREDICTIONS_COLUMNS]})
    results["persistence"] = {"sel": persist_metrics}
    log.info("Persistencia %s -> sel_weighted_rmse=%s", part.name, persist_metrics.get("weighted_rmse"))

    if not args.no_upload:
        mlflow_log.upload_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, None)
    summary_path = os.path.join(output_dir, f"refit-{part.name}-summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    log.info("Resumen de refit escrito en %s", summary_path)
    return 0


def _flag(name: str) -> list[str]:
    """`--foo-bar` + alias `--foo_bar`.

    SageMaker puede pasar los hiperparámetros como `--clave valor`; aceptar
    ambas formas evita que el job falle por el guion/underscore.
    """
    return [f"--{name}", f"--{name.replace('-', '_')}"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento del baseline LightGBM (Fase 3).")
    parser.add_argument(*_flag("data-bucket"), default=os.environ.get("DATA_BUCKET", "sinca-data"))
    parser.add_argument(*_flag("mlflow-bucket"), default=os.environ.get("MLFLOW_BUCKET", mlflow_log.MLFLOW_BUCKET_DEFAULT))
    parser.add_argument(*_flag("mlflow-prefix"), default=os.environ.get("MLFLOW_PREFIX", mlflow_log.MLFLOW_PREFIX_DEFAULT))
    parser.add_argument(*_flag("mlflow-dir"), default=os.environ.get("MLFLOW_DIR", "/opt/ml/mlflow"))
    parser.add_argument(*_flag("experiment"), default=os.environ.get("MLFLOW_EXPERIMENT", mlflow_log.EXPERIMENT_NAME))
    parser.add_argument(*_flag("local-validated-dir"), default=os.environ.get("LOCAL_VALIDATED_DIR"))
    parser.add_argument(*_flag("model-dir"), default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(*_flag("output-dir"), default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument(*_flag("search-trials"), type=int, default=SEARCH_TRIALS_DEFAULT)
    parser.add_argument(*_flag("search-windows"), type=int, default=SEARCH_WINDOWS_DEFAULT,
                        help="Cantidad de ventanas de validación usadas en el random search (0 = todas).")
    parser.add_argument(*_flag("alpha"), type=float, default=ALPHA_DEFAULT)
    parser.add_argument(*_flag("version"), type=int, default=MODEL_VERSION_DEFAULT)
    parser.add_argument(*_flag("seed"), type=int, default=0)
    parser.add_argument(*_flag("stations"), nargs="*", default=None)
    parser.add_argument(*_flag("pollutants"), nargs="*", default=None)
    parser.add_argument(*_flag("skip-persistence"), action="store_true")
    parser.add_argument(*_flag("no-upload"), action="store_true", help="No leer/escribir S3 (pruebas locales).")
    parser.add_argument(*_flag("mode"), choices=("search", "refit"), default="search",
                        help="search: búsqueda + final (actual). refit: reajuste en una partición con hiperparámetros de runs fuente.")
    parser.add_argument(*_flag("partition"), default="ae-dmd", choices=tuple(pt.PARTITIONS),
                        help="Partición para --mode refit (defecto ae-dmd).")
    parser.add_argument(*_flag("source-run-ids"), default=None,
                        help="Solo refit: slug:pollutant=run_id separados por coma (un run lightgbm por serie).")
    parser.add_argument(*_flag("mlflow-runs"), default="all",
                        help="Descarga selectiva de MLflow: all | none | IDs separados por coma (defecto all).")
    parser.add_argument(*_flag("rounds-from-source"), type=lambda s: s.lower() in ("1", "true", "yes"), default=True,
                        help="Solo refit: num_boost_round = árboles del booster v1 del run fuente, sin validación (D11).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv()
    args = parse_args(argv)
    return run(args)


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    sys.exit(main())
