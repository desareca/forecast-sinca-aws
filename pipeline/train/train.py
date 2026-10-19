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
):
    """Entrena un booster LightGBM con early stopping sobre `valid_df` (si se pasa)."""
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
        num_boost_round=NUM_BOOST_ROUND,
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


def save_booster(booster, model_dir: str, name: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.txt")
    booster.save_model(path, num_iteration=booster.best_iteration or NUM_BOOST_ROUND)
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

    if not args.no_upload:
        downloaded = mlflow_log.download_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, client)
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
