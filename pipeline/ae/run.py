"""Entry point de los autoencoders y del pase de DMD (change `ae-reconstruccion`).

Modos (`--mode`):

- `search`: búsqueda aleatoria por arquitectura sobre pocas ventanas equiespaciadas
  (`_select_windows`), con early stopping; devuelve la mejor configuración de cada
  arquitectura en un JSON (tarea 5.2).
- `final`: entrena cada arquitectura con todo lo anterior al test, evalúa una vez
  en el test held-out con 3 semillas, y registra un run por semilla en MLflow
  (tarea 5.3–5.4). Incluye compuerta, diagnósticos y ablación de `z_aux`.
- `reference`: runs `ref-mean` y `ref-pca-d{d}` sobre las mismas ventanas (5.5).
- `dmd`: pase de DMD global sobre el latente de un AE congelado, por serie, con
  runs `ae-{arq}-dmd` y el operador como artefacto del run AE (5.5).

Uso (SageMaker Training Job, código vía `source_dir`/`entry_point`):

    python ae/run.py --mode final --arch gru --d 16 --L 72 --data-bucket sinca-data

Uso local (sin escribir en S3):

    python pipeline/ae/run.py --mode final --local-validated-dir ./validated --no-upload
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

# El pipeline es la raíz de importación (`config`, `features`, `ae`, ...).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ae import data as aed  # noqa: E402
from ae import dmd as dmdm  # noqa: E402
from ae import metrics as aem  # noqa: E402
from ae.models import ARCHITECTURES, LATENT_CHOICES, LENGTH_CHOICES, AEConfig  # noqa: E402
from evaluate.evaluate import evaluate_predictions, zone_icap  # noqa: E402
from features.build import build_features, load_validated, make_rolling_windows  # noqa: E402
from train import mlflow_log  # noqa: E402
from train.train import _select_windows  # noqa: E402

log = logging.getLogger("ae")

MODEL_BASE = "sinca-zona-ae"
DEFAULT_EXPERIMENT = mlflow_log.EXPERIMENT_NAME  # `sinca-baseline` (provisional, design Open Questions)
SEEDS_DEFAULT = (0, 1, 2)
DMD_SEED = 0
VALID_FALLBACK_HOURS = 24 * 30


# --- Contexto de entrenamiento -----------------------------------------------


@dataclass
class Context:
    """Normalización y ventanas de una configuración de L y de particiones."""

    grid: aed.HourlyGrid
    norm: aed.Normalizer
    ng: aed.NormalizedGrid
    tr: aed.Windows
    va: aed.Windows | None
    te: aed.Windows | None


def _span_mask(n: int, lo: int, hi: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    return mask


def make_context(
    grid: aed.HourlyGrid,
    length: int,
    train_span: tuple[int, int],
    valid_span: tuple[int, int] | None,
    test_span: tuple[int, int] | None = None,
) -> Context:
    """Normaliza con el train de la partición y corta las ventanas de cada conjunto."""
    norm = aed.fit_normalizer(grid, _span_mask(len(grid.index), *train_span))
    ng = aed.apply_normalizer(grid, norm)
    tr = aed.select_samples(grid, ng, length, *train_span)
    va = None
    if valid_span is not None and valid_span[1] > valid_span[0]:
        va = aed.select_samples(grid, ng, length, *valid_span)
    te = aed.select_samples(grid, ng, length, *test_span) if test_span is not None else None
    return Context(grid=grid, norm=norm, ng=ng, tr=tr, va=va, te=te)


def default_valid_span(grid: aed.HourlyGrid, train_span: tuple[int, int]) -> tuple[int, int]:
    """Validación para early stopping: último bloque de la ventana deslizante dentro del train.

    Igual que el baseline (LightGBM usa la última ventana de valid). Si no hay
    ventana deslizante disponible, se usan los últimos 30 días del train.
    """
    lo, hi = train_span
    for window in reversed(make_rolling_windows()):
        v_lo, v_hi = aed.span(grid.index, window["valid_start"], window["valid_end"])
        if lo <= v_lo and v_hi <= hi and v_hi > v_lo:
            return v_lo, v_hi
    return max(lo, hi - VALID_FALLBACK_HOURS), hi


def _arrays(win: aed.Windows):
    return (win.x_mp, win.x_aux, win.x_mp, win.mp_real, win.y_aux, win.aux_real)


def fit_context(ctx: Context, cfg: AEConfig, epochs: int, patience: int):
    from ae import models as aeml

    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    valid = _arrays(ctx.va) if ctx.va is not None and len(ctx.va) else None
    history = aeml.fit(models, cfg, _arrays(ctx.tr), valid, epochs=epochs, patience=patience)
    return models, history


# --- Reconstrucción y métricas -----------------------------------------------


def stitch(values: np.ndarray, end_positions: np.ndarray, length: int, total: int) -> tuple[np.ndarray, np.ndarray]:
    """Promedio por hora de las ventanas que la cubren. Devuelve `(serie [T,C], cubierta [T])`."""
    n, _, channels = values.shape
    starts = np.asarray(end_positions) - length + 1
    idx = (starts[:, None] + np.arange(length)[None, :]).ravel()
    acc = np.zeros((total, channels))
    cnt = np.zeros(total)
    np.add.at(acc, idx, values.reshape(-1, channels))
    np.add.at(cnt, idx, 1.0)
    covered = cnt > 0
    out = np.zeros_like(acc)
    out[covered] = acc[covered] / cnt[covered, None]
    return out, covered


def _longest_run(mask: np.ndarray) -> slice:
    best, start = (0, 0), None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return slice(*best)


def split_metrics(grid: aed.HourlyGrid, win: aed.Windows, mp_ugm3: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Métricas por serie (µg/m³) y tabla de reconstrucción por hora de la partición."""
    total = len(grid.index)
    stitched, covered = stitch(mp_ugm3, win.end_positions, win.length, total)
    truth = grid.mp.to_numpy(dtype=float)
    real = grid.mp_real.to_numpy(dtype=bool)
    out: dict[str, dict[str, float]] = {}
    frames = []
    seg = _longest_run(covered)
    for c, (_, station, pollutant) in enumerate(aed.mp_channels()):
        key = f"{aed.station_slug(station)}_{pollutant}"
        label = f"{station}|{pollutant}"
        mask = covered & real[:, c]
        metrics = aem.recon_metrics(truth[:, c], stitched[:, c], mask)
        metrics.update(aem.spectral_ratio(truth[seg, c], stitched[seg, c]))
        out[key] = metrics
        pos = np.where(covered)[0]
        frames.append(pd.DataFrame({
            "timestamp": grid.index[pos],
            "serie": label,
            "y_true": np.where(real[pos, c], truth[pos, c], np.nan),
            "y_recon": stitched[pos, c],
        }))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, table


def _flatten_metrics(prefix: str, per_series: dict[str, dict[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, metrics in per_series.items():
        for name, value in metrics.items():
            if value is not None and np.isfinite(value):
                flat[f"{prefix}{name}_{key}"] = float(value)
    return flat


def _mean_metric(per_series: dict[str, dict[str, float]], name: str = "rmse") -> float:
    values = [m[name] for m in per_series.values() if name in m and np.isfinite(m[name])]
    return float(np.mean(values)) if values else float("nan")


def reference_recon_z(kind: str, ctx: Context, win: aed.Windows, d: int) -> np.ndarray:
    """Reconstrucción en espacio z de las referencias: `mean` o PCA(d) sobre el MP de train."""
    if kind == "mean":
        return np.zeros_like(win.x_mp)  # el train tiene media 0 en espacio z
    flat_tr = ctx.tr.x_mp.reshape(len(ctx.tr), -1).astype(float)
    mean, comp = aem.pca_fit(flat_tr, d)
    flat = win.x_mp.reshape(len(win), -1).astype(float)
    return aem.pca_reconstruct(flat, mean, comp).reshape(win.x_mp.shape)


# --- Latente y ablación ------------------------------------------------------


def latent_table(z: np.ndarray, win: aed.Windows, d_aux: int, split: str) -> pd.DataFrame:
    d = z.shape[1]
    cols = [f"z_mp_{i}" for i in range(d - d_aux)] + [f"z_aux_{j}" for j in range(d_aux)]
    frame = pd.DataFrame(z, columns=cols)
    frame.insert(0, "timestamp", win.index)
    frame.insert(1, "split", split)
    return frame


def ablation_delta(models, cfg: AEConfig, ctx: Context, win: aed.Windows, z: np.ndarray, base: dict) -> float:
    """Empeoramiento del RMSE medio de MP al poner `z_aux` en cero (aporte auxiliar)."""
    from ae import models as aeml

    z_ab = aeml.ablate_z_aux(z, models.d_aux)
    mp_ab = aed.mp_to_ugm3(aeml.decode_mp(models, z_ab), ctx.norm)
    per_series, _ = split_metrics(ctx.grid, win, mp_ab)
    return _mean_metric(per_series) - _mean_metric(base)


# --- MLflow ------------------------------------------------------------------


def _log_run(name: str, tags: dict, params: dict, metrics: dict, frames: dict[str, pd.DataFrame], files: dict[str, str] | None = None) -> str:
    import mlflow

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        with tempfile.TemporaryDirectory(prefix="sinca-ae-") as tmp:
            for filename, frame in frames.items():
                frame.to_parquet(os.path.join(tmp, filename), index=False)
            for dest, src in (files or {}).items():
                shutil.copy(src, os.path.join(tmp, dest))
            mlflow.log_artifacts(tmp)
        return run.info.run_id


def run_name_ae(arch: str, version: int) -> str:
    return f"{MODEL_BASE}_{arch}-v{version}"


def config_to_params(cfg: AEConfig, version: int) -> dict:
    return {**cfg.to_dict(), "version": version, "tipo": "entrenamiento"}


# --- Modos -------------------------------------------------------------------


def run_final(grid: aed.HourlyGrid, cfg: AEConfig, args, version: int, model_dir: str) -> dict:
    """Entrena una configuración (una semilla), evalúa en valid y test, registra el run."""
    from ae import models as aeml

    spans = aed.holdout_spans(grid.index)
    valid_span = default_valid_span(grid, spans["train"])
    ctx = make_context(grid, cfg.length, spans["train"], valid_span, spans["test"])
    if len(ctx.tr) == 0 or ctx.te is None or len(ctx.te) == 0:
        raise RuntimeError("Sin muestras de train o test con esta L (revisar huecos y rango de datos).")
    log.info("Entrenando %s L=%d d=%d seed=%d: train=%d valid=%d test=%d", cfg.arch, cfg.length, cfg.d, cfg.seed,
             len(ctx.tr), len(ctx.va) if ctx.va is not None else 0, len(ctx.te))
    models, history = fit_context(ctx, cfg, args.epochs, args.patience)

    results: dict[str, dict] = {}
    tables: dict[str, pd.DataFrame] = {}
    zs: dict[str, np.ndarray] = {"train": aeml.encode(models, cfg, ctx.tr.x_mp, ctx.tr.x_aux)}
    for split, win in (("valid", ctx.va), ("test", ctx.te)):
        if win is None or len(win) == 0:
            continue
        mp_z, _ = aeml.predict_mp_aux(models, cfg, win.x_mp, win.x_aux)
        per_series, table = split_metrics(grid, win, aed.mp_to_ugm3(mp_z, ctx.norm))
        zs[split] = aeml.encode(models, cfg, win.x_mp, win.x_aux)
        results[split] = per_series
        tables[split] = table.assign(split=split)
        pca_metrics, _ = split_metrics(grid, win, aed.mp_to_ugm3(reference_recon_z("pca", ctx, win, cfg.d), ctx.norm))
        results[f"{split}_pca"] = pca_metrics

    gate = aem.gate_decision(results.get("valid", {}), results.get("valid_pca", {}), args.gate_thresholds)
    latent = aem.latent_diagnostics(zs["test"], models.d_aux) if "test" in zs else {}
    ablation = float("nan")
    if "test" in zs:
        ablation = ablation_delta(models, cfg, ctx, ctx.te, zs["test"], results["test"])

    name = run_name_ae(cfg.arch, version)
    seed_dir = os.path.join(model_dir, f"{name}-s{cfg.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    enc_path = os.path.join(seed_dir, "encoder.keras")
    dec_path = os.path.join(seed_dir, "decoder_mp.keras")
    models.encoder.save(enc_path)
    models.decoder_mp.save(dec_path)
    with open(os.path.join(seed_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"cfg": cfg.to_dict(), "gate": gate, "version": version}, handle, ensure_ascii=False, indent=2, default=str)

    metrics = {
        **_flatten_metrics("recon_", results.get("test", {})),
        **_flatten_metrics("valid_recon_", results.get("valid", {})),
        **_flatten_metrics("ref_pca_test_", results.get("test_pca", {})),
        **_flatten_metrics("ref_pca_valid_", results.get("valid_pca", {})),
        "recon_rmse_mean_test": _mean_metric(results.get("test", {})),
        "recon_rmse_mean_valid": _mean_metric(results.get("valid", {})),
        "ablation_zaux_delta_rmse_test": ablation,
        "gate_beats_pca": float(gate["beats_pca"]),
        "epochs_run": float(len(history.history.get("loss", []))),
        "train_loss_final": float(history.history["loss"][-1]) if history.history.get("loss") else float("nan"),
        **{f"latent_{k}": v for k, v in latent.items()},
    }
    if gate["gate_pass"] is not None:
        metrics["gate_pass"] = float(gate["gate_pass"])

    tags = {"modelo": f"ae-{cfg.arch}", "tipo": "entrenamiento", "stage": "ae-reconstruccion", "seed": cfg.seed}
    params = {**config_to_params(cfg, version), "gate_thresholds": json.dumps(args.gate_thresholds)}
    frames = {
        "reconstruction.parquet": pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame(),
        "latent.parquet": pd.concat(
            [latent_table(zs[s], w, models.d_aux, s) for s, w in (("train", ctx.tr), ("valid", ctx.va), ("test", ctx.te)) if s in zs],
            ignore_index=True,
        ),
        "diagnostics.parquet": pd.DataFrame(
            [{"split": s, "serie": k, "metric": m, "value": v} for s, per in results.items() if not s.endswith("_pca") for k, ms in per.items() for m, v in ms.items()]
        ),
    }
    run_id = _log_run(name, tags, params, metrics, frames, files={"encoder.keras": enc_path, "decoder_mp.keras": dec_path})
    log.info("Run %s registrado (seed=%d, gate=%s)", run_id, cfg.seed, gate["gate_pass"])
    return {"run_id": run_id, "name": name, "seed": cfg.seed, "gate": gate, "test_rmse_mean": metrics["recon_rmse_mean_test"], "model_dir": seed_dir}


def run_reference(grid: aed.HourlyGrid, cfg: AEConfig, version: int) -> None:
    spans = aed.holdout_spans(grid.index)
    valid_span = default_valid_span(grid, spans["train"])
    ctx = make_context(grid, cfg.length, spans["train"], valid_span, spans["test"])
    kinds = {"ref-mean": "mean", f"ref-pca-d{cfg.d}": "pca"}
    for name, kind in kinds.items():
        metrics: dict[str, float] = {}
        tables = {}
        for split, win in (("valid", ctx.va), ("test", ctx.te)):
            if win is None or len(win) == 0:
                continue
            per_series, table = split_metrics(grid, win, aed.mp_to_ugm3(reference_recon_z(kind, ctx, win, cfg.d), ctx.norm))
            prefix = "recon_" if split == "test" else "valid_recon_"
            metrics.update(_flatten_metrics(prefix, per_series))
            metrics[f"recon_rmse_mean_{split}"] = _mean_metric(per_series)
            tables[split] = table.assign(split=split)
        tags = {"modelo": name, "tipo": "referencia", "stage": "ae-reconstruccion"}
        params = {"tipo": "referencia", "L": cfg.length, "d": cfg.d, "version": version, "referencia": name}
        frames = {"reconstruction.parquet": pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame()}
        _log_run(name, tags, params, metrics, frames)
        log.info("Referencia %s registrada", name)


def run_dmd(grid: aed.HourlyGrid, cfg: AEConfig, args, ae_run_id: str, ae_dir: str, sinca: pd.DataFrame, open_meteo, feriados) -> None:
    """Pase de DMD sobre el latente congelado; registra runs por serie y el operador."""
    from mlflow.tracking import MlflowClient
    from ae import models as aeml

    with open(os.path.join(ae_dir, "config.json"), encoding="utf-8") as handle:
        saved = json.load(handle)
    gate = saved.get("gate", {})
    if gate.get("gate_pass") is False:
        raise RuntimeError("La compuerta marcó este AE como no aprobado: no se aplica el pase de DMD.")

    import keras
    encoder = keras.saving.load_model(os.path.join(ae_dir, "encoder.keras"), compile=False)
    decoder = keras.saving.load_model(os.path.join(ae_dir, "decoder_mp.keras"), compile=False)
    models = aeml.AEModels(full=None, encoder=encoder, decoder_mp=decoder, decoder_aux=None, d=cfg.d, d_aux=cfg.resolved_d_aux(), aux_input_dim=0)

    spans = aed.holdout_spans(grid.index)
    valid_span = default_valid_span(grid, spans["train"])
    ctx = make_context(grid, cfg.length, spans["train"], valid_span, spans["test"])

    def latent(win):
        # `encode` ya recorta la entrada auxiliar según `aux_input`.
        return aeml.encode(models, cfg, win.x_mp, win.x_aux)

    z_tr = latent(ctx.tr)
    segs_tr = dmdm.split_segments(z_tr, ctx.tr.end_positions)
    fit_full = dmdm.fit_dmd(segs_tr, None)
    trunc_rank = max(2, cfg.d // 2)
    fit_trunc = dmdm.fit_dmd(segs_tr, trunc_rank)

    lin_test = dmdm.linearizability(fit_full["K"], dmdm.split_segments(latent(ctx.te), ctx.te.end_positions))
    lin_trunc = dmdm.linearizability(fit_trunc["K"], dmdm.split_segments(latent(ctx.te), ctx.te.end_positions))
    lin_valid = {}
    if ctx.va is not None and len(ctx.va):
        lin_valid = dmdm.linearizability(fit_full["K"], dmdm.split_segments(latent(ctx.va), ctx.va.end_positions))

    z_te = latent(ctx.te)
    z_pred = dmdm.predict_latent(fit_full["K"], z_te, dmdm.HORIZON_FORECAST)
    decoded = aed.mp_to_ugm3(aeml.decode_mp(models, z_pred), ctx.norm)
    forecast = dmdm.forecast_from_decoded(decoded)  # [N, 6]

    features = build_features(sinca, open_meteo, feriados)
    targets = features[["timestamp", "estacion", "pollutant", "target"]]
    pieces = []
    for c, (_, station, pollutant) in enumerate(aed.mp_channels()):
        pieces.append(pd.DataFrame({
            "timestamp": ctx.te.index, "estacion": station, "pollutant": pollutant, "y_pred": forecast[:, c],
        }))
    preds = pd.concat(pieces, ignore_index=True).merge(targets, on=["timestamp", "estacion", "pollutant"], how="inner")
    preds = preds.rename(columns={"target": "y_true"})
    preds["split"] = "test"
    preds = preds[["timestamp", "estacion", "pollutant", "y_true", "y_pred", "split"]]

    for station, pollutant in sorted({(s, p) for _, s, p in aed.mp_channels()}):
        frame = preds[(preds["estacion"] == station) & (preds["pollutant"] == pollutant)]
        name = mlflow_log.run_name(station, f"ae-{cfg.arch}-dmd", args.version)
        metrics = {f"test_{k}": v for k, v in evaluate_predictions(frame, args.alpha).items()} if not frame.empty else {}
        tags = {"serie": f"{station}|{pollutant}", "modelo": f"ae-{cfg.arch}-dmd", "stage": "ae-dmd", "tipo": "entrenamiento", "ae_run_id": ae_run_id}
        params = {"estacion": station, "pollutant": pollutant, "version": args.version, "tipo": "entrenamiento", "alpha": args.alpha, "rank": "full"}
        frames = {
            "predictions.parquet": frame,
            "zone_icap.parquet": zone_icap(frame) if not frame.empty else pd.DataFrame(columns=["timestamp", "icap_true", "icap_pred"]),
            "rolling_scores.parquet": pd.DataFrame({"window_index": pd.Series(dtype=int), "rmse": pd.Series(dtype=float)}),
        }
        _log_run(name, tags, params, metrics, frames)

    client = MlflowClient()
    ae_metrics = {f"dmd_latent_rmse_{k.rsplit('_', 1)[-1]}": v for k, v in lin_test.items()}
    ae_metrics.update({f"dmd_valid_latent_rmse_{k.rsplit('_', 1)[-1]}": v for k, v in lin_valid.items()})
    ae_metrics.update({f"dmd_trunc_latent_rmse_{k.rsplit('_', 1)[-1]}": v for k, v in lin_trunc.items()})
    ae_metrics["dmd_trunc_rank"] = float(trunc_rank)
    for key, value in ae_metrics.items():
        if np.isfinite(value):
            client.log_metric(ae_run_id, key, float(value))

    with tempfile.TemporaryDirectory(prefix="sinca-dmd-") as tmp:
        full_path = os.path.join(tmp, "dmd_operator.npz")
        trunc_path = os.path.join(tmp, "dmd_operator_trunc.npz")
        np.savez(full_path, **dmdm.dmd_operator_arrays(fit_full))
        np.savez(trunc_path, **dmdm.dmd_operator_arrays(fit_trunc))
        client.log_artifact(ae_run_id, full_path)
        client.log_artifact(ae_run_id, trunc_path)
    log.info("Pase DMD registrado para %s (linealizabilidad test: %s)", ae_run_id, lin_test)


# --- Búsqueda ----------------------------------------------------------------

SEARCH_SPACE: dict[str, list] = {
    "d": list(LATENT_CHOICES),
    "length": list(LENGTH_CHOICES),
    "hidden": [32, 64, 128],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [1e-3, 3e-3],
    "batch_size": [64, 128],
}


def run_search(grid: aed.HourlyGrid, args, windows: list[dict] | None = None) -> dict:
    """Búsqueda aleatoria por arquitectura; devuelve la mejor configuración de cada una."""
    from ae import models as aeml

    rng = np.random.default_rng(args.seed)
    windows = windows if windows is not None else make_rolling_windows()
    spans = aed.rolling_spans(grid.index, windows)
    pairs = [(w, s) for w, s in zip(windows, spans) if s["train"][1] > s["train"][0] and s["valid"][1] > s["valid"][0]]
    chosen = _select_windows(pairs, args.search_windows)
    best: dict[str, dict] = {}
    for arch in args.archs:
        history = []
        for trial in range(args.search_trials):
            choice = {name: values[int(rng.integers(len(values)))] for name, values in SEARCH_SPACE.items()}
            cfg = AEConfig(arch=arch, length=choice["length"], d=choice["d"], hidden=choice["hidden"], dropout=choice["dropout"],
                           lr=choice["lr"], batch_size=choice["batch_size"], w_mp=args.w_mp, w_aux=args.w_aux,
                           aux_input=args.aux_input, seed=args.seed + trial)
            scores = []
            for _, span_pair in chosen:
                ctx = make_context(grid, cfg.length, span_pair["train"], span_pair["valid"])
                if len(ctx.tr) < 24 or ctx.va is None or len(ctx.va) == 0:
                    continue
                models, _ = fit_context(ctx, cfg, args.search_epochs, args.patience)
                mp_z, _ = aeml.predict_mp_aux(models, cfg, ctx.va.x_mp, ctx.va.x_aux)
                recon = aed.mp_to_ugm3(mp_z, ctx.norm)
                truth = grid.mp.to_numpy()[np.add.outer(ctx.va.end_positions - cfg.length + 1, np.arange(cfg.length))]
                real = ctx.va.mp_real
                err = (recon - truth)[real]
                scores.append(float(np.sqrt(np.mean(err**2))) if err.size else float("nan"))
            score = float(np.nanmean(scores)) if scores else float("nan")
            history.append({"trial": trial, "config": cfg.to_dict(), "score": score})
            log.info("  %s trial %d/%d score=%.4f %s", arch, trial + 1, args.search_trials, score, cfg.to_dict())
        finite = [h for h in history if np.isfinite(h["score"])]
        best[arch] = {"best": min(finite, key=lambda h: h["score"]) if finite else None, "history": history}
    return best


# --- Orquestación ------------------------------------------------------------


def _load_data(args):
    data = load_validated(args.data_bucket, client=None, local_dir=args.local_validated_dir)
    return data


def _config_from_args(args, arch: str, seed: int) -> AEConfig:
    overrides = {}
    if args.search_report:
        with open(args.search_report, encoding="utf-8") as handle:
            report = json.load(handle)
        best = (report.get(arch) or {}).get("best")
        if best:
            overrides = best["config"]
    base = dict(arch=arch, length=args.length, d=args.d, d_aux=args.d_aux, hidden=args.hidden, dropout=args.dropout,
                lr=args.lr, batch_size=args.batch_size, w_mp=args.w_mp, w_aux=args.w_aux, aux_input=args.aux_input, seed=seed)
    base.update({k: v for k, v in overrides.items() if k in base and k != "seed"})
    return AEConfig(**base)


def run(args) -> int:
    model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "model")
    output_dir = args.output_dir or os.environ.get("SM_OUTPUT_DATA_DIR", "output")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    data = _load_data(args)
    grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
    log.info("Grilla: %d horas, %d canales MP, %d auxiliares", len(grid.index), aed.N_MP, aed.N_AUX)

    if args.mode == "search":
        best = run_search(grid, args)
        path = os.path.join(output_dir, "ae_search.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(best, handle, ensure_ascii=False, indent=2, default=str)
        log.info("Búsqueda escrita en %s", path)
        return 0

    if not args.no_upload:
        mlflow_log.download_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, None)
    mlflow_log.configure(args.mlflow_dir, experiment=args.experiment)

    if args.mode == "final":
        for seed in args.seeds:
            cfg = _config_from_args(args, args.arch, seed)
            run_final(grid, cfg, args, args.version, model_dir)
    elif args.mode == "reference":
        run_reference(grid, _config_from_args(args, args.arch, 0), args.version)
    elif args.mode == "dmd":
        if not args.ae_dir or not args.ae_run_id:
            raise SystemExit("--mode dmd requiere --ae-dir y --ae-run-id")
        cfg = _config_from_args(args, args.arch, DMD_SEED)
        run_dmd(grid, cfg, args, args.ae_run_id, args.ae_dir, data["sinca"], data["open_meteo"], data["feriados"])

    if not args.no_upload:
        mlflow_log.upload_state(args.mlflow_bucket, args.mlflow_prefix, args.mlflow_dir, None)
    return 0


def _flag(name: str) -> list[str]:
    return [f"--{name}", f"--{name.replace('-', '_')}"]


def _gate_thresholds(text: str | None) -> dict | None:
    return json.loads(text) if text else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoders con rama auxiliar y pase de DMD (ae-reconstruccion).")
    parser.add_argument(*_flag("mode"), choices=("search", "final", "reference", "dmd"), default="final")
    parser.add_argument(*_flag("data-bucket"), default=os.environ.get("DATA_BUCKET", "sinca-data"))
    parser.add_argument(*_flag("mlflow-bucket"), default=os.environ.get("MLFLOW_BUCKET", mlflow_log.MLFLOW_BUCKET_DEFAULT))
    parser.add_argument(*_flag("mlflow-prefix"), default=os.environ.get("MLFLOW_PREFIX", mlflow_log.MLFLOW_PREFIX_DEFAULT))
    parser.add_argument(*_flag("mlflow-dir"), default=os.environ.get("MLFLOW_DIR", "/opt/ml/mlflow"))
    parser.add_argument(*_flag("experiment"), default=os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
    parser.add_argument(*_flag("local-validated-dir"), default=os.environ.get("LOCAL_VALIDATED_DIR"))
    parser.add_argument(*_flag("model-dir"), default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(*_flag("output-dir"), default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument(*_flag("arch"), choices=ARCHITECTURES, default="mlp")
    parser.add_argument(*_flag("archs"), nargs="*", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument(*_flag("length"), type=int, choices=LENGTH_CHOICES, default=24)
    parser.add_argument(*_flag("d"), type=int, choices=LATENT_CHOICES, default=16)
    parser.add_argument(*_flag("d-aux"), type=int, default=None)
    parser.add_argument(*_flag("hidden"), type=int, default=64)
    parser.add_argument(*_flag("dropout"), type=float, default=0.1)
    parser.add_argument(*_flag("lr"), type=float, default=1e-3)
    parser.add_argument(*_flag("batch-size"), type=int, default=128)
    parser.add_argument(*_flag("w-mp"), type=float, default=1.0)
    parser.add_argument(*_flag("w-aux"), type=float, default=0.1)
    parser.add_argument(*_flag("aux-input"), type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    parser.add_argument(*_flag("seeds"), type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS_DEFAULT)
    parser.add_argument(*_flag("seed"), type=int, default=0)
    parser.add_argument(*_flag("version"), type=int, default=1)
    parser.add_argument(*_flag("alpha"), type=float, default=1.0)
    parser.add_argument(*_flag("epochs"), type=int, default=60)
    parser.add_argument(*_flag("search-epochs"), type=int, default=15)
    parser.add_argument(*_flag("patience"), type=int, default=8)
    parser.add_argument(*_flag("search-trials"), type=int, default=6)
    parser.add_argument(*_flag("search-windows"), type=int, default=4)
    parser.add_argument(*_flag("search-report"), default=None, help="JSON de `search`: usa la mejor config de --arch.")
    parser.add_argument(*_flag("gate-thresholds"), type=_gate_thresholds, default=None, help="JSON con umbrales; vacío = solo reporta.")
    parser.add_argument(*_flag("ae-dir"), default=None)
    parser.add_argument(*_flag("ae-run-id"), default=None)
    parser.add_argument(*_flag("no-upload"), action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
