"""Helpers sintéticos del change `deep-koopman` (nombre único; no choca con otros `conftest`)."""
import datetime as dt
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

_spec = importlib.util.spec_from_file_location("ae_conftest_sintetico_k", ROOT / "tests" / "ae" / "conftest.py")
_ae_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae_conftest)
synthetic_validated = _ae_conftest.synthetic_validated

ARTIFACT_LOCATION = "file:///opt/ml/mlflow/mlruns"
EXPERIMENT = "sinca-baseline"
T0 = dt.datetime(2026, 11, 18, 4, 14, 52, tzinfo=dt.timezone.utc)


class FakeS3:
    """S3 en memoria: `head_object`, `copy_object`, `upload_file`, `put_object`, `get_object`."""

    def __init__(self):
        self.last_modified = T0
        self.uploads: list[str] = []
        self.copies: list[str] = []
        self.objects: dict[str, bytes] = {}

    def head_object(self, Bucket, Key):
        if Key.endswith("mlflow.sqlite"):
            return {"LastModified": self.last_modified}
        if Key not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"LastModified": self.last_modified}

    def copy_object(self, Bucket, Key, CopySource):
        self.copies.append(Key)

    def upload_file(self, filename, bucket, key):
        self.uploads.append(key)
        self.objects[key] = pathlib.Path(filename).read_bytes()
        if key.endswith("mlflow.sqlite"):
            self.last_modified = self.last_modified + dt.timedelta(minutes=5)

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.uploads.append(Key)

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objects[Key])}


def contexto_y_posiciones(grid, n: int | None = 600):
    """Contexto `ae-dmd` (normalizador de train) y posiciones de train elegibles (recortadas a `n`)."""
    from features import partition as pt
    from operador import estado as E

    ctx = E.contexto(grid, pt.get_partition("ae-dmd"), ("train",))
    pos, split = E.posiciones(ctx, 24, ("train",))
    fin = ctx.spans["train"][1]
    if n is not None:
        pos = pos[:n]
        fin = int(pos[-1]) + 30
    return ctx, pos, fin


def modelo_chico(encoder="gru", decoder="lineal", d=10, variante="", seed=0):
    from koopman.modelo import KoopmanModelo

    return KoopmanModelo(encoder=encoder, decoder=decoder, d=d, variante=variante, hidden=8, seed=seed)


def _log(state, name, tags, params, frames=None, arrays=None, files=None):
    import shutil

    from operador import registro as G

    run_id = G.log_variante(str(state), name, tags, params, {}, frames or {}, arrays)
    for dest, src in (files or {}).items():
        shutil.copy(src, G.artifacts_dir(str(state), run_id) / dest)
    return run_id


def estado_referencias(tmp, hidden: int = 8):
    """Estado MLflow sintético con los runs de referencia que usa `deep-koopman` (sin AWS):
    persistencia/LightGBM/DMD v2 (filas fijas), `lightgbm-v1` (test), GRU 7.1 `e16` v2 (3 semillas),
    AE GRU v2 s0 (`encoder.keras`) y `e15a` s0 (`operator.npz`). Devuelve `(state, validated, grid, data)`."""
    import mlflow
    import numpy as np

    from ae import data as aed
    from ae import models as aeml
    from baseline.persistence import predict_frame
    from features import partition as pt
    from features.build import build_features

    data = synthetic_validated(start="2025-11-01", hours=24 * 280, gaps=True)
    validated = tmp / "validated"
    for key in ("sinca", "open_meteo", "feriados"):
        (validated / key).mkdir(parents=True, exist_ok=True)
        data[key].to_parquet(validated / key / "part-0.parquet", index=False)
    state = tmp / "mlflow"
    state.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(state / 'mlflow.sqlite').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
    part = pt.get_partition("ae-dmd")
    feats = build_features(data["sinca"], data["open_meteo"], data["feriados"])
    allp = predict_frame(data["sinca"], feats)
    sel = allp[pt.row_mask(allp["timestamp"], part, "sel")].assign(split="sel")
    test = allp[pt.row_mask(allp["timestamp"], part, "test")].assign(split="test")
    c = {"particion": "ae-dmd", "stage": "ae-dmd-prep"}
    _log(state, "sinca-zona-persistence-aedmd-v2", {**c, "modelo": "persistence-aedmd"}, {"version": 2},
         {"predictions.parquet": sel})
    for (st, po), g in sel.groupby(["estacion", "pollutant"]):
        gt = test[(test.estacion == st) & (test.pollutant == po)]
        src = _log(state, f"lgbm-v1-{st}-{po}", {"modelo": "lightgbm"}, {"version": 1, "estacion": st, "pollutant": po},
                   {"predictions.parquet": gt.assign(y_pred=gt.y_pred * 1.05)})
        _log(state, f"lgbm-{st}-{po}", {**c, "modelo": "lightgbm-aedmd"},
             {"version": 2, "estacion": st, "pollutant": po, "source_run_id": src},
             {"predictions.parquet": g.assign(y_pred=g.y_pred * 1.1)})
        _log(state, f"dmd-{st}-{po}", {**c, "modelo": "ae-gru-dmd", "seed": "0"},
             {"version": 2, "estacion": st, "pollutant": po}, {"predictions.parquet": g.assign(y_pred=g.y_pred * 1.2)})
    for s in (0, 1, 2):
        _log(state, f"sinca-zona-dmdop-gru-e16-aedmd-v2-s{s}", {"stage": "dmd-operador-global", "modelo": "dmdop-gru"},
             {"version": 2}, {"predictions.parquet": sel.assign(y_pred=sel.y_pred * (1.05 + 0.01 * s))})
    cfg = aeml.AEConfig(arch="gru", length=24, d=16, hidden=hidden, seed=0)
    ae = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    ae.encoder.save(tmp / "enc.keras")
    ae.decoder_mp.save(tmp / "dec.keras")
    _log(state, "sinca-zona-ae_gru-aedmd-v2-s0", {**c, "modelo": "ae-gru", "seed": "0"},
         {"version": 2, "aux_input": True, "hidden": hidden},
         files={"encoder.keras": tmp / "enc.keras", "decoder_mp.keras": tmp / "dec.keras"},
         frames={"latent.parquet": __import__("pandas").DataFrame({"a": [0]})})
    lam = np.array([0.99 * np.exp(1j * 2 * np.pi / 24), 0.995 * np.exp(1j * 2 * np.pi / 300), 0.98, 0.97, 0.9])
    lam = np.concatenate([lam, np.conj(lam[:2])])
    _log(state, "sinca-zona-dmdop-gru-e15a-aedmd-v1-s0", {"stage": "dmd-operador-global", "modelo": "dmdop-gru", "seed": "0"},
         {"version": 1}, arrays={"operator.npz": {"A": np.diag(lam.real), "eigvals_real": lam.real, "eigvals_imag": lam.imag}})
    return state, validated, grid, data
