"""Configuración de tests del change `ae-reconstruccion`.

El pipeline (`pipeline/`) es la raíz de importación (`config`, `features`, `ae`, ...),
igual que en SageMaker. Los tests no usan AWS: datos sintéticos y MLflow local.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

TZ = "America/Santiago"
STATION_NAMES = ("Padre Las Casas II", "Nielol", "Las Encinas")


def synthetic_validated(start: str = "2026-01-01", hours: int = 24 * 60, seed: int = 0, gaps: bool = True):
    """Datos validados sintéticos: SINCA por estación, Open-Meteo y feriados.

    Con `gaps=True` introduce un hueco corto (3 h) y uno largo (10 h) en Ñielol.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=hours, freq="h", tz=TZ)
    sinca_frames = []
    for station in STATION_NAMES:
        base = 40.0 if station != "Nielol" else 55.0
        pm25 = np.clip(base + 15 * np.sin(np.arange(hours) / 24 * 2 * np.pi) + rng.normal(0, 4, hours), 1, None)
        pm10 = pm25 * 1.8
        frame = pd.DataFrame({
            "estacion": station,
            "timestamp": ts,
            "pm25": pm25,
            "pm10": pm10,
            "pres": 1010 + rng.normal(0, 2, hours),
            "wspd": np.abs(rng.normal(2, 1, hours)),
            "rain": np.abs(rng.normal(0, 0.2, hours)),
            "temp": 10 + rng.normal(0, 2, hours),
        })
        if gaps and station == "Nielol":
            frame.loc[100:102, "pm25"] = np.nan   # hueco corto (3 h)
            frame.loc[200:209, "pm25"] = np.nan   # hueco largo (10 h)
        sinca_frames.append(frame)
    sinca = pd.concat(sinca_frames, ignore_index=True)
    open_meteo = pd.DataFrame({"timestamp": ts, "boundary_layer_height": 300 + rng.normal(0, 20, hours)})
    feriados = pd.DataFrame({"fecha": [pd.Timestamp("2026-01-01").date()]})
    return {"sinca": sinca, "open_meteo": open_meteo, "feriados": feriados}


@pytest.fixture
def validated():
    return synthetic_validated()
