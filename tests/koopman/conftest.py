"""Fixtures del change `deep-koopman` (sin AWS). Helpers en `sint_koopman.py`."""
import pytest

from sint_koopman import ARTIFACT_LOCATION, EXPERIMENT, _ae_conftest, synthetic_validated

# `tests/ae` importa `TZ`/`synthetic_validated` con `from conftest import ...` dentro de funciones.
TZ = _ae_conftest.TZ
STATION_NAMES = _ae_conftest.STATION_NAMES


@pytest.fixture
def estado_mlflow(tmp_path):
    import mlflow

    state = tmp_path / "mlflow"
    state.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{(state / 'mlflow.sqlite').as_posix()}")
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    return state


@pytest.fixture(scope="session")
def grilla_kp():
    from ae import data as aed

    data = synthetic_validated(start="2025-11-01", hours=24 * 280, gaps=True)
    return aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"]), data
