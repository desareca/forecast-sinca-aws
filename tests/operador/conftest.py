"""Fixtures del change `dmd-operador-global` (sin AWS). Helpers en `sint_operador.py`."""
import pytest

from sint_operador import ARTIFACT_LOCATION, EXPERIMENT, _ae_conftest, sistema_lineal, synthetic_validated

# Los tests de `tests/ae` importan `TZ`/`synthetic_validated` con `from conftest import ...` dentro de
# funciones: con ambos `conftest` cargados, cualquiera de los dos debe exponerlos.
TZ = _ae_conftest.TZ
STATION_NAMES = _ae_conftest.STATION_NAMES


@pytest.fixture
def lineal():
    return sistema_lineal()


@pytest.fixture
def grilla_280():
    """Grilla horaria de 280 días que cubre los 4 tramos de la partición `ae-dmd`."""
    from ae import data as aed

    data = synthetic_validated(start="2025-11-01", hours=24 * 280, gaps=True)
    return aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"]), data


@pytest.fixture
def estado_mlflow(tmp_path):
    """Estado MLflow vacío con `artifact_location` de `/opt/ml` (como el descargado de S3)."""
    import mlflow

    state = tmp_path / "mlflow-op"
    state.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{(state / 'mlflow.sqlite').as_posix()}")
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    return state
