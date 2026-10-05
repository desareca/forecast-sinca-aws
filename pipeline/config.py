"""Configuración estática del pipeline de datos.

Contiene las estaciones SINCA de la zona saturada Temuco/Padre Las Casas, el
mapeo de parámetros SINCA a columnas/unidades, los endpoints de las fuentes y
las constantes de particionado S3.

No contiene credenciales: todas las fuentes son HTTP público sin key y el
acceso a S3 usa el credential chain estándar (task role en Fargate, o
`AWS_PROFILE` explícito en local).
"""
from __future__ import annotations

# --- Estaciones --------------------------------------------------------------
# `id`   -> ID de la página SINCA (index.php/estacion/index/id/<id>)
# `code` -> código de red usado en el `macropath` (./<code>/Cal/...)
# `slug` -> nombre seguro para particiones S3
STATIONS: dict[str, dict[str, str]] = {
    "Padre Las Casas II": {
        "id": "263",
        "code": "RIX/902",
        "slug": "padre_las_casas_ii",
    },
    "Nielol": {
        "id": "237",
        "code": "RIX/905",
        "slug": "nielol",
    },
    "Las Encinas": {
        "id": "186",
        "code": "RIX/901",
        "slug": "las_encinas",
    },
}

# --- Parámetros SINCA --------------------------------------------------------
# Segmento final del `macropath` -> (columna, unidad).
CAL_VARS: dict[str, tuple[str, str]] = {
    "PM10": ("pm10", "ug/m3"),
    "PM25": ("pm25", "ug/m3"),
    "0001": ("so2", "ppm"),
    "0002": ("no", "ppm"),
    "0003": ("no2", "ppm"),
    "0004": ("co", "ppm"),
    "0008": ("o3", "ppm"),
    "0NOX": ("nox", "ppm"),
}

MET_VARS: dict[str, tuple[str, str]] = {
    "TEMP": ("temp", "C"),
    "RHUM": ("rhum", "%"),
    "PRES": ("pres", "hPa"),
    "RAIN": ("rain", "mm"),
    "WDIR": ("wdir", "deg"),
    "WSPD": ("wspd", "m/s"),
    "GLOB": ("glob", "W/m2"),
}

VAR_UNITS: dict[str, str] = {
    **{v[0]: v[1] for v in CAL_VARS.values()},
    **{v[0]: v[1] for v in MET_VARS.values()},
}

# Rangos plausibles por variable (dimensión "validez").
# Los gases admiten un pequeño margen negativo por ruido instrumental
# (el SINCA entrega valores levemente negativos en horas limpias).
VAR_RANGES: dict[str, tuple[float, float]] = {
    "pm10": (0.0, 1000.0),
    "pm25": (0.0, 1000.0),
    "so2": (-50.0, 2000.0),
    "no": (-50.0, 2000.0),
    "no2": (-50.0, 2000.0),
    "co": (-50.0, 100.0),
    "o3": (-50.0, 1000.0),
    "nox": (-50.0, 3000.0),
    "temp": (-30.0, 50.0),
    "rhum": (0.0, 100.0),
    "pres": (900.0, 1100.0),
    "rain": (0.0, 500.0),
    "wdir": (0.0, 360.0),
    "wspd": (0.0, 50.0),
    "glob": (0.0, 1500.0),
}

# Series target (6 series): MP2.5 y MP10 por estación. Completitud estricta.
TARGET_VARS: list[str] = ["pm25", "pm10"]

# --- Fuentes -----------------------------------------------------------------
SINCA_BASE = "https://sinca.mma.gob.cl"
SINCA_CGI = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi"
SINCA_STATION_PAGE = "https://sinca.mma.gob.cl/index.php/estacion/index/id/{id}"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Punto de referencia para la altura de capa límite (Padre Las Casas II).
OPEN_METEO_LAT = -38.7648
OPEN_METEO_LON = -72.5988

NAGER_HOLIDAYS = "https://date.nager.at/api/v3/PublicHolidays/{year}/CL"
# Región de La Araucanía (ISO 3166-2:CL-AR); se conservan los feriados
# nacionales (global=true) + los regionales de La Araucanía.
NAGER_REGION_ARAUCANIA = "CL-AR"

TIMEZONE = "America/Santiago"

# --- Persistencia ------------------------------------------------------------
DATA_BUCKET_DEFAULT = "sinca-data"

# Umbral de detención: si la tasa de cuarentena de una estación supera este
# valor, el pipeline NO persiste `validated` para esa estación (dimensión
# crítica de calidad). Configurable por env `MAX_QUARANTINE_RATE`.
MAX_QUARANTINE_RATE_DEFAULT = 0.2

# Umbral de completitud (dimensión "completitud") para las series target.
MAX_NULL_RATE_TARGET = 0.05

# Frescura (dimensión "oportunidad"): el último timestamp debe estar dentro de
# las últimas N horas. Solo aplica en modo incremental.
FRESHNESS_HOURS = 48

# Ventana GEC (gestión de episodios críticos): 1 abril - 15 septiembre.
GEC_START = (4, 1)
GEC_END = (9, 15)
