"""Tests de la partición `ae-dmd` (change `ae-dmd-prep`, tarea 1.3)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ae import data as aed
from conftest import synthetic_validated
from features import partition as pt
from features.build import TEST_END, TEST_START


def test_tramos_y_token_de_ae_dmd():
    part = pt.get_partition("ae-dmd")
    assert pt.TRAMOS == ("train", "es", "sel", "test")
    assert set(part.bounds) == set(pt.TRAMOS)
    assert part.bounds["train"] == (date(2021, 1, 1), date(2026, 3, 1))
    assert part.bounds["es"] == (date(2026, 3, 1), date(2026, 4, 1))
    assert part.bounds["sel"] == (date(2026, 4, 1), date(2026, 7, 1))
    assert part.token == "aedmd"


def test_test_es_el_held_out_de_features_build():
    part = pt.PARTITIONS["ae-dmd"]
    assert part.bounds["test"] == (TEST_START, TEST_END + timedelta(days=1))


def test_row_mask_excluye_ultimas_24h_y_acepta_inicio():
    part = pt.get_partition("ae-dmd")
    stamps = pd.DatetimeIndex(["2026-03-01 00:00", "2026-03-30 23:00", "2026-03-31 00:00", "2026-03-31 23:00"])
    mask = pt.row_mask(stamps, part, "es")
    assert mask.tolist() == [True, True, False, False]


def test_row_mask_naive_y_con_tz_dan_lo_mismo():
    part = pt.get_partition("ae-dmd")
    naive = pd.date_range("2026-03-25", "2026-04-05", freq="6h")
    aware = naive.tz_localize("America/Santiago")
    assert np.array_equal(pt.row_mask(naive, part, "es"), pt.row_mask(aware, part, "es"))


def test_partition_spans_contiguos_sin_solape():
    part = pt.get_partition("ae-dmd")
    data = synthetic_validated(start="2026-01-01", hours=280 * 24)
    grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
    spans = aed.partition_spans(grid.index, part)
    assert set(spans) == set(pt.TRAMOS)
    order = [spans[t] for t in pt.TRAMOS]
    for (_, hi), (lo, _) in zip(order, order[1:]):
        assert hi == lo  # contiguos: el fin de un tramo es el inicio del siguiente
    assert all(lo < hi for lo, hi in order)
    assert spans["train"][0] == 0


def test_get_partition_desconocida_lanza_valueerror():
    with pytest.raises(ValueError, match="ae-dmd"):
        pt.get_partition("x")


def test_to_json_es_estable_y_tiene_bounds():
    import json

    part = pt.get_partition("ae-dmd")
    payload = json.loads(pt.to_json(part))
    assert payload["name"] == "ae-dmd"
    assert payload["bounds"]["sel"] == ["2026-04-01", "2026-07-01"]
    assert payload["embargo_hours"] == pt.EMBARGO_HOURS
