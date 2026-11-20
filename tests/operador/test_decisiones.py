"""Tarea 1.8: arrastre de decisiones entre etapas."""
import json

import pytest

from operador import registro as G


def test_etapa_sin_decision_previa_falla(tmp_path):
    path = tmp_path / "decisiones.json"
    path.write_text("{}\n")
    assert G.exigir_decision("1.0", G.leer_decisiones(path)) == {}
    with pytest.raises(SystemExit, match="1.0"):
        G.exigir_decision("1.1", G.leer_decisiones(path))


def test_registrar_y_arrastrar(tmp_path):
    path = tmp_path / "decisiones.json"
    G.registrar_decision("1.0", "AE siguen", {"ae": True}, "regla D9", "2026-11-19", path)
    data = json.loads(path.read_text())
    assert set(data["1.0"]) == {"ganador", "config", "fecha", "nota"}
    assert G.exigir_decision("1.1", G.leer_decisiones(path)) == {"ae": True}
    with pytest.raises(SystemExit):
        G.exigir_decision("1.2", G.leer_decisiones(path))
    with pytest.raises(ValueError):
        G.registrar_decision("cierre", "x", {}, "", "", path)


def test_cli_rechaza_etapa_sin_decision(tmp_path):
    from operador import run as R

    path = tmp_path / "decisiones.json"
    path.write_text("{}")
    args = R.parse_args(["--etapa", "1.2", "--state-dir", str(tmp_path), "--local-validated-dir", str(tmp_path),
                         "--no-download", "--decisiones", str(path)])
    with pytest.raises(SystemExit, match="1.1"):
        R.run(args)
