"""Tarea 3.3: checkpoints por época y retoma."""
import pytest

tf = pytest.importorskip("tensorflow")

from sint_koopman import contexto_y_posiciones, modelo_chico
from koopman import entrenar as T

CFG = {"congeladas": 1, "curriculo": ((2, 1), (2, 6))}
CLAVES = ("epoca", "H", "congelado", "lr", "orden")


def _datos(grid):
    ctx, pos, fin = contexto_y_posiciones(grid, 260)
    cols = [c for c, _, _ in T.aed.mp_channels()]
    return T.Datos(grid, ctx.ng, pos, fin), (ctx.norm.mean[cols].to_numpy(), ctx.norm.std[cols].to_numpy())


def test_retoma_tras_la_epoca_2(grilla_kp, tmp_path):
    datos, esc = _datos(grilla_kp[0])
    sin = T.entrenar(modelo_chico(seed=4), datos, "f1", esc, en_caliente=True, epocas=4, cfg_perdida=CFG,
                     ckpt_dir=tmp_path / "a" / "principal", seed=4, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="época 2"):
        T.entrenar(modelo_chico(seed=4), datos, "f1", esc, en_caliente=True, epocas=4, cfg_perdida=CFG,
                   ckpt_dir=tmp_path / "b" / "principal", seed=4, fallar_tras_epoca=2, log=lambda *_: None)
    logs = []
    con = T.entrenar(modelo_chico(seed=4), datos, "f1", esc, en_caliente=True, epocas=4, cfg_perdida=CFG,
                     ckpt_dir=tmp_path / "b" / "principal", seed=4, log=logs.append)
    assert con["retomado_en"] == 2 and any("Retoma f1 en la época 3" in x for x in logs)
    assert len(con["historia"]) == 4
    assert [{k: f[k] for k in CLAVES} for f in con["historia"]] == [{k: f[k] for k in CLAVES} for f in sin["historia"]]
    assert [f["H"] for f in con["historia"]] == [1, 1, 6, 6]
    # variante terminada no se repite
    otra = T.entrenar(modelo_chico(seed=4), datos, "f1", esc, en_caliente=True, epocas=4, cfg_perdida=CFG,
                      ckpt_dir=tmp_path / "b" / "principal", seed=4, log=logs.append)
    assert otra["retomado_en"] is None and len(otra["historia"]) == 4
