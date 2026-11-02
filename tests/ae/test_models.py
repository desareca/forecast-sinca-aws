"""Tests de los autoencoders en Keras (tarea 2.5): formas, enmascarado y pesos."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ae import data as aed  # noqa: E402
from ae import models as aeml  # noqa: E402
from ae.models import ARCHITECTURES, AEConfig  # noqa: E402

N, L = 6, 24


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    x_mp = rng.normal(size=(N, L, aed.N_MP)).astype("float32")
    x_aux = rng.normal(size=(N, L, aed.AUX_INPUT_DIM)).astype("float32")
    y_aux = rng.normal(size=(N, L, aed.N_AUX)).astype("float32")
    real_mp = np.ones((N, L, aed.N_MP), dtype=bool)
    real_aux = np.ones((N, L, aed.N_AUX), dtype=bool)
    return x_mp, x_aux, x_mp.copy(), real_mp, y_aux, real_aux


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("aux_input", [True, False])
def test_formas_entrada_salida_por_variante(arch, aux_input):
    cfg = AEConfig(arch=arch, length=L, d=8, hidden=8, aux_input=aux_input)
    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    x_mp, x_aux, *_ = _batch()
    sl = aed.aux_input_columns(aux_input)
    z = models.encoder.predict([x_mp, x_aux[..., sl]], verbose=0)
    assert z.shape == (N, 8)
    assert models.d_aux == 2 and models.z_mp_dim == 6
    assert models.decoder_mp.predict(z, verbose=0).shape == (N, L, aed.N_MP)
    assert models.decoder_aux.predict(z, verbose=0).shape == (N, L, aed.N_AUX)
    mp, aux = aeml.predict_mp_aux(models, cfg, x_mp, x_aux)
    assert mp.shape == (N, L, aed.N_MP) and aux.shape == (N, L, aed.N_AUX)


def test_decoder_mp_ve_latente_completo():
    cfg = AEConfig(arch="mlp", length=L, d=8, hidden=8)
    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    assert models.decoder_mp.input_shape == (None, 8)


def test_salida_lineal_sin_activacion():
    cfg = AEConfig(arch="linear_decoder", length=L, d=8, hidden=8)
    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    last = models.decoder_mp.layers[-1]
    assert last.get_config().get("activation") in (None, "linear")


def test_loss_ignora_valores_enmascarados():
    y_pred = tf.constant(np.zeros((2, 5, 3), dtype="float32"))
    target = np.ones((2, 5, 3), dtype="float32")
    real = np.ones((2, 5, 3), dtype="float32")
    base = aeml.masked_channel_mse(tf.constant(np.concatenate([target, real], -1)), y_pred).numpy()
    # Cambiar el objetivo en posiciones enmascaradas no altera la pérdida.
    real_masked = real.copy()
    real_masked[:, 2, :] = 0.0
    target_changed = target.copy()
    target_changed[:, 2, :] = 1e6
    a = aeml.masked_channel_mse(tf.constant(np.concatenate([target, real_masked], -1)), y_pred).numpy()
    b = aeml.masked_channel_mse(tf.constant(np.concatenate([target_changed, real_masked], -1)), y_pred).numpy()
    assert np.allclose(a, b)
    assert np.allclose(base, 1.0)


def test_w_aux_cero_no_mueve_decoder_aux():
    cfg = AEConfig(arch="mlp", length=L, d=8, hidden=8, w_aux=0.0, batch_size=6, lr=1e-2)
    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    before_aux = [w.numpy().copy() for w in models.decoder_aux.weights]
    before_mp = [w.numpy().copy() for w in models.decoder_mp.weights]
    aeml.fit(models, cfg, _batch(), None, epochs=1, patience=1)
    for b, a in zip(before_aux, models.decoder_aux.weights):
        assert np.allclose(a.numpy(), b)  # sin peso aux, el decoder aux no recibe gradiente
    # El decoder de MP sí aprende.
    assert any(not np.allclose(a.numpy(), b) for a, b in zip(models.decoder_mp.weights, before_mp))


def test_ablacion_pone_bloque_aux_en_cero():
    z = np.arange(12, dtype=float).reshape(3, 4)
    out = aeml.ablate_z_aux(z, d_aux=2)
    assert np.all(out[:, 2:] == 0)
    assert np.all(out[:, :2] == z[:, :2])


def test_rama_aux_apagada_recibe_solo_calendario():
    cfg = AEConfig(arch="mlp", length=L, d=8, hidden=8, aux_input=False)
    models = aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)
    assert models.aux_input_dim == aed.CAL_DIM
