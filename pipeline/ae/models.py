"""Autoencoders con rama auxiliar y latente en bloques (change `ae-reconstruccion`, grupo 2).

Arquitectura (design.md D2):

    X_mp [L×6] ──► encoder_mp ──► z_mp ─┐
                                        ├─► z=[z_mp | z_aux] ─► decoder_mp  ─► X̂_mp  (peso w_mp)
    aux [L×34] ──► encoder_aux ─► z_aux ┘                  └─► decoder_aux ─► X̂_aux (peso w_aux)

- `d` es la dimensión total del latente; `d_aux` (por defecto `d/4`, ≥ 2) la del
  bloque auxiliar, y `z_mp` tiene `d - d_aux` dimensiones.
- Variantes: `mlp`, `conv1d`, `gru`, `lstm` y `linear_decoder` (encoder no lineal,
  decoder lineal). Salidas siempre lineales.
- Loss `w_mp·E_mp + w_aux·E_aux`, cada término MSE enmascarado (solo valores
  reales) promediado por canal. El calendario no tiene término de loss.

TensorFlow/Keras se importan en las funciones para que los módulos numpy
(`data`, `metrics`, `dmd`) funcionen sin TF instalado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARCHITECTURES: tuple[str, ...] = ("mlp", "conv1d", "gru", "lstm", "linear_decoder")
LATENT_CHOICES: tuple[int, ...] = (8, 16, 32)
LENGTH_CHOICES: tuple[int, ...] = (24, 72, 168)


@dataclass(frozen=True)
class AEConfig:
    arch: str = "mlp"
    length: int = 24
    d: int = 16
    d_aux: int | None = None
    hidden: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    w_mp: float = 1.0
    w_aux: float = 0.1
    aux_input: bool = True
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def resolved_d_aux(self) -> int:
        d_aux = self.d_aux if self.d_aux is not None else max(2, self.d // 4)
        if not 1 <= d_aux < self.d:
            raise ValueError(f"d_aux={d_aux} inválido para d={self.d}")
        return d_aux

    def to_dict(self) -> dict:
        return {
            "arch": self.arch, "length": self.length, "d": self.d, "d_aux": self.resolved_d_aux(),
            "hidden": self.hidden, "dropout": self.dropout, "lr": self.lr,
            "batch_size": self.batch_size, "w_mp": self.w_mp, "w_aux": self.w_aux,
            "aux_input": self.aux_input, "seed": self.seed,
        }


@dataclass
class AEModels:
    """Modelos de Keras de un autoencoder."""

    full: object          # entrenamiento: [x_mp, x_aux] -> [mp_out, aux_out]
    encoder: object       # [x_mp, x_aux] -> z
    decoder_mp: object    # z -> X̂_mp
    decoder_aux: object   # z -> X̂_aux
    d: int
    d_aux: int
    aux_input_dim: int

    @property
    def z_mp_dim(self) -> int:
        return self.d - self.d_aux


def masked_channel_mse(y_true, y_pred):
    """MSE enmascarado y promediado por canal. `y_true` = [valor | máscara real].

    Devuelve una pérdida por ventana (`[batch]`); Keras promedia el batch.
    """
    from keras import ops

    channels = y_pred.shape[-1]
    target = y_true[..., :channels]
    mask = y_true[..., channels:]
    sq = mask * ops.square(target - y_pred)
    num = ops.sum(sq, axis=1)                 # [batch, C]
    den = ops.sum(mask, axis=1)               # [batch, C]
    per_channel = num / ops.maximum(den, 1e-6)
    return ops.mean(per_channel, axis=-1)     # [batch]


def pack_targets(values: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Empaqueta objetivo y máscara real para `masked_channel_mse`."""
    return np.concatenate([values.astype("float32"), real.astype("float32")], axis=-1)


def _encoder(keras, layers, arch: str, length: int, n_in: int, z_dim: int, hidden: int, dropout: float, name: str):
    inp = keras.Input((length, n_in), name=f"{name}_in")
    if arch in ("mlp", "linear_decoder"):
        x = layers.Flatten()(inp)
        x = layers.Dense(hidden, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    elif arch == "conv1d":
        x = layers.Conv1D(hidden, 5, padding="same", activation="relu")(inp)
        x = layers.Conv1D(hidden, 5, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv1D(hidden, 5, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(dropout)(x)
    elif arch in ("gru", "lstm"):
        cell = layers.GRU if arch == "gru" else layers.LSTM
        x = cell(hidden)(inp)
        x = layers.Dropout(dropout)(x)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
    z = layers.Dense(z_dim, name=f"{name}_z")(x)
    return keras.Model(inp, z, name=f"encoder_{name}")


def _decoder(keras, layers, arch: str, length: int, z_dim: int, n_out: int, hidden: int, dropout: float, name: str):
    zin = keras.Input((z_dim,), name=f"{name}_z_in")
    if arch == "mlp":
        x = layers.Dense(hidden, activation="relu")(zin)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(length * n_out)(x)
        out = layers.Reshape((length, n_out))(x)
    elif arch == "conv1d":
        x = layers.Dense(length * hidden, activation="relu")(zin)
        x = layers.Reshape((length, hidden))(x)
        x = layers.Conv1D(hidden, 5, padding="same", activation="relu")(x)
        out = layers.Conv1D(n_out, 1, padding="same")(x)
    elif arch in ("gru", "lstm"):
        cell = layers.GRU if arch == "gru" else layers.LSTM
        x = layers.RepeatVector(length)(zin)
        x = cell(hidden, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(n_out))(x)
    elif arch == "linear_decoder":
        # Decoder lineal: cada dimensión del latente tiene una forma fija sobre las series.
        x = layers.Dense(length * n_out)(zin)
        out = layers.Reshape((length, n_out))(x)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
    return keras.Model(zin, out, name=f"decoder_{name}")


def build_models(cfg: AEConfig, n_mp: int, n_aux_out: int, n_aux_all: int, n_cal: int) -> AEModels:
    """Construye encoder, decoders y el modelo de entrenamiento."""
    import keras
    from keras import layers

    if cfg.arch not in ARCHITECTURES:
        raise ValueError(f"Arquitectura desconocida: {cfg.arch}")
    d_aux = cfg.resolved_d_aux()
    z_mp_dim = cfg.d - d_aux
    aux_in = n_aux_all if cfg.aux_input else n_cal

    enc_mp = _encoder(keras, layers, cfg.arch, cfg.length, n_mp, z_mp_dim, cfg.hidden, cfg.dropout, "mp")
    enc_aux = _encoder(keras, layers, cfg.arch, cfg.length, aux_in, d_aux, cfg.hidden, cfg.dropout, "aux")

    x_mp = keras.Input((cfg.length, n_mp), name="x_mp")
    x_aux = keras.Input((cfg.length, aux_in), name="x_aux")
    z = layers.Concatenate(name="z")([enc_mp(x_mp), enc_aux(x_aux)])
    encoder = keras.Model([x_mp, x_aux], z, name="encoder")

    dec_mp = _decoder(keras, layers, cfg.arch, cfg.length, cfg.d, n_mp, cfg.hidden, cfg.dropout, "mp")
    dec_aux = _decoder(keras, layers, cfg.arch, cfg.length, cfg.d, n_aux_out, cfg.hidden, cfg.dropout, "aux")

    mp_out = layers.Activation("linear", name="mp_out")(dec_mp(z))
    aux_out = layers.Activation("linear", name="aux_out")(dec_aux(z))
    full = keras.Model([x_mp, x_aux], [mp_out, aux_out], name="ae")
    return AEModels(
        full=full,
        encoder=encoder,
        decoder_mp=dec_mp,
        decoder_aux=dec_aux,
        d=cfg.d,
        d_aux=d_aux,
        aux_input_dim=aux_in,
    )


def compile_models(models: AEModels, cfg: AEConfig) -> None:
    import keras

    models.full.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
        loss={"mp_out": masked_channel_mse, "aux_out": masked_channel_mse},
        loss_weights={"mp_out": cfg.w_mp, "aux_out": cfg.w_aux},
    )


def fit(
    models: AEModels,
    cfg: AEConfig,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid: tuple | None = None,
    epochs: int = 60,
    patience: int = 8,
    verbose: int = 0,
):
    """Entrena con early stopping sobre `valid`.

    `train`/`valid` = `(x_mp, x_aux_all, y_mp, real_mp, y_aux, real_aux)`; la
    entrada auxiliar se recorta con `aux_input_columns`.
    """
    import keras

    from .data import aux_input_columns

    keras.utils.set_random_seed(cfg.seed)
    compile_models(models, cfg)
    sl = aux_input_columns(cfg.aux_input)

    def _pack(data):
        x_mp, x_aux, y_mp, real_mp, y_aux, real_aux = data
        return (
            [x_mp, x_aux[..., sl]],
            {"mp_out": pack_targets(y_mp, real_mp), "aux_out": pack_targets(y_aux, real_aux)},
        )

    x_tr, y_tr = _pack(train)
    kwargs = {}
    callbacks = []
    if valid is not None:
        kwargs["validation_data"] = _pack(valid)
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    history = models.full.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs,
    )
    return history


def predict_mp_aux(models: AEModels, cfg: AEConfig, x_mp: np.ndarray, x_aux: np.ndarray, batch_size: int = 512):
    """Reconstrucciones `(mp_z, aux_z)` en espacio normalizado."""
    from .data import aux_input_columns

    sl = aux_input_columns(cfg.aux_input)
    mp, aux = models.full.predict([x_mp, x_aux[..., sl]], batch_size=batch_size, verbose=0)
    return np.asarray(mp), np.asarray(aux)


def encode(models: AEModels, cfg: AEConfig, x_mp: np.ndarray, x_aux: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Latente `z = [z_mp | z_aux]` por ventana."""
    from .data import aux_input_columns

    sl = aux_input_columns(cfg.aux_input)
    return np.asarray(models.encoder.predict([x_mp, x_aux[..., sl]], batch_size=batch_size, verbose=0))


def decode_mp(models: AEModels, z: np.ndarray, batch_size: int = 512) -> np.ndarray:
    return np.asarray(models.decoder_mp.predict(z, batch_size=batch_size, verbose=0))


def ablate_z_aux(z: np.ndarray, d_aux: int) -> np.ndarray:
    """Copia de `z` con el bloque auxiliar en cero (ablación de `z_aux`)."""
    out = np.array(z, copy=True)
    out[:, out.shape[1] - d_aux:] = 0.0
    return out
