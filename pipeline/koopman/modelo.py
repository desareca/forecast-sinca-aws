"""Modelo del Koopman entrenado (`deep-koopman`, D4–D7).

- Entrada del encoder: ventana `L = 24` de MP (6) + meteo (12) + máscaras de disponibilidad (12),
  normalizadas como el AE (`ae.data`); sin calendario (D4).
- Encoders: `gru` = GRU(64) + Dropout(0.1) + Dense(d); `conv1ds` = Conv1D(64, k=5, same, relu) +
  Flatten + Dropout(0.1) + Dense(d).
- Decoders: `lineal` = Dense(24·6) (principal); `gru` = RepeatVector + GRU(64) + TimeDistributed(Dense(6)).
- Operador `K` diagonal por bloques: diario (θ = 2π/24 fijo), `n_libres` bloques 2×2 libres
  (θ = π·sigmoid(b)) y `n_reales` reales, con `ρ = exp(−(δ + softplus(a)))`, `δ = ln 2/1000`
  (vida media ≤ 1000 h). Control `B·u_{t+1}`, `u` = `dow_sin`, `dow_cos`, `is_holiday`, `gec_window`.
- Variantes: `kden` (K denso libre, sin techo), `perlib` (θ diario libre), `sinbu` (sin `B·u`).

Orden de coordenadas de `z`: `[diario(2) | libre_1(2) | … | real_1 | … ]`.
"""
from __future__ import annotations

import math

import numpy as np

from ae import data as aed

HORIZONTE = 24
N_MP = aed.N_MP
N_AUX = aed.N_AUX
N_IN = N_MP + 2 * N_AUX  # 30
U_CHANNELS = ("dow_sin", "dow_cos", "is_holiday", "gec_window")
U_IDX = [aed.CALENDAR_CHANNELS.index(c) for c in U_CHANNELS]
N_U = len(U_CHANNELS)
DELTA = math.log(2.0) / 1000.0
RHO_MAX = math.exp(-DELTA)  # 2^(-1/1000)
THETA_DIARIO = 2.0 * math.pi / 24.0
ENCODERS = ("gru", "conv1ds")
DECODERS = ("lineal", "gru")
VARIANTES_K = ("", "kden", "perlib", "sinbu")


# --- Estructura de bloques ------------------------------------------------------


def estructura(d: int) -> tuple[int, int]:
    """`(n_libres, n_reales)` para `d`: 10 -> (2, 4); 8 (B4) -> (2, 2)."""
    if d == 10:
        return 2, 4
    if d == 8:
        return 2, 2
    raise ValueError(f"d = {d} no soportado (10 u 8)")


def bloques(d: int) -> list[np.ndarray]:
    """Índices de `z` por bloque: diario, libres, reales (en ese orden)."""
    n_lib, n_real = estructura(d)
    out = [np.array([0, 1])]
    for i in range(n_lib):
        out.append(np.array([2 + 2 * i, 3 + 2 * i]))
    base = 2 + 2 * n_lib
    for j in range(n_real):
        out.append(np.array([base + j]))
    return out


def tipos_bloque(d: int) -> list[str]:
    n_lib, n_real = estructura(d)
    return ["diario"] + ["libre"] * n_lib + ["real"] * n_real


def softplus_inv(y: float) -> float:
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y)) if y < 30 else y


def a_desde_rho(rho: float) -> float:
    """Parámetro `a` tal que `exp(−(δ + softplus(a))) = rho` (recortado al techo)."""
    rho = min(max(float(rho), 1e-6), RHO_MAX * (1 - 1e-12))
    return softplus_inv(-math.log(rho) - DELTA)


def b_desde_theta(theta: float) -> float:
    p = min(max(abs(float(theta)) / math.pi, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def rho_desde_vida_media(h: float) -> float:
    return 0.5 ** (1.0 / float(h))


# --- Operador ----------------------------------------------------------------------


class OperadorBloques:
    """`K` estructurado (o denso en `kden`) y `B` como variables TF."""

    def __init__(self, d: int = 10, variante: str = "", seed: int = 0):
        import tensorflow as tf

        if variante not in VARIANTES_K:
            raise ValueError(f"Variante de K desconocida: {variante}")
        self.d, self.variante = d, variante
        self.n_lib, self.n_real = estructura(d)
        init = self.parametros_genericos()
        self.a = tf.Variable(np.array(init["a"], dtype="float32"), name="k_a")
        self.b = tf.Variable(np.array(init["b"], dtype="float32"), name="k_b")
        self.b_diario = tf.Variable(np.float32(b_desde_theta(THETA_DIARIO)), name="k_b_diario",
                                    trainable=variante == "perlib")
        self.kden = None
        if variante == "kden":
            self.kden = tf.Variable(self.matriz_np(), name="k_denso")
        self.B = tf.Variable(np.zeros((d, N_U), dtype="float32"), name="k_B") if variante != "sinbu" else None

    # parámetros iniciales genéricos (conv1d): diario 300 h; libres (168, 100) y (48, 48); reales 24, 72, 168, 500 h
    def parametros_genericos(self) -> dict:
        hl = [300.0] + [168.0, 48.0][: self.n_lib] + [24.0, 72.0, 168.0, 500.0][-self.n_real:]
        per_lib = [100.0, 48.0][: self.n_lib]
        return {"a": [a_desde_rho(rho_desde_vida_media(h)) for h in hl],
                "b": [b_desde_theta(2 * math.pi / p) for p in per_lib]}

    @property
    def variables(self) -> list:
        out = [self.a, self.b, self.b_diario]
        if self.kden is not None:
            out.append(self.kden)
        if self.B is not None:
            out.append(self.B)
        return out

    @property
    def trainable(self) -> list:
        if self.kden is not None:
            return [self.kden] + ([self.B] if self.B is not None else [])
        out = [self.a, self.b]
        if self.variante == "perlib":
            out.append(self.b_diario)
        if self.B is not None:
            out.append(self.B)
        return out

    def rho(self):
        import tensorflow as tf

        return tf.exp(-(DELTA + tf.nn.softplus(self.a)))

    def thetas(self):
        import tensorflow as tf

        th_d = math.pi * tf.sigmoid(self.b_diario) if self.variante == "perlib" else tf.constant(THETA_DIARIO)
        return tf.concat([tf.reshape(th_d, [1]), math.pi * tf.sigmoid(self.b)], axis=0)

    def matriz(self):
        """`K` (d×d) como tensor."""
        import tensorflow as tf

        if self.kden is not None:
            return self.kden
        rho, th = self.rho(), self.thetas()
        idx, vals = [], []
        for i in range(1 + self.n_lib):
            r, t = rho[i], th[i]
            c, s = r * tf.cos(t), r * tf.sin(t)
            p = 2 * i
            idx += [[p, p], [p, p + 1], [p + 1, p], [p + 1, p + 1]]
            vals += [c, -s, s, c]
        base = 2 + 2 * self.n_lib
        for j in range(self.n_real):
            idx.append([base + j, base + j])
            vals.append(rho[1 + self.n_lib + j])
        return tf.scatter_nd(tf.constant(idx), tf.stack(vals), [self.d, self.d])

    def matriz_np(self) -> np.ndarray:
        return np.asarray(self.matriz(), dtype="float32")

    def paso(self, z, u):
        """`z_{t+1} = K z_t + B u_{t+1}` por fila (`z [N, d]`, `u [N, 4]`)."""
        import tensorflow as tf

        out = tf.matmul(z, self.matriz(), transpose_b=True)
        if self.B is not None:
            out = out + tf.matmul(u, self.B, transpose_b=True)
        return out

    def autovalores(self) -> np.ndarray:
        return np.linalg.eigvals(self.matriz_np().astype(float))

    def asignar(self, a=None, b=None, b_diario=None, B=None) -> None:
        if a is not None:
            self.a.assign(np.asarray(a, dtype="float32"))
        if b is not None:
            self.b.assign(np.asarray(b, dtype="float32"))
        if b_diario is not None:
            self.b_diario.assign(np.float32(b_diario))
        if B is not None and self.B is not None:
            self.B.assign(np.asarray(B, dtype="float32"))


# --- Redes --------------------------------------------------------------------------


def construir_encoder(tipo: str, d: int, hidden: int = 64, dropout: float = 0.1):
    import keras
    from keras import layers

    inp = keras.Input((HORIZONTE, N_IN), name="x_enc")
    if tipo == "gru":
        x = layers.GRU(hidden, name="enc_gru")(inp)
    elif tipo == "conv1ds":
        x = layers.Conv1D(hidden, 5, padding="same", activation="relu", name="enc_conv")(inp)
        x = layers.Flatten()(x)
    else:
        raise ValueError(f"Encoder desconocido: {tipo}")
    x = layers.Dropout(dropout, name="enc_drop")(x)
    z = layers.Dense(d, name="enc_z")(x)
    return keras.Model(inp, z, name=f"koop_enc_{tipo}")


def construir_decoder(tipo: str, d: int, hidden: int = 64):
    import keras
    from keras import layers

    zin = keras.Input((d,), name="z")
    if tipo == "lineal":
        x = layers.Dense(HORIZONTE * N_MP, name="dec_lin")(zin)
        out = layers.Reshape((HORIZONTE, N_MP))(x)
    elif tipo == "gru":
        x = layers.RepeatVector(HORIZONTE)(zin)
        x = layers.GRU(hidden, return_sequences=True, name="dec_gru")(x)
        out = layers.TimeDistributed(layers.Dense(N_MP), name="dec_out")(x)
    else:
        raise ValueError(f"Decoder desconocido: {tipo}")
    return keras.Model(zin, out, name=f"koop_dec_{tipo}")


class KoopmanModelo:
    """Encoder + operador + decoder. `config` = `{encoder, decoder, d, variante, hidden, dropout, seed}`."""

    def __init__(self, encoder: str = "gru", decoder: str = "lineal", d: int = 10, variante: str = "",
                 hidden: int = 64, dropout: float = 0.1, seed: int = 0):
        import keras

        keras.utils.set_random_seed(seed)
        self.config = {"encoder": encoder, "decoder": decoder, "d": d, "variante": variante,
                       "hidden": hidden, "dropout": dropout, "seed": seed}
        self.d = d
        self.enc = construir_encoder(encoder, d, hidden, dropout)
        self.dec = construir_decoder(decoder, d, hidden)
        self.op = OperadorBloques(d, variante, seed)
        self.mascara = None  # máscara de bloques conservados (eliminación de modos, D13)

    # --- grupos de variables
    @property
    def vars_enc(self) -> list:
        return list(self.enc.trainable_variables)

    @property
    def vars_dec(self) -> list:
        return list(self.dec.trainable_variables)

    @property
    def vars_op(self) -> list:
        return list(self.op.trainable)

    @property
    def entrenables(self) -> list:
        return self.vars_enc + self.vars_dec + self.vars_op

    def pesos(self) -> dict[str, np.ndarray]:
        out = {}
        for i, w in enumerate(self.enc.get_weights()):
            out[f"enc_{i}"] = w
        for i, w in enumerate(self.dec.get_weights()):
            out[f"dec_{i}"] = w
        for v in self.op.variables:
            out[f"op_{v.name.split(':')[0]}"] = np.asarray(v)
        return out

    def cargar_pesos(self, pesos: dict) -> None:
        n_enc = len(self.enc.get_weights())
        n_dec = len(self.dec.get_weights())
        self.enc.set_weights([pesos[f"enc_{i}"] for i in range(n_enc)])
        self.dec.set_weights([pesos[f"dec_{i}"] for i in range(n_dec)])
        for v in self.op.variables:
            key = f"op_{v.name.split(':')[0]}"
            if key in pesos:
                v.assign(pesos[key])

    # --- cálculo
    def encode(self, x, training: bool = False):
        return self.enc(x, training=training)

    def decode(self, z, training: bool = False):
        if self.mascara is not None:
            z = z * self.mascara
        return self.dec(z, training=training)

    def paso(self, z, u):
        out = self.op.paso(z, u)
        if self.mascara is not None:
            out = out * self.mascara
        return out

    def rollout(self, z0, U, pasos: int = HORIZONTE) -> list:
        """`[z_{t+1}, …, z_{t+pasos}]` con `U [N, ≥pasos, 4]` (`u_{t+1}…`)."""
        out, z = [], z0
        for k in range(pasos):
            z = self.paso(z, U[:, k, :])
            out.append(z)
        return out

    def fijar_bloques(self, conservar_idx: list[int] | None) -> None:
        """Conserva solo los bloques dados (índices de `bloques(d)`); `None` = todos."""
        import tensorflow as tf

        if conservar_idx is None:
            self.mascara = None
            return
        m = np.zeros(self.d, dtype="float32")
        for i in conservar_idx:
            m[bloques(self.d)[i]] = 1.0
        self.mascara = tf.constant(m)

    def kernel_decoder(self) -> np.ndarray:
        """Pesos que actúan linealmente sobre `z` en el decoder (`d × ·`)."""
        return np.asarray(self.dec.get_layer("dec_lin" if self.config["decoder"] == "lineal" else "dec_gru").get_weights()[0])


# --- Entradas -----------------------------------------------------------------------


def entradas(ng: aed.NormalizedGrid, positions) -> np.ndarray:
    """Entrada del encoder `[N, 24, 30]` (MP + meteo + máscaras) de las ventanas que terminan en `positions`."""
    positions = np.asarray(positions, dtype=int)
    idx = (positions - HORIZONTE + 1)[:, None] + np.arange(HORIZONTE)[None, :]
    mp = ng.mp_z.fillna(0.0).to_numpy(dtype=np.float32)[idx]
    aux = ng.aux_z.to_numpy(dtype=np.float32)[idx]
    avail = ng.aux_avail.to_numpy(dtype=np.float32)[idx]
    return np.concatenate([mp, aux, avail], axis=-1)


def ventanas_mp(ng: aed.NormalizedGrid, positions) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas MP normalizadas `[N, 24, 6]` (0 en huecos) y máscara de valores reales."""
    positions = np.asarray(positions, dtype=int)
    idx = (positions - HORIZONTE + 1)[:, None] + np.arange(HORIZONTE)[None, :]
    return ng.mp_z.fillna(0.0).to_numpy(dtype=np.float32)[idx], ng.mp_real.to_numpy()[idx].astype("float32")


def u_futuro(ng: aed.NormalizedGrid, positions, pasos: int = HORIZONTE) -> np.ndarray:
    """`u_{t+1} … u_{t+pasos}` `[N, pasos, 4]` (calendario conocido de `t+k`)."""
    positions = np.asarray(positions, dtype=int)
    idx = np.clip(positions[:, None] + np.arange(1, pasos + 1)[None, :], 0, len(ng.index) - 1)
    return ng.cal.to_numpy(dtype=np.float32)[:, U_IDX][idx]


def pronostico(modelo: KoopmanModelo, ng: aed.NormalizedGrid, norm: aed.Normalizer, positions,
               lote: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Rollout de 24 pasos con `u` conocido -> `dec(z_{t+24})` -> promedio. Devuelve `(pron [N,6] µg/m³,
    ventana log1p [N,24,6])`. No usa meteo posterior a `t`."""
    positions = np.asarray(positions, dtype=int)
    logs = []
    mean, std = _escala(norm)
    for i in range(0, len(positions), lote):
        p = positions[i:i + lote]
        z = modelo.encode(entradas(ng, p))
        zf = modelo.rollout(z, u_futuro(ng, p))[-1]
        logs.append(np.asarray(modelo.decode(zf), dtype=float) * std + mean)
    lp = np.concatenate(logs) if logs else np.zeros((0, HORIZONTE, N_MP))
    return np.clip(np.expm1(lp), 0.0, None).mean(axis=1), lp


def _escala(norm: aed.Normalizer) -> tuple[np.ndarray, np.ndarray]:
    cols = [name for name, _, _ in aed.mp_channels()]
    return norm.mean[cols].to_numpy(dtype=float), norm.std[cols].to_numpy(dtype=float)
