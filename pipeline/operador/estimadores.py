"""Estimadores del operador lineal (`dmd-operador-global`, D5). Solo numpy.

Todos devuelven un `Operador` en coordenadas reducidas: `K = U·A·Uᵀ` (`U = None` = rango
completo, `K = A`). Los pares `(z_t, z_{t+lag})` se forman solo dentro de cada segmento de horas
consecutivas. `pydmd` solo contrasta el DMD exacto (tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ae import dmd as dmdm

EPS_RECORTE = 1e-3


@dataclass
class Operador:
    A: np.ndarray                 # r × r
    U: np.ndarray | None          # d × r (None = identidad)
    rank: int
    estimador: str
    pasos: int = 1                # 1 = horario; 24 = directo (A₂₄)
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> np.ndarray:
        return self.A if self.U is None else self.U @ self.A @ self.U.T

    def eigvals(self) -> np.ndarray:
        if "eig_bloques" in self.meta:
            return np.concatenate(self.meta["eig_bloques"])
        return np.linalg.eigvals(self.A)

    def avanzar(self, z: np.ndarray, k: int) -> np.ndarray:
        """`z_{t+k·pasos} = K^k z_t` por fila (`z [N, d]`)."""
        ak = np.linalg.matrix_power(self.A, int(k))
        z = np.asarray(z, dtype=float)
        if self.U is None:
            return z @ ak.T
        return ((z @ self.U) @ ak.T) @ self.U.T

    def arrays(self) -> dict[str, np.ndarray]:
        lam = self.eigvals()
        out = {"A": self.A, "rank": np.asarray(self.rank), "pasos": np.asarray(self.pasos),
               "eigvals_real": lam.real, "eigvals_imag": lam.imag}
        if self.U is not None:
            out["U"] = self.U
        for key, value in self.meta.items():
            if isinstance(value, np.ndarray):
                out[key] = value
        return out


# --- Pares y rango ------------------------------------------------------------


def pares(segments, lag: int = 1, weights=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(Z0, Z1, w)` con `Z* [d, N]`. `weights`: lista paralela a `segments` (peso por fila `t`)."""
    if isinstance(segments, np.ndarray):
        segments = [segments]
        weights = None if weights is None else [weights]
    z0, z1, ws = [], [], []
    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=float)
        if seg.shape[0] > lag:
            z0.append(seg[:-lag])
            z1.append(seg[lag:])
            ws.append(np.ones(seg.shape[0] - lag) if weights is None else np.asarray(weights[i], dtype=float)[:-lag])
    if not z0:
        raise ValueError("Se necesitan al menos dos instantes separados por `lag` dentro de un segmento.")
    return np.concatenate(z0).T, np.concatenate(z1).T, np.concatenate(ws)


def _omega(beta: float) -> float:
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def gavish_donoho_rank(sigma, shape) -> int:
    """Umbral óptimo de Gavish–Donoho con σ desconocido: `ω(β)·mediana(σ)` (como `pydmd`, `svd_rank=0`)."""
    sigma = np.asarray(sigma, dtype=float)
    beta = min(shape) / max(shape)
    tau = np.median(sigma) * _omega(beta)
    return max(1, int(np.sum(sigma > tau)))


def rango_gd(z0: np.ndarray) -> int:
    return gavish_donoho_rank(np.linalg.svd(z0, compute_uv=False), z0.shape)


def _resolve_rank(z0: np.ndarray, rank) -> int | None:
    if rank is None or rank == "full":
        return None
    if rank == "gd":
        rank = rango_gd(z0)
    rank = int(rank)
    return None if rank >= min(z0.shape) else rank


def _fit_pairs(z0: np.ndarray, z1: np.ndarray, rank, estimador: str, pasos: int = 1) -> Operador:
    r = _resolve_rank(z0, rank)
    if r is None:
        return Operador(A=z1 @ np.linalg.pinv(z0), U=None, rank=z0.shape[0], estimador=estimador, pasos=pasos)
    u, s, vt = np.linalg.svd(z0, full_matrices=False)
    u_r, s_r, v_r = u[:, :r], s[:r], vt[:r].T
    a = u_r.T @ z1 @ v_r @ np.diag(1.0 / s_r)
    return Operador(A=a, U=u_r, rank=r, estimador=estimador, pasos=pasos)


# --- Estimadores ----------------------------------------------------------------


def exacto(segments, rank=None) -> Operador:
    """DMD exacto (envuelve `ae.dmd.fit_dmd`). `rank`: `None` completo, entero o `"gd"`."""
    if isinstance(segments, np.ndarray):
        segments = [segments]
    z0, _, _ = pares(segments)
    r = _resolve_rank(z0, rank)
    fit = dmdm.fit_dmd(segments, r)
    if r is None:
        return Operador(A=fit["K"], U=None, rank=fit["rank"], estimador="exacto")
    u_r = np.linalg.svd(z0, full_matrices=False)[0][:, :r]
    return Operador(A=u_r.T @ fit["K"] @ u_r, U=u_r, rank=r, estimador="exacto")


def truncado(segments, rank: int) -> Operador:
    return exacto(segments, int(rank))


def ponderado(segments, weights, rank=None) -> Operador:
    """`K = (Z₁WZ₀ᵀ)(Z₀WZ₀ᵀ)⁺` (= DMD exacto sobre pares escalados por `√w`)."""
    z0, z1, w = pares(segments, weights=weights)
    sw = np.sqrt(w)[None, :]
    op = _fit_pairs(z0 * sw, z1 * sw, rank, "ponderado")
    return op


def directo24(segments, rank=None, weights=None) -> Operador:
    """`A₂₄`: pares `(z_t, z_{t+24})` dentro de segmentos (referencia de precisión)."""
    z0, z1, w = pares(segments, lag=24, weights=weights)
    sw = np.sqrt(w)[None, :]
    return _fit_pairs(z0 * sw, z1 * sw, rank, "directo24", pasos=24)


def _recortar_lam(lam: np.ndarray, eps: float) -> np.ndarray:
    mod = np.abs(lam)
    tope = 1.0 - eps
    return np.where(mod > tope, tope * lam / np.where(mod > 0, mod, 1.0), lam)


def _rearmar(vecs: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return np.real(vecs @ np.diag(lam) @ np.linalg.inv(vecs))


def recorte_matriz(a: np.ndarray, eps: float = EPS_RECORTE) -> np.ndarray:
    lam, vecs = np.linalg.eig(a)
    if np.all(np.abs(lam) <= 1.0 - eps):
        return np.array(a, copy=True)
    return _rearmar(vecs, _recortar_lam(lam, eps))


def recorte(op: Operador, eps: float = EPS_RECORTE) -> Operador:
    """`|λ| > 1−ε` -> `(1−ε)·λ/|λ|`, reconstrucción real (en coordenadas reducidas)."""
    nombre = op.estimador if op.estimador.endswith("+recorte") else f"{op.estimador}+recorte"
    meta = dict(op.meta)
    meta["eps"] = np.asarray(eps)
    return Operador(A=recorte_matriz(op.A, eps), U=op.U, rank=op.rank, estimador=nombre, pasos=op.pasos, meta=meta)


def fb_formula(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """fbDMD sobre coordenadas dadas: `K = (K_f·K_b⁻¹)^{1/2}`, raíz más cercana a `eig(K_f)`, real."""
    k_f = y @ np.linalg.pinv(x)
    k_b = x @ np.linalg.pinv(y)
    m = k_f @ np.linalg.inv(k_b)
    mu, vecs = np.linalg.eig(m)
    lam_f = np.linalg.eigvals(k_f)
    root = np.sqrt(mu.astype(complex))
    cands = np.stack([root, -root], axis=1)
    dist = np.abs(cands[:, :, None] - lam_f[None, None, :]).min(axis=2)
    lam = cands[np.arange(len(mu)), dist.argmin(axis=1)]
    return _rearmar(vecs, lam)


def fbdmd(segments, weights=None, rank=None, eps: float | None = EPS_RECORTE) -> Operador:
    """fbDMD en coordenadas de rango `r` (pares escalados por `√w` si hay pesos); con recorte."""
    z0, z1, w = pares(segments, weights=weights)
    sw = np.sqrt(w)[None, :]
    z0, z1 = z0 * sw, z1 * sw
    r = _resolve_rank(z0, rank)
    if r is None:
        u_r, x, y = None, z0, z1
    else:
        u_r = np.linalg.svd(z0, full_matrices=False)[0][:, :r]
        x, y = u_r.T @ z0, u_r.T @ z1
    op = Operador(A=fb_formula(x, y), U=u_r, rank=x.shape[0], estimador="fbdmd")
    return recorte(op, eps) if eps is not None else op


def por_bloques(segments, bloques: list[np.ndarray], rank="gd", weights=None) -> Operador:
    """Un operador por bloque de coordenadas (Hankel por estación); `K` bloque-diagonal completo."""
    if isinstance(segments, np.ndarray):
        segments = [segments]
    d = np.asarray(segments[0]).shape[1]
    k = np.zeros((d, d))
    ranks, eigs = [], []
    for idx in bloques:
        z0, z1, w = pares([np.asarray(s)[:, idx] for s in segments], weights=weights)
        sw = np.sqrt(w)[None, :]
        op = _fit_pairs(z0 * sw, z1 * sw, rank, "exacto")
        k[np.ix_(idx, idx)] = op.K
        ranks.append(op.rank)
        eigs.append(np.linalg.eigvals(op.A))
    return Operador(A=k, U=None, rank=int(sum(ranks)), estimador="exacto-bloques",
                    meta={"ranks_bloques": np.asarray(ranks), "eig_bloques": eigs})


def dmdc(x_segments, u_segments, weights=None, rank=None) -> Operador:
    """`x_{t+1} = A x_t + B u_t` por mínimos cuadrados (pares dentro de segmentos; `√w` si hay pesos).

    `rank`: `None` = pseudoinversa completa de `Ω = [X; U]`; entero o `"gd"` = SVD truncada de `Ω`
    (Gavish–Donoho sobre `[X; U]`, D6). `B` queda en `meta["K_ma"]`.
    """
    x0, x1, w = pares(x_segments, weights=weights)
    u0 = np.concatenate([np.asarray(u, float)[:-1] for u in u_segments if np.asarray(u).shape[0] > 1]).T
    sw = np.sqrt(w)[None, :]
    x0, x1, u0 = x0 * sw, x1 * sw, u0 * sw
    omega = np.concatenate([x0, u0], axis=0)
    r = _resolve_rank(omega, rank)
    if r is None:
        g = x1 @ np.linalg.pinv(omega)
        used = omega.shape[0]
    else:
        uu, s, vt = np.linalg.svd(omega, full_matrices=False)
        g = x1 @ vt[:r].T @ np.diag(1.0 / s[:r]) @ uu[:, :r].T
        used = r
    n = x0.shape[0]
    return Operador(A=g[:, :n], U=None, rank=int(used), estimador="dmdc", meta={"K_ma": g[:, n:]})
