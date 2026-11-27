"""The S21 notch of a hanger resonator: the line's delay, and the fit of Probst et al. (2015).

S. Probst et al., "Efficient and robust analysis of complex scattering data under noise in microwave
resonators", Rev. Sci. Instrum. 86, 024706 (2015): the notch-type S21 is
``a e^{i alpha} e^{-2 pi i f tau} [1 - (Q_l / |Q_c|) e^{i phi} / (1 + 2 i Q_l (f / f_r - 1))]``.  The delay
``tau`` is taken from the band away from the resonance and removed first; the other six parameters are fitted
together by least squares on the complex residual.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares


def band_delay(
    frequencies: np.ndarray, s21: np.ndarray, resonance: np.ndarray
) -> float | None:
    """The line's delay in s, from the slope of arg S21 over the frequencies outside ``resonance``.

    ``resonance`` marks the frequencies near the resonance, left out.  None when fewer than 10 are left.
    """
    away = ~resonance
    if away.sum() < 10:
        return None
    phase = np.unwrap(np.angle(s21[away]))
    slope = np.polyfit(frequencies[away], phase, 1)[0]
    return float(-slope / (2.0 * math.pi))


def fit_notch(
    f: np.ndarray, z: np.ndarray, f0_hz: float, q_guess: float
) -> dict[str, Any]:
    """Probst et al. (2015) notch model, ``a e^{i alpha} [1 - (Q_l / |Q_c|) e^{i phi} / (1 + 2 i Q_l (f / f_r - 1))]``.

    ``z`` is S21 with the delay removed.  ``1 / Q_i = 1 / Q_l - cos(phi) / |Q_c|``.  The resonance's offset from
    ``f0_hz`` is fitted in units of ``f0_hz / q_guess``, so every parameter is of order one.
    """
    edge = max(1, len(z) // 20)
    background = np.concatenate((z[:edge], z[-edge:])).mean()
    depth = 1.0 - np.abs(z).min() / abs(background)
    start = [
        abs(background),
        np.angle(background),
        0.0,
        1.0,
        1.0 / max(depth, 1e-3),
        0.0,
    ]

    def model(p: np.ndarray) -> np.ndarray:
        a, alpha, offset, q_l, q_c, phi = p
        f_r = f0_hz * (1.0 + offset / q_guess)
        q_loaded = q_l * q_guess
        return (
            a
            * np.exp(1j * alpha)
            * (
                1.0
                - (q_l / q_c)
                * np.exp(1j * phi)
                / (1.0 + 2j * q_loaded * (f / f_r - 1.0))
            )
        )

    def residual(p: np.ndarray) -> np.ndarray:
        difference = model(p) - z
        return np.concatenate((difference.real, difference.imag))

    solution = least_squares(
        residual, start, x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    a, alpha, offset, q_l, q_c, phi = solution.x
    q_loaded, q_coupling = q_l * q_guess, q_c * q_guess
    inverse_internal = 1.0 / q_loaded - math.cos(phi) / q_coupling
    return {
        "fitted": bool(solution.success),
        "f0_hz": f0_hz * (1.0 + offset / q_guess),
        "q_loaded": q_loaded,
        "q_coupling_abs": q_coupling,
        "phi_rad": phi,
        "q_coupling_real": q_coupling / math.cos(phi),
        "q_internal": 1.0 / inverse_internal if inverse_internal > 0 else math.inf,
        "inverse_q_internal": inverse_internal,
        "amplitude": a,
        "alpha_rad": alpha,
        "rms_residual": float(np.sqrt(np.mean(np.abs(model(solution.x) - z) ** 2))),
    }
