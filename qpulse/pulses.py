"""Microwave pulse envelopes for transmon control.

All envelopes return amplitudes in GHz (angular frequency units, ħ = 1).
Time arguments are in ns.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianPulse:
    """Gaussian envelope calibrated for a π-rotation on the |0⟩↔|1⟩ transition.

    Parameters
    ----------
    gate_time : float
        Total pulse duration in ns.
    n_sigma : float
        Number of standard deviations fitting in half the gate window.
        σ = gate_time / (2 * n_sigma). Default 2 means ±2σ truncation.
    """

    gate_time: float
    n_sigma: float = 2.0

    def __post_init__(self):
        self.sigma = self.gate_time / (2 * self.n_sigma)
        self.t_center = self.gate_time / 2

        # DC offset so the pulse starts/ends at zero
        self._dc_offset = np.exp(-self.n_sigma**2 / 2)

        # Calibrate amplitude so ∫ Ω_I(t) dt = π  (for a π-pulse)
        # Analytic: ∫ A[exp(-(t-tc)²/2σ²) - offset] dt over [0, tg]
        # ≈ A * σ * √(2π) * erf(n_σ/√2) - A * offset * tg
        from scipy.special import erf

        norm = self.sigma * np.sqrt(2 * np.pi) * erf(
            self.n_sigma / np.sqrt(2)
        ) - self._dc_offset * self.gate_time
        self.amp = np.pi / norm

    def I_envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        """In-phase (I) Gaussian envelope Ω_I(t)."""
        gauss = np.exp(-((t - self.t_center) ** 2) / (2 * self.sigma**2))
        return self.amp * (gauss - self._dc_offset)

    def Q_envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        """Quadrature (Q) envelope — zero for a plain Gaussian."""
        if np.ndim(t) == 0:
            return 0.0
        return np.zeros_like(np.asarray(t, dtype=float))

    def d_I_envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        """Time derivative of the I-channel envelope dΩ_I/dt."""
        return (
            -self.amp
            * (t - self.t_center)
            / self.sigma**2
            * np.exp(-((t - self.t_center) ** 2) / (2 * self.sigma**2))
        )


@dataclass
class DRAGPulse(GaussianPulse):
    """DRAG-corrected Gaussian pulse (Motzoi et al., PRL 103, 110501, 2009).

    Adds a Q-channel component Ω_Q(t) = -β/α · dΩ_I/dt that cancels
    leakage to |2⟩ via destructive interference on the |1⟩↔|2⟩ transition.

    Parameters
    ----------
    gate_time : float
        Total pulse duration in ns.
    beta : float
        Dimensionless DRAG scaling coefficient (λ in the theory).
        β = 1 is the first-order analytic optimum.
    alpha : float
        Transmon anharmonicity in GHz (negative, e.g., -0.3).
    n_sigma : float
        Truncation parameter (default 2).
    """

    beta: float = 1.0
    alpha: float = -0.3

    def Q_envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        """DRAG quadrature envelope: Ω_Q(t) = -β/α · dΩ_I/dt."""
        return -self.beta / self.alpha * self.d_I_envelope(t)
