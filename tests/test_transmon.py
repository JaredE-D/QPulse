"""Tests for the transmon Hamiltonian and energy spectrum."""

import numpy as np
import pytest
from qutip import basis

from qpulse.transmon import TransmonDRAG
from qpulse.pulses import GaussianPulse, DRAGPulse
from qpulse.utils import projector


class TestTransmonHamiltonian:
    """Verify that the static Hamiltonian gives the correct energy spectrum."""

    def setup_method(self):
        self.omega_q = 5.0   # GHz
        self.alpha = -0.3    # GHz
        self.n_levels = 4
        self.transmon = TransmonDRAG(
            omega_q=self.omega_q,
            alpha=self.alpha,
            n_levels=self.n_levels,
        )

    def test_energy_spectrum_rotating_frame(self):
        """In the rotating frame at ω_q, energies are E_n = (α/2)n(n-1)."""
        H0 = self.transmon._H_static()
        eigvals = H0.eigenenergies()
        eigvals_sorted = np.sort(np.real(eigvals))

        # Expected: E_0 = 0, E_1 = 0, E_2 = α, E_3 = 3α
        expected = np.array([
            0.5 * self.alpha * n * (n - 1) for n in range(self.n_levels)
        ])
        expected_sorted = np.sort(expected)

        np.testing.assert_allclose(eigvals_sorted, expected_sorted, atol=1e-12)

    def test_n_levels_minimum(self):
        """Must raise for n_levels < 3."""
        with pytest.raises(ValueError, match="n_levels must be ≥ 3"):
            TransmonDRAG(n_levels=2)


class TestPulseCalibration:
    """Verify that the Gaussian pulse integrates to π."""

    def test_pi_pulse_area(self):
        tg = 20.0  # ns
        pulse = GaussianPulse(gate_time=tg, n_sigma=2.0)
        t = np.linspace(0, tg, 10000)
        area = np.trapezoid(pulse.I_envelope(t), t)
        np.testing.assert_allclose(area, np.pi, rtol=1e-4)

    def test_drag_inherits_area(self):
        """DRAG pulse should have the same I-channel area as the Gaussian."""
        tg = 20.0
        pulse = DRAGPulse(gate_time=tg, beta=1.0, alpha=-0.3)
        t = np.linspace(0, tg, 10000)
        area = np.trapezoid(pulse.I_envelope(t), t)
        np.testing.assert_allclose(area, np.pi, rtol=1e-4)

    def test_gaussian_starts_and_ends_at_zero(self):
        tg = 20.0
        pulse = GaussianPulse(gate_time=tg, n_sigma=2.0)
        assert abs(pulse.I_envelope(0.0)) < 1e-10
        assert abs(pulse.I_envelope(tg)) < 1e-10


class TestRabiOscillation:
    """Validate solver against exact 2-level Rabi oscillations.

    With |α| >> Ω (drive amplitude), the |2⟩ state is far off-resonant
    and the system behaves as an effective two-level qubit.
    """

    def test_rabi_half_period(self):
        """A calibrated π-pulse with large |α| should cleanly invert."""
        # Large |α| makes |2⟩ far detuned → effectively 2-level
        transmon = TransmonDRAG(omega_q=5.0, alpha=-5.0, n_levels=3)
        result = transmon.simulate(gate_time=20.0, beta=0.0, n_steps=500)

        P0_final = result.expect[0][-1]
        P1_final = result.expect[1][-1]
        P2_final = result.expect[2][-1]

        assert P1_final > 0.99, f"P1 = {P1_final:.4f}, expected > 0.99"
        assert P0_final < 0.01, f"P0 = {P0_final:.4f}, expected < 0.01"
        assert P2_final < 0.001, f"P2 = {P2_final:.6f}, expected < 0.001"

    def test_drag_reduces_leakage(self):
        """DRAG should reduce leakage vs. plain Gaussian at realistic α."""
        transmon = TransmonDRAG(omega_q=5.0, alpha=-0.3, n_levels=3)
        res_gauss = transmon.simulate(gate_time=20.0, beta=0.0, n_steps=300)
        res_drag = transmon.simulate(gate_time=20.0, beta=1.0, n_steps=300)

        leak_gauss = res_gauss.expect[2][-1]
        leak_drag = res_drag.expect[2][-1]

        assert leak_drag < leak_gauss, (
            f"DRAG leakage ({leak_drag:.4e}) should be less than "
            f"Gaussian leakage ({leak_gauss:.4e})"
        )
