"""Transmon Hamiltonian builder and DRAG simulation driver.

Frame convention
----------------
We work in the frame rotating at the drive frequency ω_d = ω_q.
After the rotating-wave approximation (RWA), the static Hamiltonian
contains only the anharmonicity, and the drive Hamiltonian has
slowly-varying envelopes.

Units: ħ = 1, frequencies in GHz, times in ns.
       (product GHz · ns = dimensionless ✓)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qutip import basis, mesolve, Qobj

from qpulse.utils import annihilation, creation, number_op, identity, projector
from qpulse.pulses import GaussianPulse, DRAGPulse

if TYPE_CHECKING:
    from qutip import Result


class TransmonDRAG:
    """Transmon qubit simulator with Gaussian and DRAG pulse support.

    Parameters
    ----------
    omega_q : float
        Qubit |0⟩→|1⟩ frequency in GHz (e.g. 5.0).
    alpha : float
        Anharmonicity in GHz (negative for transmon, e.g. -0.3).
    n_levels : int
        Hilbert space truncation dimension (≥ 3).
    t1 : float or None
        Energy relaxation time T₁ in ns. None = closed system.
    t2 : float or None
        Dephasing time T₂ in ns. None = no pure dephasing.
    """

    def __init__(
        self,
        omega_q: float = 5.0,
        alpha: float = -0.3,
        n_levels: int = 3,
        t1: float | None = None,
        t2: float | None = None,
    ):
        if n_levels < 3:
            raise ValueError("n_levels must be ≥ 3 to capture leakage to |2⟩")

        self.omega_q = omega_q
        self.alpha = alpha
        self.n_levels = n_levels
        self.t1 = t1
        self.t2 = t2

        # Pre-build operators (reused across simulations)
        self._a = annihilation(n_levels)
        self._a_dag = creation(n_levels)
        self._n = number_op(n_levels)
        self._I = identity(n_levels)

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def _H_static(self) -> Qobj:
        """Static Hamiltonian in the rotating frame (drive resonant: Δ = 0).

        H_0 = (α/2) â†â(â†â - I)

        This is the anharmonic part only; the linear term ω_q â†â vanishes
        in the frame rotating at ω_d = ω_q.
        """
        return 0.5 * self.alpha * self._n * (self._n - self._I)

    def _H_drive_operators(self) -> tuple[Qobj, Qobj]:
        """Drive coupling operators for I and Q channels (RWA).

        In the rotating frame after RWA:
          H_d = (1/2)[Ω_I(t)(â + â†) + Ω_Q(t)·i(â† - â)]

        Returns (â + â†)/2 and i(â† - â)/2.
        """
        x_op = 0.5 * (self._a + self._a_dag)      # I-channel coupling
        y_op = 0.5j * (self._a_dag - self._a)      # Q-channel coupling
        return x_op, y_op

    def _collapse_operators(self) -> list[Qobj]:
        """Lindblad collapse operators for T₁ relaxation and T_φ dephasing."""
        c_ops = []
        if self.t1 is not None and self.t1 > 0:
            # Relaxation: L₁ = √(1/T₁) â
            c_ops.append(np.sqrt(1.0 / self.t1) * self._a)
        if self.t2 is not None and self.t1 is not None:
            # Pure dephasing rate: 1/T_φ = 1/T₂ - 1/(2T₁)
            gamma_phi = 1.0 / self.t2 - 1.0 / (2.0 * self.t1)
            if gamma_phi > 0:
                c_ops.append(np.sqrt(gamma_phi) * self._n)
        return c_ops

    def build_hamiltonian(self, pulse: GaussianPulse) -> list:
        """Construct the full time-dependent Hamiltonian in QuTiP list format.

        Returns [H_0, [H_x, Ω_I(t, args)], [H_y, Ω_Q(t, args)]]
        suitable for passing directly to qutip.mesolve.
        """
        H0 = self._H_static()
        Hx, Hy = self._H_drive_operators()

        def coeff_I(t, **kw):
            return pulse.I_envelope(t)

        def coeff_Q(t, **kw):
            return pulse.Q_envelope(t)

        return [H0, [Hx, coeff_I], [Hy, coeff_Q]]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        gate_time: float,
        beta: float = 0.0,
        n_sigma: float = 2.0,
        n_steps: int = 500,
        psi0_index: int = 0,
    ) -> Result:
        """Run a single DRAG gate simulation.

        Parameters
        ----------
        gate_time : float
            Pulse duration in ns.
        beta : float
            DRAG coefficient (0 = pure Gaussian).
        n_sigma : float
            Gaussian truncation parameter.
        n_steps : int
            Number of time steps in the output.
        psi0_index : int
            Initial Fock state index (default |0⟩).

        Returns
        -------
        qutip.Result
            Contains .times, .expect (populations), and .states.
        """
        # Build pulse
        if beta == 0.0:
            pulse = GaussianPulse(gate_time=gate_time, n_sigma=n_sigma)
        else:
            pulse = DRAGPulse(
                gate_time=gate_time,
                beta=beta,
                alpha=self.alpha,
                n_sigma=n_sigma,
            )

        # Build Hamiltonian
        H = self.build_hamiltonian(pulse)

        # Initial state
        psi0 = basis(self.n_levels, psi0_index)

        # Expectation operators: projectors |k⟩⟨k| for population tracking
        e_ops = [projector(self.n_levels, k) for k in range(self.n_levels)]

        # Collapse operators (empty list for closed system)
        c_ops = self._collapse_operators()

        # Time grid
        tlist = np.linspace(0, gate_time, n_steps)

        # Solve
        result = mesolve(
            H,
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=e_ops,
            options={"store_states": True, "max_step": gate_time / n_steps},
        )
        return result

    def sweep_beta(
        self,
        gate_time: float,
        beta_values: np.ndarray,
        n_sigma: float = 2.0,
        n_steps: int = 300,
    ) -> dict:
        """Sweep β and record final-state populations and leakage.

        Returns
        -------
        dict with keys:
            "beta" : array of β values
            "P0"   : final |0⟩ population for each β
            "P1"   : final |1⟩ population for each β
            "P2"   : final |2⟩ population for each β (= leakage)
        """
        P0 = np.empty_like(beta_values)
        P1 = np.empty_like(beta_values)
        P2 = np.empty_like(beta_values)

        for i, beta in enumerate(beta_values):
            res = self.simulate(
                gate_time=gate_time,
                beta=beta,
                n_sigma=n_sigma,
                n_steps=n_steps,
            )
            P0[i] = res.expect[0][-1]
            P1[i] = res.expect[1][-1]
            P2[i] = res.expect[2][-1]

        return {"beta": beta_values, "P0": P0, "P1": P1, "P2": P2}

    def sweep_gate_time(
        self,
        gate_times: np.ndarray,
        beta: float = 0.0,
        n_sigma: float = 2.0,
        n_steps: int = 300,
    ) -> dict:
        """Sweep gate time and record final-state leakage and fidelity.

        Returns
        -------
        dict with keys:
            "gate_time" : array of gate durations (ns)
            "P0", "P1", "P2" : final populations
            "leakage" : P2 (= 1 - P0 - P1 if n_levels=3)
            "fidelity" : P1 (population transfer fidelity for π-pulse from |0⟩)
        """
        n = len(gate_times)
        P0, P1, P2 = np.empty(n), np.empty(n), np.empty(n)

        for i, tg in enumerate(gate_times):
            res = self.simulate(
                gate_time=tg,
                beta=beta,
                n_sigma=n_sigma,
                n_steps=n_steps,
            )
            P0[i] = res.expect[0][-1]
            P1[i] = res.expect[1][-1]
            P2[i] = res.expect[2][-1]

        return {
            "gate_time": gate_times,
            "P0": P0,
            "P1": P1,
            "P2": P2,
            "leakage": P2,
            "fidelity": P1,
        }
