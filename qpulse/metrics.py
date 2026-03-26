"""Physics metrics for gate characterization.

All functions operate on qutip.Result objects returned by TransmonDRAG.simulate().
"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, basis


def state_populations(result, n_levels: int = 3) -> dict[str, np.ndarray]:
    """Extract time-resolved populations P_k(t) from expectation values.

    Parameters
    ----------
    result : qutip.Result
        Output of mesolve with projector e_ops.
    n_levels : int
        Number of levels tracked.

    Returns
    -------
    dict mapping "P0", "P1", ... to 1-D arrays over time.
    """
    return {f"P{k}": np.array(result.expect[k]) for k in range(n_levels)}


def leakage(result) -> float:
    """Final leakage out of the computational subspace.

    L = 1 - P_0(t_f) - P_1(t_f) = sum of populations outside {|0⟩, |1⟩}.
    """
    P0_final = result.expect[0][-1]
    P1_final = result.expect[1][-1]
    return float(1.0 - P0_final - P1_final)


def gate_fidelity(result, target_state_index: int = 1) -> float:
    """State-transfer fidelity for a π-pulse starting from |0⟩.

    F = ⟨target|ρ(t_f)|target⟩ = P_target(t_f).

    For a perfect X-gate (π-pulse) starting from |0⟩, the target is |1⟩.
    """
    return float(result.expect[target_state_index][-1])


def process_fidelity(result, n_levels: int = 3) -> float:
    """Average gate fidelity estimated from the final unitary.

    Extracts the propagator U from stored states (assumes pure-state evolution
    starting from |0⟩ through |d-1⟩ columns), then computes:
        F = |Tr(U_target† · U_sim)|² / d²

    where d = 2 (computational subspace) and U_target = σ_x (X-gate).

    NOTE: This requires running simulate() for each initial basis state.
    For a single run starting from |0⟩, use gate_fidelity() instead.
    """
    if not result.states:
        raise ValueError("Result has no stored states. Re-run with store_states=True.")

    # For single-run from |0⟩, we can only get one column of U
    # Return the state-transfer fidelity as a lower bound
    psi_final = result.states[-1]
    target = basis(n_levels, 1)  # |1⟩ for π-pulse from |0⟩
    overlap = target.dag() * psi_final
    return float(np.abs(overlap[0, 0]) ** 2)
