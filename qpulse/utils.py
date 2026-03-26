"""Operator constructors and physical constants.

Convention: ħ = 1, frequencies in GHz, times in ns.
Energy unit: GHz (i.e., E = ω since ħ = 1).
"""

import numpy as np
from qutip import destroy, create, num, qeye, basis

# ---------- Unit conversions ----------
GHZ_TO_MHZ = 1e3
MHZ_TO_GHZ = 1e-3
NS_TO_US = 1e-3


def annihilation(n_levels: int):
    """Bosonic annihilation operator truncated to n_levels."""
    return destroy(n_levels)


def creation(n_levels: int):
    """Bosonic creation operator truncated to n_levels."""
    return create(n_levels)


def number_op(n_levels: int):
    """Number operator â†â truncated to n_levels."""
    return num(n_levels)


def identity(n_levels: int):
    """Identity operator for n_levels."""
    return qeye(n_levels)


def fock_state(n_levels: int, n: int):
    """Fock state |n⟩ in n_levels-dimensional Hilbert space."""
    return basis(n_levels, n)


def projector(n_levels: int, n: int):
    """Projector |n⟩⟨n| in n_levels-dimensional Hilbert space."""
    state = fock_state(n_levels, n)
    return state * state.dag()
