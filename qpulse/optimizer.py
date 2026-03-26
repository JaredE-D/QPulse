"""DRAG β parameter optimization via scipy.optimize.

Minimizes leakage to |2⟩ (or maximizes state-transfer fidelity)
as a function of the DRAG coefficient β.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar, minimize

if TYPE_CHECKING:
    from qpulse.transmon import TransmonDRAG


def optimize_beta(
    transmon: TransmonDRAG,
    gate_time: float,
    method: str = "bounded",
    bounds: tuple[float, float] = (0.0, 3.0),
    objective: str = "leakage",
    n_sigma: float = 2.0,
) -> dict:
    """Find the β that minimizes leakage or maximizes fidelity.

    Parameters
    ----------
    transmon : TransmonDRAG
        Configured transmon instance.
    gate_time : float
        Gate duration in ns.
    method : str
        Optimization method for scipy ("bounded" for scalar, "Nelder-Mead" for general).
    bounds : tuple
        Search range for β.
    objective : str
        "leakage" to minimize P₂(t_f), or "infidelity" to minimize 1-P₁(t_f).
    n_sigma : float
        Gaussian truncation parameter.

    Returns
    -------
    dict with "beta_opt", "cost", "result" (full scipy result).
    """

    def cost_function(beta: float) -> float:
        res = transmon.simulate(
            gate_time=gate_time,
            beta=float(beta),
            n_sigma=n_sigma,
            n_steps=200,
        )
        if objective == "leakage":
            return float(res.expect[2][-1])  # minimize P₂
        else:
            return float(1.0 - res.expect[1][-1])  # minimize 1 - P₁

    if method == "bounded":
        opt = minimize_scalar(cost_function, bounds=bounds, method="bounded")
        return {
            "beta_opt": opt.x,
            "cost": opt.fun,
            "result": opt,
        }
    else:
        x0 = np.array([0.5 * (bounds[0] + bounds[1])])
        opt = minimize(cost_function, x0, method=method)
        return {
            "beta_opt": opt.x[0],
            "cost": opt.fun,
            "result": opt,
        }
