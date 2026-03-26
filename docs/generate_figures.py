"""Generate all figures for the QPulse results summary.

Run from repo root: python docs/generate_figures.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpulse.transmon import TransmonDRAG
from qpulse.pulses import GaussianPulse, DRAGPulse
from qpulse.metrics import state_populations, leakage, gate_fidelity
from qpulse.optimizer import optimize_beta

FIGDIR = "docs/figures"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "lines.linewidth": 2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ── System parameters ──
OMEGA_Q = 5.0
ALPHA = -0.3
N_LEVELS = 3
GATE_TIME = 40.0
N_SIGMA = 2.0

transmon = TransmonDRAG(omega_q=OMEGA_Q, alpha=ALPHA, n_levels=N_LEVELS)

# ── Pre-compute optimized beta ──
print("Optimizing β...")
opt = optimize_beta(transmon, gate_time=GATE_TIME, bounds=(0.0, 3.0), objective="leakage")
beta_star = opt["beta_opt"]
print(f"  β* = {beta_star:.6f}, min leakage = {opt['cost']:.2e}")

opt_fid = optimize_beta(transmon, gate_time=GATE_TIME, bounds=(0.0, 3.0), objective="infidelity")
beta_star_fid = opt_fid["beta_opt"]
print(f"  β* (infidelity) = {beta_star_fid:.6f}, min infidelity = {opt_fid['cost']:.2e}")


# =====================================================================
# Figure 1: Pulse envelopes (time + frequency domain)
# =====================================================================
print("Figure 1: Pulse envelopes...")

gauss = GaussianPulse(gate_time=GATE_TIME, n_sigma=N_SIGMA)
drag = DRAGPulse(gate_time=GATE_TIME, beta=1.0, alpha=ALPHA, n_sigma=N_SIGMA)
t = np.linspace(0, GATE_TIME, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t, gauss.I_envelope(t), "C0", label=r"Gaussian $\Omega_I(t)$")
ax.plot(t, drag.Q_envelope(t), "C1", label=r"DRAG $\Omega_Q(t) = -\beta/\alpha \cdot \dot{\Omega}_I$")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Amplitude (GHz)")
ax.set_title("Pulse Envelopes (Time Domain)")
ax.legend(loc="upper right", fontsize=10)

ax = axes[1]
dt = t[1] - t[0]
freqs = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
fft_gauss = np.fft.fftshift(np.fft.fft(gauss.I_envelope(t)))
fft_drag_Q = np.fft.fftshift(np.fft.fft(drag.Q_envelope(t)))
ax.semilogy(freqs, np.abs(fft_gauss)**2, "C0", label=r"Gaussian $|\tilde{\Omega}_I|^2$")
ax.semilogy(freqs, np.abs(fft_drag_Q)**2, "C1", alpha=0.7, label=r"DRAG $|\tilde{\Omega}_Q|^2$")
ax.axvline(ALPHA, color="red", ls="--", lw=1.5, label=rf"$\alpha = {ALPHA}$ GHz (leakage freq)")
ax.set_xlim(-1.0, 0.5)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Power Spectral Density (arb.)")
ax.set_title("Spectral Content")
ax.legend(loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/01_pulse_envelopes.png")
plt.close(fig)


# =====================================================================
# Figure 2: Population dynamics — Gaussian vs DRAG
# =====================================================================
print("Figure 2: Population dynamics...")

res_gauss = transmon.simulate(gate_time=GATE_TIME, beta=0.0, n_steps=500)
res_drag = transmon.simulate(gate_time=GATE_TIME, beta=1.0, n_steps=500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, res, title in [
    (axes[0], res_gauss, r"Gaussian ($\beta = 0$)"),
    (axes[1], res_drag,  r"DRAG ($\beta = 1$)"),
]:
    pops = state_populations(res, N_LEVELS)
    t_ns = np.array(res.times)
    ax.plot(t_ns, pops["P0"], "C0", label=r"$P_0$")
    ax.plot(t_ns, pops["P1"], "C1", label=r"$P_1$")
    ax.plot(t_ns, pops["P2"], "C3", label=r"$P_2$ (leakage)", ls="--")
    L = leakage(res)
    F = gate_fidelity(res)
    ax.set_title(f"{title}\nLeakage = {L:.2e}, Fidelity = {F:.6f}")
    ax.set_xlabel("Time (ns)")
    ax.legend(loc="center right")

axes[0].set_ylabel("Population")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/02_population_dynamics.png")
plt.close(fig)


# =====================================================================
# Figure 3: β sweep
# =====================================================================
print("Figure 3: β sweep...")

beta_values = np.linspace(0, 2.0, 60)
sweep = transmon.sweep_beta(gate_time=GATE_TIME, beta_values=beta_values)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.semilogy(sweep["beta"], sweep["P2"], "C3o-", ms=3, label=r"$P_2(t_f)$ — leakage")
ax1.set_ylabel(r"Leakage $P_2(t_f)$")
ax1.set_title(rf"$\beta$ Sweep at $t_g = {GATE_TIME:.0f}$ ns, $\alpha/2\pi = {ALPHA*1e3:.0f}$ MHz")
ax1.grid(True, alpha=0.3)

ax2.plot(sweep["beta"], sweep["P1"], "C1s-", ms=3, label=r"$P_1(t_f)$ — fidelity")
ax2.set_xlabel(r"DRAG coefficient $\beta$")
ax2.set_ylabel(r"Fidelity $P_1(t_f)$")
ax2.grid(True, alpha=0.3)

idx_opt = np.argmin(sweep["P2"])
beta_opt_approx = sweep["beta"][idx_opt]
for ax in (ax1, ax2):
    ax.axvline(beta_opt_approx, color="red", ls=":", lw=1,
               label=rf"$\beta^* \approx {beta_opt_approx:.2f}$")
    ax.legend()

fig.tight_layout()
fig.savefig(f"{FIGDIR}/03_beta_sweep.png")
plt.close(fig)


# =====================================================================
# Figure 4: Three-panel comparison (Gaussian / DRAG β=1 / optimized)
# =====================================================================
print("Figure 4: Three-panel comparison...")

res_opt = transmon.simulate(gate_time=GATE_TIME, beta=beta_star, n_steps=500)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

configs = [
    (res_gauss, r"Gaussian ($\beta = 0$)"),
    (res_drag,  r"DRAG ($\beta = 1.0$)"),
    (res_opt,   rf"DRAG ($\beta^* = {beta_star:.3f}$)"),
]

for ax, (res, title) in zip(axes, configs):
    pops = state_populations(res, N_LEVELS)
    t_ns = np.array(res.times)
    ax.plot(t_ns, pops["P0"], "C0", label=r"$P_0$")
    ax.plot(t_ns, pops["P1"], "C1", label=r"$P_1$")
    ax.plot(t_ns, pops["P2"], "C3", ls="--", label=r"$P_2$ (leakage)")
    L = leakage(res)
    F = gate_fidelity(res)
    ax.set_title(f"{title}\nL = {L:.2e}, F = {F:.6f}")
    ax.set_xlabel("Time (ns)")
    ax.legend(fontsize=9)

axes[0].set_ylabel("Population")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/04_three_panel_comparison.png")
plt.close(fig)


# =====================================================================
# Figure 5: Leakage & fidelity vs gate time
# =====================================================================
print("Figure 5: Gate time sweep...")

gate_times = np.linspace(10, 80, 30)
sweep_gauss = transmon.sweep_gate_time(gate_times, beta=0.0)
sweep_drag = transmon.sweep_gate_time(gate_times, beta=beta_star)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.semilogy(sweep_gauss["gate_time"], sweep_gauss["leakage"], "C0o-", ms=4, label="Gaussian")
ax1.semilogy(sweep_drag["gate_time"], sweep_drag["leakage"], "C1s-", ms=4,
             label=rf"DRAG ($\beta^* = {beta_star:.2f}$)")
ax1.set_ylabel(r"Leakage $P_2(t_f)$")
ax1.set_title(r"Speed–Fidelity Tradeoff: Gaussian vs. DRAG")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(sweep_gauss["gate_time"], sweep_gauss["fidelity"], "C0o-", ms=4, label="Gaussian")
ax2.plot(sweep_drag["gate_time"], sweep_drag["fidelity"], "C1s-", ms=4,
         label=rf"DRAG ($\beta^* = {beta_star:.2f}$)")
ax2.axhline(0.999, color="gray", ls=":", lw=1, label="99.9% threshold")
ax2.set_xlabel("Gate time (ns)")
ax2.set_ylabel(r"Fidelity $P_1(t_f)$")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/05_gate_time_sweep.png")
plt.close(fig)


# =====================================================================
# Figure 6: Convergence check d=3 vs d=4
# =====================================================================
print("Figure 6: Convergence check...")

transmon_4 = TransmonDRAG(omega_q=OMEGA_Q, alpha=ALPHA, n_levels=4)
res_3 = transmon.simulate(gate_time=GATE_TIME, beta=beta_star, n_steps=500)
res_4 = transmon_4.simulate(gate_time=GATE_TIME, beta=beta_star, n_steps=500)

fig, ax = plt.subplots(figsize=(10, 5))
t3 = np.array(res_3.times)
t4 = np.array(res_4.times)

for k, (color, ls) in enumerate([("C0", "-"), ("C1", "-"), ("C3", "--")]):
    ax.plot(t3, res_3.expect[k], color=color, ls=ls, label=rf"$P_{k}$, $d=3$")
    ax.plot(t4, res_4.expect[k], color=color, ls=ls, lw=3, alpha=0.3, label=rf"$P_{k}$, $d=4$")

ax.plot(t4, res_4.expect[3], "C4", ls=":", label=r"$P_3$, $d=4$")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Population")
ax.set_title("Convergence Check: Hilbert Space Truncation")
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/06_convergence_check.png")
plt.close(fig)


# =====================================================================
# Figure 7: Open system — Gaussian vs DRAG with T1/T2
# =====================================================================
print("Figure 7: Open system...")

transmon_open = TransmonDRAG(
    omega_q=OMEGA_Q, alpha=ALPHA, n_levels=N_LEVELS,
    t1=50_000.0, t2=70_000.0,
)
res_open_gauss = transmon_open.simulate(gate_time=GATE_TIME, beta=0.0, n_steps=500)
res_open_drag = transmon_open.simulate(gate_time=GATE_TIME, beta=beta_star, n_steps=500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, res, title in [
    (axes[0], res_open_gauss, r"Gaussian (open, $T_1=50\mu$s)"),
    (axes[1], res_open_drag,  rf"DRAG $\beta^*={beta_star:.2f}$ (open)"),
]:
    pops = state_populations(res, N_LEVELS)
    t_ns = np.array(res.times)
    ax.plot(t_ns, pops["P0"], "C0", label=r"$P_0$")
    ax.plot(t_ns, pops["P1"], "C1", label=r"$P_1$")
    ax.plot(t_ns, pops["P2"], "C3", ls="--", label=r"$P_2$")
    L = leakage(res)
    F = gate_fidelity(res)
    ax.set_title(f"{title}\nL = {L:.2e}, F = {F:.6f}")
    ax.set_xlabel("Time (ns)")
    ax.legend()

axes[0].set_ylabel("Population")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/07_open_system.png")
plt.close(fig)


# =====================================================================
# Figure 8: Infidelity vs T1
# =====================================================================
print("Figure 8: Infidelity vs T1...")

t1_values = np.logspace(3, 5.5, 20)
fid_gauss = np.empty_like(t1_values)
fid_drag = np.empty_like(t1_values)

for i, t1_ns in enumerate(t1_values):
    t2_ns = min(2 * t1_ns, 1.4 * t1_ns)
    tr_open = TransmonDRAG(
        omega_q=OMEGA_Q, alpha=ALPHA, n_levels=N_LEVELS,
        t1=t1_ns, t2=t2_ns,
    )
    res_g = tr_open.simulate(gate_time=GATE_TIME, beta=0.0, n_steps=200)
    res_d = tr_open.simulate(gate_time=GATE_TIME, beta=beta_star, n_steps=200)
    fid_gauss[i] = gate_fidelity(res_g)
    fid_drag[i] = gate_fidelity(res_d)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(t1_values / 1e3, 1 - fid_gauss, "C0o-", ms=5, label="Gaussian")
ax.semilogx(t1_values / 1e3, 1 - fid_drag, "C1s-", ms=5,
            label=rf"DRAG ($\beta^* = {beta_star:.2f}$)")
ax.set_xlabel(r"$T_1$ ($\mu$s)")
ax.set_ylabel(r"Infidelity $1 - F$")
ax.set_title(r"Gate Infidelity vs. $T_1$ — Leakage-Limited Regime")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/08_infidelity_vs_t1.png")
plt.close(fig)


# =====================================================================
# Print summary table for the markdown
# =====================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Gate time:            {GATE_TIME} ns")
print(f"Peak Omega/|alpha|:   {gauss.amp / abs(ALPHA):.3f}")
print(f"Optimized beta*:      {beta_star:.6f}")
print(f"")
print(f"Gaussian leakage:     {leakage(res_gauss):.4e}")
print(f"DRAG(1) leakage:      {leakage(res_drag):.4e}")
print(f"DRAG(opt) leakage:    {leakage(res_opt):.4e}")
print(f"")
print(f"Gaussian fidelity:    {gate_fidelity(res_gauss):.6f}")
print(f"DRAG(1) fidelity:     {gate_fidelity(res_drag):.6f}")
print(f"DRAG(opt) fidelity:   {gate_fidelity(res_opt):.6f}")
print(f"")
print(f"Leakage reduction:    {leakage(res_gauss)/leakage(res_drag):.0f}x (beta=1)")
print(f"                      {leakage(res_gauss)/leakage(res_opt):.0f}x (beta*)")
print(f"")
L3 = leakage(res_3)
L4 = 1.0 - res_4.expect[0][-1] - res_4.expect[1][-1]
print(f"Convergence d=3:      {L3:.6e}")
print(f"Convergence d=4:      {L4:.6e}")
print(f"P3 (d=4):             {res_4.expect[3][-1]:.6e}")

print(f"\nAll figures saved to {FIGDIR}/")
