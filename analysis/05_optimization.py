"""
05_optimization.py
==================

Multi-algorithm optimisation of TMPC design parameters.
Runs SciPy Powell, Genetic Algorithm, and Particle Swarm Optimisation.

Optimises: R, H, M_halfLaps, w0, R_ring
Objective: maximise OPL × throughput (with spot overlap penalty)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc.optimize import optimise_scipy, optimise_ga, optimise_pso, OptResult

os.makedirs("figures", exist_ok=True)

N = 9  # Fixed: N=9 as identified by chord_skip sweep

print("=" * 60)
print("TMPC Multi-Algorithm Optimisation (N=9)")
print("=" * 60)

def print_result(res: OptResult, label: str):
    cfg = res.cfg
    print(f"\n{label}:")
    print(f"  OPL        = {res.OPL:.3f} m")
    print(f"  Throughput = {res.throughput*100:.1f}%")
    print(f"  Utility    = {res.utility:.4f}")
    print(f"  N evals    = {res.n_evals}")
    print(f"  R = {cfg.R*1e3:.1f}mm, H = {cfg.H*1e3:.1f}mm")
    print(f"  M_halfLaps = {cfg.M_halfLaps}, chord_skip = {cfg.chord_skip}")
    print(f"  w0 = {cfg.w0*1e3:.2f}mm")

# ── 1. Powell optimisation ────────────────────────────────────────────────────
print("\nRunning SciPy Powell...")
result_powell = optimise_scipy(N=N, method="Powell", maxiter=500)
print_result(result_powell, "SciPy/Powell")

# ── 2. Genetic Algorithm ──────────────────────────────────────────────────────
print("\nRunning Genetic Algorithm (pop=30, gen=40)...")
result_ga = optimise_ga(N=N, pop_size=30, n_generations=40, seed=42)
print_result(result_ga, "Genetic Algorithm")

# ── 3. Particle Swarm Optimisation ───────────────────────────────────────────
print("\nRunning Particle Swarm Optimisation (30 particles, 60 iters)...")
result_pso = optimise_pso(N=N, n_particles=30, n_iters=60, seed=7)
print_result(result_pso, "PSO")

# ── Comparison plot ───────────────────────────────────────────────────────────
results = [result_powell, result_ga, result_pso]
labels = ['Powell', 'GA', 'PSO']
colors = ['C0', 'C1', 'C2']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].bar(labels, [r.OPL for r in results], color=colors)
axes[0].set_ylabel('OPL [m]')
axes[0].set_title('Optical Path Length')

axes[1].bar(labels, [r.throughput*100 for r in results], color=colors)
axes[1].set_ylabel('Throughput [%]')
axes[1].set_title('Throughput')
axes[1].set_ylim(0, 100)

axes[2].bar(labels, [r.utility for r in results], color=colors)
axes[2].set_ylabel('OPL × Throughput')
axes[2].set_title('Combined Utility')

fig.suptitle(f'TMPC Multi-Algorithm Optimisation (N={N})', fontsize=13)
fig.tight_layout()
fig.savefig("figures/05_optimization.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nFigure saved to figures/05_optimization.png")

# Best result
best = max(results, key=lambda r: r.utility)
print(f"\nBest method: {best.method}")
print(f"Best OPL = {best.OPL:.3f} m, Throughput = {best.throughput*100:.1f}%")
