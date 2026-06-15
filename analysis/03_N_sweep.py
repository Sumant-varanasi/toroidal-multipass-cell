"""
03_N_sweep.py
=============

Sweep mirror count N from 6 to 16, using optimal chord_skip for each N.
Compares OPL, AOI, throughput across different mirror counts.

Key finding: N=9 with chord_skip=4 provides the best balance of
long OPL, low AOI, and high throughput.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc import TMPCConfig, trace_cell, LossModel
from tmpc.geometry import incident_angle_deg
from tmpc.losses import per_pass_throughput

os.makedirs("figures", exist_ok=True)

R_ring = 0.999
M_halfLaps = 66

results = []
for N in range(6, 17):
    # Find best chord_skip for this N (maximise OPL * throughput)
    best = None
    best_utility = -1
    for s in range(1, N):
        if math.gcd(s, N) != 1:
            continue
        try:
            cfg = TMPCConfig(N=N, R=50e-3, H=40e-3, D_mirror=25.4e-3,
                             M_halfLaps=M_halfLaps, chord_skip=s,
                             wavelength=1.654e-6, w0=1.0e-3, M2=1.2)
            data = trace_cell(cfg)
            OPL = data["path_length"][-1]
            n_ref = len(data["pass_no"])
            aperture = cfg.D_mirror / 2.0
            _, cumul = per_pass_throughput(data["spot_radius"], aperture,
                                           R_ring=R_ring, R_top=R_ring,
                                           top_bounce_index=n_ref // 2)
            T = cumul[-1]
            utility = OPL * T
            if utility > best_utility:
                best_utility = utility
                best = {
                    "N": N, "s": s, "OPL": OPL, "n_ref": n_ref,
                    "aoi": incident_angle_deg(cfg),
                    "chord_mm": cfg.L_chord * 1e3,
                    "throughput": T,
                    "utility": utility,
                }
        except Exception:
            pass
    if best:
        results.append(best)

print("N sweep (optimal chord_skip per N):")
print(f"{'N':>4} {'s':>3} {'OPL':>8} {'AOI':>7} {'chord':>8} {'T%':>6} {'utility':>10}")
for r in results:
    print(f"{r['N']:>4} {r['s']:>3} {r['OPL']:>8.2f}m {r['aoi']:>7.1f}° "
          f"{r['chord_mm']:>7.1f}mm {r['throughput']*100:>5.1f}% {r['utility']:>10.3f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
N_vals = [r['N'] for r in results]

axes[0,0].bar(N_vals, [r['OPL'] for r in results])
axes[0,0].set_xlabel('N (mirrors)'); axes[0,0].set_ylabel('OPL [m]')
axes[0,0].set_title('OPL vs N')

axes[0,1].bar(N_vals, [r['aoi'] for r in results], color='C1')
axes[0,1].set_xlabel('N (mirrors)'); axes[0,1].set_ylabel('AOI [deg]')
axes[0,1].set_title('Angle of incidence vs N')

axes[1,0].bar(N_vals, [r['throughput']*100 for r in results], color='C2')
axes[1,0].set_xlabel('N (mirrors)'); axes[1,0].set_ylabel('Throughput [%]')
axes[1,0].set_title(f'Throughput vs N (R={R_ring})')

axes[1,1].bar(N_vals, [r['utility'] for r in results], color='C3')
axes[1,1].set_xlabel('N (mirrors)'); axes[1,1].set_ylabel('OPL × Throughput')
axes[1,1].set_title('Combined utility vs N')

for ax in axes.flat:
    ax.set_xticks(N_vals)

fig.tight_layout()
fig.savefig("figures/03_N_sweep.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nFigure saved to figures/03_N_sweep.png")
