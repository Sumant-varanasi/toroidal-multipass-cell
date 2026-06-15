"""
07_chord_skip_comparison.py
============================

Side-by-side comparison: N=8 baseline (chord_skip=1) vs N=9 final (chord_skip=4).
The definitive summary figure for the internship report / conference paper.

Shows:
- Top-down beam patterns
- Spot patterns on mirrors
- Beam evolution with ABCD analysis
- Throughput comparison

This is the key result figure: chord_skip=4 in N=9 gives 9-pointed
star pattern through cell centre, 3× volume utilisation, 67% more OPL.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc import TMPCConfig, trace_cell, LossModel
from tmpc.geometry import incident_angle_deg, volume_utilisation
from tmpc.losses import per_pass_throughput
from tmpc.plotting import plot_topdown, plot_spot_patterns

os.makedirs("figures", exist_ok=True)

# ── Designs ───────────────────────────────────────────────────────────────────
cfg_baseline = TMPCConfig(
    N=8, R=50e-3, H=40e-3, D_mirror=25.4e-3,
    M_halfLaps=66, chord_skip=1,
    wavelength=1.654e-6, w0=1.0e-3, M2=1.2,
)
cfg_final = TMPCConfig(
    N=9, R=50e-3, H=40e-3, D_mirror=25.4e-3,
    M_halfLaps=19, chord_skip=4,
    wavelength=1.654e-6, w0=1.5e-3, M2=1.2,
)

data_b = trace_cell(cfg_baseline)
data_f = trace_cell(cfg_final)

R_ring = 0.999
aperture = cfg_baseline.D_mirror / 2.0

_, cumul_b = per_pass_throughput(data_b["spot_radius"], aperture,
                                  R_ring=R_ring, R_top=R_ring,
                                  top_bounce_index=len(data_b["pass_no"]) // 2)
_, cumul_f = per_pass_throughput(data_f["spot_radius"], aperture,
                                  R_ring=R_ring, R_top=R_ring,
                                  top_bounce_index=len(data_f["pass_no"]) // 2)

vol_b = volume_utilisation(data_b, cfg_baseline, n_samples=10_000, seed=2026)
vol_f = volume_utilisation(data_f, cfg_final, n_samples=10_000, seed=2026)

# ── Print comparison table ────────────────────────────────────────────────────
OPL_b = data_b["path_length"][-1]
OPL_f = data_f["path_length"][-1]
n_b = len(data_b["pass_no"])
n_f = len(data_f["pass_no"])

print("=" * 70)
print("DEFINITIVE COMPARISON: N=8 baseline vs N=9 final (chord_skip=4)")
print("=" * 70)
print(f"{'Parameter':<30} {'N=8 baseline':>15} {'N=9 final':>15} {'Change':>10}")
print("-" * 70)
print(f"{'OPL [m]':<30} {OPL_b:>15.2f} {OPL_f:>15.2f} {(OPL_f/OPL_b-1)*100:>+9.0f}%")
print(f"{'Reflections':<30} {n_b:>15} {n_f:>15} {(n_f/n_b-1)*100:>+9.0f}%")
print(f"{'Chord length [mm]':<30} {cfg_baseline.L_chord*1e3:>15.1f} {cfg_final.L_chord*1e3:>15.1f} {(cfg_final.L_chord/cfg_baseline.L_chord-1)*100:>+9.0f}%")
print(f"{'AOI [deg]':<30} {incident_angle_deg(cfg_baseline):>15.1f} {incident_angle_deg(cfg_final):>15.1f}")
print(f"{'Throughput (R=0.999) [%]':<30} {cumul_b[-1]*100:>15.1f} {cumul_f[-1]*100:>15.1f} {(cumul_f[-1]-cumul_b[-1])*100:>+9.1f}pp")
print(f"{'Volume utilisation [%]':<30} {vol_b["utilisation"]*100:>15.0f} {vol_f["utilisation"]*100:>15.0f} {(vol_f["utilisation"]-vol_b["utilisation"])*100:>+9.0f}pp")
print("=" * 70)

# ── Comparison figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-down views
plot_topdown(data_b, cfg_baseline, ax=axes[0, 0],
             title=f"N=8, s=1 — OPL={OPL_b:.1f}m, Vol={vol_b['utilisation']*100:.0f}%")
plot_topdown(data_f, cfg_final, ax=axes[0, 1],
             title=f"N=9, s=4 — OPL={OPL_f:.1f}m, Vol={vol_f['utilisation']*100:.0f}%")

# Spot patterns
plot_spot_patterns(data_b, cfg_baseline, ax=axes[1, 0],
                   title="N=8 spot pattern")
plot_spot_patterns(data_f, cfg_final, ax=axes[1, 1],
                   title="N=9 spot pattern")

fig.suptitle("TMPC: N=8 baseline vs N=9 final design (chord_skip=4)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("figures/07_chord_skip_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nFigure saved to figures/07_chord_skip_comparison.png")
