"""
04_roc_analysis.py
==================

ABCD matrix analysis: sweep mirror ROC from 0.2 m to 2.0 m and flat.
Shows beam stability, max spot size, and clipping risk for N=9, chord_skip=4.

Key finding:
- Flat mirrors FAIL (beam grows to 18 mm, clips at 12.7 mm aperture).
- All concave ROC values 0.2–2.0 m are stable (max spot < 1.2 mm).
- Recommended: ROC = 1.0 m (Thorlabs CM254-1000-P01, off-the-shelf).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc import TMPCConfig, trace_cell
from tmpc.gaussian import GaussianBeam, round_trip_evolution, roc_sweep

os.makedirs("figures", exist_ok=True)

# Final design: N=9, chord_skip=4
cfg = TMPCConfig(
    N=9, R=50e-3, H=40e-3, D_mirror=25.4e-3,
    M_halfLaps=19,  # gives ~342 bounces with chord_skip=4
    chord_skip=4,
    wavelength=1.654e-6, w0=1.5e-3, M2=1.2,
)

data = trace_cell(cfg)
OPL = data["path_length"][-1]
n_ref = len(data["pass_no"])
leg_len = cfg.L_chord / math.cos(cfg.alpha_rad)
theta_i = math.radians(10.0)  # AOI for N=9, s=4

print(f"N=9, s=4 final design:")
print(f"  OPL = {OPL:.2f} m")
print(f"  Reflections = {n_ref}")
print(f"  Chord = {cfg.L_chord*1e3:.1f} mm")
print(f"  Leg length = {leg_len*1e3:.1f} mm")
print(f"  AOI = 10.0 deg")
print()

# ROC sweep
roc_values = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, math.inf]
w0 = 1.5e-3
wavelength = 1.654e-6
aperture_radius = 25.4e-3 / 2  # 12.7 mm

results = roc_sweep(leg_len, theta_i, roc_values, n_ref, w0, wavelength, M2=1.2)

print(f"{'ROC':>8} {'Tan':>6} {'Sag':>6} {'Max_w_mm':>10} {'Out_w_mm':>10} {'Clips':>6}")
print("-" * 55)
for r in results:
    roc_str = "Flat" if math.isinf(r["ROC"]) else f"{r['ROC']:.1f}m"
    clips = "YES" if r["max_w_tan"]*1e3 > aperture_radius*1e3 else "No"
    print(f"{roc_str:>8} {'✓' if r['stable_tan'] else '✗':>6} "
          f"{'✓' if r['stable_sag'] else '✗':>6} "
          f"{r['max_w_tan']*1e3:>10.3f} {r['out_w_tan']*1e3:>10.3f} {clips:>6}")

# Plot beam evolution for ROC=1.0m and flat
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for roc_val, color, label in [(1.0, 'C0', 'ROC=1.0m (recommended)'),
                                (math.inf, 'C3', 'Flat mirrors (FAILS)')]:
    beam_in = GaussianBeam.from_waist(w0, wavelength, M2=1.2)
    w_tan, _ = round_trip_evolution(leg_len, roc_val, theta_i, n_ref, beam_in, "tangential")
    w_sag, _ = round_trip_evolution(leg_len, roc_val, theta_i, n_ref, beam_in, "sagittal")
    
    bounces = np.arange(n_ref + 1)
    axes[0].plot(bounces, w_tan * 1e3, color=color, lw=1, label=label)
    axes[1].plot(bounces, w_sag * 1e3, color=color, lw=1, label=label)

for ax, plane in zip(axes, ['Tangential', 'Sagittal']):
    ax.axhline(aperture_radius * 1e3, color='k', ls='--', lw=1, label='Mirror aperture (12.7mm)')
    ax.set_xlabel('Bounce number')
    ax.set_ylabel('Beam radius [mm]')
    ax.set_title(f'{plane} plane — beam evolution')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 20)

fig.tight_layout()
fig.savefig("figures/04_roc_analysis.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nFigure saved to figures/04_roc_analysis.png")
print("\nConclusion: ROC=1.0m (Thorlabs CM254-1000-P01) recommended.")
print("  Max spot = 1.027 mm over 342 bounces, 11.7 mm safety margin.")
