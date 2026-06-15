"""
01_baseline.py
==============

Baseline characterisation of the N=8, chord_skip=1 reference design.

Outputs
-------
- Prints OPL, reflections, chord length, AOI, alpha, spot size, throughput
- Saves top-down and spot-pattern plots to figures/01_baseline_*.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc import TMPCConfig, trace_cell, LossModel
from tmpc.geometry import incident_angle_deg
from tmpc.losses import per_pass_throughput, throughput_after_n
from tmpc.plotting import plot_topdown, plot_spot_patterns, plot_beam_evolution, plot_throughput

# ── Baseline design ──────────────────────────────────────────────────────────
cfg = TMPCConfig(
    N=8,
    R=50e-3,
    H=40e-3,
    D_mirror=25.4e-3,
    M_halfLaps=66,
    chord_skip=1,
    wavelength=1.654e-6,
    w0=1.0e-3,
    M2=1.2,
    waist_placement="center",
)

data = trace_cell(cfg)

# ── Print summary ─────────────────────────────────────────────────────────────
OPL = data["path_length"][-1]
n_ref = len(data["pass_no"])
chord_mm = cfg.L_chord * 1e3
aoi = incident_angle_deg(cfg)
alpha_deg = math.degrees(cfg.alpha_rad)
w_max_mm = data["spot_radius"].max() * 1e3
aperture = cfg.D_mirror / 2.0

lm = LossModel(R_ring=0.999, R_top=0.999)
_, cumul = per_pass_throughput(data["spot_radius"], aperture,
                               R_ring=lm.R_ring, R_top=lm.R_top,
                               top_bounce_index=n_ref // 2)

print(f"OPL design   : {cfg.OPL_design:.4f} m")
print(f"OPL traced   : {OPL:.4f} m")
print(f"Reflections (total) : {n_ref}")
print(f"Chord length : {chord_mm:.3f} mm")
print(f"AOI          : {aoi:.3f} deg")
print(f"alpha (out-of-plane) : {alpha_deg:.4f} deg")
print(f"Max spot radius : {w_max_mm:.4f} mm  (aperture = {aperture*1e3:.1f} mm)")
print(f"Throughput (R=0.999) : {cumul[-1]*100:.1f}%")
print()
print("N=8 baseline summary:")
print(f"  N={cfg.N}, chord_skip={cfg.chord_skip}, M_halfLaps={cfg.M_halfLaps}")
print(f"  R={cfg.R*1e3:.0f} mm, H={cfg.H*1e3:.0f} mm")
print(f"  OPL = {OPL:.2f} m, Reflections = {n_ref}")

# ── Save plots ─────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(6, 6))
plot_topdown(data, cfg, ax=ax1, title="N=8 baseline — top-down view")
fig1.savefig("figures/01_baseline_topdown.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
plot_spot_patterns(data, cfg, ax=ax2, title="N=8 baseline — spot pattern")
fig2.savefig("figures/01_baseline_spots.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4))
plot_beam_evolution(data, cfg, ax=ax3, title="N=8 baseline — beam evolution")
fig3.savefig("figures/01_baseline_beam.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

print("Figures saved to figures/")
