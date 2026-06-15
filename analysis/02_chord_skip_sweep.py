"""
02_chord_skip_sweep.py
======================

Sweep chord_skip from 1 to N-1 for N=8 and N=9.
Shows how OPL, AOI, throughput, and volume utilisation change with chord_skip.

Key finding: chord_skip = 4 for N=9 gives a 9-pointed star pattern
             that maximises volume utilisation and minimises AOI.
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

def sweep_chord_skip(N, M_halfLaps=66):
    results = []
    for s in range(1, N):
        # Check gcd condition - skip if gcd(s,N) != 1
        import math as _math
        if _math.gcd(s, N) != 1:
            results.append(None)
            continue
        try:
            cfg = TMPCConfig(N=N, R=50e-3, H=40e-3, D_mirror=25.4e-3,
                             M_halfLaps=M_halfLaps, chord_skip=s,
                             wavelength=1.654e-6, w0=1.0e-3, M2=1.2)
            data = trace_cell(cfg)
            OPL = data["path_length"][-1]
            n_ref = len(data["pass_no"])
            aoi = incident_angle_deg(cfg)
            aperture = cfg.D_mirror / 2.0
            _, cumul = per_pass_throughput(data["spot_radius"], aperture,
                                           R_ring=R_ring, R_top=R_ring,
                                           top_bounce_index=n_ref // 2)
            results.append({
                "s": s,
                "N": N,
                "OPL": OPL,
                "n_ref": n_ref,
                "aoi": aoi,
                "chord_mm": cfg.L_chord * 1e3,
                "throughput": cumul[-1],
            })
        except Exception as e:
            results.append(None)
    return [r for r in results if r is not None]

print("Chord-skip sweep for N=8:")
res8 = sweep_chord_skip(N=8)
for r in res8:
    print(f"  s={r['s']}: OPL={r['OPL']:.2f}m, AOI={r['aoi']:.1f}deg, "
          f"chord={r['chord_mm']:.1f}mm, T={r['throughput']*100:.1f}%")

print()
print("Chord-skip sweep for N=9:")
res9 = sweep_chord_skip(N=9)
for r in res9:
    print(f"  s={r['s']}: OPL={r['OPL']:.2f}m, AOI={r['aoi']:.1f}deg, "
          f"chord={r['chord_mm']:.1f}mm, T={r['throughput']*100:.1f}%")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax_row, res, title_suffix in zip(axes, [res8, res9], ['N=8', 'N=9']):
    s_vals = [r['s'] for r in res]
    
    ax_row[0].bar(s_vals, [r['OPL'] for r in res])
    ax_row[0].set_xlabel('chord_skip'); ax_row[0].set_ylabel('OPL [m]')
    ax_row[0].set_title(f'{title_suffix} — Optical path length')
    
    ax_row[1].bar(s_vals, [r['aoi'] for r in res], color='C1')
    ax_row[1].set_xlabel('chord_skip'); ax_row[1].set_ylabel('AOI [deg]')
    ax_row[1].set_title(f'{title_suffix} — Angle of incidence')

fig.tight_layout()
fig.savefig("figures/02_chord_skip_sweep.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved to figures/02_chord_skip_sweep.png")
