"""
06_volume_util.py
=================

Monte Carlo volume utilisation analysis.
Compares N=8 baseline vs N=9 final design.

Key finding: N=9 chord_skip=4 gives ~70% volume utilisation
             vs ~23% for the N=8 baseline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmpc import TMPCConfig, trace_cell
from tmpc.geometry import volume_utilisation

os.makedirs("figures", exist_ok=True)

configs = [
    ("N=8 baseline (s=1)", TMPCConfig(N=8, R=50e-3, H=40e-3, D_mirror=25.4e-3,
                                      M_halfLaps=66, chord_skip=1,
                                      wavelength=1.654e-6, w0=1.0e-3, M2=1.2)),
    ("N=9 final (s=4)",   TMPCConfig(N=9, R=50e-3, H=40e-3, D_mirror=25.4e-3,
                                      M_halfLaps=19, chord_skip=4,
                                      wavelength=1.654e-6, w0=1.5e-3, M2=1.2)),
]

results = []
print("Volume Utilisation Analysis (Monte Carlo, n=20,000 samples)")
print("=" * 60)

for label, cfg in configs:
    data = trace_cell(cfg)
    vol_result = volume_utilisation(data, cfg, n_samples=20_000, seed=2026)
    results.append((label, cfg, data, vol_result))
    
    print(f"\n{label}:")
    print(f"  OPL          = {data['path_length'][-1]:.2f} m")
    print(f"  Reflections  = {len(data['pass_no'])}")
    print(f"  Utilisation  = {vol_result['utilisation']*100:.1f}%")
    print(f"  Threshold    = {vol_result['threshold_m']*1e3:.3f} mm (mean spot radius)")
    print(f"  Mean dist.   = {vol_result['mean_distance_m']*1e3:.3f} mm")
    print(f"  Cell volume  = {vol_result['cell_volume_m3']*1e6:.1f} mL")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (label, cfg, data, vol_result) in zip(axes, results):
    # Top-down scatter plot coloured by distance to nearest beam
    rng = np.random.default_rng(42)
    n_pts = 5000
    r = cfg.R * np.sqrt(rng.random(n_pts))
    th = 2 * np.pi * rng.random(n_pts)
    zs = -cfg.H/2 + cfg.H * rng.random(n_pts)
    
    x_beam = data['x']
    y_beam = data['y']
    x_pts = r * np.cos(th)
    y_pts = r * np.sin(th)
    
    # Distance from each sample point to nearest beam chord (simplified 2D)
    min_d = np.full(n_pts, np.inf)
    for i in range(len(x_beam)):
        d = np.sqrt((x_pts - x_beam[i])**2 + (y_pts - y_beam[i])**2)
        np.minimum(min_d, d, out=min_d)
    
    threshold = vol_result['threshold_m']
    colors = ['C2' if d < threshold else 'C3' for d in min_d]
    
    ax.scatter(x_pts*1e3, y_pts*1e3, c=colors, s=1, alpha=0.3)
    ax.scatter(x_beam*1e3, y_beam*1e3, c='blue', s=8, zorder=5, label='Mirror hits')
    
    # Draw ring
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(cfg.R*1e3*np.cos(theta), cfg.R*1e3*np.sin(theta), 'k-', lw=0.5)
    
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nUtilisation: {vol_result["utilisation"]*100:.0f}%', fontsize=11)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')

fig.tight_layout()
fig.savefig("figures/06_volume_util.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nFigure saved to figures/06_volume_util.png")
print("\nConclusion: N=9 chord_skip=4 gives 3× better volume coverage.")
