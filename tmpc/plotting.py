"""
plotting.py
===========

Matplotlib visualisation helpers for the toroidal multipass cell simulation.

Functions
---------
plot_topdown       : Top-down (x-y) view of mirror ring and beam chords.
plot_spot_patterns : Mirror z-height vs mirror index spot pattern.
plot_beam_evolution: Beam radius as function of path length / bounce number.
plot_throughput    : Cumulative throughput vs bounce number.
plot_roc_sweep     : ROC sweep results (max spot size, stability).
plot_sweep_1d      : Generic 1D parameter sweep plot.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


def _mm(x):
    """Convert metres to millimetres for plot labels."""
    return x * 1e3


def plot_topdown(data: dict, cfg, ax=None, title: str = None) -> plt.Axes:
    """
    Top-down (x-y) view of the multipass cell.
    Shows mirror ring, mirror positions, and beam chord pattern.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw cell boundary circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(_mm(cfg.R) * np.cos(theta), _mm(cfg.R) * np.sin(theta),
            'k-', lw=0.5, alpha=0.3, label='Ring')
    
    # Draw mirror positions
    for k in range(cfg.N):
        phi = 2 * math.pi * k / cfg.N
        mx = _mm(cfg.R) * math.cos(phi)
        my = _mm(cfg.R) * math.sin(phi)
        ax.plot(mx, my, 'bs', ms=8, zorder=5)
        ax.text(mx * 1.12, my * 1.12, str(k), ha='center', va='center',
                fontsize=7, color='blue')
    
    # Draw beam chords (first half-lap only to avoid clutter)
    x_mm = _mm(data["x"])
    y_mm = _mm(data["y"])
    j_up = len(x_mm) // 2
    
    # Entry point (approximately mirror 0 location)
    phi0 = 0.0
    x0 = _mm(cfg.R) * math.cos(phi0)
    y0 = _mm(cfg.R) * math.sin(phi0)
    
    pts = [(x0, y0)] + list(zip(x_mm[:j_up], y_mm[:j_up]))
    segs = [[pts[i], pts[i+1]] for i in range(len(pts)-1)]
    lc = LineCollection(segs, colors=['C1'], linewidths=0.8, alpha=0.6)
    ax.add_collection(lc)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(title or f'Top-down view (N={cfg.N}, skip={cfg.chord_skip})')
    ax.set_xlim(-_mm(cfg.R)*1.3, _mm(cfg.R)*1.3)
    ax.set_ylim(-_mm(cfg.R)*1.3, _mm(cfg.R)*1.3)
    
    return ax


def plot_spot_patterns(data: dict, cfg, ax=None, title: str = None) -> plt.Axes:
    """
    Plot spot positions (z vs mirror index) for all bounces.
    Colour-coded by going-up (orange) vs coming-down (blue).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    going_up = ~data["is_returning"]
    going_dn = data["is_returning"]
    
    ax.scatter(data["mirror_id"][going_up], _mm(data["z"][going_up]),
               s=4, c='C1', alpha=0.5, label='Up-going')
    ax.scatter(data["mirror_id"][going_dn], _mm(data["z"][going_dn]),
               s=4, c='C0', alpha=0.5, label='Down-going')
    
    ax.set_xlabel('Mirror index')
    ax.set_ylabel('z [mm]')
    ax.set_title(title or f'Spot pattern (N={cfg.N}, skip={cfg.chord_skip}, M={cfg.M_halfLaps})')
    ax.legend(markerscale=3, fontsize=8)
    ax.axhline(_mm(-cfg.H/2), color='k', lw=0.5, ls='--')
    ax.axhline(_mm(cfg.H/2), color='k', lw=0.5, ls='--')
    
    return ax


def plot_beam_evolution(data: dict, cfg, ax=None, title: str = None) -> plt.Axes:
    """
    Plot beam spot radius vs cumulative path length.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    opl_m = data["path_length"]
    w_mm = _mm(data["spot_radius"])
    aperture_mm = _mm(cfg.D_mirror / 2)
    
    ax.plot(opl_m, w_mm, 'C0-', lw=0.8, label='Spot radius 1/e²')
    ax.axhline(aperture_mm, color='r', ls='--', lw=1, label=f'Mirror aperture ({aperture_mm:.1f} mm)')
    
    ax.set_xlabel('Optical path length [m]')
    ax.set_ylabel('Beam radius [mm]')
    ax.set_title(title or f'Beam evolution (N={cfg.N}, ROC={cfg.ROC:.2f} m)')
    ax.legend(fontsize=8)
    
    return ax


def plot_throughput(data: dict, cfg, R_ring: float = 0.999, ax=None,
                    title: str = None) -> plt.Axes:
    """
    Plot cumulative throughput vs bounce number.
    """
    from .losses import per_pass_throughput
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    aperture = cfg.D_mirror / 2
    _, cumul = per_pass_throughput(data["spot_radius"], aperture, R_ring=R_ring)
    
    bounces = data["pass_no"]
    ax.plot(bounces, cumul * 100, 'C2-', lw=1.0)
    ax.set_xlabel('Bounce number')
    ax.set_ylabel('Cumulative throughput [%]')
    ax.set_title(title or f'Throughput (R={R_ring:.4f}, N={cfg.N})')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(cumul[-1]*100, color='r', ls='--',
               label=f'Final: {cumul[-1]*100:.1f}%')
    ax.legend(fontsize=9)
    
    return ax


def plot_roc_sweep(results: list, aperture_radius_mm: float = 12.7,
                   fig=None) -> plt.Figure:
    """
    Plot ROC sweep results: max spot size and stability indicators.
    """
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = fig.axes
    
    roc_vals = [r["ROC"] for r in results]
    max_w = [_mm(r.get("max_w_tan", r.get("max_w", 0))) for r in results]
    stable = [r.get("stable_tan", True) and r.get("stable_sag", True) for r in results]
    
    colors = ['C2' if s else 'C3' for s in stable]
    
    ax = axes[0]
    ax.bar([f'{r:.2f}' for r in roc_vals], max_w, color=colors)
    ax.axhline(aperture_radius_mm, color='r', ls='--', label=f'Aperture ({aperture_radius_mm} mm)')
    ax.set_xlabel('ROC [m]')
    ax.set_ylabel('Max spot radius [mm]')
    ax.set_title('Max beam size vs ROC')
    ax.legend()
    
    # Stability map
    ax2 = axes[1]
    g_tan = [r.get("g_tan", 0) for r in results]
    g_sag = [r.get("g_sag", 0) for r in results]
    ax2.scatter(g_tan, g_sag, c=colors, s=80, zorder=5)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.5, alpha=0.5)
    ax2.set_xlabel('g_tangential')
    ax2.set_ylabel('g_sagittal')
    ax2.set_title('Stability diagram')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axvline(0, color='k', lw=0.3)
    ax2.axhline(0, color='k', lw=0.3)
    
    green_patch = mpatches.Patch(color='C2', label='Stable')
    red_patch = mpatches.Patch(color='C3', label='Unstable')
    ax2.legend(handles=[green_patch, red_patch])
    
    fig.tight_layout()
    return fig


def plot_sweep_1d(x_vals, y_vals, xlabel: str = 'x', ylabel: str = 'y',
                  title: str = None, ax=None) -> plt.Axes:
    """Generic 1D parameter sweep plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, 'o-', ms=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
