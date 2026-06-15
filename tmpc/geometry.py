"""
geometry.py
===========

Toroidal multipass cell geometry and ray tracing.

The cell consists of N flat mirrors arranged on a ring of radius R, with their
normals pointing radially inward. The beam enters through a small hole in
mirror 0 at the bottom of the cell (z = -H/2), spirals upward as it bounces
around the polygon, hits a flat top retroreflector at z = +H/2, then spirals
back down and exits through the same hole.

Chord-skip pattern
------------------
The ``chord_skip`` parameter controls which mirror is visited at each bounce:

chord_skip = 1 (default)
    Beam hops to the neighbouring mirror: 0 -> 1 -> 2 -> ... -> N-1 -> 0
    Each chord is short (length 2R sin(pi/N)) and stays near the ring
    perimeter -- the centre of the cell receives no light. AOI is large
    (pi/2 - pi/N).

chord_skip = s (general)
    Beam hops s mirrors at a time, modulo N: 0 -> s -> 2s -> 3s -> ...
    Each chord is longer (length 2R sin(s*pi/N)) and passes through the
    cell interior. AOI shrinks to pi/2 - s*pi/N. For s = N/2 the chords
    are diameters and AOI -> 0 (normal incidence).

Choose gcd(s, N) = 1 so that every mirror is visited within the first
N bounces -- otherwise only a subset of the ring is used.

This generalisation matches the Tuzson/Graf/Chang toroidal-MPC literature,
where diagonal chords sample the entire cell volume rather than skimming
the perimeter.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


@dataclass
class TMPCConfig:
    """Geometric and optical configuration for a toroidal multipass cell."""

    N: int = 8
    R: float = 50e-3
    H: float = 40e-3
    D_mirror: float = 25.4e-3
    ROC: float = math.inf

    M_halfLaps: int = 66
    chord_skip: int = 1

    z_entry: Optional[float] = None

    wavelength: float = 1.654e-6
    w0: float = 1.0e-3
    M2: float = 1.2
    waist_placement: str = "center"

    L_chord: float = field(init=False)
    alpha_rad: float = field(init=False)
    total_reflections: int = field(init=False)
    OPL_design: float = field(init=False)

    def __post_init__(self) -> None:
        if self.chord_skip < 1 or self.chord_skip >= self.N:
            raise ValueError(
                f"chord_skip must be in [1, N-1]; got {self.chord_skip}"
            )
        self.L_chord = 2.0 * self.R * math.sin(
            self.chord_skip * math.pi / self.N
        )
        if (self.N * self.M_halfLaps) % 2 != 0:
            raise ValueError("N * M_halfLaps must be even (j_up must be int).")
        j_up = self.N * self.M_halfLaps // 2
        if self.z_entry is None:
            self.z_entry = -self.H / 2.0
        self.alpha_rad = math.atan2(self.H, j_up * self.L_chord)
        self.total_reflections = 2 * j_up
        leg_len = self.L_chord / math.cos(self.alpha_rad)
        self.OPL_design = 2.0 * j_up * leg_len


def mirror_centre(k: int, cfg: TMPCConfig) -> np.ndarray:
    phi = 2.0 * math.pi * k / cfg.N
    return np.array([cfg.R * math.cos(phi), cfg.R * math.sin(phi), 0.0])


def mirror_normal(k: int, cfg: TMPCConfig) -> np.ndarray:
    phi = 2.0 * math.pi * k / cfg.N
    return np.array([-math.cos(phi), -math.sin(phi), 0.0])


def incident_angle_deg(cfg: TMPCConfig) -> float:
    s_eff = min(cfg.chord_skip, cfg.N - cfg.chord_skip)
    theta = math.pi / 2.0 - s_eff * math.pi / cfg.N
    return math.degrees(theta)


def trace_cell(cfg: TMPCConfig, *, gaussian: bool = True) -> dict:
    N = cfg.N
    j_up = cfg.N * cfg.M_halfLaps // 2
    dz_per_chord = cfg.L_chord * math.tan(cfg.alpha_rad)
    leg_len = cfg.L_chord / math.cos(cfg.alpha_rad)

    total = 2 * j_up
    pass_no = np.arange(1, total + 1)
    mirror_id = np.empty(total, dtype=int)
    x = np.empty(total)
    y = np.empty(total)
    z = np.empty(total)
    inc = np.full(total, incident_angle_deg(cfg))
    spotR = np.empty(total)
    path = np.empty(total)
    is_back = np.zeros(total, dtype=bool)

    z_run = cfg.z_entry
    path_run = 0.0
    s = cfg.chord_skip

    for j in range(j_up):
        z_run += dz_per_chord
        path_run += leg_len
        mid = ((j + 1) * s) % N
        mirror_id[j] = mid
        p = mirror_centre(mid, cfg)
        x[j] = p[0]; y[j] = p[1]; z[j] = z_run
        path[j] = path_run

    for j in range(j_up, total):
        z_run -= dz_per_chord
        path_run += leg_len
        k_down = j - j_up + 1
        mid = ((j_up - k_down + 1) * s) % N
        is_back[j] = True
        mirror_id[j] = mid
        p = mirror_centre(mid, cfg)
        x[j] = p[0]; y[j] = p[1]; z[j] = z_run
        path[j] = path_run

    if gaussian:
        if cfg.waist_placement == "center":
            z_waist = path[-1] / 2.0
        elif cfg.waist_placement == "entrance":
            z_waist = 0.0
        else:
            raise ValueError("waist_placement must be 'center' or 'entrance'.")
        zR = math.pi * cfg.w0 ** 2 / (cfg.M2 * cfg.wavelength)
        spotR[:] = cfg.w0 * np.sqrt(1.0 + ((path - z_waist) / zR) ** 2)
    else:
        spotR[:] = cfg.w0

    return {
        "pass_no": pass_no,
        "mirror_id": mirror_id,
        "x": x, "y": y, "z": z,
        "incident_angle_deg": inc,
        "spot_radius": spotR,
        "path_length": path,
        "is_returning": is_back,
    }


def per_mirror_z_spots(data: dict, N: int) -> dict:
    out = {}
    for k in range(N):
        sel = data["mirror_id"] == k
        out[k] = data["z"][sel]
    return out


def distinct_spots_per_mirror(data: dict, N: int,
                              tol: float = 1e-6) -> dict:
    out = {}
    for k in range(N):
        sel = data["mirror_id"] == k
        zs = np.sort(data["z"][sel])
        if len(zs) == 0:
            out[k] = zs
            continue
        keep = [zs[0]]
        for z in zs[1:]:
            if z - keep[-1] > tol:
                keep.append(float(z))
        out[k] = np.array(keep)
    return out


def min_spot_separation(data: dict, N: int) -> float:
    grouped = distinct_spots_per_mirror(data, N)
    dmin = math.inf
    for k, zs in grouped.items():
        if len(zs) < 2:
            continue
        diffs = np.diff(zs)
        if diffs.min() < dmin:
            dmin = float(diffs.min())
    return dmin if math.isfinite(dmin) else 0.0


def spots_overlap(data: dict, cfg: TMPCConfig,
                  sep_factor: float = 2.0) -> bool:
    grouped = distinct_spots_per_mirror(data, cfg.N)
    for k, zs in grouped.items():
        if len(zs) < 2:
            continue
        sel = data["mirror_id"] == k
        w_mean = float(data["spot_radius"][sel].mean())
        diffs = np.diff(zs)
        if diffs.min() < sep_factor * w_mean:
            return True
    return False


def mirror_fill_fraction(data: dict, cfg: TMPCConfig) -> float:
    A_mirror = math.pi * (cfg.D_mirror / 2.0) ** 2
    grouped = distinct_spots_per_mirror(data, cfg.N)
    worst = 0.0
    for k, zs in grouped.items():
        if len(zs) == 0:
            continue
        sel = data["mirror_id"] == k
        w_mean = float(data["spot_radius"][sel].mean())
        coverage = len(zs) * math.pi * w_mean ** 2 / A_mirror
        if coverage > worst:
            worst = coverage
    return min(worst, 1.0)


def volume_utilisation(data: dict, cfg: TMPCConfig, *,
                       threshold: Optional[float] = None,
                       n_samples: int = 20_000,
                       seed: int = 2026) -> dict:
    """Monte Carlo volume utilisation estimate."""
    if threshold is None:
        threshold = float(np.asarray(data["spot_radius"]).mean())

    rng = np.random.default_rng(seed)
    r = cfg.R * np.sqrt(rng.random(n_samples))
    th = 2.0 * np.pi * rng.random(n_samples)
    zs = -cfg.H / 2.0 + cfg.H * rng.random(n_samples)
    pts = np.stack([r * np.cos(th), r * np.sin(th), zs], axis=1)

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    z = np.asarray(data["z"])

    min_d = np.full(n_samples, np.inf)
    for i in range(len(x) - 1):
        a = np.array([x[i], y[i], z[i]])
        b = np.array([x[i+1], y[i+1], z[i+1]])
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-30:
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        closest = a + np.outer(t, ab)
        d = np.linalg.norm(pts - closest, axis=1)
        np.minimum(min_d, d, out=min_d)

    covered = float((min_d < threshold).mean())
    cell_volume = math.pi * cfg.R ** 2 * cfg.H
    return {
        "utilisation": covered,
        "threshold_m": float(threshold),
        "mean_distance_m": float(min_d.mean()),
        "median_distance_m": float(np.median(min_d)),
        "cell_volume_m3": float(cell_volume),
    }
