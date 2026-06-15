"""
gaussian.py
===========

Gaussian beam optics and ABCD matrix analysis for the toroidal multipass cell.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class GaussianBeam:
    q: complex
    wavelength: float
    M2: float = 1.0

    @classmethod
    def from_waist(cls, w0: float, wavelength: float,
                   distance_from_waist: float = 0.0,
                   M2: float = 1.0) -> "GaussianBeam":
        zR = math.pi * w0 ** 2 / (M2 * wavelength)
        q = complex(distance_from_waist, zR)
        return cls(q=q, wavelength=wavelength, M2=M2)

    @property
    def w(self) -> float:
        inv_q = 1.0 / self.q
        return math.sqrt(-self.M2 * self.wavelength / (math.pi * inv_q.imag))

    @property
    def R(self) -> float:
        inv_q = 1.0 / self.q
        return math.inf if abs(inv_q.real) < 1e-30 else 1.0 / inv_q.real

    @property
    def rayleigh(self) -> float:
        return self.q.imag

    @property
    def divergence_halfangle(self) -> float:
        w0 = math.sqrt(self.q.imag * self.M2 * self.wavelength / math.pi)
        return self.M2 * self.wavelength / (math.pi * w0)


def free_space(d: float) -> np.ndarray:
    return np.array([[1.0, d], [0.0, 1.0]])


def thin_lens(f: float) -> np.ndarray:
    if math.isinf(f):
        return np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def mirror_focal_lengths(ROC: float, theta_i_rad: float) -> tuple:
    """Return (f_tangential, f_sagittal) for a spherical mirror at AOI theta_i_rad."""
    if math.isinf(ROC):
        return math.inf, math.inf
    f_tan = ROC * math.cos(theta_i_rad) / 2.0
    f_sag = ROC / (2.0 * math.cos(theta_i_rad))
    return f_tan, f_sag


def apply_abcd(beam: GaussianBeam, M: np.ndarray) -> GaussianBeam:
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]
    q_new = (A * beam.q + B) / (C * beam.q + D)
    return GaussianBeam(q=q_new, wavelength=beam.wavelength, M2=beam.M2)


def propagate_beam(beam: GaussianBeam, distance: float) -> GaussianBeam:
    return apply_abcd(beam, free_space(distance))


def abcd_unit_cell(L_leg: float, ROC: float,
                   theta_i_rad: float, plane: str = "tangential"
                   ) -> np.ndarray:
    """ABCD matrix for one mirror + free-space leg."""
    f_tan, f_sag = mirror_focal_lengths(ROC, theta_i_rad)
    f = f_tan if plane == "tangential" else f_sag
    return thin_lens(f) @ free_space(L_leg)


def stability(unit_cell: np.ndarray) -> float:
    """Return g = (A+D)/2 for stability check: |g| <= 1 for stable."""
    return float((unit_cell[0, 0] + unit_cell[1, 1]) / 2.0)


def round_trip_evolution(L_leg: float, ROC: float, theta_i_rad: float,
                         n_reflections: int, beam_in: GaussianBeam,
                         plane: str = "tangential"
                         ) -> tuple:
    """Evolve beam through n_reflections bounces, return (w_array, R_array)."""
    unit = abcd_unit_cell(L_leg, ROC, theta_i_rad, plane=plane)
    w = np.empty(n_reflections + 1)
    R = np.empty(n_reflections + 1)
    b = beam_in
    w[0], R[0] = b.w, b.R
    for i in range(n_reflections):
        b = apply_abcd(b, unit)
        w[i + 1], R[i + 1] = b.w, b.R
    return w, R


def roc_sweep(L_leg: float, theta_i_rad: float,
              roc_values, n_reflections: int,
              w0: float, wavelength: float, M2: float = 1.0) -> list:
    """Sweep over ROC values and return stability/beam-size info."""
    results = []
    for ROC in roc_values:
        unit_tan = abcd_unit_cell(L_leg, ROC, theta_i_rad, "tangential")
        unit_sag = abcd_unit_cell(L_leg, ROC, theta_i_rad, "sagittal")
        g_tan = stability(unit_tan)
        g_sag = stability(unit_sag)
        stable_tan = abs(g_tan) <= 1.0
        stable_sag = abs(g_sag) <= 1.0
        beam_in = GaussianBeam.from_waist(w0, wavelength, M2=M2)
        w_tan, _ = round_trip_evolution(L_leg, ROC, theta_i_rad, n_reflections,
                                        beam_in, plane="tangential")
        w_sag, _ = round_trip_evolution(L_leg, ROC, theta_i_rad, n_reflections,
                                        beam_in, plane="sagittal")
        results.append({
            "ROC": ROC,
            "stable_tan": stable_tan,
            "stable_sag": stable_sag,
            "g_tan": g_tan,
            "g_sag": g_sag,
            "max_w_tan": float(w_tan.max()),
            "max_w_sag": float(w_sag.max()),
            "out_w_tan": float(w_tan[-1]),
            "out_w_sag": float(w_sag[-1]),
        })
    return results
