"""
losses.py
=========

Loss model for the toroidal multipass cell.
Includes mirror reflectivity losses, clipping losses, and throughput calculation.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LossModel:
    R_ring: float = 0.97          # reflectivity of ring mirrors
    R_top: float = 0.97           # reflectivity of top retroreflector
    absorption_fraction: float = 0.40   # fraction of loss due to absorption
    scatter_fraction: float = 0.60      # fraction of loss due to scatter
    sigma_theta_rad: float = 0.0        # angular misalignment std dev [rad]
    sigma_d_m: float = 0.0              # lateral misalignment std dev [m]


def mirror_loss_breakdown(model: LossModel, top: bool = False) -> dict:
    """Break down total mirror loss into absorption and scatter components."""
    R = model.R_top if top else model.R_ring
    total = 1.0 - R
    return {
        "reflectivity_loss": total,
        "absorption_loss": model.absorption_fraction * total,
        "scatter_loss": model.scatter_fraction * total,
        "transmission": R,
    }


def clipping_loss(spot_w: float, aperture_radius: float) -> float:
    """
    Power fraction lost to clipping for a Gaussian beam on a circular aperture.
    Returns fraction LOST (0 = no clipping, 1 = all clipped).
    Uses 1 - exp(-2*a^2/w^2) approximation (fraction outside aperture).
    """
    if spot_w <= 0:
        return 0.0
    return math.exp(-2.0 * aperture_radius ** 2 / spot_w ** 2)


def misalignment_coupling_loss(sigma_theta: float, sigma_d: float,
                               w0: float, wavelength: float) -> float:
    """
    Estimate of coupling efficiency loss due to beam misalignment.
    Returns fractional power loss (0 = no loss).
    """
    return (sigma_d / w0) ** 2 + (math.pi * w0 * sigma_theta / wavelength) ** 2


def throughput_after_n(R_per_bounce: float, n: int,
                       clipping_per_bounce: float = 0.0) -> float:
    """
    Simple throughput model: T = (R * (1 - clip))^n
    """
    survive = R_per_bounce * (1.0 - clipping_per_bounce)
    return survive ** n


def per_pass_throughput(spot_radii: np.ndarray, aperture_radius: float,
                        R_ring: float, R_top: float = None,
                        top_bounce_index: int = None
                        ) -> tuple:
    """
    Calculate per-bounce and cumulative throughput arrays.

    Parameters
    ----------
    spot_radii : array of spot 1/e^2 radii at each bounce [m]
    aperture_radius : mirror aperture radius [m]
    R_ring : ring mirror reflectivity
    R_top : top mirror reflectivity (optional)
    top_bounce_index : index of top mirror bounce (optional)

    Returns
    -------
    per_pass : array of per-bounce transmission factors
    cumulative : cumulative product (total throughput after each bounce)
    """
    n = len(spot_radii)
    clip = np.array([1.0 - clipping_loss(w, aperture_radius)
                     for w in spot_radii])
    refl = np.full(n, R_ring)
    if R_top is not None and top_bounce_index is not None:
        refl[top_bounce_index] *= R_top
    per_pass = refl * clip
    cumulative = np.cumprod(per_pass)
    return per_pass, cumulative


def reflectivity_for_target_throughput(n_reflections: int,
                                       target: float) -> float:
    """
    Calculate the required mirror reflectivity to achieve a target throughput
    after n_reflections bounces (assuming no clipping loss).
    """
    return target ** (1.0 / n_reflections)


def snr_estimate(OPL_m: float, throughput: float,
                 laser_power_mW: float = 1.0,
                 detector_NEP_pW: float = 100.0,
                 absorption_coeff_per_m: float = 1e-4) -> dict:
    """
    Rough SNR estimate for absorption spectroscopy measurement.

    Parameters
    ----------
    OPL_m : effective optical path length [m]
    throughput : fraction of input power reaching detector
    laser_power_mW : laser output power [mW]
    detector_NEP_pW : noise-equivalent power [pW/rtHz]
    absorption_coeff_per_m : target gas absorption coefficient [1/m]

    Returns
    -------
    dict with signal, noise, and SNR estimates
    """
    P_in = laser_power_mW * 1e-3  # W
    P_det = P_in * throughput      # W at detector
    
    # Beer-Lambert: dI/I = alpha * L
    delta_P = P_det * absorption_coeff_per_m * OPL_m  # signal power
    
    noise = detector_NEP_pW * 1e-12  # W/rtHz
    snr = delta_P / noise if noise > 0 else float('inf')
    
    return {
        "P_input_W": P_in,
        "P_detector_W": P_det,
        "signal_W": delta_P,
        "noise_W_per_rtHz": noise,
        "SNR_per_rtHz": snr,
        "min_detectable_alpha": noise / (P_det * OPL_m) if P_det * OPL_m > 0 else float('inf'),
    }
