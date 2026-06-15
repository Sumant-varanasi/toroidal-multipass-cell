"""
Toroidal Multipass Cell (TMPC) simulation package.

Modules
-------
geometry : Ring-of-N-mirrors geometry, ray tracing, spot patterns.
gaussian : Gaussian beam propagation, ABCD matrices, stability analysis.
losses : Mirror, clipping, coupling, and total throughput model.
optimize : SciPy + simple genetic-algorithm optimisation.
plotting : Visualisation helpers for spot patterns, beam evolution, sweeps.

Designed for CH4 leak-detection on a drone-mounted platform
(see Student 3 internship reference design v1, Aston / IITD Abu Dhabi, June 2026).
"""

from .geometry import TMPCConfig, trace_cell
from .gaussian import GaussianBeam, propagate_beam, abcd_unit_cell, stability
from .losses import LossModel, throughput_after_n
from .optimize import optimise_scipy, optimise_ga

__all__ = [
    "TMPCConfig",
    "trace_cell",
    "GaussianBeam",
    "propagate_beam",
    "abcd_unit_cell",
    "stability",
    "LossModel",
    "throughput_after_n",
    "optimise_scipy",
    "optimise_ga",
]
