"""
optimize.py
===========

Design optimisation for the toroidal multipass cell.
Three optimisers: SciPy Powell, real-coded GA, PSO.

Decision vector: x = [R, H, M_halfLaps, w0, R_ring]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
from scipy.optimize import minimize

from .geometry import TMPCConfig, trace_cell, spots_overlap, min_spot_separation
from .losses import LossModel, per_pass_throughput


@dataclass
class OptResult:
    x: np.ndarray
    cfg: TMPCConfig
    OPL: float
    throughput: float
    utility: float
    method: str
    n_evals: int


DEFAULT_BOUNDS = {
    "R": (30e-3, 100e-3),
    "H": (20e-3, 60e-3),
    "M_halfLaps": (10, 200),
    "w0": (0.3e-3, 2.0e-3),
    "R_ring": (0.95, 0.999),
}


def _decode(x: np.ndarray, N: int, *, ROC: float = math.inf,
            wavelength: float = 1.654e-6, M2: float = 1.2,
            D_mirror: float = 25.4e-3) -> tuple:
    R = float(np.clip(x[0], DEFAULT_BOUNDS["R"][0], DEFAULT_BOUNDS["R"][1]))
    H = float(np.clip(x[1], DEFAULT_BOUNDS["H"][0], DEFAULT_BOUNDS["H"][1]))
    M_half = float(np.clip(x[2], DEFAULT_BOUNDS["M_halfLaps"][0],
                            DEFAULT_BOUNDS["M_halfLaps"][1]))
    w0 = float(np.clip(x[3], DEFAULT_BOUNDS["w0"][0], DEFAULT_BOUNDS["w0"][1]))
    Rref = float(np.clip(x[4], DEFAULT_BOUNDS["R_ring"][0], DEFAULT_BOUNDS["R_ring"][1]))
    M_half_int = int(round(M_half))
    if (N * M_half_int) % 2 != 0:
        M_half_int += 1
    cfg = TMPCConfig(
        N=N, R=float(R), H=float(H), D_mirror=D_mirror,
        ROC=ROC, M_halfLaps=M_half_int,
        wavelength=wavelength, w0=float(w0), M2=M2,
    )
    lm = LossModel(R_ring=float(Rref), R_top=float(Rref))
    return cfg, lm


def _utility(cfg: TMPCConfig, lm: LossModel,
             *, overlap_strict: bool = True) -> tuple:
    data = trace_cell(cfg)
    OPL = float(data["path_length"][-1])
    aperture = cfg.D_mirror / 2.0
    _, cumul = per_pass_throughput(
        data["spot_radius"], aperture,
        R_ring=lm.R_ring, R_top=lm.R_top, top_bounce_index=len(data["pass_no"]) // 2,
    )
    throughput = float(cumul[-1])
    penalty = 1.0
    w_max = data["spot_radius"].max()
    if w_max > 0.45 * cfg.D_mirror:
        penalty *= math.exp(-(w_max / (0.45 * cfg.D_mirror) - 1.0) * 5.0)
    if overlap_strict:
        sep = min_spot_separation(data, cfg.N)
        if sep < 2.5 * w_max:
            penalty *= max(1e-6, sep / (2.5 * w_max))
    utility = OPL * throughput * penalty
    return utility, OPL, throughput


def _neg_objective(x: np.ndarray, N: int,
                   overlap_strict: bool = True) -> float:
    try:
        cfg, lm = _decode(x, N)
        U, _, _ = _utility(cfg, lm, overlap_strict=overlap_strict)
        return -U
    except Exception:
        return 1e6


def optimise_scipy(N: int = 8, x0: np.ndarray = None,
                   method: str = "Powell",
                   maxiter: int = 1000,
                   overlap_strict: bool = True) -> OptResult:
    """Optimise using SciPy (default: Powell method)."""
    if x0 is None:
        x0 = np.array([50e-3, 40e-3, 66, 1.0e-3, 0.995])
    bounds = [(DEFAULT_BOUNDS[k][0], DEFAULT_BOUNDS[k][1])
              for k in ("R", "H", "M_halfLaps", "w0", "R_ring")]
    res = minimize(_neg_objective, x0, args=(N, overlap_strict), method=method,
                   bounds=bounds,
                   options={"maxiter": maxiter, "xtol": 1e-6, "ftol": 1e-6})
    cfg, lm = _decode(res.x, N)
    U, OPL, T = _utility(cfg, lm, overlap_strict=overlap_strict)
    return OptResult(x=res.x, cfg=cfg, OPL=OPL, throughput=T,
                     utility=U, method=f"scipy/{method}", n_evals=res.nfev)


def optimise_ga(N: int = 8, pop_size: int = 40, n_generations: int = 60,
                mutation_sigma: float = 0.10, elite: int = 4,
                seed: int = 42,
                overlap_strict: bool = True) -> OptResult:
    """Optimise using a real-coded genetic algorithm."""
    rng = np.random.default_rng(seed)
    lb = np.array([DEFAULT_BOUNDS[k][0] for k in
                   ("R", "H", "M_halfLaps", "w0", "R_ring")])
    ub = np.array([DEFAULT_BOUNDS[k][1] for k in
                   ("R", "H", "M_halfLaps", "w0", "R_ring")])
    pop = lb + (ub - lb) * rng.random((pop_size, 5))
    pop[0] = np.array([50e-3, 40e-3, 66, 1.0e-3, 0.995])
    pop[1] = np.array([50e-3, 40e-3, 66, 1.5e-3, 0.999])

    def eval_pop(pop):
        return np.array([-_neg_objective(ind, N, overlap_strict=overlap_strict)
                         for ind in pop])

    fitness = eval_pop(pop)
    n_evals = pop_size

    for gen in range(n_generations):
        order = np.argsort(-fitness)
        pop = pop[order]; fitness = fitness[order]
        new_pop = [pop[i].copy() for i in range(elite)]
        while len(new_pop) < pop_size:
            a, b = rng.integers(0, pop_size, size=2)
            p1 = pop[a] if fitness[a] > fitness[b] else pop[b]
            c, d = rng.integers(0, pop_size, size=2)
            p2 = pop[c] if fitness[c] > fitness[d] else pop[d]
            w = rng.random()
            child = w * p1 + (1 - w) * p2
            sigma = mutation_sigma * (ub - lb)
            child += rng.normal(0, sigma)
            child = np.clip(child, lb, ub)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = eval_pop(pop)
        n_evals += pop_size

    best_idx = int(np.argmax(fitness))
    best = pop[best_idx]
    cfg, lm = _decode(best, N)
    U, OPL, T = _utility(cfg, lm, overlap_strict=overlap_strict)
    return OptResult(x=best, cfg=cfg, OPL=OPL, throughput=T,
                     utility=U, method="ga", n_evals=n_evals)


def optimise_pso(N: int = 8, n_particles: int = 30, n_iters: int = 80,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 seed: int = 7,
                 overlap_strict: bool = True) -> OptResult:
    """Optimise using Particle Swarm Optimisation (PSO)."""
    rng = np.random.default_rng(seed)
    lb = np.array([DEFAULT_BOUNDS[k][0] for k in
                   ("R", "H", "M_halfLaps", "w0", "R_ring")])
    ub = np.array([DEFAULT_BOUNDS[k][1] for k in
                   ("R", "H", "M_halfLaps", "w0", "R_ring")])
    x = lb + (ub - lb) * rng.random((n_particles, 5))
    x[0] = np.array([50e-3, 40e-3, 66, 1.0e-3, 0.995])
    x[1] = np.array([50e-3, 40e-3, 66, 1.5e-3, 0.999])
    v = (ub - lb) * (rng.random((n_particles, 5)) - 0.5) * 0.1
    p_best = x.copy()
    p_best_fit = np.array([-_neg_objective(xi, N, overlap_strict=overlap_strict)
                           for xi in x])
    g_best = p_best[np.argmax(p_best_fit)].copy()
    g_best_fit = float(p_best_fit.max())
    n_evals = n_particles
    for _ in range(n_iters):
        r1 = rng.random(x.shape); r2 = rng.random(x.shape)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = np.clip(x + v, lb, ub)
        fit = np.array([-_neg_objective(xi, N, overlap_strict=overlap_strict)
                        for xi in x])
        n_evals += n_particles
        improved = fit > p_best_fit
        p_best[improved] = x[improved]
        p_best_fit[improved] = fit[improved]
        if p_best_fit.max() > g_best_fit:
            g_best_fit = float(p_best_fit.max())
            g_best = p_best[np.argmax(p_best_fit)].copy()
    cfg, lm = _decode(g_best, N)
    U, OPL, T = _utility(cfg, lm, overlap_strict=overlap_strict)
    return OptResult(x=g_best, cfg=cfg, OPL=OPL, throughput=T,
                     utility=U, method="pso", n_evals=n_evals)
