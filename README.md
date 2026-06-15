# Toroidal Multipass Gas Cell (TMPC)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade simulation, optimisation, and ML platform for **Toroidal Multipass Gas Cells (TMPC)** designed for drone-mounted methane (CH₄) leak detection.

**Internship Project** — Student 3 | Aston University / IIT Delhi Abu Dhabi | June 2026

---

## Project Overview

This project models a compact toroidal multipass cell where a laser beam spirals around a ring of N mirrors, achieving long effective optical path lengths (30+ m) inside a 100 mm × 40 mm puck-shaped cell. The target application is drone-mounted CH₄ detection at 1.654 μm.

### Key Results

| Parameter | N=8 baseline | N=9 final (chord_skip=4) | Change |
|-----------|-------------|--------------------------|--------|
| Optical path length | 20.2 m | 33.7 m | +67% |
| Total bounces | 528 | 342 | −35% |
| Chord length | 38.3 mm | 98.5 mm | +157% |
| Angle of incidence | 67.5° | 10.0° | −85% |
| Throughput (R=99.9%) | 59% | 71% | +12% |
| Volume utilisation | 23% | 70% | +3× |

**Recommended mirror**: Concave, ROC = 1.0 m (e.g. Thorlabs CM254-1000-P01)

---

## Repository Structure

```
toroidal-multipass-cell/
├── tmpc/                       # Core simulation package
│   ├── __init__.py
│   ├── geometry.py             # Ring geometry, ray tracing, chord-skip pattern
│   ├── gaussian.py             # Gaussian beam optics, ABCD matrices
│   ├── losses.py               # Mirror loss, clipping, throughput model
│   ├── optimize.py             # Powell, GA, PSO optimisers
│   └── plotting.py             # Matplotlib visualisation helpers
├── analysis/                   # Analysis scripts (run in order)
│   ├── 01_baseline.py          # N=8 baseline characterisation
│   ├── 02_chord_skip_sweep.py  # Chord-skip pattern sweep
│   ├── 03_N_sweep.py           # Mirror count sweep
│   ├── 04_roc_analysis.py      # ROC / concave mirror ABCD analysis
│   ├── 05_optimization.py      # Multi-algorithm optimisation
│   ├── 06_volume_util.py       # Volume utilisation Monte Carlo
│   └── 07_chord_skip_comparison.py  # Baseline vs final comparison
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas

# Run baseline analysis
python analysis/01_baseline.py

# Run full analysis pipeline
for i in 01 02 03 04 05 06 07; do python analysis/${i}_*.py; done
```

---

## Physics Background

The **chord_skip** parameter controls which mirror the beam visits at each bounce:
- `chord_skip=1`: beam hops to adjacent mirror — short chords, 67.5° AOI, poor volume coverage
- `chord_skip=4` (N=9): beam crosses nearly full diameter — 98.5 mm chord, 10° AOI, star pattern

The star pattern means every chord passes through the cell centre, giving 70% volume utilisation vs 23% for the adjacent-hop design.

ABCD matrix analysis shows flat mirrors fail for this geometry (beam grows to 18 mm, exceeds mirror aperture). Concave mirrors with ROC = 1.0 m keep the beam at ≤ 1.03 mm diameter throughout all 342 bounces.

---

## Dependencies

```
numpy
scipy
matplotlib
pandas
```

---

## Reference

Based on:
- Graf, M. & Tuzson, B. et al. (2017). *Compact multipass cell for laser absorption spectroscopy*
- Chang, Y. et al. (2020). *Toroidal multipass absorption cell for trace gas detection*

---

## Author

Harthik (Student 3) — Internship at Aston University / IIT Delhi Abu Dhabi, June 2026
