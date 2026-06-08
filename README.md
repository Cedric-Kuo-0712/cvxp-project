# Convex Optimization for 3D IC Placement

Final project for **CVXP** (NTU EE). This repository implements a 3D IC placement engine based on [ICCAD 2023 CAD Contest Problem B](https://drive.google.com/file/d/1PJOSECe0sCDGzJoQrQWGIzTnIyUVOr65/view).

**Author:** Yu-Chia Kuo (B11901047), Department of Electrical Engineering, NTU

## Overview

The placer partitions cells across top and bottom dies, runs analytical global placement, legalizes cells onto rows, and places cross-die terminals. The flow combines:

- Spectral partitioning (min-cut convex relaxation)
- Nesterov-accelerated global placement with electrostatic density penalty
- Quadratic or Log-Sum-Exp (LSE) wirelength models
- FFT or DCT Poisson solvers for density spreading
- Row legalization with optional ADMM refinement

Additional scripts explore LP/QP/SOCP formulations and SDP-based partitioning using CVXPY.

## Requirements

- Python 3
- NumPy, SciPy, Matplotlib
- CVXPY (for analysis scripts such as `solve_socp.py`, `solve_sdp_partition.py`)

## Usage

Run the main placement flow:

```bash
python main.py <input_file> <output_file>
```

Example:

```bash
python main.py testcase_official/ProblemB_case1_0522.txt output_case1.txt
```

### Options

| Flag | Description |
|------|-------------|
| `--density-solver {fft,dct}` | Poisson solver for density penalty (default: `fft`) |
| `--wirelength-model {quadratic,lse}` | Wirelength model (default: `quadratic`) |
| `--gamma FLOAT` | LSE smoothing parameter (default: `0.5`) |
| `--use-admm` | Enable ADMM-based legalization refinement |
| `--target-top-ratio FLOAT` | Target top-die instance/area ratio |
| `--check-kkt` | Print KKT checks for internal QPs |

## Project Structure

| File | Role |
|------|------|
| `main.py` | Main placement pipeline |
| `parser.py` | Input/output file parser |
| `partitioner.py` | Spectral die partitioning |
| `nesterov_placer.py` | Global placement (NAG + density) |
| `density_solver.py` | FFT/DCT electrostatic density solver |
| `qp_solver.py` | QP wirelength solver utilities |
| `legalizer.py` | Row legalization and terminal placement |
| `evaluator.py` | Evaluate placement quality (HPWL + terminal cost) |
| `visualize_placement.py` | Render placement diagrams |
| `solve_socp.py` | SOCP formulation experiments (CVXPY) |
| `solve_sdp_partition.py` | SDP partitioning relaxation (CVXPY) |
| `testcase_official/` | Official contest test cases |
