# 2D Coupled THMC Two-Phase Flow Model for Tectonic Shear Pumping and Hard-Rock Lithium Mineralization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete Julia simulation codes, parameter-sweep driver, and Python visualization pipeline used to generate **Figure 3** and the coupled two-phase flow results in the manuscript:

> **No Deformation, No Lithium Pegmatites: Tectonic Shear Pumping Governs Global Hard-Rock Lithium Mineralization** *(update final title, authors, and journal reference upon acceptance)*

The model solves 2D coupled thermomechanical–hydrodynamic–chemical (THMC) two-phase compaction, dextral simple shear, stress-shadowed oblique ($\sim 67^\circ$) hydrofracture diking, enthalpy-conserving crystallization, and volatile-driven ($\text{H}_2\text{O}+\text{B}+\text{F}$) melt viscosity collapse in a cooling peraluminous granitic crystal mush ($2 \times 2\text{ km}$).

---

## Repository Structure

```text
.
├── solver_2d_thmc_mush.jl          # Main 2D THMC solver: 4 ablation benchmarks (96x96; Fig. 3a,b)
├── run_fig3_cd_sweeps.jl           # Lock-free task-parallel sweep driver: 145 2D runs (48x48; Fig. 3c,d)
├── Plot_fig3_layout_skeleton.py    # Publication-ready 183 x 132 mm vector PDF & 600 dpi PNG renderer
├── output_2d/                      # Simulation outputs (.npz snapshots, time series, sweep grids & CSV)
│   ├── snapshots_peak_*.npz        # Peak Li-export 2D snapshots (used in Fig. 3a)
│   ├── snapshots_active_*.npz      # Synchronous t/tau_cool = 0.45 2D snapshots
│   ├── snapshots_*.npz             # Final state (t/tau_cool = 1.20) 2D snapshots
│   ├── time_series.npz             # Time evolution of trap Li, channel Li, export & melt viscosity (Fig. 3b)
│   ├── cross_regime_collapse.csv   # 24 cross-regime 2D runs across EXT / STRIKE_SLIP / COMP_TRANS (Fig. 3c)
│   └── phase_diagram_grid.npz      # 11x11 (121 runs) global dynamical phase diagram & Pi_p = 1 line (Fig. 3d)
└── README.md
```

---

## System Requirements

### Software Dependencies
* **Julia** ($\ge 1.10$ recommended; tested on Windows 11 and Linux x86\_64)
  * External packages: `ParallelStencil`, `NPZ`
  * Standard library modules: `Printf`, `Random`, `Statistics`
* **Python** ($\ge 3.9$ recommended)
  * Packages: `numpy`, `matplotlib`, `scipy`

### Hardware Recommendations
All simulations run on standard multi-core desktop CPUs without requiring a GPU:
* **4 physical CPU cores** recommended for `solver_2d_thmc_mush.jl` (`96×96` grid; avoids small-grid thread-barrier overhead).
* **6–8 physical CPU cores** recommended for `run_fig3_cd_sweeps.jl` (true lock-free task-level parallelism across independent 2D runs).
* **RAM:** $< 2\text{ GB}$ peak memory usage.

---

## Installation

### 1. Install Julia Packages
Open a terminal (PowerShell on Windows or Bash on Linux/macOS) and run:

```bash
julia -e 'using Pkg; Pkg.add(["ParallelStencil", "NPZ"])'
```

### 2. Install Python Packages
```bash
pip install numpy matplotlib scipy
```

---

## Reproducing Figure 3 Step by Step

### Step 1 — Run the Four End-Member Ablation Benchmarks (Fig. 3a, b)
Executes the four $96 \times 96$ benchmark cases (`static`, `filter`, `shear_const`, and `coupled`) using a fused 3-kernel Pseudo-Maxwell pseudo-transient (PT) Stokes–Darcy solver with Jacobi-PCG elliptic $P_f$ polishing and second-order TVD (van Leer) conservative advection:

```bash
julia -t 4 solver_2d_thmc_mush.jl
```

* **Runtime:** ~25 min total on 4 CPU threads (`static`: ~50 s, `filter`: ~31 s, `shear_const`: ~7.8 min, `coupled`: ~16.2 min).
* **Outputs:** Writes `snapshots_{peak,active,final}_*.npz` and `time_series.npz` into `./output_2d/`.

### Step 2 — Run the 145-Simulation Parameter Sweeps (Fig. 3c, d)
Executes the $11 \times 11$ global dynamical phase diagram ($n = 121$ runs spanning $\dot{\varepsilon} = 10^{-16}\text{–}10^{-11}\text{ s}^{-1}$ and $\eta_{m0} = 10^4\text{–}10^8\text{ Pa s}$) followed by the cross-regime kinematic collapse suite ($n = 24$ runs across extensional, strike-slip, and compressional–transpressional settings) on a $48 \times 48$ grid. Existing Panel **a/b** outputs in `./output_2d/` are left untouched:

```bash
julia -t 8 run_fig3_cd_sweeps.jl
```

* **Runtime:** ~5.5 min total on 8 CPU threads (Panel **d**: ~4.5 min; Panel **c**: ~1.0 min).
* **Outputs:** Writes `output_2d/phase_diagram_grid.npz` and `output_2d/cross_regime_collapse.csv`.

### Step 3 — Render Publication Figure 3 (Vector PDF & 600 dpi PNG)
Loads all real simulation datasets from `./output_2d/`, performs the robust soft-$L_1$ log-logistic master regression for Panel **c**, and renders the 183 mm $\times$ 132 mm Nature full-width layout:

```bash
python Plot_fig3_layout_skeleton.py
```

* **Runtime:** $< 10\text{ s}$.
* **Outputs:**
  * `Fig3_Numerical_Model_Skeleton_peak.pdf` / `.png` — **Manuscript Figure 3** (peak Li-export snapshots in Panel **a**).
  * `Fig3_Numerical_Model_Skeleton_active.pdf` / `.png` — Alternative synchronous ($t/\tau_{\text{cool}} = 0.45$) snapshot view.

> **Fast figure-only preview:** If you downloaded the precomputed `output_2d/` archive from Zenodo, you can skip Steps 1–2 and run Step 3 directly.

---

## Expected Benchmark Verification Results

Running `solver_2d_thmc_mush.jl` with default `ModelParams(nx = 96, nz = 96)` and random seed `42` reproduces the following reference metrics:

| Benchmark Case | Switches (`shear` / `squeeze` / `dilatancy` / `visc_fb`) | Final BDT Export (m) | Peak BDT Trap Li (ppm) | Max Channel/Vein Li (ppm) | Min Channel $\eta_m$ (Pa s) | Physical Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `(i) static` | `F / F / F / F` | `0.09` | `161.0` | `221` | `6.00e+06` | Locked |
| `(ii) filter` | `F / T / F / F` | `2.70` | `208.7` | `269` | `6.00e+06` | Self-sealed |
| `(iii) shear_const` | `T / F / T / F` | `0.99` | `243.2` | `761` | `6.00e+06` | Sub-critical |
| `(iv) coupled` | `T / F / T / T` | `19.99` | `6021.3` | `7732` | `1.00e+03` | Runaway LCT |

---

## Output Data Dictionary

### 1. 2D Snapshot Files (`output_2d/snapshots_*.npz`)
All 2D arrays are stored in `(nz, nx)` layout (row-major depth index from base $z=0$ to roof $z=L_z$):
* `x`, `z`: 1D cell-centre coordinates in metres (`[0, 2000]`).
* `phi`: Melt volume fraction (porosity).
* `C_Li`, `Cf`: Melt Li concentration (ppm) and normalized volatile flux concentration ($\text{H}_2\text{O}+\text{B}+\text{F}$).
* `eta_m`: Local melt viscosity ($\text{Pa s}$).
* `hf`: Anchored oblique hydrofracture dike aperture state (`[0, 1]`).
* `T`: Temperature ($^\circ\text{C}$).
* `Pc_MPa`, `Pf_MPa`: Compaction pressure $P_t - P_f$ and excess fluid pressure $P_f$ (MPa).
* `qDx`, `qDz`, `Vsx`, `Vsz`: Cell-centred Darcy melt flux and solid matrix velocity ($\text{m s}^{-1}$).
* `t_norm`, `export_m`, `trap_li`: Dimensionless time $t/\tau_{\text{cool}}$, cumulative exported melt sheet thickness (m), and shallow trap Li (ppm) at snapshot capture.

### 2. Time-Series File (`output_2d/time_series.npz`)
For each case `k` $\in$ `{static, filter, shear_const, coupled}`:
* `t_norm_<k>`: Dimensionless time array $t/\tau_{\text{cool}}$.
* `trap_li_<k>`: Shallow palaeo-BDT pegmatite trap Li concentration (ppm).
* `peak_li_<k>`: Maximum of active channel/vein Li and trap Li (ppm).
* `min_eta_<k>`: Effective channel/vein melt viscosity (geometric mean of the 15 lowest-viscosity mobile cells; $\text{Pa s}$).
* `bdt_vol_<k>`: Cumulative exported melt sheet thickness (m).

### 3. Parameter-Sweep Files (`output_2d/phase_diagram_grid.npz` & `cross_regime_collapse.csv`)
* `phase_diagram_grid.npz`: Contains 2D meshgrids `EPS`, `ETA`, peak trap Li `Li` (ppm), `chan_li_grid`, `export_grid`, `pi_p_grid`, discrete run coordinates `run_eps`/`run_eta`, and the analytical criticality boundary `eps_line`/`eta_crit_line` ($\Pi_p = 1$).
* `cross_regime_collapse.csv`: Columns `regime, dim, Pi_p, enrichment, eps_bg, eta_m0, export_m, trap_li, chan_li` for all 24 cross-regime runs.

---

## Resolution & Customization Notes

* **Higher-resolution mesh verification:** To test mesh convergence on the benchmark suite, edit `params = ModelParams(nx = 160, nz = 160)` in `main()` of `solver_2d_thmc_mush.jl` (increase `max_steps = 4500` accordingly).
* **Custom sweep density:** Edit `length = 11` or `nx = 48, nz = 48` inside `run_panel_d_phase_diagram()` and `run_panel_c_cross_regime()` in `run_fig3_cd_sweeps.jl`.

---

## Citation & License

This codebase is released under the **MIT License** (see `LICENSE`). If you use or adapt these solvers or datasets, please cite both the accompanying manuscript and the archived Zenodo release:

```bibtex
@software{liu_2026_thmc_lithium,
  author       = {Liu, Xiaochi and co-authors},
  title        = {{2D Coupled THMC Two-Phase Flow Model for Tectonic Shear Pumping and Hard-Rock Lithium Mineralization}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {[https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)}
}
```

### Contact
**Xiaochi Liu** — [https://liu-xiaochi.github.io/](https://liu-xiaochi.github.io/)