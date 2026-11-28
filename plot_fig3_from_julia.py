# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:38:20 2026

@author: liuxi
"""

# -*- coding: utf-8 -*-
"""
Fig. 3 Publication Figure Renderer (183 mm x 132 mm Nature Full-Width)
Asymmetric "Top 4, Bottom 3" GridSpec with locked right-edge LogNorm colorbar rail:
  - Panel a (i-iv): 2D two-phase simulation snapshots (pure Li field + phi contours + log-scaled Darcy quiver)
  - Panel b       : Dual-stacked time evolution (Peak extractable Li & effective channel melt viscosity eta_m)
  - Panel c       : Cross-regime kinematic collapse (real 2D runs from cross_regime_collapse.csv + fitted master curve)
  - Panel d       : Global dynamical phase diagram (real 2D sweep from phase_diagram_grid.npz + analytical Pi_p = 1)

Outputs (generated automatically for each mode in SNAP_MODES):
  - Fig3_Numerical_Model_Skeleton_{peak,active}.pdf  (183 x 132 mm vector PDF, editable in Illustrator)
  - Fig3_Numerical_Model_Skeleton_{peak,active}.png  (600 dpi raster preview)
"""

import csv
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# 1. EXACT MILLIMETER LAYOUT GEOMETRY (183 mm x 132 mm)
# =============================================================================
OUTPUT_DIR = Path(".")
OUTPUT_PREFIX = "Fig3_Numerical_Model_Skeleton"
USE_DEMO_DATA = False             # False: 读取 ./output_2d/ 真实模拟输出；True: 合成演示数据
DATA_DIR = Path("output_2d")      # Julia 输出目录（同时含 a/b 快照与 c/d 扫参数据）
SNAP_MODES = ("peak", "active")   # 优先输出 peak（推荐定稿）与 active 两套成图
TIME_LI_KEY = "trap_li"           # "trap_li" (浅部 BDT 捕集可采锂) | "peak_li" (晶粥/捕集峰值锂)

# Panel a 共享对数达西流矢跨度（覆盖 4 个量级黏度坍缩，保证四图采用严格统一的物理标尺）
Q_DYNAMIC_DECADES = 4.0

FIG_W_MM = 183.0
FIG_H_MM = 132.0

# 水平毫米锚点（严格锁定 3a(iv) 与 3d 右边框 = 167.0 mm，双色标轨道 = 169.2–172.0 mm）
LEFT_MM = 12.0
MAIN_RIGHT_MM = 167.0             # 主绘图区净宽 155.0 mm
CBAR_LEFT_MM = 169.2
CBAR_RIGHT_MM = 172.0             # 色标条宽 2.8 mm，右侧留 11.0 mm 给刻度与标题

# 顶行 (Panel a)：4 幅 36.5 x 36.5 mm 精确正方形，间距 3.0 mm (4*36.5 + 3*3.0 = 155.0 mm)
TOP_BOT_MM = 88.0
TOP_TOP_MM = 124.5                # 高度 36.5 mm，顶部留 7.5 mm 给标题与角标 a

# 底行 (Panels b, c, d)：高度 58.0 mm，行间留白 18.5 mm（容纳 3a 横轴、4 组公共图例及 b/c/d 标题）
BOT_BOT_MM = 11.5
BOT_TOP_MM = 69.5

# 底行三面板水平分配（总宽 155.0 mm，面板间距 12.0 mm 确保纵轴标题零拥挤）
B_LEFT_MM, B_RIGHT_MM = 12.0, 60.5       # 3b 宽 48.5 mm
C_LEFT_MM, C_RIGHT_MM = 72.5, 114.0      # 3c 宽 41.5 mm
D_LEFT_MM, D_RIGHT_MM = 126.0, 167.0     # 3d 宽 41.0 mm（右缘与 3a(iv) 严丝合缝对齐）


def fx(mm: float) -> float:
    return mm / FIG_W_MM


def fy(mm: float) -> float:
    return mm / FIG_H_MM


# =============================================================================
# 2. NATURE COLOR PALETTE & PERCEPTUAL COLORMAP (与 Fig. 1 / 2 / 4 严格同源)
# =============================================================================
LI_CMAP = LinearSegmentedColormap.from_list(
    "nature_lithium_muted",
    [
        (0.00, "#EAEFF3"),   # 10^2 ppm: 低 Li 锁死晶粥（冷灰蓝）
        (0.32, "#C9D6DF"),   # 过渡弱富集
        (0.58, "#E8C58E"),   # 中等富集（暖米黄）
        (0.82, "#B84A39"),   # 逼近锂辉石饱和（暖陶土红）
        (1.00, "#6E1811"),   # >10^4 ppm: 极端富锂通道（暗绯红）
    ],
)
LI_NORM = LogNorm(vmin=1.0e2, vmax=2.0e4)

CASE_META = {
    "static": {
        "title": "(i) Static buoyancy",
        "tag": "Locked",
        "legend": "(i) Static buoyancy",
        "color": "#7D848A",
        "ls": ":",
        "lw": 1.15,
    },
    "filter": {
        "title": "(ii) Filter pressing",
        "tag": "Self-sealed",
        "legend": "(ii) Filter pressing",
        "color": "#5A768F",
        "ls": "--",
        "lw": 1.20,
    },
    "shear_const": {
        "title": "(iii) Shear (fixed η)",
        "tag": "Sub-critical",
        "legend": "(iii) Shear (fixed η)",
        "color": "#D49A38",
        "ls": "-.",
        "lw": 1.25,
    },
    "coupled": {
        "title": "(iv) Full coupled",
        "tag": "Runaway",
        "legend": "(iv) Full coupled pump",
        "color": "#B84A39",
        "ls": "-",
        "lw": 1.65,
    },
}

REGIME_STYLES = {
    "EXT":         {"label": "Extensional",  "color": "#D49A38", "marker": "^"},
    "COMP_TRANS":  {"label": "Comp.–Trans.", "color": "#B84A39", "marker": "o"},
    "STRIKE_SLIP": {"label": "Strike-slip",  "color": "#386890", "marker": "s"},
}

CRIMSON_CRIT = "#7A2317"


# =============================================================================
# 3. DATA INTERFACE (真实 Julia 输出加载 + Panel c 自动主曲线拟合)
# =============================================================================
def generate_synthetic_demo_data() -> dict:
    """Generate physically consistent synthetic arrays for fallback/preview."""
    rng = np.random.default_rng(42)

    nx, nz = 128, 128
    x = np.linspace(0.0, 2.0, nx)
    z = np.linspace(0.0, 2.0, nz)
    X, Z = np.meshgrid(x, z)
    noise = rng.normal(0.0, 1.0, size=(nz, nx))

    theta = np.radians(22.0)
    s_coord = X * np.sin(theta) + Z * np.cos(theta)
    bands = (
        np.exp(-((s_coord - 0.52) / 0.085) ** 2)
        + np.exp(-((s_coord - 1.18) / 0.095) ** 2)
        + 0.75 * np.exp(-((s_coord - 1.78) / 0.08) ** 2)
    )
    upward_grad = np.clip(1.15 - 0.45 * Z, 0.25, 1.15)

    snapshots = {}
    phi_st = np.clip(0.14 + 0.03 * np.exp(-((Z - 0.6) / 0.4) ** 2) + 0.005 * noise, 0.05, 0.25)
    li_st = np.clip(145.0 * (1.0 + 0.35 * np.exp(-Z / 0.8)) * (1.0 + 0.04 * noise), 100.0, 350.0)
    snapshots["static"] = {"X": X, "Z": Z, "Li": li_st, "phi": phi_st, "vx": np.zeros_like(X), "vz": -0.02 * np.ones_like(X), "q_norm_mode": "linear"}

    seal = np.exp(-((Z - 0.45) / 0.16) ** 2)
    phi_fp = np.clip(0.16 * (1.0 - 0.75 * seal) + 0.06 * np.exp(-((Z - 1.1) / 0.35) ** 2), 0.02, 0.26)
    li_fp = np.clip(160.0 + 380.0 * np.exp(-((Z - 0.85) / 0.32) ** 2) * (1.0 + 0.05 * noise), 110.0, 950.0)
    snapshots["filter"] = {"X": X, "Z": Z, "Li": li_fp, "phi": phi_fp, "vx": np.zeros_like(X), "vz": -0.12 * (1.0 - seal), "q_norm_mode": "linear"}

    phi_sc = np.clip(0.15 + 0.14 * bands * upward_grad + 0.006 * noise, 0.06, 0.34)
    li_sc = np.clip(180.0 * np.exp(1.85 * bands * upward_grad) * (1.0 + 0.05 * noise), 120.0, 2200.0)
    vx_sc = 0.35 * bands * np.cos(theta)
    vz_sc = -0.35 * bands * np.sin(theta) - 0.10 * bands
    snapshots["shear_const"] = {"X": X, "Z": Z, "Li": li_sc, "phi": phi_sc, "vx": vx_sc, "vz": vz_sc, "q_norm_mode": "linear"}

    sharp_bands = np.clip(bands ** 1.35, 0.0, 1.3)
    phi_cp = np.clip(0.12 + 0.36 * sharp_bands * upward_grad + 0.008 * noise, 0.05, 0.56)
    li_cp = np.clip(210.0 * np.exp(4.25 * sharp_bands * upward_grad) * (1.0 + 0.06 * noise), 120.0, 1.95e4)
    vx_cp = 1.15 * sharp_bands * np.cos(theta)
    vz_cp = -1.25 * sharp_bands * np.sin(theta) - 0.55 * sharp_bands
    snapshots["coupled"] = {"X": X, "Z": Z, "Li": li_cp, "phi": phi_cp, "vx": vx_cp, "vz": vz_cp, "q_norm_mode": "linear"}

    t = np.linspace(0.0, 1.30, 160)
    li_t = {
        "static":      110.0 + 75.0 * (1.0 - np.exp(-2.2 * t)),
        "filter":      110.0 + 440.0 * (1.0 - np.exp(-3.5 * t)) * np.exp(-0.25 * np.maximum(t - 0.35, 0.0)),
        "shear_const": 110.0 * np.exp(2.35 * (1.0 - np.exp(-2.0 * t))),
        "coupled":     110.0 * np.exp(4.78 / (1.0 + np.exp(-7.5 * (t - 0.56)))),
    }
    eta_t = {
        "static":      1.0e6 * np.exp(2.2 * np.minimum(t, 1.15)),
        "filter":      1.0e6 * np.exp(1.8 * np.minimum(t, 1.15)),
        "shear_const": 1.0e6 * np.ones_like(t),
        "coupled":     1.0e6 * (1.0e-3 + 0.999 / (1.0 + np.exp(8.5 * (t - 0.54)))),
    }

    pi_curve = np.logspace(-2.0, 2.0, 200)
    enrich_mean = 1.5 + 115.0 / (1.0 + pi_curve ** (-2.35))
    enrich_lo = enrich_mean * 0.72
    enrich_hi = enrich_mean * 1.38

    collapse_pts = []
    for reg in ("EXT", "COMP_TRANS", "STRIKE_SLIP"):
        pis_2d = 10.0 ** rng.uniform(-1.85, 1.85, size=8)
        for p in pis_2d:
            val = (1.5 + 115.0 / (1.0 + p ** (-2.35))) * 10.0 ** rng.normal(0.0, 0.085)
            collapse_pts.append({"regime": reg, "dim": "2D", "Pi_p": p, "enrichment": np.clip(val, 1.1, 190.0)})
        for p3 in (10.0 ** rng.uniform(-0.8, -0.3), 10.0 ** rng.uniform(0.45, 1.25)):
            val3 = (1.5 + 115.0 / (1.0 + p3 ** (-2.35))) * 10.0 ** rng.normal(0.0, 0.06)
            collapse_pts.append({"regime": reg, "dim": "3D", "Pi_p": p3, "enrichment": np.clip(val3, 1.1, 190.0)})

    eps_grid = np.logspace(-16.0, -11.0, 65)
    eta_grid = np.logspace(4.0, 8.0, 65)
    EPS, ETA = np.meshgrid(eps_grid, eta_grid)

    dp_eff = 20.0 * (EPS / 1.0e-13) / (1.0 + (EPS / 2.5e-12) ** 1.1)
    pi_grid = dp_eff / (20.0 * (ETA / 1.0e6))
    peak_li_grid = np.clip(130.0 * (1.0 + 135.0 / (1.0 + pi_grid ** (-2.5))), 110.0, 1.95e4)

    run_eps, run_eta = np.meshgrid(np.logspace(-15.7, -11.3, 9), np.logspace(4.3, 7.7, 9))
    eps_line = np.logspace(-16.0, -11.0, 200)
    eta_crit_line = 1.0e6 * (eps_line / 1.0e-13) / (1.0 + (eps_line / 2.5e-12) ** 1.1)

    return {
        "snapshots": snapshots,
        "time": {"t": t, "Li": li_t, "eta": eta_t},
        "collapse": {
            "pi_curve": pi_curve,
            "mean": enrich_mean,
            "lo": enrich_lo,
            "hi": enrich_hi,
            "points": collapse_pts,
        },
        "phase": {
            "EPS": EPS,
            "ETA": ETA,
            "Li": peak_li_grid,
            "run_eps": run_eps.ravel(),
            "run_eta": run_eta.ravel(),
            "eps_line": eps_line,
            "eta_crit_line": eta_crit_line,
        },
    }


def fit_collapse_master_curve(pts: list) -> dict:
    """
    Fit a physically constrained log-logistic master curve + +/-1-sigma scatter band
    directly to the real cross-regime numerical data in Panel c.
    """
    pi_curve = np.logspace(-2.0, 2.0, 240)
    pis = np.array([pt["Pi_p"] for pt in pts], dtype=float)
    enrs = np.array([pt["enrichment"] for pt in pts], dtype=float)
    valid = np.isfinite(pis) & np.isfinite(enrs) & (pis > 0.0) & (enrs > 0.0)
    pis, enrs = pis[valid], enrs[valid]

    if pis.size < 6 or not HAS_SCIPY:
        mean_c = 1.0 + 140.0 / (1.0 + pi_curve ** (-2.2))
        return {"pi_curve": pi_curve, "mean": mean_c, "lo": mean_c * 0.72, "hi": mean_c * 1.38}

    log_y = np.log10(np.maximum(enrs, 1.0))

    def _eval_curve(params, p_arr):
        y_lo, y_hi, p0_log, slope = params
        return y_lo + (y_hi - y_lo) / (1.0 + (p_arr / (10.0 ** p0_log)) ** (-slope))

    y_hi_init = float(np.clip(np.percentile(enrs, 92), 30.0, 220.0))
    res = least_squares(
        lambda prm: np.log10(np.maximum(_eval_curve(prm, pis), 1.0e-3)) - log_y,
        x0=[1.05, y_hi_init, 0.0, 2.2],
        bounds=([0.95, 15.0, -0.65, 0.8], [2.5, 240.0, 0.65, 5.0]),
        loss="soft_l1",
        f_scale=0.18,
    )
    y_lo, y_hi, p0_log, slope = res.x
    mean_c = _eval_curve(res.x, pi_curve)
    # Robust 1-sigma log10 scatter band (MAD-based, clamped to sensible visual envelope)
    resid = np.log10(np.maximum(_eval_curve(res.x, pis), 1.0e-3)) - log_y
    mad_sig = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    sig = float(np.clip(max(mad_sig, 0.10), 0.10, 0.28))
    lo_c = np.maximum(1.0, mean_c * (10.0 ** (-sig)))
    hi_c = mean_c * (10.0 ** (+sig))

    print(
        f"[INFO] Fitted Panel c master curve on {pis.size} real runs: "
        f"y_lo={y_lo:.2f}, y_hi={y_hi:.1f}, Pi_p0={10.0**p0_log:.2f}, slope={slope:.2f}, 1-sigma={sig:.2f} dex"
    )
    return {"pi_curve": pi_curve, "mean": mean_c, "lo": lo_c, "hi": hi_c}


def _resolve_snapshot_path(data_dir: Path, case_key: str, mode: str) -> Path:
    """Resolve snapshot file according to snapshot mode with graceful fallback."""
    priority = {
        "active": [f"snapshots_active_{case_key}.npz", f"snapshots_peak_{case_key}.npz", f"snapshots_{case_key}.npz"],
        "peak":   [f"snapshots_peak_{case_key}.npz", f"snapshots_active_{case_key}.npz", f"snapshots_{case_key}.npz"],
        "final":  [f"snapshots_{case_key}.npz"],
    }
    if mode not in priority:
        raise ValueError(f"Unsupported snap_mode='{mode}'. Choose from 'active', 'peak', or 'final'.")

    for fname in priority[mode]:
        fpath = data_dir / fname
        if fpath.exists():
            return fpath
    raise FileNotFoundError(
        f"Missing snapshot for case '{case_key}' in '{data_dir.resolve()}' (tried: {priority[mode]})."
    )


def load_real_simulation_data(
    data_dir: Path,
    snap_mode: str = "peak",
    time_li_key: str = TIME_LI_KEY,
) -> dict:
    """
    Load real 2D coupled THMC outputs produced by `solver_2d_thmc_mush.jl`
    (Panels a & b) and `run_fig3_cd_sweeps.jl` (Panels c & d).
    """
    case_keys = ["static", "filter", "shear_const", "coupled"]
    demo_fallback = generate_synthetic_demo_data()

    # -------------------------------------------------------------------------
    # 1) Panel a: 4 x 2D snapshots (z: base->top converted to Depth: 0->2 km)
    # -------------------------------------------------------------------------
    raw_snaps = {}
    for ckey in case_keys:
        fpath = _resolve_snapshot_path(data_dir, ckey, snap_mode)
        with np.load(fpath) as d:
            x_km = np.asarray(d["x"], dtype=float) / 1.0e3
            z_up_km = np.asarray(d["z"], dtype=float) / 1.0e3
            lz_km = float(z_up_km[0] + z_up_km[-1])
            depth_km = (lz_km - z_up_km)[::-1]
            X, Z = np.meshgrid(x_km, depth_km, indexing="xy")

            li_arr = np.asarray(d["C_Li"], dtype=float)[::-1, :]
            phi_arr = np.asarray(d["phi"], dtype=float)[::-1, :]
            qdx_arr = np.asarray(d["qDx"], dtype=float)[::-1, :]
            qdz_arr = np.asarray(d["qDz"], dtype=float)[::-1, :]
            t_snap = float(np.asarray(d["t_norm"]).ravel()[0]) if "t_norm" in d else np.nan

        raw_snaps[ckey] = {
            "file": fpath.name,
            "t_norm": t_snap,
            "X": X,
            "Z": Z,
            "Li": np.clip(li_arr, 1.0e2, 2.0e4),
            "phi": phi_arr,
            "qDx": qdx_arr,
            "qDz": qdz_arr,
        }

    # Shared log-compressed Darcy-flux reference anchored to `coupled` 98th percentile
    q_cp_spd = np.hypot(raw_snaps["coupled"]["qDx"], raw_snaps["coupled"]["qDz"])
    q_pos = q_cp_spd[q_cp_spd > 0.0]
    q_ref = float(np.percentile(q_pos, 98)) if q_pos.size > 0 else 1.0
    q_ref = max(q_ref, 1.0e-30)
    q_floor = q_ref * (10.0 ** (-Q_DYNAMIC_DECADES))

    snapshots = {}
    print(f"\n[INFO] Loaded 2D snapshots (snap_mode='{snap_mode}') from '{data_dir.resolve()}':")
    print(f"       Shared log-quiver reference: q_ref={q_ref:.2e} m/s, q_floor={q_floor:.2e} m/s ({Q_DYNAMIC_DECADES:.1f} decades)")
    for ckey in case_keys:
        rs = raw_snaps[ckey]
        q_mag = np.hypot(rs["qDx"], rs["qDz"])
        log_len = np.where(
            q_mag > q_floor,
            np.clip(np.log10(np.maximum(q_mag, q_floor) / q_floor) / Q_DYNAMIC_DECADES, 0.0, 1.15),
            0.0,
        )
        safe_mag = np.maximum(q_mag, 1.0e-30)
        vx_log = (rs["qDx"] / safe_mag) * log_len
        vz_log = (-rs["qDz"] / safe_mag) * log_len

        snapshots[ckey] = {
            "X": rs["X"],
            "Z": rs["Z"],
            "Li": rs["Li"],
            "phi": rs["phi"],
            "vx": vx_log,
            "vz": vz_log,
            "q_norm_mode": "log",
        }
        print(
            f"  - {ckey:11s} <- {rs['file']:30s} | t/tc={rs['t_norm']:.3f} | "
            f"phi=[{rs['phi'].min():.3f}, {rs['phi'].max():.3f}] | "
            f"Li=[{rs['Li'].min():.0f}, {rs['Li'].max():.0f}] ppm | "
            f"|qD|_95={np.percentile(q_mag, 95):.2e} m/s"
        )

    # -------------------------------------------------------------------------
    # 2) Panel b: Per-case adaptive-CFL time series from time_series.npz
    # -------------------------------------------------------------------------
    ts_path = data_dir / "time_series.npz"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing time series file: '{ts_path.resolve()}'.")

    t_dict, li_dict, eta_dict = {}, {}, {}
    with np.load(ts_path) as d:
        for ckey in case_keys:
            t_dict[ckey] = np.asarray(d[f"t_norm_{ckey}"], dtype=float)
            li_key = f"{time_li_key}_{ckey}"
            if li_key not in d:
                raise KeyError(f"Key '{li_key}' not found in '{ts_path.name}'. Available: {list(d.keys())}")
            li_dict[ckey] = np.asarray(d[li_key], dtype=float)
            eta_dict[ckey] = np.asarray(d[f"min_eta_{ckey}"], dtype=float)

    # -------------------------------------------------------------------------
    # 3) Panel c: Cross-regime collapse + direct master-curve regression
    # -------------------------------------------------------------------------
    csv_path = data_dir / "cross_regime_collapse.csv"
    if csv_path.exists():
        pts = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pts.append({
                    "regime": row["regime"].strip(),
                    "dim": row["dim"].strip(),
                    "Pi_p": float(row["Pi_p"]),
                    "enrichment": float(row["enrichment"]),
                })
        fit_dict = fit_collapse_master_curve(pts)
        collapse_data = {
            "pi_curve": fit_dict["pi_curve"],
            "mean": fit_dict["mean"],
            "lo": fit_dict["lo"],
            "hi": fit_dict["hi"],
            "points": pts,
        }
        print(f"[INFO] Loaded Panel c collapse data ({len(pts)} real runs) from '{csv_path.name}'.")
    else:
        collapse_data = demo_fallback["collapse"]
        print(f"[WARN] '{csv_path.name}' not found in '{data_dir}' -> Panel c using skeleton preview data.")

    # -------------------------------------------------------------------------
    # 4) Panel d: 2D phase diagram sweep (real 11x11 or 9x9 grid)
    # -------------------------------------------------------------------------
    pd_path = data_dir / "phase_diagram_grid.npz"
    if pd_path.exists():
        with np.load(pd_path) as d:
            if "eps_grid" in d and "eta_m0_grid" in d:
                eps_1d = np.asarray(d["eps_grid"], dtype=float)
                eta_1d = np.asarray(d["eta_m0_grid"], dtype=float)
                EPS, ETA = np.meshgrid(eps_1d, eta_1d, indexing="xy")
                li_grid = np.clip(np.asarray(d["trap_li_grid"], dtype=float), 1.0e2, 2.0e4)
                phase_data = {
                    "EPS": EPS,
                    "ETA": ETA,
                    "Li": li_grid,
                    "run_eps": EPS.ravel(),
                    "run_eta": ETA.ravel(),
                    "eps_line": demo_fallback["phase"]["eps_line"],
                    "eta_crit_line": demo_fallback["phase"]["eta_crit_line"],
                }
            else:
                phase_data = {
                    "EPS": np.asarray(d["EPS"], dtype=float),
                    "ETA": np.asarray(d["ETA"], dtype=float),
                    "Li": np.clip(np.asarray(d["Li"], dtype=float), 1.0e2, 2.0e4),
                    "run_eps": np.asarray(d["run_eps"], dtype=float),
                    "run_eta": np.asarray(d["run_eta"], dtype=float),
                    "eps_line": np.asarray(d["eps_line"], dtype=float),
                    "eta_crit_line": np.asarray(d["eta_crit_line"], dtype=float),
                }
        print(f"[INFO] Loaded Panel d phase diagram from '{pd_path.name}' (shape={phase_data['Li'].shape}).")
    else:
        phase_data = demo_fallback["phase"]
        print(f"[WARN] '{pd_path.name}' not found in '{data_dir}' -> Panel d using skeleton preview data.")

    return {
        "snapshots": snapshots,
        "time": {"t": t_dict, "Li": li_dict, "eta": eta_dict},
        "collapse": collapse_data,
        "phase": phase_data,
    }


# =============================================================================
# 4. NATURE-STYLE FIGURE 3 RENDERER (183 mm x 132 mm)
# =============================================================================
def plot_fig3(data: dict, suffix: str = "") -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "pdf.fonttype": 42,        # TrueType 矢量嵌入，确保 Illustrator 可直接选中修改文字
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=300, facecolor="white")

    # -------------------------------------------------------------------------
    # PANEL a: Top Row — 4 Square 2D Snapshots + Right-Edge Colorbar Rail
    # -------------------------------------------------------------------------
    gs_a = fig.add_gridspec(
        nrows=1, ncols=4,
        left=fx(LEFT_MM), right=fx(MAIN_RIGHT_MM),
        bottom=fy(TOP_BOT_MM), top=fy(TOP_TOP_MM),
        wspace=3.0 / 36.5,         # 精确对应 3.0 mm 子图水平间距
    )

    case_keys = ["static", "filter", "shear_const", "coupled"]
    im_a = None
    for idx, ckey in enumerate(case_keys):
        ax = fig.add_subplot(gs_a[0, idx])
        snap = data["snapshots"][ckey]
        meta = CASE_META[ckey]

        # 纯净 Li 浓度场（gouraud 平滑渲染消除网格像素锯齿）
        im_a = ax.pcolormesh(
            snap["X"], snap["Z"], snap["Li"],
            cmap=LI_CMAP, norm=LI_NORM, shading="gouraud", rasterized=True, zorder=1
        )

        # 高孔隙熔体通道等值线 (phi = 0.25, 0.40)
        phi_min_v, phi_max_v = float(np.nanmin(snap["phi"])), float(np.nanmax(snap["phi"]))
        cand_levels = [0.25, 0.40]
        valid_levels = [lv for lv in cand_levels if phi_min_v + 0.005 < lv < phi_max_v - 0.005]
        if valid_levels:
            lws = [0.55 if lv < 0.35 else 0.85 for lv in valid_levels]
            lss = ["--" if lv < 0.35 else "-" for lv in valid_levels]
            ax.contour(
                snap["X"], snap["Z"], snap["phi"],
                levels=valid_levels, colors="#FFFFFF",
                linewidths=lws, linestyles=lss, alpha=0.90, zorder=2
            )

        # 稀疏达西流速箭头（四图严格采用共享物理对数标尺）
        is_log_q = snap.get("q_norm_mode") == "log"
        base_div = 8 if is_log_q else 9
        step = max(1, snap["X"].shape[0] // base_div)
        vx_s = snap["vx"][step // 2 :: step, step // 2 :: step]
        vz_s = snap["vz"][step // 2 :: step, step // 2 :: step]
        xs_q = snap["X"][step // 2 :: step, step // 2 :: step]
        zs_q = snap["Z"][step // 2 :: step, step // 2 :: step]
        spd = np.hypot(vx_s, vz_s)
        thresh = 0.22 if is_log_q else 0.15
        mask_q = spd > thresh
        if np.any(mask_q):
            mean_spd = float(np.mean(spd[mask_q]))
            q_alpha = float(np.clip(0.32 + 0.56 * mean_spd, 0.35, 0.88)) if is_log_q else 0.80
            ax.quiver(
                xs_q[mask_q], zs_q[mask_q],
                vx_s[mask_q], -vz_s[mask_q],   # 注意 y 轴向下反转，取 -vz 对应屏幕向上流动
                color="#1E2124", scale=7.8 if is_log_q else 8.5,
                width=0.009, headwidth=3.6, headlength=4.0,
                alpha=q_alpha, zorder=3
            )

        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(2.0, 0.0)      # 深度向下递增
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_yticks([0.0, 1.0, 2.0])
        ax.tick_params(labelsize=6.0, length=2.2, pad=1.8)
        ax.set_title(meta["title"], fontsize=6.6, fontweight="bold", color="#1E2124", pad=3.5)

        if idx == 0:
            ax.set_ylabel("Depth (km)", fontsize=6.8, labelpad=2.5)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=1.8)

        # 底部极简状态角标
        ax.text(
            0.05, 0.06, meta["tag"], transform=ax.transAxes,
            fontsize=5.6, fontweight="bold",
            color="#FFFFFF" if ckey == "coupled" else "#222528",
            ha="left", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=(0.43, 0.09, 0.07, 0.80) if ckey == "coupled" else (1.0, 1.0, 1.0, 0.84),
                edgecolor="none",
            ),
            zorder=5,
        )

    # Panel a 居中公共横轴标签与面板角标 "a"
    fig.text(
        fx(0.5 * (LEFT_MM + MAIN_RIGHT_MM)), fy(TOP_BOT_MM - 5.2),
        "Horizontal distance (km)", fontsize=6.8, ha="center", va="top", color="#1E2124"
    )
    fig.text(fx(2.5), fy(TOP_TOP_MM + 6.2), "a", fontsize=8.5, fontweight="bold", ha="left", va="top")

    # 顶行右侧专用 LogNorm 色标轨道
    cax_a = fig.add_axes([fx(CBAR_LEFT_MM), fy(TOP_BOT_MM), fx(CBAR_RIGHT_MM - CBAR_LEFT_MM), fy(TOP_TOP_MM - TOP_BOT_MM)])
    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_ticks([1e2, 1e3, 1e4])
    cb_a.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cb_a.ax.tick_params(labelsize=5.8, length=2.0, pad=1.8)
    cb_a.ax.minorticks_off()
    cb_a.set_label("Melt Li (ppm)", fontsize=6.4, labelpad=2.5)
    cb_a.outline.set_linewidth(0.55)

    # -------------------------------------------------------------------------
    # SHARED 4-CASE LEGEND STRIP (位于上下行留白走廊中)
    # -------------------------------------------------------------------------
    leg_handles = [
        Line2D([0], [0], color=CASE_META[k]["color"], ls=CASE_META[k]["ls"], lw=CASE_META[k]["lw"], label=CASE_META[k]["legend"])
        for k in case_keys
    ]
    fig.legend(
        handles=leg_handles,
        loc="center", bbox_to_anchor=(fx(0.5 * (LEFT_MM + MAIN_RIGHT_MM)), fy(76.8)),
        ncol=4, frameon=False, fontsize=6.1, handlelength=1.8, handletextpad=0.4, columnspacing=1.2
    )

    # -------------------------------------------------------------------------
    # PANEL b: Bottom-Left — Dual-Stacked Time Evolution (Li & Viscosity)
    # -------------------------------------------------------------------------
    gs_b = fig.add_gridspec(
        nrows=2, ncols=1,
        left=fx(B_LEFT_MM), right=fx(B_RIGHT_MM),
        bottom=fy(BOT_BOT_MM), top=fy(BOT_TOP_MM),
        hspace=2.8 / 27.6,         # 精确 2.8 mm 紧凑子图间距
    )
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[1, 0], sharex=ax_b1)

    t_data = data["time"]["t"]
    for ckey in case_keys:
        m = CASE_META[ckey]
        t_arr = t_data[ckey] if isinstance(t_data, dict) else t_data
        ax_b1.plot(t_arr, data["time"]["Li"][ckey], color=m["color"], ls=m["ls"], lw=m["lw"], zorder=4 if ckey == "coupled" else 3)
        ax_b2.plot(t_arr, data["time"]["eta"][ckey], color=m["color"], ls=m["ls"], lw=m["lw"], zorder=4 if ckey == "coupled" else 3)

    # 锂辉石饱和阈值线 & 热固结死亡线 (t / tau_cool = 1)
    ax_b1.axhline(3500.0, color="#6E757C", lw=0.75, ls=":", zorder=2)
    ax_b1.text(0.03, 4200.0, "Spod. saturation", fontsize=5.4, color="#4A4E52", ha="left", va="bottom")

    for ax_sub in (ax_b1, ax_b2):
        ax_sub.axvline(1.0, color="#8C939A", lw=0.80, ls="--", zorder=1.5)
        ax_sub.axvspan(1.0, 1.30, color="#EEF1F4", alpha=0.65, zorder=0)
        ax_sub.set_yscale("log")
        ax_sub.set_xlim(0.0, 1.30)
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.tick_params(which="major", labelsize=6.0, length=2.3, pad=1.8)
        ax_sub.tick_params(which="minor", length=0)

    ax_b1.set_ylim(70.0, 2.5e4)
    ax_b1.yaxis.set_major_locator(FixedLocator([1e2, 1e3, 1e4]))
    ax_b1.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_b1.set_ylabel("Peak Li (ppm)", fontsize=6.6, labelpad=2.5)
    ax_b1.tick_params(labelbottom=False)
    ax_b1.text(1.02, 1.4e4, "Thermal\nlock-in", fontsize=5.2, color="#4A4E52", ha="left", va="top", linespacing=1.08)

    ax_b2.set_ylim(5.0e2, 3.0e7)
    ax_b2.yaxis.set_major_locator(FixedLocator([1e3, 1e5, 1e7]))
    ax_b2.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_b2.set_ylabel(r"Melt $\eta_m$ (Pa s)", fontsize=6.6, labelpad=2.5)
    ax_b2.set_xlabel(r"Normalized time, $t / \tau_{\mathrm{cool}}$", fontsize=6.8, labelpad=2.5)
    ax_b2.set_xticks([0.0, 0.5, 1.0])

    fig.text(fx(2.5), fy(BOT_TOP_MM + 4.0), "b", fontsize=8.5, fontweight="bold", ha="left", va="top")

    # -------------------------------------------------------------------------
    # PANEL c: Bottom-Center — Cross-Regime Kinematic Collapse
    # -------------------------------------------------------------------------
    gs_c = fig.add_gridspec(
        nrows=1, ncols=1,
        left=fx(C_LEFT_MM), right=fx(C_RIGHT_MM),
        bottom=fy(BOT_BOT_MM), top=fy(BOT_TOP_MM),
    )
    ax_c = fig.add_subplot(gs_c[0, 0])

    col = data["collapse"]
    ax_c.fill_between(col["pi_curve"], col["lo"], col["hi"], color="#E6B8AF", alpha=0.35, edgecolor="none", zorder=1)
    ax_c.plot(col["pi_curve"], col["mean"], color=CRIMSON_CRIT, lw=1.15, zorder=2)
    ax_c.axvline(1.0, color=CRIMSON_CRIT, lw=1.0, ls="--", zorder=2.5)

    has_3d = any(pt["dim"] == "3D" for pt in col["points"])
    for pt in col["points"]:
        st = REGIME_STYLES[pt["regime"]]
        is_3d = pt["dim"] == "3D"
        ax_c.scatter(
            pt["Pi_p"], pt["enrichment"],
            marker=st["marker"],
            s=22 if is_3d else 18,
            facecolors=st["color"] if (is_3d or not has_3d) else "#FFFFFF",
            edgecolors="#FFFFFF" if (not has_3d) else st["color"],
            linewidths=0.45 if (not has_3d) else (0.85 if not is_3d else 0.5),
            alpha=0.92,
            zorder=5 if is_3d else 4,
        )

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(1.0e-2, 1.0e2)
    ax_c.set_ylim(0.85, 2.8e2)
    ax_c.xaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1e0, 1e1, 1e2]))
    ax_c.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_c.yaxis.set_major_locator(FixedLocator([1e0, 1e1, 1e2]))
    ax_c.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_c.set_xlabel(r"Tectonic pumping number, $\Pi_p$", fontsize=6.8, labelpad=2.5)
    ax_c.set_ylabel(r"Li enrichment, $C_{\mathrm{Li}} / C_0$", fontsize=6.8, labelpad=2.0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.tick_params(which="major", labelsize=6.0, length=2.3, pad=1.8)
    ax_c.tick_params(which="minor", length=0)

    ax_c.text(1.22, 1.15, r"$\Pi_p = 1$", fontsize=5.8, fontweight="bold", color=CRIMSON_CRIT, ha="left", va="bottom")

    c_handles = [
        Line2D([0], [0], marker=REGIME_STYLES[r]["marker"], color="none",
               markerfacecolor=REGIME_STYLES[r]["color"], markeredgecolor=REGIME_STYLES[r]["color"],
               markersize=4.1, label=REGIME_STYLES[r]["label"])
        for r in ("EXT", "COMP_TRANS", "STRIKE_SLIP")
    ]
    if has_3d:
        c_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#FFFFFF", markeredgecolor="#4A4E52",
                   markersize=3.8, label="2D / 3D (filled)")
        )
    ax_c.legend(
        handles=c_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor=(1, 1, 1, 0.86), edgecolor="none",
        fontsize=5.3, handletextpad=0.28, labelspacing=0.24, borderpad=0.22
    )

    fig.text(fx(C_LEFT_MM - 9.5), fy(BOT_TOP_MM + 4.0), "c", fontsize=8.5, fontweight="bold", ha="left", va="top")

    # -------------------------------------------------------------------------
    # PANEL d: Bottom-Right — Dynamical Phase Diagram + Aligned Colorbar Rail
    # -------------------------------------------------------------------------
    gs_d = fig.add_gridspec(
        nrows=1, ncols=1,
        left=fx(D_LEFT_MM), right=fx(D_RIGHT_MM),
        bottom=fy(BOT_BOT_MM), top=fy(BOT_TOP_MM),
    )
    ax_d = fig.add_subplot(gs_d[0, 0])

    ph = data["phase"]
    # Gouraud 平滑渲染真实 2D 相图网格，消除方块马赛克感
    im_d = ax_d.pcolormesh(
        ph["EPS"], ph["ETA"], ph["Li"],
        cmap=LI_CMAP, norm=LI_NORM, shading="gouraud", rasterized=True, zorder=1
    )

    # 叠加真实 2D 数值模拟采样点（低调透明黑点，展示离散扫参支撑密度）
    ax_d.scatter(
        ph["run_eps"], ph["run_eta"], s=2.6, c="#1E2124", alpha=0.25, linewidths=0, zorder=2
    )

    # 叠加与求解器严格自洽的解析临界线 Pi_p = 1
    valid_l = (ph["eta_crit_line"] >= 1e4) & (ph["eta_crit_line"] <= 1e8)
    ax_d.plot(
        ph["eps_line"][valid_l], ph["eta_crit_line"][valid_l],
        color=CRIMSON_CRIT, lw=1.35, ls="--", zorder=4
    )

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlim(1.0e-16, 1.0e-11)
    ax_d.set_ylim(1.0e4, 1.0e8)
    ax_d.xaxis.set_major_locator(FixedLocator([1e-16, 1e-14, 1e-12]))
    ax_d.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_d.yaxis.set_major_locator(FixedLocator([1e4, 1e6, 1e8]))
    ax_d.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_d.set_xlabel(r"Shear strain rate, $\dot{\varepsilon}$ ($\mathrm{s}^{-1}$)", fontsize=6.8, labelpad=2.5)
    ax_d.set_ylabel(r"Initial viscosity, $\eta_{m0}$ (Pa s)", fontsize=6.8, labelpad=2.0)
    ax_d.tick_params(which="major", labelsize=6.0, length=2.3, pad=1.8)
    ax_d.tick_params(which="minor", length=0)

    # 自动将 Pi_p = 1 标注贴合在临界线浅色侧（避免掉进深红区看不清）
    eps_v = ph["eps_line"][valid_l]
    eta_v = ph["eta_crit_line"][valid_l]
    if eps_v.size > 10:
        idx_lbl = int(np.argmin(np.abs(np.log10(eps_v) - (-13.7))))
        lbl_eps = eps_v[idx_lbl] * 0.62
        lbl_eta = eta_v[idx_lbl] * 1.90
    else:
        lbl_eps, lbl_eta = 1.6e-14, 1.8e6

    ax_d.text(2.2e-16, 4.2e7, "Mush lock\n" + r"($\Pi_p < 1$)", fontsize=5.8, fontweight="bold", color="#2F4B63", ha="left", va="top", linespacing=1.12, zorder=5)
    ax_d.text(6.5e-12, 2.0e4, "Shear-driven\nLCT " + r"($\Pi_p > 1$)", fontsize=5.8, fontweight="bold", color="#FFFFFF", ha="right", va="bottom", linespacing=1.12, zorder=5)
    ax_d.text(lbl_eps, lbl_eta, r"$\Pi_p = 1$", fontsize=5.8, fontweight="bold", color=CRIMSON_CRIT, rotation=39, ha="center", va="bottom", zorder=5)

    fig.text(fx(D_LEFT_MM - 9.5), fy(BOT_TOP_MM + 4.0), "d", fontsize=8.5, fontweight="bold", ha="left", va="top")

    # 底行右侧对齐 LogNorm 色标轨道（与 cax_a 共享同一水平坐标）
    cax_d = fig.add_axes([fx(CBAR_LEFT_MM), fy(BOT_BOT_MM), fx(CBAR_RIGHT_MM - CBAR_LEFT_MM), fy(BOT_TOP_MM - BOT_BOT_MM)])
    cb_d = fig.colorbar(im_d, cax=cax_d)
    cb_d.set_ticks([1e2, 1e3, 1e4])
    cb_d.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cb_d.ax.tick_params(labelsize=5.8, length=2.0, pad=1.8)
    cb_d.ax.minorticks_off()
    cb_d.set_label("Peak melt Li (ppm)", fontsize=6.4, labelpad=2.5)
    cb_d.outline.set_linewidth(0.55)

    # -------------------------------------------------------------------------
    # 5. EXPORT VECTOR PDF & 600 DPI RASTER PREVIEW
    # -------------------------------------------------------------------------
    tag = f"_{suffix}" if suffix else ""
    pdf_out = OUTPUT_DIR / f"{OUTPUT_PREFIX}{tag}.pdf"
    png_out = OUTPUT_DIR / f"{OUTPUT_PREFIX}{tag}.png"
    fig.savefig(pdf_out, format="pdf")
    fig.savefig(png_out, format="png", dpi=600)
    plt.close(fig)
    print(f"[DONE] Saved 183 x 132 mm Nature Fig. 3 vector PDF -> {pdf_out}")
    print(f"[DONE] Saved 600 dpi raster preview                -> {png_out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    if USE_DEMO_DATA:
        plot_fig3(generate_synthetic_demo_data(), suffix="demo")
    else:
        for mode in SNAP_MODES:
            sim_data = load_real_simulation_data(DATA_DIR, snap_mode=mode, time_li_key=TIME_LI_KEY)
            plot_fig3(sim_data, suffix=mode)