"""
sensitivity_analysis_025deg.py

Computes ILD/BLD for 6 threshold+method configurations on the 0.25° CMEMS
monthly reanalysis (1993-2024), using two approaches:

  Gradient method  — ILD = first depth below 5 m where dT/dz <= threshold
  Difference method — ILD = first depth below 10 m where T < T(10m) - ΔT

Strategy: each year's data is downloaded from CMEMS once (.compute()), then
all 6 configs are run on the in-memory array.  This avoids 6× redundant
downloads.

Output (one Zarr per config):
  data/025deg/sensitivity/bld_gradient_015_1993_2024.zarr
  data/025deg/sensitivity/bld_gradient_025_1993_2024.zarr
  data/025deg/sensitivity/bld_gradient_100_1993_2024.zarr
  data/025deg/sensitivity/bld_difference_02_1993_2024.zarr
  data/025deg/sensitivity/bld_difference_05_1993_2024.zarr
  data/025deg/sensitivity/bld_difference_08_1993_2024.zarr
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import copernicusmarine as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import compute_global_bld
from scripts.plot_bld_summary import summarize_zarr, compare_configs

# ── Configuration ────────────────────────────────────────────────────────────

YEARS      = range(1993, 2025)
OUTPUT_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (name, method, threshold)
# gradient  threshold: negative  (dT/dz <= threshold, e.g. -0.025 °C/m)
# difference threshold: positive (ΔT from 10 m reference, e.g. 0.2 °C)
CONFIGS = [
    ("gradient_015",   "gradient",   -0.015),
    ("gradient_025",   "gradient",   -0.025),
    ("gradient_100",   "gradient",   -0.1  ),
    ("difference_02",  "difference",  0.2  ),
    ("difference_05",  "difference",  0.5  ),
    ("difference_08",  "difference",  0.8  ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def zarr_path(name):
    return OUTPUT_DIR / f"bld_{name}_1993_2024.zarr"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def year_stored(path, year):
    """Return True if all 12 months of year are present in the Zarr store."""
    if not path.exists():
        return False
    try:
        ds = xr.open_zarr(path, consolidated=False)
        stored = [t.year for t in pd.DatetimeIndex(ds.time.values)]
        return stored.count(year) == 12
    except Exception:
        return False


def all_configs_done(year):
    return all(year_stored(zarr_path(name), year) for name, _, _ in CONFIGS)


def write_result(result, path):
    if not path.exists():
        result.to_zarr(path, mode="w", zarr_format=2, safe_chunks=False)
    else:
        result.to_zarr(path, mode="a", append_dim="time",
                       zarr_format=2, safe_chunks=False)


# ── Open remote dataset (lazy) ────────────────────────────────────────────────

log("Opening full monthly dataset (lazy) ...")
ds_full = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
    variables=["thetao_glor", "mlotst_glor"],
    minimum_longitude=-180,             maximum_longitude=179.75,
    minimum_latitude=-80,               maximum_latitude=90,
    minimum_depth=0.5057600140571594,
    maximum_depth=508.639892578125,
    coordinates_selection_method="strict-inside",
)
t0 = pd.Timestamp(ds_full.time.values[0]).strftime("%Y-%m")
t1 = pd.Timestamp(ds_full.time.values[-1]).strftime("%Y-%m")
log(f"Dataset ready: {t0} to {t1}")
log(f"Running {len(CONFIGS)} configs: {[c[0] for c in CONFIGS]}")

# ── Main loop ─────────────────────────────────────────────────────────────────

for year in YEARS:
    if all_configs_done(year):
        log(f"Skipping {year} (all configs already stored)")
        continue

    # Download this year once into RAM — ~200 MB per year
    log(f"Loading {year} into memory ...")
    ds_year = ds_full.sel(time=str(year)).compute()
    log(f"Loaded {year}  →  running configs ...")

    for name, method, threshold in CONFIGS:
        path = zarr_path(name)

        if year_stored(path, year):
            log(f"  [{name}] {year} already stored — skipping")
            continue

        log(f"  [{name}] method={method}  threshold={threshold}")
        result = compute_global_bld(
            ds_year,
            temp_var="thetao_glor",
            mld_var="mlotst_glor",
            method=method,
            threshold=threshold,
            use_abs=True,
        )
        # Quick per-year stats
        bld_vals = result["bld"].values
        mean_bld = float(np.nanmean(bld_vals))
        pct_pos  = float(np.mean(bld_vals > 0) * 100)
        write_result(result, path)
        log(f"  [{name}] {year} saved  |  mean BLD={mean_bld:.1f} m  |  {pct_pos:.1f}% > 0")

    log(f"Year {year} complete")

log("All 6 configurations complete. Generating summaries ...")

# Individual summary per config
for name, _, _ in CONFIGS:
    summarize_zarr(zarr_path(name), label=name, out_dir=ROOT / "displays")

# Side-by-side comparison of all 6
compare_configs(
    {name: zarr_path(name) for name, _, _ in CONFIGS},
    out_dir=ROOT / "displays",
)

log("Done.")
