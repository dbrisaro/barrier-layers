"""
compute_bld_083deg_2024.py

Downloads the 0.083° monthly reanalysis for 2024 and computes BLD
(same method as compute_all_years_025deg.py: gradient threshold = −0.1°C/m).

Output: data/083deg/bld_083deg_monthly_2024.zarr

Resume-safe: skips months already stored in the Zarr.

Usage:
    python scripts/compute_bld_083deg_2024.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import copernicusmarine as cm
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import compute_global_bld

OUTPUT_PATH = ROOT / "data" / "083deg" / "bld_083deg_monthly_2024.zarr"

YEAR        = 2024
DATASET_ID  = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
TEMP_VAR    = "thetao"
MLD_VAR     = "mlotst"
THRESHOLD   = -0.1        # gradient °C/m  — same as 0.25° main zarr
METHOD      = "gradient"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def month_stored(zarr_path, year, month):
    """Return True if this month is already saved in the Zarr."""
    if not zarr_path.exists():
        return False
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False, decode_times=False)
        t  = ds.time.values
        return any(
            (int(str(pd.Timestamp(v))[:4]) == year and
             int(str(pd.Timestamp(v))[5:7]) == month)
            for v in t
        )
    except Exception:
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

months = pd.date_range(f"{YEAR}-01", f"{YEAR}-12", freq="MS")
log(f"Computing 0.083° BLD for {YEAR}  ({len(months)} months)")
log(f"Method: {METHOD}, threshold={THRESHOLD} °C/m")
log(f"Output: {OUTPUT_PATH.relative_to(ROOT)}\n")

for t0 in months:
    m = t0.month

    if month_stored(OUTPUT_PATH, YEAR, m):
        log(f"  {t0.strftime('%Y-%m')} already stored — skip")
        continue

    start = t0.strftime("%Y-%m-%dT00:00:00")
    end   = (t0 + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%dT23:59:59")

    log(f"  {t0.strftime('%Y-%m')}  downloading ...")
    ds = cm.open_dataset(
        dataset_id=DATASET_ID,
        variables=[TEMP_VAR, MLD_VAR],
        minimum_longitude=-180, maximum_longitude=179.916671752929,
        minimum_latitude=-80,   maximum_latitude=90,
        start_datetime=start,
        end_datetime=end,
        minimum_depth=0.49402499198913574,
        maximum_depth=541.0889282226562,
        coordinates_selection_method="strict-inside",
    )

    log(f"  {t0.strftime('%Y-%m')}  computing BLD ...")
    result = compute_global_bld(
        ds, temp_var=TEMP_VAR, mld_var=MLD_VAR,
        method=METHOD, threshold=THRESHOLD,
    )

    # ── Save to Zarr ──────────────────────────────────────────────────────────
    if not OUTPUT_PATH.exists():
        result.to_zarr(OUTPUT_PATH, mode="w", safe_chunks=False)
        log(f"  {t0.strftime('%Y-%m')}  created Zarr store")
    else:
        result.to_zarr(OUTPUT_PATH, mode="a", append_dim="time",
                       safe_chunks=False)
        log(f"  {t0.strftime('%Y-%m')}  appended to Zarr")

    # Quick stats
    bld = result["bld"].isel(time=0).values
    pos = bld[bld > 0]
    log(f"  {t0.strftime('%Y-%m')}  BLD: mean={np.nanmean(bld):.1f} m  "
        f"max={np.nanmax(bld):.1f} m  "
        f"coverage={100*len(pos)/np.isfinite(bld).sum():.1f}%\n")

log("All months done.")
log(f"Zarr: {OUTPUT_PATH}")

# Quick summary
ds_out = xr.open_zarr(OUTPUT_PATH, consolidated=False, decode_times=False)
log(f"Total time steps stored: {ds_out.dims['time']}")
log(f"Spatial grid: {ds_out.dims['latitude']} lat × {ds_out.dims['longitude']} lon")
