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
from scripts.plot_bld_summary import summarize_zarr

YEARS = range(1993, 2025)
OUTPUT_DIR = ROOT / "data" / "025deg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZARR_PATH = OUTPUT_DIR / "bld_025deg_monthly_1993_2024.zarr"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def year_already_stored(zarr_path, year):
    if not zarr_path.exists():
        return False
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        stored = [t.year for t in pd.DatetimeIndex(ds.time.values)]
        return stored.count(year) == 12
    except Exception:
        return False


log("Opening full monthly dataset (lazy) ...")
ds_full = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
    variables=["thetao_glor", "mlotst_glor"],
    minimum_longitude=-180,               maximum_longitude=179.75,
    minimum_latitude=-80,                 maximum_latitude=90,
    minimum_depth=0.5057600140571594,
    maximum_depth=508.639892578125,
    coordinates_selection_method="strict-inside",
)
t0 = pd.Timestamp(ds_full.time.values[0]).strftime("%Y-%m")
t1 = pd.Timestamp(ds_full.time.values[-1]).strftime("%Y-%m")
log(f"Dataset ready: {t0} to {t1}")

for year in YEARS:
    if year_already_stored(ZARR_PATH, year):
        log(f"Skipping {year} (already stored)")
        continue

    log(f"Processing {year} ...")
    ds_year = ds_full.sel(time=str(year))

    result = compute_global_bld(ds_year, temp_var="thetao_glor", mld_var="mlotst_glor", threshold=-0.1)

    log(f"Writing {year} to Zarr ...")
    if not ZARR_PATH.exists():
        result.to_zarr(ZARR_PATH, mode="w", zarr_format=2, safe_chunks=False)
    else:
        result.to_zarr(ZARR_PATH, mode="a", append_dim="time", zarr_format=2, safe_chunks=False)

    # Quick per-year stats
    bld_vals = result["bld"].values
    mean_bld = float(np.nanmean(bld_vals))
    pct_pos  = float(np.mean(bld_vals > 0) * 100)
    log(f"Saved: {year}  |  mean BLD={mean_bld:.1f} m  |  {pct_pos:.1f}% pixels > 0")

log("Done. Generating summary figure ...")
summarize_zarr(ZARR_PATH, label="bld_025deg_monthly_1993_2024")
log("All done.")
