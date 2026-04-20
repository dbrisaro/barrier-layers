import sys
import calendar
from pathlib import Path
from datetime import datetime, date
import numpy as np
import xarray as xr
import pandas as pd
import copernicusmarine as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import compute_global_bld

YEAR = 2025
OUTPUT_DIR = ROOT / "data" / "083deg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZARR_PATH = OUTPUT_DIR / f"bld_083deg_monthly_{YEAR}.zarr"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def month_already_stored(zarr_path, year, month):
    if not zarr_path.exists():
        return False
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        return any(t.year == year and t.month == month for t in times)
    except Exception:
        return False


for month in range(1, 13):
    month_start = date(YEAR, month, 1)
    month_end   = date(YEAR, month, calendar.monthrange(YEAR, month)[1])

    if month_start > date.today():
        log(f"Skipping {month_start.strftime('%Y-%m')} (future)")
        continue

    if month_already_stored(ZARR_PATH, YEAR, month):
        log(f"Skipping {month_start.strftime('%Y-%m')} (already stored)")
        continue

    start_dt = month_start.strftime("%Y-%m-%dT00:00:00")
    end_dt   = month_end.strftime("%Y-%m-%dT23:59:59")

    log(f"Downloading {month_start.strftime('%Y-%m')} ({month_end.day} days) ...")

    ds_temp = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=-180,               maximum_longitude=179.9166717529297,
        minimum_latitude=-80,                 maximum_latitude=90,
        start_datetime=start_dt,
        end_datetime=end_dt,
        minimum_depth=0.49402499198913574,
        maximum_depth=541.0889282226562,
        coordinates_selection_method="strict-inside",
    )

    ds_mld = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["mlotst"],
        minimum_longitude=-180,               maximum_longitude=179.9166717529297,
        minimum_latitude=-80,                 maximum_latitude=90,
        start_datetime=start_dt,
        end_datetime=end_dt,
        coordinates_selection_method="strict-inside",
    )

    ds = xr.merge([ds_temp, ds_mld])

    log("Computing BLD ...")
    daily = compute_global_bld(ds, temp_var="thetao", mld_var="mlotst", threshold=-0.1)

    log("Computing monthly mean ...")
    monthly = daily.mean(dim="time").expand_dims(
        dim={"time": [pd.Timestamp(month_start)]}
    )

    log("Writing to Zarr ...")
    if not ZARR_PATH.exists():
        monthly.to_zarr(ZARR_PATH, mode="w", zarr_format=2)
    else:
        monthly.to_zarr(ZARR_PATH, mode="a", append_dim="time", zarr_format=2)

    log(f"Saved: {month_start.strftime('%Y-%m')}")

log("Done.")
