"""
Seasonal BLD statistics from ensemble mean across 6 sensitivity configs.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/daniela/Documents/Barrier Layers/data/025deg/sensitivity"
FILES = [
    "bld_gradient_015_1993_2024.zarr",
    "bld_gradient_025_1993_2024.zarr",
    "bld_gradient_100_1993_2024.zarr",
    "bld_difference_02_1993_2024.zarr",
    "bld_difference_05_1993_2024.zarr",
    "bld_difference_08_1993_2024.zarr",
]

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# ── step 1: load all 6 bld arrays, set <= 0 to NaN ────────────────────────
print("Loading 6 zarr files …")
bld_list = []
for fname in FILES:
    path = f"{DATA_DIR}/{fname}"
    ds = xr.open_zarr(path)
    bld = ds["bld"]
    bld = bld.where(bld > 0)   # set <= 0 → NaN
    bld_list.append(bld)
    print(f"  loaded {fname}  shape={bld.shape}")

# stack into (6, time, lat, lon) numpy array — compute in one shot
print("\nComputing ensemble mean (nanmean across 6 configs) …")
# Use dask-backed stacking then compute
stacked = xr.concat(bld_list, dim="config")          # (6, time, lat, lon)
# nanmean across configs axis=0; keep NaN only where ALL are NaN
ens_mean = stacked.reduce(np.nanmean, dim="config")  # (time, lat, lon)
ens_mean = ens_mean.compute()
print(f"  ensemble mean shape: {ens_mean.shape}")

# ── ocean mask: pixel is ocean if finite in ANY config at ANY time ─────────
print("Building ocean mask …")
ocean_mask = np.zeros((ens_mean.shape[1], ens_mean.shape[2]), dtype=bool)
for bld in bld_list:
    arr = bld.compute().values          # (time, lat, lon)
    has_data = np.any(np.isfinite(arr), axis=0)
    ocean_mask |= has_data

n_total   = ocean_mask.size
n_ocean   = ocean_mask.sum()
print(f"\n  Total grid points (lat × lon): {n_total:,}")
print(f"  Ocean pixels (at least one config ever finite): {n_ocean:,}")

# ── step 3: seasonal climatology ──────────────────────────────────────────
time_months = ens_mean.time.dt.month.values   # (384,)
ens_vals    = ens_mean.values                 # (384, lat, lon)

print("\n" + "="*70)
print("SEASONAL STATISTICS")
print("="*70)
print(f"\nTotal grid points (lat × lon) : {n_total:,}")
print(f"Total ocean pixels             : {n_ocean:,}")

for season, months in SEASONS.items():
    # select months belonging to this season
    sel = np.isin(time_months, months)
    seasonal_vals = ens_vals[sel, :, :]       # (n_months_sel, lat, lon)

    # climatological mean (nanmean over time)
    clim = np.nanmean(seasonal_vals, axis=0)  # (lat, lon)

    # restrict to ocean pixels
    clim_ocean = clim[ocean_mask]

    # counts
    valid_mask = np.isfinite(clim_ocean) & (clim_ocean > 0)
    n_nan      = np.sum(~np.isfinite(clim_ocean))
    n_valid    = np.sum(valid_mask)
    frac_nan   = 100.0 * n_nan / n_ocean

    valid_vals = clim_ocean[valid_mask]

    pcts = np.percentile(valid_vals, [5, 25, 50, 75, 95, 99])

    print(f"\n── {season} ──────────────────────────────────────────────────")
    print(f"  Ocean pixels                      : {n_ocean:,}")
    print(f"  NaN pixels (no barrier layer)     : {n_nan:,}")
    print(f"  Valid pixels (BLD > 0)            : {n_valid:,}")
    print(f"  Fraction ocean that is NaN        : {frac_nan:.2f} %")
    print(f"  Min BLD (valid)                   : {valid_vals.min():.4f} m")
    print(f"  Max BLD (valid)                   : {valid_vals.max():.4f} m")
    print(f"  Mean BLD (valid)                  : {valid_vals.mean():.4f} m")
    print(f"  Percentiles (5/25/50/75/95/99)    : "
          f"{pcts[0]:.4f} / {pcts[1]:.4f} / {pcts[2]:.4f} / "
          f"{pcts[3]:.4f} / {pcts[4]:.4f} / {pcts[5]:.4f} m")

print("\nDone.")
