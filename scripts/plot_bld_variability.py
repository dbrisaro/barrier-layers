"""
plot_bld_variability.py

Interannual variability of BLD (1993–2024) using the default ILD config
(gradient −0.1°C/m).

Computes:
  1. Temporal std of monthly BLD (total variability)
  2. Seasonal amplitude: max − min of the monthly climatology
  3. Interannual std: std of annual mean BLD across years (after removing
     the seasonal cycle)
  4. Year-to-year anomaly maps for selected years

Outputs (displays/):
  bld_variability_maps.png   — 3-panel global maps (total std, seasonal amp, interannual std)
  bld_anomaly_maps.png       — annual BLD anomaly maps for every 5th year + extremes
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = "gradient_100"

BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)


def zarr_path(name):
    for f in SENS_DIR.iterdir():
        if name in f.name:
            return f
    raise FileNotFoundError(f"No sensitivity zarr found for '{name}' in {SENS_DIR}")


def map_panel(ax, data, lons, lats, cmap, vmin, vmax, title, unit, norm=None):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")
    kw = dict(cmap=cmap, transform=ccrs.PlateCarree(), rasterized=True)
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin
        kw["vmax"] = vmax
    im = ax.pcolormesh(lons, lats, data, **kw)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading default config (gradient −0.1°C/m)…")
ds = xr.open_zarr(zarr_path(DEFAULT_CONFIG))
bld = ds["bld"].where(ds["bld"] > 0)   # NaN where no barrier layer

lats = bld.latitude.values
lons = bld.longitude.values

# ── 1. Total temporal std ─────────────────────────────────────────────────────
print("Computing total temporal std…")
std_total = bld.std("time").compute()   # (nlat, nlon)

# ── 2. Seasonal amplitude ────────────────────────────────────────────────────
print("Computing seasonal amplitude…")
clim = bld.groupby("time.month").mean("time").compute()   # (12, nlat, nlon)
seasonal_amp = clim.max("month") - clim.min("month")      # (nlat, nlon)

# ── 3. Interannual std ────────────────────────────────────────────────────────
print("Computing interannual std (after removing seasonal cycle)…")
bld_deseas = bld.groupby("time.month") - clim              # anomaly vs seasonal mean
bld_ann    = bld_deseas.resample(time="YE").mean("time").compute()  # (32, nlat, nlon)
std_interann = bld_ann.std("time").compute()

# ── Figure 1: 3-panel variability maps ────────────────────────────────────────

fig, axes = plt.subplots(
    3, 1, figsize=(14, 18),
    subplot_kw={"projection": ccrs.Robinson()},
)

vmax_std = float(np.nanpercentile(std_total.values, 98))
im0 = map_panel(axes[0], std_total.values,  lons, lats, BLD_CMAP, 0, vmax_std,
                "Total temporal std of BLD (m)", "m")
plt.colorbar(im0, ax=axes[0], orientation="vertical", pad=0.02, shrink=0.8, label="std (m)")

vmax_amp = float(np.nanpercentile(seasonal_amp.values, 98))
im1 = map_panel(axes[1], seasonal_amp.values, lons, lats, BLD_CMAP, 0, vmax_amp,
                "Seasonal amplitude of BLD (m)  [max clim − min clim]", "m")
plt.colorbar(im1, ax=axes[1], orientation="vertical", pad=0.02, shrink=0.8, label="amplitude (m)")

vmax_ia = float(np.nanpercentile(std_interann.values, 98))
im2 = map_panel(axes[2], std_interann.values, lons, lats, BLD_CMAP, 0, vmax_ia,
                "Interannual std of BLD (m)  [std of deseasonalised annual means]", "m")
plt.colorbar(im2, ax=axes[2], orientation="vertical", pad=0.02, shrink=0.8, label="std (m)")

fig.suptitle(
    "Barrier Layer Depth — Variability 1993–2024\n"
    "ILD method: Gradient −0.1°C/m",
    fontsize=14, fontweight="bold", y=1.01,
)
fig.tight_layout()
out1 = OUT_DIR / "bld_variability_maps.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out1}")


# ── Figure 2: Annual anomaly maps ────────────────────────────────────────────

print("\nBuilding annual anomaly maps…")

years_da  = bld_ann.time.dt.year.values   # 1993–2024
clim_ann  = bld_ann.mean("time")          # long-term annual mean
anom_ann  = bld_ann - clim_ann            # (32, nlat, nlon)

# Choose which years to display: every 5th + the min/max anomaly years
global_anom = anom_ann.mean(["latitude", "longitude"]).values
yr_max = int(years_da[np.nanargmax(global_anom)])
yr_min = int(years_da[np.nanargmin(global_anom)])

select_years = sorted(set(
    list(years_da[::5]) + [yr_min, yr_max, int(years_da[-1])]
))
n = len(select_years)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig2, axes2 = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, 4 * nrows),
    subplot_kw={"projection": ccrs.Robinson()},
)
axes2 = axes2.ravel()

vmax_anom = float(np.nanpercentile(np.abs(anom_ann.values), 98))
norm_anom  = TwoSlopeNorm(vmin=-vmax_anom, vcenter=0, vmax=vmax_anom)
cmap_anom  = plt.cm.RdBu_r

for idx, yr in enumerate(select_years):
    ax = axes2[idx]
    yr_idx = int(np.where(years_da == yr)[0][0])
    data = anom_ann.values[yr_idx]
    ga   = global_anom[yr_idx]

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    ax.gridlines(linewidth=0.3, color="grey", alpha=0.3, linestyle="--")
    im = ax.pcolormesh(
        lons, lats, data,
        norm=norm_anom, cmap=cmap_anom,
        transform=ccrs.PlateCarree(), rasterized=True,
    )
    tag = ""
    if yr == yr_max: tag = " ▲ max"
    if yr == yr_min: tag = " ▼ min"
    ax.set_title(f"{yr}{tag}  (Δ={ga:+.1f} m)", fontsize=10, fontweight="bold")

# Remove unused axes
for idx in range(len(select_years), len(axes2)):
    axes2[idx].set_visible(False)

# Shared colorbar
fig2.subplots_adjust(right=0.88, hspace=0.3, wspace=0.05)
cbar_ax = fig2.add_axes([0.90, 0.15, 0.018, 0.7])
plt.colorbar(im, cax=cbar_ax, label="BLD anomaly (m)")

fig2.suptitle(
    "Annual BLD Anomaly Maps (relative to 1993–2024 mean)\n"
    "ILD method: Gradient −0.1°C/m",
    fontsize=13, fontweight="bold",
)

out2 = OUT_DIR / "bld_anomaly_maps.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2}")

print("\nDone.")
print(f"  Max positive anomaly year: {yr_max}  (Δ = {global_anom[np.where(years_da==yr_max)[0][0]]:+.2f} m)")
print(f"  Max negative anomaly year: {yr_min}  (Δ = {global_anom[np.where(years_da==yr_min)[0][0]]:+.2f} m)")
