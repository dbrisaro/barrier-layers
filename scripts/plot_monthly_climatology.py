"""
plot_monthly_climatology.py

Monthly BLD climatology maps (Jan–Dec, averaged 1993–2024) for all 6 sensitivity
configurations. Uses white = 0 (no barrier layer), blue gradient for positive BLD.

Outputs (displays/):
  monthly_clim_{name}.png              — 12-panel (Jan-Dec) per config  (6 files)
  monthly_clim_seasonal_comparison.png — 4 seasons (DJF/MAM/JJA/SON) × 6 configs
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("gradient_015",  "Gradient −0.015°C/m", "#1a3a6b"),
    ("gradient_025",  "Gradient −0.025°C/m", "#2e6fad"),
    ("gradient_100",  "Gradient −0.1°C/m",   "#7eb8d4"),
    ("difference_02", "Diff 0.2°C",           "#1a5c2a"),
    ("difference_05", "Diff 0.5°C",           "#4aab5e"),
    ("difference_08", "Diff 0.8°C",           "#a8dbb0"),
]

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# Seasons: (label, month indices 1-based)
SEASONS = [
    ("DJF (Dec-Feb)", [12, 1, 2]),
    ("MAM (Mar-May)", [3, 4, 5]),
    ("JJA (Jun-Aug)", [6, 7, 8]),
    ("SON (Sep-Nov)", [9, 10, 11]),
]

# ── BLD colormap: pure white at 0, blues for positive ─────────────────────────
BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)

proj = ccrs.Robinson()
pc   = ccrs.PlateCarree()


def zarr_path(name):
    return SENS_DIR / f"bld_{name}_1993_2024.zarr"


def plot_bld_panel(ax, lon, lat, data, vmax, title, fig):
    """Single BLD map panel: white at 0, blue gradient for positive BLD."""
    data_m = np.ma.masked_where(data <= 0, data)
    img = ax.pcolormesh(lon, lat, data_m, vmin=0, vmax=vmax,
                        cmap=BLD_CMAP, transform=pc, rasterized=True)
    ax.add_feature(cfeature.LAND,      facecolor="#d4d4d4", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=9, pad=4)
    return img


# ── 1. Per-config: 12-month climatology (4 rows × 3 cols) ────────────────────
print("Computing monthly climatologies ...")

for name, label, _ in CONFIGS:
    print(f"  {name} ...", end=" ", flush=True)

    ds  = xr.open_zarr(zarr_path(name), consolidated=False)
    bld = ds["bld"]
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # Monthly climatology: mean of all Januaries, all Februaries, etc.
    clim = bld.groupby("time.month").mean(dim="time").compute()  # (12, lat, lon)

    # Common vmax across months
    pos_vals = clim.values[clim.values > 0]
    vmax = float(np.nanpercentile(pos_vals, 97)) if len(pos_vals) else 50
    vmax = max(20, vmax)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.08,
                            left=0.02, right=0.98, top=0.93, bottom=0.03)

    for m in range(12):
        row, col = divmod(m, 3)
        ax  = fig.add_subplot(gs[row, col], projection=proj)
        img = plot_bld_panel(ax, lon, lat, clim.isel(month=m).values,
                             vmax, MONTH_LABELS[m], fig)

    # One shared colorbar
    cax = fig.add_axes([0.25, 0.01, 0.5, 0.015])
    cb  = fig.colorbar(img, cax=cax, orientation="horizontal")
    cb.set_label("BLD (m)  —  white = no barrier layer", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Monthly BLD climatology  |  {label}  (1993–2024)",
                 fontsize=13, y=0.97)
    out = OUT_DIR / f"monthly_clim_{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out.name}")


# ── 2. Seasonal comparison: 4 rows (seasons) × 6 cols (configs) ──────────────
print("\nBuilding seasonal comparison figure ...")

# Load all seasonal means
seas_maps = {}   # config → season_label → (lat, lon) array
lat_ref = lon_ref = None

for name, label, _ in CONFIGS:
    ds  = xr.open_zarr(zarr_path(name), consolidated=False)
    bld = ds["bld"]
    if lat_ref is None:
        lat_ref = ds["latitude"].values
        lon_ref = ds["longitude"].values

    clim = bld.groupby("time.month").mean(dim="time").compute()  # (12, lat, lon)
    seas_maps[name] = {}
    for slabel, months in SEASONS:
        # Average over the months that form this season
        month_indices = [m - 1 for m in months]  # 0-based
        seas_maps[name][slabel] = clim.isel(month=month_indices).mean(dim="month").values

# Common vmax across all seasons and configs
all_pos = np.concatenate([
    v[v > 0].ravel()
    for cfg_d in seas_maps.values()
    for v in cfg_d.values()
])
vmax_all = max(20, float(np.nanpercentile(all_pos, 97))) if len(all_pos) else 50

n_seas = len(SEASONS)
n_cfg  = len(CONFIGS)

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(n_seas, n_cfg,
                        hspace=0.10, wspace=0.06,
                        left=0.02, right=0.98,
                        top=0.93, bottom=0.06)

for r, (slabel, _) in enumerate(SEASONS):
    for c, (name, cfg_label, _) in enumerate(CONFIGS):
        ax  = fig.add_subplot(gs[r, c], projection=proj)
        title = cfg_label if r == 0 else ""
        img = plot_bld_panel(ax, lon_ref, lat_ref,
                             seas_maps[name][slabel],
                             vmax_all, title, fig)
        # Season label on the left
        if c == 0:
            ax.set_ylabel(slabel, fontsize=9, labelpad=4)
            ax.text(-0.06, 0.5, slabel, transform=ax.transAxes,
                    fontsize=9, va="center", ha="right", rotation=90)

# One shared colorbar at the bottom
cax = fig.add_axes([0.20, 0.02, 0.60, 0.015])
cb  = fig.colorbar(img, cax=cax, orientation="horizontal")
cb.set_label("BLD (m)  —  white = no barrier layer", fontsize=10)
cb.ax.tick_params(labelsize=9)

fig.suptitle("Seasonal BLD climatology — all 6 configs  (1993–2024)",
             fontsize=14, y=0.97)
out_seas = OUT_DIR / "monthly_clim_seasonal_comparison.png"
fig.savefig(out_seas, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_seas.name}")

print("\nAll done. Outputs:")
for p in sorted(OUT_DIR.glob("monthly_clim_*.png")):
    print(f"  displays/{p.name}")
