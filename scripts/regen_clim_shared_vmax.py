"""
regen_clim_shared_vmax.py

Regenerates monthly_clim_difference_02.png and monthly_clim_gradient_025.png
with a SHARED colorbar vmax so the two slides are directly comparable.

Shared vmax = 98th percentile of positive BLD across both configs combined.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_TO_PLOT = [
    ("difference_02", "Diff 0.2°C"),
    ("gradient_025",  "Gradient −0.025°C/m"),
]

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

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
    data_m = np.ma.masked_where(data <= 0, data)
    img = ax.pcolormesh(lon, lat, data_m, vmin=0, vmax=vmax,
                        cmap=BLD_CMAP, transform=pc, rasterized=True)
    ax.add_feature(cfeature.LAND,      facecolor="#d4d4d4", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=9, pad=4)
    return img


# ── Step 1: compute shared vmax ───────────────────────────────────────────────
print("Computing shared vmax across difference_02 and gradient_025 ...")

all_pos = []
clims = {}

for name, label in CONFIGS_TO_PLOT:
    print(f"  Loading {name} ...", end=" ", flush=True)
    ds  = xr.open_zarr(zarr_path(name), consolidated=False)
    bld = ds["bld"]
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    clim = bld.groupby("time.month").mean(dim="time").compute()  # (12, lat, lon)
    clims[name] = (clim, lat, lon)

    pos_vals = clim.values[clim.values > 0]
    all_pos.append(pos_vals)
    print(f"done  (n_pos={len(pos_vals):,})")

all_pos_combined = np.concatenate(all_pos)
shared_vmax = float(np.nanpercentile(all_pos_combined, 98))
shared_vmax = max(20.0, shared_vmax)
print(f"\nShared vmax (98th pct) = {shared_vmax:.1f} m")


# ── Step 2: regenerate both figures ──────────────────────────────────────────
for name, label in CONFIGS_TO_PLOT:
    print(f"\nPlotting {name} ...")
    clim, lat, lon = clims[name]

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(4, 3, hspace=0.25, wspace=0.08,
                            left=0.02, right=0.98, top=0.93, bottom=0.04)

    for m in range(12):
        row, col = divmod(m, 3)
        ax  = fig.add_subplot(gs[row, col], projection=proj)
        img = plot_bld_panel(ax, lon, lat, clim.isel(month=m).values,
                             shared_vmax, MONTH_LABELS[m], fig)

    # One shared colorbar — with explicit ticks so both figures show identical scale
    import matplotlib.ticker as mticker
    cax = fig.add_axes([0.25, 0.015, 0.5, 0.012])
    cb  = fig.colorbar(img, cax=cax, orientation="horizontal")
    cb.set_label("BLD (m)  —  white = no barrier layer", fontsize=9)
    # Force ticks to cover the full shared range so both colorbars look identical
    tick_step = 20 if shared_vmax <= 100 else 25
    cb.set_ticks([t for t in range(0, int(shared_vmax) + 1, tick_step)])
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Monthly BLD climatology  |  {label}  (1993–2024)",
                 fontsize=13, y=0.97)

    out = OUT_DIR / f"monthly_clim_{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")

print("\nDone. Both figures use shared vmax =", round(shared_vmax, 1), "m")
