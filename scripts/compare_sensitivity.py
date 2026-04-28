"""
compare_sensitivity.py

Compares the 6 ILD detection configurations across the 1993-2024 BLD dataset.
Memory-efficient: never loads all 6 full arrays simultaneously — works on
time-mean maps and regional time series instead.

Outputs (displays/):
  sensitivity_ensemble_maps.png      — mean / spread / detection agreement maps
  sensitivity_regional_timeseries.png — all 6 configs per key region
  sensitivity_global_timeseries.png  — global mean time series, all 6 configs
  sensitivity_stats_table.png        — summary table of key metrics
"""

import sys
import copy
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── BLD colormap: pure white at 0, increasing blue for positive values ────────
BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

# Key regions: (label, lat_min, lat_max, lon_min, lon_max)
REGIONS = [
    ("Tropical Indian Ocean", -20,  20,  40, 100),
    ("Bay of Bengal",          5,  25,  80, 100),
    ("W. Pacific Warm Pool",  -10,  10, 130, 180),
    ("Amazon Plume",           -5,  10, -55, -30),
]

def zarr_path(name):
    return SENS_DIR / f"bld_{name}_1993_2024.zarr"

def open_bld(name):
    return xr.open_zarr(zarr_path(name), consolidated=False)["bld"]

def area_weights(lat):
    w = xr.DataArray(np.cos(np.deg2rad(lat)), dims=["latitude"],
                     coords={"latitude": lat})
    return w

print("Loading time-mean BLD maps for all configs ...")
lat = xr.open_zarr(zarr_path("gradient_015"), consolidated=False)["latitude"].values
lon = xr.open_zarr(zarr_path("gradient_015"), consolidated=False)["longitude"].values
times = pd.DatetimeIndex(
    xr.open_zarr(zarr_path("gradient_015"), consolidated=False)["time"].values
)
weights = area_weights(lat)

# ── 1. Time-mean maps (small: 681×1440 per config) ───────────────────────────
mean_maps   = {}   # config → (lat, lon) array
detect_maps = {}   # config → fraction of months with BLD > 0

for name, label, _ in CONFIGS:
    print(f"  {name} ...", end=" ", flush=True)
    bld = open_bld(name)
    mean_maps[name]   = bld.mean(dim="time").compute().values
    detect_maps[name] = (bld > 0).mean(dim="time").compute().values
    print("done")

# Ensemble statistics across 6 configs
stack       = np.stack(list(mean_maps.values()), axis=0)   # (6, lat, lon)
ens_mean    = np.nanmean(stack, axis=0)
ens_std     = np.nanstd(stack,  axis=0)
detect_all  = np.stack(list(detect_maps.values()), axis=0)
detect_agree = np.mean(detect_all > 0.5, axis=0) * 100     # % configs that detect BLD > 0 most of the time

# ── 2. Global mean time series (per config) ───────────────────────────────────
print("\nComputing global time series ...")
global_ts = {}
for name, label, _ in CONFIGS:
    print(f"  {name} ...", end=" ", flush=True)
    bld = open_bld(name)
    ts  = bld.weighted(weights).mean(dim=["latitude", "longitude"]).compute()
    global_ts[name] = ts.values
    print("done")

# ── 3. Regional time series (per config × region) ────────────────────────────
print("\nComputing regional time series ...")
regional_ts = {name: {} for name, _, _ in CONFIGS}

for name, label, _ in CONFIGS:
    bld = open_bld(name)
    for rname, lat_min, lat_max, lon_min, lon_max in REGIONS:
        reg = bld.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
        w_reg = area_weights(reg["latitude"].values)
        ts    = reg.weighted(w_reg).mean(dim=["latitude", "longitude"]).compute()
        regional_ts[name][rname] = ts.values
    print(f"  {name} done")

# ── Figure 1: Mean BLD / Spread / Agreement  (3 panels) ──────────────────────
print("\nPlotting ensemble maps ...")

proj = ccrs.Robinson()
pc   = ccrs.PlateCarree()

def add_map(ax, fig, data, vmin, vmax, cmap, title, cbar_label, norm=None):
    if norm is not None:
        img = ax.pcolormesh(lon, lat, data, norm=norm,
                            cmap=cmap, transform=pc, rasterized=True)
    else:
        img = ax.pcolormesh(lon, lat, data, vmin=vmin, vmax=vmax,
                            cmap=cmap, transform=pc, rasterized=True)
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=11, pad=6)
    cb = fig.colorbar(img, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.82, aspect=30)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def add_bld_map(ax, fig, data, vmax, title):
    """BLD map: white = 0 (no barrier layer), blues = positive BLD."""
    data_m = np.ma.masked_where(data <= 0, data)
    img = ax.pcolormesh(lon, lat, data_m, vmin=0, vmax=vmax,
                        cmap=BLD_CMAP, transform=pc, rasterized=True)
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=11, pad=6)
    cb = fig.colorbar(img, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.82, aspect=30)
    cb.set_label("BLD (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5),
                            subplot_kw={"projection": proj})
fig1.patch.set_facecolor("white")
fig1.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08,
                     wspace=0.08)

vmax_mean = max(20, float(np.nanpercentile(ens_mean[ens_mean > 0], 95))) if np.any(ens_mean > 0) else 50
add_bld_map(axes1[0], fig1, ens_mean, vmax_mean, "A   Mean BLD")

vmax_std = float(np.nanpercentile(ens_std[np.isfinite(ens_std)], 95))
add_map(axes1[1], fig1, ens_std, 0, vmax_std, "YlOrRd",
        "B   Ensemble spread  σ(BLD)", "σ (m)")

add_map(axes1[2], fig1, detect_agree, 0, 100, "RdYlGn",
        "C   Detection agreement", "% configs with BLD > 0")

fig1.suptitle("Sensitivity analysis — mean BLD, spread and method agreement  (1993–2024)",
              fontsize=12, y=0.97)
out1 = OUT_DIR / "sensitivity_ensemble_maps.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1.name}")

# ── Figure 2: Regional time series ───────────────────────────────────────────
print("Plotting regional time series ...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig.patch.set_facecolor("white")

for ax, (rname, *_) in zip(axes.flatten(), REGIONS):
    for name, label, color in CONFIGS:
        ts = regional_ts[name][rname]
        ax.plot(times, ts, lw=0.8, color=color, alpha=0.7, label=label)
        # 12-month running mean
        rm = pd.Series(ts).rolling(12, center=True).mean().values
        ax.plot(times, rm, lw=2.0, color=color)

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title(rname, fontsize=11)
    ax.set_ylabel("BLD (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

# shared legend
handles = [plt.Line2D([0], [0], color=c, lw=2, label=l)
           for _, l, c in CONFIGS]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.6, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("BLD regional time series — all 6 configs  (1993–2024)", fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])

out2 = OUT_DIR / "sensitivity_regional_timeseries.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2.name}")

# ── Figure 3: Global time series ─────────────────────────────────────────────
print("Plotting global time series ...")

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("white")

for name, label, color in CONFIGS:
    ts = global_ts[name]
    ax.plot(times, ts, lw=0.8, color=color, alpha=0.5)
    rm = pd.Series(ts).rolling(12, center=True).mean().values
    ax.plot(times, rm, lw=2.2, color=color, label=label)

ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Area-weighted global mean BLD (m)", fontsize=10)
ax.set_title("Global mean BLD — all 6 configs  (1993–2024)", fontsize=12)
ax.legend(fontsize=9, framealpha=0.6, ncol=2)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", lw=0.4, alpha=0.4)
plt.tight_layout()

out3 = OUT_DIR / "sensitivity_global_timeseries.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3.name}")

# ── Figure 4: Stats summary table ────────────────────────────────────────────
print("Plotting stats table ...")

rows = []
for name, label, _ in CONFIGS:
    bm   = mean_maps[name]
    ts   = global_ts[name]
    rows.append([
        label,
        f"{np.nanmean(ts):.1f}",
        f"{np.nanmax(bm):.1f}",
        f"{float(np.nanmean(detect_maps[name]) * 100):.1f}",
        f"{np.nanstd(ts):.1f}",
    ])

col_labels = ["Config", "Global\nmean BLD (m)", "Max BLD\n(time-mean, m)",
              "% ocean\nBLD > 0", "Temporal\nstd (m)"]

fig, ax = plt.subplots(figsize=(12, 3.5))
fig.patch.set_facecolor("white")
ax.set_axis_off()

tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

# Header styling
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row colors — gradient = blue tones, difference = green tones
row_colors = ["#dce8f5","#dce8f5","#dce8f5","#d8f0dc","#d8f0dc","#d8f0dc"]
for i, rc in enumerate(row_colors):
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(rc)

fig.suptitle("Sensitivity analysis — key metrics  (1993–2024)", fontsize=12, y=0.98)
plt.tight_layout()

out4 = OUT_DIR / "sensitivity_stats_table.png"
fig.savefig(out4, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4.name}")

print("\nAll done. Outputs:")
for p in [out1, out2, out3, out4]:
    print(f"  displays/{p.name}")
