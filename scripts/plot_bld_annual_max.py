"""
plot_bld_annual_max.py

Reproduces the diagnostics of Pan et al. (2018) Fig. 7 for the full global ocean:

  (a) Annual maximum of the monthly BLD climatology (m)
  (b) Month in which BLD reaches its annual maximum
  (c) Number of months in which BLD > 10 m

Blank / masked areas: grid points where BLD never exceeds 10 m in any month
of the climatology.

Outputs (displays/):
  fig_bld_annual_max_ensemble.png     — 3-panel, ensemble mean
  fig_bld_annual_max_per_method.png   — 3 rows × 6 cols, one column per config
"""

from pathlib import Path
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("gradient_015",  "Gradient −0.015°C/m"),
    ("gradient_025",  "Gradient −0.025°C/m"),
    ("gradient_100",  "Gradient −0.1°C/m"),
    ("difference_02", "Diff ΔT = 0.2°C"),
    ("difference_05", "Diff ΔT = 0.5°C"),
    ("difference_08", "Diff ΔT = 0.8°C"),
]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

proj = ccrs.Robinson()
pc   = ccrs.PlateCarree()


# ── Month colormap (12 discrete colors, circular) ────────────────────────────
_month_colors = plt.cm.twilight_shifted(np.linspace(0, 1, 13)[:-1])
MONTH_CMAP  = ListedColormap(_month_colors, name="months12")
MONTH_NORM  = BoundaryNorm(np.arange(0.5, 13.5, 1), ncolors=12)

# ── N-months colormap: light→dark so persistent regions stand out ─────────────
NMON_LEVELS = np.arange(1, 14)   # 1..12
NMON_CMAP   = plt.cm.get_cmap("Greens", 12)
NMON_NORM   = BoundaryNorm(np.arange(0.5, 13.5, 1), ncolors=12)

# ── BLD max colormap (Blues discrete) ────────────────────────────────────────
MAX_LEVELS = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250]
MAX_CMAP   = plt.cm.get_cmap("Blues", len(MAX_LEVELS) - 1)
MAX_NORM   = BoundaryNorm(MAX_LEVELS, ncolors=MAX_CMAP.N, clip=True)


def load_clim(name):
    """Load monthly climatology (12, lat, lon) for a config, bld > 0 only."""
    bld  = xr.open_zarr(SENS_DIR / f"bld_{name}_1993_2024.zarr",
                        consolidated=False)["bld"]
    bld  = bld.where(bld > 0)
    clim = bld.groupby("time.month").mean("time").compute()   # (12, lat, lon)
    return clim.values, bld.latitude.values, bld.longitude.values


def compute_diagnostics(clim, threshold=0):
    """
    clim      : (12, nlat, nlon) — monthly climatology, NaN where no BL
    threshold : BLD value (m) used to define 'barrier layer present'

    Returns
    -------
    bld_max  : (nlat, nlon)  annual max BLD (m); NaN where never > threshold
    mon_max  : (nlat, nlon)  month of max (1-12); NaN where never > threshold
    n_months : (nlat, nlon)  number of months with BLD > threshold; 0 where none
    """
    n_months = np.sum(clim > threshold, axis=0).astype(float)
    never_above = n_months == 0

    # Annual max
    bld_max = np.nanmax(clim, axis=0)
    bld_max[never_above] = np.nan

    # Month of max (1-based); use nanargmax → 0-based, +1
    with np.errstate(all="ignore"):
        mon_max = np.full(bld_max.shape, np.nan)
        valid   = ~never_above & np.any(np.isfinite(clim), axis=0)
        clim_masked = np.where(np.isfinite(clim), clim, -np.inf)
        mon_max[valid] = np.nanargmax(clim_masked[:, valid], axis=0) + 1

    return bld_max, mon_max, n_months


def add_map(ax, lon, lat, data, norm, cmap, title, mask_zero=False):
    """Single map panel."""
    d = np.ma.masked_invalid(data)
    if mask_zero:
        d = np.ma.masked_where(d == 0, d)
    ax.pcolormesh(lon, lat, d, norm=norm, cmap=cmap,
                  transform=pc, rasterized=True)
    ax.add_feature(cfeature.LAND, facecolor="#d4d4d4", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=9, pad=3, loc="left")


def colorbars_row(fig, axes_row, imgs, labels):
    for ax, img, lbl in zip(axes_row, imgs, labels):
        cb = fig.colorbar(img, ax=ax, orientation="horizontal",
                          pad=0.04, shrink=0.80, aspect=30)
        cb.set_label(lbl, fontsize=7)
        cb.ax.tick_params(labelsize=6)


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — ensemble mean (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing ensemble mean climatology...")
clim_stack = []
lat = lon = None
for name, _ in CONFIGS:
    print(f"  {name}...", end=" ", flush=True)
    c, lat, lon = load_clim(name)
    clim_stack.append(c)
    print("ok")

ens_clim = np.nanmean(np.stack(clim_stack, axis=0), axis=0)   # (12, nlat, nlon)
bld_max, mon_max, n_months = compute_diagnostics(ens_clim)

fig, axes = plt.subplots(1, 3, figsize=(17, 5),
                         subplot_kw={"projection": proj})
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.01, right=0.99, top=0.93,
                    bottom=0.12, wspace=0.05)

# Dummy ScalarMappables for colorbars
sm_max  = plt.cm.ScalarMappable(norm=MAX_NORM,   cmap=MAX_CMAP)
sm_mon  = plt.cm.ScalarMappable(norm=MONTH_NORM, cmap=MONTH_CMAP)
sm_nmon = plt.cm.ScalarMappable(norm=NMON_NORM,  cmap=NMON_CMAP)

add_map(axes[0], lon, lat, bld_max,   MAX_NORM,   MAX_CMAP,
        "a)  Annual max BLD (m)")
add_map(axes[1], lon, lat, mon_max,   MONTH_NORM, MONTH_CMAP,
        "b)  Month of BLD maximum")
add_map(axes[2], lon, lat, n_months,  NMON_NORM,  NMON_CMAP,
        "c)  Months with BLD > 10 m", mask_zero=True)

# Colorbars
cb0 = fig.colorbar(sm_max,  ax=axes[0], orientation="horizontal",
                   pad=0.04, shrink=0.8, aspect=30)
cb0.set_label("BLD (m)", fontsize=8)
cb0.set_ticks(MAX_LEVELS)
cb0.ax.tick_params(labelsize=7)

cb1 = fig.colorbar(sm_mon,  ax=axes[1], orientation="horizontal",
                   pad=0.04, shrink=0.8, aspect=30)
cb1.set_label("Month", fontsize=8)
cb1.set_ticks(np.arange(1, 13))
cb1.set_ticklabels(MONTH_NAMES)
cb1.ax.tick_params(labelsize=6, rotation=45)

cb2 = fig.colorbar(sm_nmon, ax=axes[2], orientation="horizontal",
                   pad=0.04, shrink=0.8, aspect=30)
cb2.set_label("N months", fontsize=8)
cb2.set_ticks(np.arange(1, 13))
cb2.ax.tick_params(labelsize=7)

fig.suptitle("Annual BLD diagnostics — ensemble mean (1993–2024)\n"
             "Blank = BLD never exceeds 10 m",
             fontsize=11, fontweight="bold")

out = OUT_DIR / "fig_bld_annual_max_ensemble.png"
fig.savefig(out, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — per method (3 rows × 6 cols)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding per-method figure...")

diags = {}
for i, (name, label) in enumerate(CONFIGS):
    c = clim_stack[i]
    diags[name] = compute_diagnostics(c, threshold=0)   # BLD > 0

# N-months colormap for BLD > 0: can reach 12 months
NMON_CMAP2 = plt.cm.get_cmap("Greens", 12)
NMON_NORM2 = BoundaryNorm(np.arange(0.5, 13.5, 1), ncolors=12)
sm_nmon2   = plt.cm.ScalarMappable(norm=NMON_NORM2, cmap=NMON_CMAP2)

ROW_LABELS = ["Annual max BLD (m)",
              "Month of BLD maximum",
              "Months with BLD > 0"]
ROW_NORMS  = [MAX_NORM,   MONTH_NORM, NMON_NORM2]
ROW_CMAPS  = [MAX_CMAP,   MONTH_CMAP, NMON_CMAP2]
ROW_MASK0  = [False,      False,      True]
ROW_SM     = [sm_max,     sm_mon,     sm_nmon2]
ROW_TICKS  = [MAX_LEVELS, np.arange(1, 13), np.arange(1, 13)]
ROW_TKLBL  = [None,       MONTH_NAMES,      None]
ROW_UNITS  = ["BLD (m)",  "Month",          "N months"]

fig2, axes2 = plt.subplots(3, 6, figsize=(22, 10),
                           subplot_kw={"projection": proj})
fig2.patch.set_facecolor("white")
fig2.subplots_adjust(left=0.06, right=0.94, top=0.96,
                     bottom=0.01, hspace=0.04, wspace=0.04)

for ci, (name, label) in enumerate(CONFIGS):
    bm, mm, nm = diags[name]
    data_rows  = [bm, mm, nm]
    for ri in range(3):
        ax = axes2[ri, ci]
        add_map(ax, lon, lat, data_rows[ri],
                ROW_NORMS[ri], ROW_CMAPS[ri],
                label if ri == 0 else "",
                mask_zero=ROW_MASK0[ri])
        if ci == 0:
            ax.text(-0.04, 0.5, ROW_LABELS[ri], transform=ax.transAxes,
                    fontsize=9, va="center", ha="right", rotation=90,
                    fontweight="bold")

# Colorbars: manual positioning after layout is fixed
fig2.canvas.draw()
for ri in range(3):
    ax0 = axes2[ri, -1]           # rightmost axes in this row
    pos = ax0.get_position()      # (x0, y0, width, height) in fig coords
    cbar_height = pos.height * 0.85
    cbar_y0     = pos.y0 + (pos.height - cbar_height) / 2
    cax = fig2.add_axes([pos.x1 + 0.005, cbar_y0, 0.012, cbar_height])
    cb  = fig2.colorbar(ROW_SM[ri], cax=cax, orientation="vertical")
    cb.set_label(ROW_UNITS[ri], fontsize=8)
    cb.set_ticks(ROW_TICKS[ri])
    if ROW_TKLBL[ri]:
        cb.set_ticklabels(ROW_TKLBL[ri])
    cb.ax.tick_params(labelsize=7)

fig2.suptitle("Annual BLD diagnostics per detection method (1993–2024)\n"
              "Blank = BLD never detected",
              fontsize=11, fontweight="bold")

out2 = OUT_DIR / "fig_bld_annual_max_per_method.png"
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2.name}")

print("\nDone.")
