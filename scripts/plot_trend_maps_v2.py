"""
plot_trend_maps_v2.py

Generates 4 trend map PNGs for two ILD configs (difference_02, gradient_025):
  - {label}_all.png     : all pixels, non-significant shown washed out (white overlay)
  - {label}_sig.png     : only significant pixels (p < 0.05), rest masked white
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT      = ROOT / "displays"

CONFIGS = [
    ("difference_02", "Diferencia ΔT = 0.2°C",     12),
    ("gradient_025",  "Gradiente −0.025°C/m",        8),
]

def compute_trend(zarr_f):
    ds  = xr.open_zarr(zarr_f)
    bld = ds["bld"].where(ds["bld"] > 0)
    ann = bld.resample(time="YE").mean("time").compute()
    arr  = ann.values
    nt, nlat, nlon = arr.shape
    x    = np.arange(nt, dtype=float)
    lats = ann.latitude.values
    lons = ann.longitude.values
    trend = np.full((nlat, nlon), np.nan)
    pval  = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            yv = arr[:, i, j]
            vm = np.isfinite(yv)
            if vm.sum() < 5:
                continue
            s, _, _, p, _ = linregress(x[vm], yv[vm])
            trend[i, j] = s * 10
            pval[i, j]  = p
    return trend, pval, lats, lons

def make_plot(trend_data, mask_sig, lats, lons, vlim, title, mode, outpath):
    """
    mode='all'  : show all trends, non-significant areas overlaid with transparent white
    mode='sig'  : show only significant trends, rest is white (NaN)
    """
    proj = ccrs.Robinson()
    pc   = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(13, 6),
                           subplot_kw={"projection": proj})
    ax.set_global()
    ax.add_feature(cfeature.LAND,      color="#d4d4d4", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,   zorder=4)

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    LON2, LAT2 = np.meshgrid(lons, lats)

    if mode == "all":
        # Draw all trends
        ax.pcolormesh(LON2, LAT2, trend_data, cmap=cmap, norm=norm,
                      transform=pc, zorder=1)
        # Overlay non-significant areas with semi-transparent white
        alpha_mask = np.where(mask_sig, 0.55, 0.0).astype(float)
        white_rgba = np.ones((*trend_data.shape, 4))
        white_rgba[..., 3] = alpha_mask
        ax.imshow(white_rgba,
                  extent=[-180, 180, -90, 90],
                  origin="lower",
                  transform=pc,
                  interpolation="nearest",
                  zorder=2)
        subtitle = "(áreas lavadas = p ≥ 0.05)"

    else:  # mode == "sig"
        # Only significant pixels
        trend_sig = np.where(~mask_sig, trend_data, np.nan)
        ax.pcolormesh(LON2, LAT2, trend_sig, cmap=cmap, norm=norm,
                      transform=pc, zorder=1)
        subtitle = "(solo píxeles con p < 0.05)"

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                        pad=0.04, shrink=0.65, aspect=32)
    cbar.set_label("Tendencia BLD (m/década)", fontsize=11)
    tick_step = 2 if vlim <= 8 else 4
    cbar.set_ticks(np.arange(-vlim, vlim + tick_step, tick_step))
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(f"Tendencia BLD 1993–2024 — {title}\n{subtitle}",
                 fontsize=12, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath.name}")


for label, title, vlim in CONFIGS:
    print(f"\n{label}:")
    zarr_f = next(f for f in SENS_DIR.iterdir() if label in f.name)
    trend, pval, lats, lons = compute_trend(zarr_f)
    print(f"  Computed trends  (vlim = ±{vlim} m/dec)")
    mask_nonsig = (pval >= 0.05) & np.isfinite(pval)  # True where NOT significant

    make_plot(trend, mask_nonsig, lats, lons, vlim, title,
              mode="all",
              outpath=OUT / f"bld_trend_map_{label}_all.png")

    make_plot(trend, mask_nonsig, lats, lons, vlim, title,
              mode="sig",
              outpath=OUT / f"bld_trend_map_{label}_sig.png")

print("\nAll 4 figures done.")
