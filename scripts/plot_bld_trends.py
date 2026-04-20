"""
plot_bld_trends.py

Per-pixel linear trend in BLD (1993–2024) using annual means.
Uses the default ILD config (gradient −0.1 °C/m) from the sensitivity ensemble,
plus an ensemble-mean trend derived from all 6 configs.

Outputs (displays/):
  bld_trend_map.png          — global trend map (m/decade) with significance mask
  bld_trend_regions.png      — annual mean time series + trend lines for 4 key regions
  bld_trend_ensemble.png     — ensemble spread of per-pixel trends across all 6 configs
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("gradient_015",  "Gradient −0.015°C/m", "#1a3a6b"),
    ("gradient_025",  "Gradient −0.025°C/m", "#2e6fad"),
    ("gradient_100",  "Gradient −0.1°C/m",   "#7eb8d4"),
    ("difference_02", "Diff ΔT 0.2°C",        "#1a5c2a"),
    ("difference_05", "Diff ΔT 0.5°C",        "#4aab5e"),
    ("difference_08", "Diff ΔT 0.8°C",        "#a8dbb0"),
]

DEFAULT_CONFIG = "gradient_100"

REGIONS = [
    ("Tropical Indian Ocean", -20,  20,  40, 100),
    ("Bay of Bengal",           5,  25,  80, 100),
    ("W. Pacific Warm Pool",  -10,  10, 130, 180),
    ("Amazon Plume",           -5,  10, -55, -30),
]


def zarr_path(name):
    for f in SENS_DIR.iterdir():
        if name in f.name:
            return f
    raise FileNotFoundError(f"No sensitivity zarr found for '{name}' in {SENS_DIR}")


def annual_mean(ds):
    """Monthly → annual mean BLD, masking negatives first."""
    bld = ds["bld"].where(ds["bld"] > 0)
    return bld.resample(time="YE").mean("time")


def pixel_trend(arr_2d_time_lat_lon):
    """
    arr_2d_time_lat_lon : (n_years, nlat, nlon) numpy array
    Returns slope (m/yr), intercept, p-value — all (nlat, nlon).
    """
    nt, nlat, nlon = arr_2d_time_lat_lon.shape
    x = np.arange(nt, dtype=float)
    slope = np.full((nlat, nlon), np.nan)
    pval  = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            y = arr_2d_time_lat_lon[:, i, j]
            valid = np.isfinite(y)
            if valid.sum() < 5:
                continue
            s, _, _, p, _ = linregress(x[valid], y[valid])
            slope[i, j] = s
            pval[i, j]  = p
    return slope, pval


# ── Figure 1: Default config trend map ────────────────────────────────────────

print("Loading default config (gradient −0.1°C/m)…")
ds_default = xr.open_zarr(zarr_path(DEFAULT_CONFIG))
ann_default = annual_mean(ds_default).compute()   # (32, nlat, nlon)

lats = ann_default.latitude.values
lons = ann_default.longitude.values
years = ann_default.time.dt.year.values  # 1993–2024

print("Computing per-pixel trend…")
slope_m_yr, pval = pixel_trend(ann_default.values)
slope_m_dec = slope_m_yr * 10.0   # convert to m/decade

sig_mask = pval < 0.05             # significant pixels

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.03], wspace=0.04)
ax  = fig.add_subplot(gs[0], projection=ccrs.Robinson())
cax = fig.add_subplot(gs[1])

ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")

vmax = np.nanpercentile(np.abs(slope_m_dec), 98)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r

# All pixels — full alpha
im = ax.pcolormesh(
    lons, lats, slope_m_dec,
    norm=norm, cmap=cmap,
    transform=ccrs.PlateCarree(),
    rasterized=True,
)
# Stippling for non-significant pixels
insig_lat, insig_lon = np.where(~sig_mask & np.isfinite(slope_m_dec))
if insig_lat.size > 0:
    step = max(1, insig_lat.size // 8000)   # thin stippling
    ax.scatter(
        lons[insig_lon[::step]], lats[insig_lat[::step]],
        s=0.3, color="k", alpha=0.25, transform=ccrs.PlateCarree(),
        zorder=5,
    )

cb = plt.colorbar(im, cax=cax)
cb.set_label("BLD trend (m decade⁻¹)", fontsize=11)
ax.set_title(
    "Barrier Layer Depth — Linear Trend 1993–2024\n"
    f"ILD method: Gradient −0.1°C/m  |  stippling = p ≥ 0.05",
    fontsize=12,
)

pct_sig = 100 * sig_mask[np.isfinite(slope_m_dec)].mean()
pct_pos = 100 * (slope_m_dec[np.isfinite(slope_m_dec)] > 0).mean()
fig.text(0.12, 0.04,
         f"Significant (p<0.05): {pct_sig:.0f}% of ocean  |  "
         f"Positive trend: {pct_pos:.0f}% of ocean",
         fontsize=9, color="#444")

out1 = OUT_DIR / "bld_trend_map.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out1}")


# ── Figure 2: Regional time series + trend lines ───────────────────────────────

print("\nBuilding regional trend time series…")

fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for ax, (label, lat0, lat1, lon0, lon1) in zip(axes, REGIONS):
    # Mask longitude: handle dateline wrap
    da = ann_default
    if lon0 < 0 and lon1 < 0:
        lon0 += 360; lon1 += 360
    da_reg = da.sel(
        latitude=slice(lat0, lat1),
        longitude=slice(lon0, lon1),
    ).mean(["latitude", "longitude"])
    y = da_reg.values
    x = np.arange(len(years), dtype=float)

    valid = np.isfinite(y)
    ax.plot(years, y, "o-", ms=4, color="#2171b5", lw=1.5, label="Annual mean BLD")

    if valid.sum() >= 5:
        s, intercept, r, p, se = linregress(x[valid], y[valid])
        trend_line = intercept + s * x
        trend_label = (
            f"Trend: {s*10:+.2f} m/decade"
            + ("*" if p < 0.05 else "")
            + f"  (R={r:.2f})"
        )
        ax.plot(years, trend_line, "--", color="#d94701", lw=1.8, label=trend_label)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Mean BLD (m)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig2.suptitle(
    "Regional Annual Mean BLD with Linear Trend (1993–2024)\n"
    "ILD method: Gradient −0.1°C/m",
    fontsize=13, fontweight="bold",
)
fig2.tight_layout()

out2 = OUT_DIR / "bld_trend_regions.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2}")


# ── Figure 3: Ensemble spread of trends (all 6 configs) ─────────────────────

print("\nComputing ensemble trend spread across all 6 configs…")

trend_stack = []
for name, clabel, color in CONFIGS:
    ds = xr.open_zarr(zarr_path(name))
    ann = annual_mean(ds).compute()
    s, p = pixel_trend(ann.values)
    trend_stack.append(s * 10.0)   # m/decade
    print(f"  {name} done")

trend_stack = np.array(trend_stack)   # (6, nlat, nlon)
trend_ensmean = np.nanmean(trend_stack, axis=0)
trend_ensstd  = np.nanstd(trend_stack, axis=0)

fig3, axes3 = plt.subplots(
    1, 2, figsize=(18, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

vmax3 = np.nanpercentile(np.abs(trend_ensmean), 98)
norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)

for ax3, data, title, cmap3 in zip(
    axes3,
    [trend_ensmean, trend_ensstd],
    ["Ensemble mean trend (m decade⁻¹)", "Ensemble std of trend (m decade⁻¹)"],
    [plt.cm.RdBu_r,
     LinearSegmentedColormap.from_list("spread", ["white", "#fc8d59", "#b30000"], N=256)],
):
    ax3.set_global()
    ax3.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    ax3.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")

    if title.startswith("Ensemble mean"):
        norm_use = norm3
        vmin_use, vmax_use = -vmax3, vmax3
    else:
        norm_use = None
        vmin_use = 0
        vmax_use = np.nanpercentile(trend_ensstd, 98)

    im3 = ax3.pcolormesh(
        lons, lats, data,
        norm=norm_use, vmin=vmin_use if norm_use is None else None,
        vmax=vmax_use if norm_use is None else None,
        cmap=cmap3,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    plt.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.04, shrink=0.7, label=title)
    ax3.set_title(title, fontsize=12)

fig3.suptitle(
    "BLD Trend Ensemble — All 6 ILD Configurations (1993–2024)\n"
    "Left: mean trend across configs  |  Right: spread (std) — uncertainty due to method choice",
    fontsize=12, fontweight="bold",
)
fig3.tight_layout()

out3 = OUT_DIR / "bld_trend_ensemble.png"
fig3.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"Saved → {out3}")

print("\nDone.")
