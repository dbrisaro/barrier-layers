"""
plot_ild_mld_bld_trends.py

Trend maps for ILD, MLD and BLD (1993-2024) from use_abs=True zarrs.
BLD = ILD - MLD decomposition + per-method ILD trend comparison.

Outputs  (displays/):
  trends_ild_mld_bld_global.png    — 3-panel: ILD / MLD / BLD global trends
  trends_ild_mld_bld_regional.png  — regional time series (5 regions × 3 variables)
  trends_ild_ensemble.png          — ensemble mean + spread (ILD, all 6 configs)
  trends_ild_per_method.png        — 6-panel ILD trend map, one per method
"""
import warnings, sys
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
from scipy.stats import t as t_dist, linregress as scipy_linreg
from matplotlib.colors import LinearSegmentedColormap
import shapefile as shp

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data/025deg/sensitivity"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 9, "axes.titlesize": 9,
})

NE_LAND  = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"
NE_COAST = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"

DEFAULT_CONFIG = "gradient_100"

CONFIGS = [
    ("gradient_015",  "Gradient −0.015°C/m"),
    ("gradient_025",  "Gradient −0.025°C/m"),
    ("gradient_100",  "Gradient −0.1°C/m"),
    ("difference_02", "Difference ΔT 0.2°C"),
    ("difference_05", "Difference ΔT 0.5°C"),
    ("difference_08", "Difference ΔT 0.8°C"),
]

REGIONS = [
    dict(label="Tropical Atlantic",    lat0=-5,  lat1=10,  lon0=-55, lon1=-30, color="#e74c3c"),
    dict(label="Bay of Bengal",        lat0=5,   lat1=25,  lon0=80,  lon1=100, color="#8e44ad"),
    dict(label="W. Pacific Warm Pool", lat0=-10, lat1=10,  lon0=130, lon1=180, color="#e67e22"),
    dict(label="N. Argentine Shelf",   lat0=-45, lat1=-30, lon0=-65, lon1=-50, color="#16a085"),
    dict(label="Southern Ocean",       lat0=-65, lat1=-45, lon0=20,  lon1=100, color="#2980b9"),
]

YEARS = np.arange(1993, 2025)

# ── helpers ───────────────────────────────────────────────────────────────────
def draw_land_coast(ax):
    if NE_LAND.exists():
        sf = shp.Reader(str(NE_LAND))
        patches = []
        for shape in sf.shapes():
            if shape.shapeType == 0: continue
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                patches.append(plt.Polygon(pts[parts[k]:parts[k+1]], closed=True))
        ax.add_collection(mcollections.PatchCollection(
            patches, facecolor="lightgray", edgecolor="none", zorder=2))
    if NE_COAST.exists():
        sf = shp.Reader(str(NE_COAST))
        for shape in sf.shapes():
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                ax.plot(seg[:,0], seg[:,1], lw=0.4, color="#555", zorder=3)

def style_map(ax):
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

def vect_linregress(arr_tyx, x):
    """Vectorised per-pixel OLS — returns slope (m/yr), p-value."""
    xm   = x.mean(); Sxx = np.sum((x - xm)**2)
    ym   = np.nanmean(arr_tyx, axis=0)
    Sxy  = np.nansum((arr_tyx - ym[None]) * (x - xm)[:, None, None], axis=0)
    slope = Sxy / Sxx
    inter = ym - slope * xm
    yhat  = inter[None] + slope[None] * x[:, None, None]
    resid = arr_tyx - yhat
    n     = np.sum(np.isfinite(arr_tyx), axis=0).astype(float)
    SSres = np.nansum(resid**2, axis=0)
    se    = np.sqrt(SSres / np.where(n > 2, n - 2, np.nan) / Sxx)
    tstat = slope / np.where(se > 0, se, np.nan)
    df    = np.where(n > 2, n - 2, 1).astype(float)
    pval  = 2 * t_dist.sf(np.abs(tstat), df=df)
    slope[n < 5] = np.nan; pval[n < 5] = np.nan
    return slope, pval

def load_annual_var(cname, varname, positive_only=False):
    """Load zarr variable → annual mean (32, nlat, nlon)."""
    print(f"    {cname}/{varname} …", end="", flush=True)
    z    = zarr.open(SENS_DIR / f"bld_{cname}_1993_2024.zarr", mode="r")
    nt   = z[varname].shape[0]
    nlat, nlon = z[varname].shape[1], z[varname].shape[2]
    ann  = np.full((nt // 12, nlat, nlon), np.nan, dtype=np.float32)
    for yi in range(nt // 12):
        t0    = yi * 12
        chunk = z[varname][t0:t0+12][:]
        if positive_only:
            chunk = np.where(chunk > 0, chunk, np.nan)
        ann[yi] = np.nanmean(chunk, axis=0)
        if yi % 8 == 0: print(".", end="", flush=True)
    print(" ok")
    return ann

def region_ts(ann_tyx, lat, lon, lat0, lat1, lon0, lon1):
    ilat = np.where((lat >= lat0) & (lat <= lat1))[0]
    ilon = np.where((lon >= lon0) & (lon <= lon1))[0]
    if ilat.size == 0 or ilon.size == 0:
        return np.full(ann_tyx.shape[0], np.nan)
    sub = ann_tyx[:, ilat[:, None], ilon[None, :]]
    return np.nanmean(sub.reshape(ann_tyx.shape[0], -1), axis=1)

# ── Load lat/lon ──────────────────────────────────────────────────────────────
ref = zarr.open(SENS_DIR / "bld_gradient_015_1993_2024.zarr", mode="r")
lat = ref["latitude"][:]
lon = ref["longitude"][:]
x   = np.arange(YEARS.size, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 3-panel global: ILD / MLD / BLD trends (default config)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 1: ILD / MLD / BLD global trend maps ===")
print(f"  Loading {DEFAULT_CONFIG} …")

ann_ild = load_annual_var(DEFAULT_CONFIG, "ild", positive_only=True)
ann_mld = load_annual_var(DEFAULT_CONFIG, "mld", positive_only=False)
ann_bld = load_annual_var(DEFAULT_CONFIG, "bld", positive_only=True)

print("  Computing trends …", end="", flush=True)
slope_ild, pval_ild = vect_linregress(ann_ild.astype(np.float64), x)
slope_mld, pval_mld = vect_linregress(ann_mld.astype(np.float64), x)
slope_bld, pval_bld = vect_linregress(ann_bld.astype(np.float64), x)
trend_ild = slope_ild * 10
trend_mld = slope_mld * 10
trend_bld = slope_bld * 10
print(" done")

vmax_ib = np.nanpercentile(
    np.abs(np.concatenate([trend_ild[np.isfinite(trend_ild)],
                           trend_bld[np.isfinite(trend_bld)]])), 98)
vmax_m  = np.nanpercentile(np.abs(trend_mld[np.isfinite(trend_mld)]), 98)
norm_ib = mcolors.TwoSlopeNorm(vmin=-vmax_ib, vcenter=0, vmax=vmax_ib)
norm_m  = mcolors.TwoSlopeNorm(vmin=-vmax_m,  vcenter=0, vmax=vmax_m)

fig1, axes1 = plt.subplots(3, 1, figsize=(14, 16))
fig1.patch.set_facecolor("white")
panels = [
    (axes1[0], trend_ild, pval_ild, norm_ib, "ILD trend (m decade⁻¹)",
     "A   Isothermal Layer Depth (ILD) — trend 1993–2024"),
    (axes1[1], trend_mld, pval_mld, norm_m,  "MLD trend (m decade⁻¹)",
     "B   Mixed Layer Depth (MLD) — trend 1993–2024"),
    (axes1[2], trend_bld, pval_bld, norm_ib, "BLD trend (m decade⁻¹)",
     "C   Barrier Layer Depth (BLD = ILD − MLD) — trend 1993–2024"),
]
for ax, data, pv, norm, cblabel, title in panels:
    im = ax.pcolormesh(lon, lat, data, norm=norm, cmap="RdBu_r", rasterized=True)
    draw_land_coast(ax); style_map(ax)
    insig_i, insig_j = np.where((pv >= 0.05) & np.isfinite(data))
    step = max(1, insig_i.size // 5000)
    if insig_i.size > 0:
        ax.scatter(lon[insig_j[::step]], lat[insig_i[::step]],
                   s=0.4, color="k", alpha=0.3, zorder=4)
    cb = fig1.colorbar(im, ax=ax, pad=0.01, shrink=0.7, aspect=25)
    cb.set_label(cblabel, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    pct_sig = 100 * np.nanmean((pv < 0.05)[np.isfinite(data)])
    pct_pos = 100 * np.nanmean(data[np.isfinite(data)] > 0)
    ax.text(0.01, 0.04, f"sig: {pct_sig:.0f}%  |  positive: {pct_pos:.0f}%",
            transform=ax.transAxes, fontsize=7, color="#444", zorder=6)

plt.subplots_adjust(hspace=0.12)
out1 = OUT_DIR / "trends_ild_mld_bld_global.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {out1.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Regional time series: ILD / MLD / BLD per region
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 2: Regional time series ===")
col_colors = {"ILD": "#c0392b", "MLD": "#2471a3", "BLD": "#1e8449"}
col_vars   = [("ILD", ann_ild), ("MLD", ann_mld), ("BLD", ann_bld)]

fig2, axes2 = plt.subplots(len(REGIONS), 3, figsize=(18, 4*len(REGIONS)))
fig2.patch.set_facecolor("white")
for ri, r in enumerate(REGIONS):
    for ci, (vname, ann) in enumerate(col_vars):
        ax = axes2[ri, ci]
        ts = region_ts(ann, lat, lon, r["lat0"], r["lat1"], r["lon0"], r["lon1"])
        ax.plot(YEARS, ts, "o-", ms=3, lw=1.4, color=col_colors[vname], alpha=0.85)
        valid = np.isfinite(ts)
        if valid.sum() >= 5:
            s, intr, rv, pv, _ = scipy_linreg(x[valid], ts[valid])
            ax.plot(YEARS, intr + s * x, "--", lw=1.8, color="k",
                    label=f"{s*10:+.2f} m/dec{'*' if pv<0.05 else ''}  R={rv:.2f}")
            ax.legend(fontsize=7, frameon=False)
        if ri == 0:
            ax.set_title(vname, fontsize=10, fontweight="bold", color=col_colors[vname])
        if ci == 0:
            ax.set_ylabel(f"{r['label']}\nDepth (m)", fontsize=8)
        else:
            ax.set_ylabel("Depth (m)", fontsize=8)
        ax.set_xlabel("Year" if ri == len(REGIONS)-1 else "", fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(r["color"]); sp.set_linewidth(1.5)

fig2.suptitle("Regional Annual Mean — ILD, MLD, BLD  (ILD method: Gradient −0.1°C/m)",
              fontsize=10, fontweight="bold")
plt.subplots_adjust(hspace=0.35, wspace=0.25)
out2 = OUT_DIR / "trends_ild_mld_bld_regional.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {out2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — ILD trend per method (2×3 panels, shared colorscale)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 3: ILD trend per method (2×3) ===")

ild_trends  = []
ild_pvals   = []

for cname, clabel in CONFIGS:
    print(f"  {cname}:")
    ann = load_annual_var(cname, "ild", positive_only=True)
    s_yr, pv = vect_linregress(ann.astype(np.float64), x)
    ild_trends.append(s_yr * 10.0)
    ild_pvals.append(pv)

# shared symmetric colorscale across all 6 methods
all_vals = np.concatenate([t[np.isfinite(t)] for t in ild_trends])
vmax_pm = np.nanpercentile(np.abs(all_vals), 98)
norm_pm = mcolors.TwoSlopeNorm(vmin=-vmax_pm, vcenter=0, vmax=vmax_pm)

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 9))
fig3.patch.set_facecolor("white")

for ax, (cname, clabel), trend, pv in zip(axes3.flat, CONFIGS, ild_trends, ild_pvals):
    im = ax.pcolormesh(lon, lat, trend, norm=norm_pm, cmap="RdBu_r", rasterized=True)
    draw_land_coast(ax); style_map(ax)
    # stipple non-significant
    insig_i, insig_j = np.where((pv >= 0.05) & np.isfinite(trend))
    step = max(1, insig_i.size // 4000)
    if insig_i.size > 0:
        ax.scatter(lon[insig_j[::step]], lat[insig_i[::step]],
                   s=0.4, color="k", alpha=0.3, zorder=4)
    pct_sig = 100 * np.nanmean((pv < 0.05)[np.isfinite(trend)])
    pct_pos = 100 * np.nanmean(trend[np.isfinite(trend)] > 0)
    ax.set_title(clabel, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.04, f"sig: {pct_sig:.0f}%  pos: {pct_pos:.0f}%",
            transform=ax.transAxes, fontsize=7, color="#333", zorder=5)

# shared colorbar
cb3 = fig3.colorbar(
    plt.cm.ScalarMappable(norm=norm_pm, cmap="RdBu_r"),
    ax=axes3, location="bottom", shrink=0.45, pad=0.04, aspect=40,
)
cb3.set_label("ILD trend (m decade⁻¹)", fontsize=9)
cb3.ax.tick_params(labelsize=8)

fig3.suptitle("ILD Linear Trend 1993–2024 — one panel per detection method\n"
              "stippling = p ≥ 0.05  |  shared color scale",
              fontsize=10, fontweight="bold")
plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.10,
                    hspace=0.08, wspace=0.04)
out3 = OUT_DIR / "trends_ild_per_method.png"
fig3.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved → {out3.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — ILD ensemble: mean + spread
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 4: ILD ensemble trend ===")

ild_trend_stack = np.array(ild_trends)            # (6, nlat, nlon)
ild_trend_mean  = np.nanmean(ild_trend_stack, axis=0)
ild_trend_std   = np.nanstd( ild_trend_stack, axis=0)

vmax_im = np.nanpercentile(np.abs(ild_trend_mean[np.isfinite(ild_trend_mean)]), 98)
vmax_is = np.nanpercentile(ild_trend_std[np.isfinite(ild_trend_std)], 97)
cmap_spread = LinearSegmentedColormap.from_list(
    "spread", ["white", "#fc8d59", "#b30000"], N=256)

fig4, axes4 = plt.subplots(1, 2, figsize=(18, 5))
fig4.patch.set_facecolor("white")

norm_im = mcolors.TwoSlopeNorm(vmin=-vmax_im, vcenter=0, vmax=vmax_im)
im4a = axes4[0].pcolormesh(lon, lat, ild_trend_mean, norm=norm_im,
                            cmap="RdBu_r", rasterized=True)
draw_land_coast(axes4[0]); style_map(axes4[0])
cb4a = fig4.colorbar(im4a, ax=axes4[0], pad=0.02, shrink=0.85, aspect=30)
cb4a.set_label("Ensemble mean ILD trend (m decade⁻¹)", fontsize=8)
cb4a.ax.tick_params(labelsize=7)
axes4[0].set_title("A   ILD ensemble mean trend — all 6 configs",
                    fontsize=9, fontweight="bold", loc="left")
axes4[0].set_xlabel("Longitude", fontsize=8); axes4[0].set_ylabel("Latitude", fontsize=8)

im4b = axes4[1].pcolormesh(lon, lat, ild_trend_std, vmin=0, vmax=vmax_is,
                            cmap=cmap_spread, rasterized=True)
draw_land_coast(axes4[1]); style_map(axes4[1])
cb4b = fig4.colorbar(im4b, ax=axes4[1], pad=0.02, shrink=0.85, aspect=30)
cb4b.set_label("Ensemble std of ILD trend (m decade⁻¹)", fontsize=8)
cb4b.ax.tick_params(labelsize=7)
axes4[1].set_title("B   ILD trend uncertainty — std across 6 methods",
                    fontsize=9, fontweight="bold", loc="left")
axes4[1].set_xlabel("Longitude", fontsize=8); axes4[1].set_ylabel("Latitude", fontsize=8)

plt.subplots_adjust(wspace=0.08)
out4 = OUT_DIR / "trends_ild_ensemble.png"
fig4.savefig(out4, dpi=200, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved → {out4.name}")

print("\nDone.")
