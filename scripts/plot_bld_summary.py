"""
plot_bld_summary.py

Standalone summary script — can be run directly on any BLD Zarr store,
or imported as a module by other scripts.

Usage (standalone):
    python scripts/plot_bld_summary.py data/025deg/bld_025deg_monthly_1993_2024.zarr

Generates in displays/:
    summary_{label}.png  — 3-panel: global map + time series + seasonal cycle
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.collections as mcollections
import shapefile as shp

# Natural Earth shapefiles (cartopy cache dir)
_NE_COAST = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"
_NE_LAND  = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"

def _draw_map(ax, land_color="lightgray", coast_lw=0.4):
    """Draw land polygons and coastlines using pyshp (no cartopy)."""
    if _NE_LAND.exists():
        sf = shp.Reader(str(_NE_LAND))
        patches = []
        for shape in sf.shapes():
            if shape.shapeType == 0:
                continue
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                patches.append(plt.Polygon(seg, closed=True))
        ax.add_collection(mcollections.PatchCollection(
            patches, facecolor=land_color, edgecolor="none", zorder=2))
    if _NE_COAST.exists():
        sf = shp.Reader(str(_NE_COAST))
        for shape in sf.shapes():
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                ax.plot(seg[:, 0], seg[:, 1], lw=coast_lw, color="#555555", zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")   # plate carrée: 1°lat = 1°lon

ROOT = Path(__file__).resolve().parent.parent

# BLD colormap: pure white at 0, increasing blue for positive BLD
BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)


# ── Core summary function ─────────────────────────────────────────────────────

def summarize_zarr(zarr_path, label=None, out_dir=None):
    """
    Load a BLD Zarr, print key stats, and save a 3-panel summary figure.

    Parameters
    ----------
    zarr_path : str or Path
    label     : short name used in title and filename (auto-derived if None)
    out_dir   : where to save the PNG (default: ROOT/displays/)
    """
    zarr_path = Path(zarr_path)
    if label is None:
        label = zarr_path.stem
    if out_dir is None:
        out_dir = ROOT / "displays"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Summary: {label}")
    print(f"{'─'*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    ds = xr.open_zarr(zarr_path, consolidated=False)
    bld = ds["bld"]                         # (time, lat, lon)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = pd.DatetimeIndex(ds["time"].values)

    n_months = len(times)
    yr0, yr1 = times[0].year, times[-1].year

    # ── Area weights (cos lat) ────────────────────────────────────────────────
    weights_da = xr.DataArray(
        np.cos(np.deg2rad(lat)), coords={"latitude": lat}, dims=["latitude"]
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    bld_mean_map  = bld.mean(dim="time").compute()          # (lat, lon)
    bld_positive  = (bld > 0).mean(dim=["latitude","longitude"]).compute() * 100

    # Area-weighted global mean per month — use xarray .weighted() which
    # correctly handles NaN masking and 2-D normalization
    bld_ts        = bld.weighted(weights_da).mean(
                        dim=["latitude","longitude"]
                    ).compute()                             # (time,)

    # Seasonal climatology
    bld_seasonal  = bld_ts.groupby("time.month").mean().compute()

    # Print stats
    valid_frac = float(np.isfinite(bld_mean_map).mean()) * 100
    print(f"  Period           : {yr0}–{yr1}  ({n_months} months)")
    print(f"  Global mean BLD  : {float(bld_ts.mean()):.1f} m")
    print(f"  Global max BLD   : {float(bld_mean_map.max()):.1f} m  (time-mean map)")
    print(f"  % ocean w/ BLD>0 : {float(bld_positive.mean()):.1f}%  (time-mean)")
    print(f"  Valid ocean frac : {valid_frac:.1f}%")
    print(f"  Seasonal range   : {float(bld_seasonal.min()):.1f}–{float(bld_seasonal.max()):.1f} m")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1.6, 1],
        hspace=0.38, wspace=0.28,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    # ── Panel 1: global BLD map ───────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    data   = bld_mean_map.values
    vmax   = max(20, float(np.nanpercentile(data[data > 0], 95))) if np.any(data > 0) else 50
    data_m = np.ma.masked_where(data <= 0, data)   # white = no barrier layer

    img = ax_map.pcolormesh(
        lon, lat, data_m,
        vmin=0, vmax=vmax,
        cmap=BLD_CMAP,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    ax_map.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax_map.set_global()
    cb = fig.colorbar(img, ax=ax_map, orientation="horizontal",
                      pad=0.04, shrink=0.7, aspect=35)
    cb.set_label("BLD (m)  —  white = no barrier layer", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax_map.set_title(f"Time-mean Barrier Layer Depth  |  {label}  ({yr0}–{yr1})",
                     fontsize=12, pad=8)

    # ── Panel 2: monthly time series ─────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[1, 0])
    ts_values = bld_ts.values
    ax_ts.plot(times, ts_values, lw=0.8, color="steelblue", alpha=0.6, label="Monthly")

    # 12-month running mean
    if len(ts_values) >= 12:
        rm = pd.Series(ts_values).rolling(12, center=True).mean().values
        ax_ts.plot(times, rm, lw=1.8, color="navy", label="12-month mean")

    ax_ts.set_ylabel("Area-weighted BLD (m)", fontsize=9)
    ax_ts.set_xlabel("Year", fontsize=9)
    ax_ts.set_title("Global mean BLD — time series", fontsize=10)
    ax_ts.tick_params(labelsize=8)
    ax_ts.legend(fontsize=8, framealpha=0.5)
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)
    ax_ts.grid(axis="y", lw=0.4, alpha=0.5)

    # ── Panel 3: seasonal climatology ────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    months     = np.arange(1, 13)
    month_lbls = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    sc_values  = bld_seasonal.values

    ax_sc.bar(months, sc_values, color="steelblue", alpha=0.75, width=0.7)
    ax_sc.set_xticks(months)
    ax_sc.set_xticklabels(month_lbls, fontsize=9)
    ax_sc.set_ylabel("BLD (m)", fontsize=9)
    ax_sc.set_title("Seasonal climatology (global mean)", fontsize=10)
    ax_sc.tick_params(labelsize=8)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.grid(axis="y", lw=0.4, alpha=0.5)
    ax_sc.axhline(0, color="black", lw=0.6)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / f"summary_{label}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path.relative_to(ROOT)}")
    print(f"{'─'*60}\n")
    return out_path


def compare_configs(config_zarr_map, out_dir=None):
    """
    2×3 comparison figure of global mean BLD maps for multiple configs.

    Parameters
    ----------
    config_zarr_map : dict  {label: zarr_path}
    out_dir         : where to save (default ROOT/displays/)
    """
    if out_dir is None:
        out_dir = ROOT / "displays"
    out_dir = Path(out_dir)

    labels = list(config_zarr_map.keys())
    n      = len(labels)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(18, 4.5 * nrows + 0.8))
    fig.patch.set_facecolor("white")

    # Reserve bottom strip for the shared colorbar
    gs = fig.add_gridspec(nrows, ncols,
                          left=0.04, right=0.97,
                          top=0.97,  bottom=0.14,
                          hspace=0.08, wspace=0.04)
    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(nrows) for c in range(ncols)]

    # compute a common vmax across all configs
    vmax_vals = []
    maps      = {}
    for lbl, zpath in config_zarr_map.items():
        ds  = xr.open_zarr(zpath, consolidated=False)
        bm  = ds["bld"].mean(dim="time").compute().values
        maps[lbl] = (ds["latitude"].values, ds["longitude"].values, bm)
        pos = bm[bm > 0]
        if len(pos):
            vmax_vals.append(np.nanpercentile(pos, 95))
    vmax = max(20, float(np.mean(vmax_vals))) if vmax_vals else 50

    last_img = None
    for i, lbl in enumerate(labels):
        ax  = axes_flat[i]
        lat, lon, data = maps[lbl]
        data_m = np.ma.masked_where(data <= 0, data)
        last_img = ax.pcolormesh(lon, lat, data_m, vmin=0, vmax=vmax,
                                 cmap=BLD_CMAP, rasterized=True, zorder=1)
        _draw_map(ax)
        ax.set_title(lbl, fontsize=10, pad=4)

    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # single shared colorbar below all panels — does not overlap maps
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.025])
    cb = fig.colorbar(last_img, cax=cbar_ax, orientation="horizontal")
    cb.set_label("BLD media anual (m)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    out_path = out_dir / "summary_sensitivity_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure saved: {out_path.relative_to(ROOT)}")
    return out_path


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_bld_summary.py <zarr_path> [label]")
        sys.exit(1)
    zpath = Path(sys.argv[1])
    lbl   = sys.argv[2] if len(sys.argv) > 2 else zpath.stem
    summarize_zarr(zpath, label=lbl)
