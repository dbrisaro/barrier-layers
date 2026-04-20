"""
compare_lr_hr.py

Side-by-side comparison of LR (0.25°) vs HR (0.083°) Barrier Layer Depth for 2024.

Produces three figures:
  1. displays/compare_lr_hr_annual_mean.png
       Global annual-mean BLD maps, 2 columns (LR / HR), 3 rows (MLD / ILD / BLD)
       + difference panel  BLD_HR − BLD_LR
  2. displays/compare_lr_hr_zoom_regions.png
       4 key regions × 2 columns = 8 zoom panels showing spatial detail gain at HR
  3. displays/compare_lr_hr_stats.png
       Monthly time-series comparison for 2024 (global mean BLD) + 1-panel stat table

Usage:
    python scripts/compare_lr_hr.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
from cftime import num2date
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LR_PATH  = ROOT / "data" / "025deg" / "bld_025deg_monthly_1993_2024.zarr"
HR_PATH  = ROOT / "data" / "083deg" / "bld_083deg_monthly_2024.zarr"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(exist_ok=True)

# ── Colormap ──────────────────────────────────────────────────────────────────
BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)
DIFF_CMAP = "RdBu_r"   # for HR − LR difference


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_lr_2024():
    """Load LR zarr, extract 2024, decode times, return Dataset."""
    ds = xr.open_zarr(LR_PATH, consolidated=False, decode_times=False)
    t  = ds.time.values
    units = ds.time.attrs["units"]
    cal   = ds.time.attrs["calendar"]
    dates = num2date(t, units=units, calendar=cal)
    idx   = [i for i, d in enumerate(dates) if d.year == 2024]
    months = [dates[i].month for i in idx]
    ds24  = ds.isel(time=idx)
    ds24  = ds24.assign_coords(time=months)
    ds24["time"].attrs = {"long_name": "Month (1-12)"}
    return ds24


def load_hr_2024():
    """Load HR zarr for 2024, decode times, return Dataset."""
    ds = xr.open_zarr(HR_PATH, consolidated=False, decode_times=False)
    t  = ds.time.values
    units = ds.time.attrs.get("units", "seconds since 1950-01-01")
    cal   = ds.time.attrs.get("calendar", "gregorian")
    try:
        dates  = num2date(t, units=units, calendar=cal)
        months = [d.month for d in dates]
    except Exception:
        # Fallback: assume months 1-12 in order
        months = list(range(1, len(t) + 1))
    ds = ds.assign_coords(time=months)
    ds["time"].attrs = {"long_name": "Month (1-12)"}
    return ds


# ── Helpers ───────────────────────────────────────────────────────────────────

def annual_mean(ds):
    """Time-mean over all available months → Dataset."""
    return ds.mean(dim="time")


def mask_bld(data):
    """Return masked array: values ≤ 0 → masked (shown as white)."""
    return np.ma.masked_where(data <= 0, data)


def add_bld_panel(ax, lon, lat, data, vmax, title, fig,
                  extent=None, show_land=True):
    """Draw a BLD map panel on a PlateCarree axis."""
    data_m = mask_bld(data)

    im = ax.pcolormesh(
        lon, lat, data_m,
        vmin=0, vmax=vmax,
        cmap=BLD_CMAP,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    if show_land:
        ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=4)
    ax.gridlines(linewidth=0.2, color="gray", alpha=0.4,
                 draw_labels=False, zorder=5)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=8, pad=3)
    return im


def add_diff_panel(ax, lon, lat, data, vlim, title, fig, extent=None):
    """Draw a difference map panel (HR − LR)."""
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.pcolormesh(
        lon, lat, data,
        norm=norm,
        cmap=DIFF_CMAP,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=4)
    ax.gridlines(linewidth=0.2, color="gray", alpha=0.4,
                 draw_labels=False, zorder=5)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=8, pad=3)
    return im


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Annual mean global maps
# ══════════════════════════════════════════════════════════════════════════════

def fig_annual_mean(ds_lr, ds_hr):
    log("Figure 1: annual mean maps ...")

    am_lr = annual_mean(ds_lr)
    am_hr = annual_mean(ds_hr)

    lon_lr = ds_lr.longitude.values
    lat_lr = ds_lr.latitude.values
    lon_hr = ds_hr.longitude.values
    lat_hr = ds_hr.latitude.values

    proj = ccrs.Robinson()
    pc   = ccrs.PlateCarree()

    # Rows: MLD, ILD, BLD + diff
    vars_cfg = [
        ("mld", "MLD", 120, 40),
        ("ild", "ILD", 200, 60),
        ("bld", "BLD", 100, 30),
    ]

    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor("white")

    # 3 rows, 3 cols: LR | HR | HR−LR
    gs = gridspec.GridSpec(
        3, 3,
        left=0.03, right=0.97,
        top=0.93, bottom=0.06,
        hspace=0.15, wspace=0.04,
        width_ratios=[1, 1, 1],
    )

    col_labels = ["LR 0.25°", "HR 0.083°", "Difference (HR − LR)"]
    for j, lab in enumerate(col_labels):
        fig.text(
            0.03 + j * 0.315 + 0.155, 0.955,
            lab, ha="center", va="center",
            fontsize=11, fontweight="bold",
        )

    for row, (var, vname, vmax_global, vlim_diff) in enumerate(vars_cfg):
        data_lr = am_lr[var].values
        data_hr = am_hr[var].values

        # ── LR column ──
        ax0 = fig.add_subplot(gs[row, 0], projection=proj)
        im0 = add_bld_panel(ax0, lon_lr, lat_lr, data_lr,
                             vmax=vmax_global,
                             title=f"{vname}  (LR 0.25°)",
                             fig=fig)

        # ── HR column ──
        ax1 = fig.add_subplot(gs[row, 1], projection=proj)
        im1 = add_bld_panel(ax1, lon_hr, lat_hr, data_hr,
                             vmax=vmax_global,
                             title=f"{vname}  (HR 0.083°)",
                             fig=fig)

        # ── Colourbar for LR & HR ──
        # One shared colorbar placed at the right side of col 1
        pos0 = ax0.get_position()
        pos1 = ax1.get_position()
        cb_ax = fig.add_axes([
            pos1.x1 + 0.005,
            pos1.y0,
            0.012,
            pos1.height,
        ])
        cb = fig.colorbar(im1, cax=cb_ax, orientation="vertical")
        cb.set_label(f"{vname} (m)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # ── Difference column ──
        # Interpolate HR onto LR grid for clean difference
        lon_hr_2d, lat_hr_2d = np.meshgrid(lon_hr, lat_hr)

        # Regrid HR → LR using nearest-grid-point for speed
        da_hr = xr.DataArray(data_hr, dims=["latitude", "longitude"],
                             coords={"latitude": lat_hr, "longitude": lon_hr})
        da_hr_interp = da_hr.interp(
            latitude=lat_lr, longitude=lon_lr, method="linear"
        )
        diff = da_hr_interp.values - data_lr

        ax2 = fig.add_subplot(gs[row, 2], projection=proj)
        im2 = add_diff_panel(ax2, lon_lr, lat_lr, diff,
                             vlim=vlim_diff,
                             title=f"{vname}  HR − LR",
                             fig=fig)
        pos2 = ax2.get_position()
        cb_ax2 = fig.add_axes([
            pos2.x1 + 0.005,
            pos2.y0,
            0.012,
            pos2.height,
        ])
        cb2 = fig.colorbar(im2, cax=cb_ax2, orientation="vertical")
        cb2.set_label("Δ (m)", fontsize=7)
        cb2.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Annual Mean  2024  —  LR (0.25°) vs HR (0.083°)\n"
        "Method: gradient threshold = −0.1 °C/m",
        fontsize=12, y=0.99,
    )

    out = OUT_DIR / "compare_lr_hr_annual_mean.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {out.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Zoom regions
# ══════════════════════════════════════════════════════════════════════════════

ZOOM_REGIONS = {
    "W. Pacific\nWarm Pool":  [-5,  30, 130, 175],    # lat_min, lat_max, lon_min, lon_max
    "Bay of Bengal":          [  5,  25,  78, 100],
    "Amazon Plume":           [ -8,  15, -62, -30],
    "Arabian Sea":            [  8,  28,  52,  78],
}


def fig_zoom_regions(ds_lr, ds_hr):
    log("Figure 2: zoom region maps ...")

    am_lr = annual_mean(ds_lr)
    am_hr = annual_mean(ds_hr)

    lon_lr = ds_lr.longitude.values
    lat_lr = ds_lr.latitude.values
    lon_hr = ds_hr.longitude.values
    lat_hr = ds_hr.latitude.values

    data_lr = am_lr["bld"].values
    data_hr = am_hr["bld"].values

    pc  = ccrs.PlateCarree()
    n   = len(ZOOM_REGIONS)   # 4 regions → 4 columns
    region_names = list(ZOOM_REGIONS.keys())
    region_extents = list(ZOOM_REGIONS.values())

    # Layout: 2 rows (LR top, HR bottom) × n columns (one per region)
    fig = plt.figure(figsize=(4.5 * n, 9))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(
        2, n,
        left=0.02, right=0.98, top=0.91, bottom=0.04,
        hspace=0.18, wspace=0.06,
    )

    # Row headers
    for row_i, lab in enumerate(["LR 0.25°", "HR 0.083°"]):
        fig.text(0.005, 0.73 - row_i * 0.48, lab,
                 ha="left", va="center", fontsize=12, fontweight="bold",
                 rotation=90)

    # Column headers — region names
    col_x_positions = [0.02 + (0.96 / n) * (col + 0.5) for col in range(n)]
    for col_i, rname in enumerate(region_names):
        fig.text(col_x_positions[col_i], 0.945,
                 rname.replace("\n", " "), ha="center", fontsize=10,
                 fontweight="bold")

    last_im = None
    for col, (rname, (lat_min, lat_max, lon_min, lon_max)) in enumerate(
            ZOOM_REGIONS.items()):

        extent = [lon_min - 3, lon_max + 3, lat_min - 3, lat_max + 3]

        lon_mask_lr = (lon_lr >= lon_min) & (lon_lr <= lon_max)
        lat_mask_lr = (lat_lr >= lat_min) & (lat_lr <= lat_max)
        region_lr_data = data_lr[np.ix_(lat_mask_lr, lon_mask_lr)]
        vmax_reg = float(np.nanpercentile(region_lr_data[region_lr_data > 0], 97)) \
                   if np.any(region_lr_data > 0) else 80
        vmax_reg = max(vmax_reg, 20)

        ax_lr = fig.add_subplot(gs[0, col], projection=pc)
        ax_hr = fig.add_subplot(gs[1, col], projection=pc)

        im0 = add_bld_panel(ax_lr, lon_lr, lat_lr, data_lr,
                            vmax=vmax_reg,
                            title=f"LR 0.25°",
                            fig=fig, extent=extent)

        im1 = add_bld_panel(ax_hr, lon_hr, lat_hr, data_hr,
                            vmax=vmax_reg,
                            title=f"HR 0.083°",
                            fig=fig, extent=extent)
        last_im = im1

        # Shared colorbar to the right of each column pair (last column only)
        if col == n - 1:
            pos1 = ax_hr.get_position()
            cb_ax = fig.add_axes([pos1.x1 + 0.006, pos1.y0, 0.010, pos1.height * 2.18])
            cb    = fig.colorbar(last_im, cax=cb_ax, orientation="vertical")
            cb.set_label("BLD (m)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        # Region bounding box on both panels
        for ax in (ax_lr, ax_hr):
            rect = mpatches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=1.2,
                edgecolor="red",
                facecolor="none",
                transform=pc,
                zorder=6,
            )
            ax.add_patch(rect)

    fig.suptitle(
        "Annual Mean BLD 2024 — Regional zoom  |  LR 0.25° vs HR 0.083°\n"
        "Method: gradient threshold = −0.1 °C/m",
        fontsize=11, y=0.995,
    )
    out = OUT_DIR / "compare_lr_hr_zoom_regions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {out.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Monthly time series + statistics table
# ══════════════════════════════════════════════════════════════════════════════

def fig_monthly_stats(ds_lr, ds_hr):
    log("Figure 3: monthly stats ...")

    lon_lr = ds_lr.longitude.values
    lat_lr = ds_lr.latitude.values
    lon_hr = ds_hr.longitude.values
    lat_hr = ds_hr.latitude.values

    # Area weights
    def area_weights(lat):
        w = np.cos(np.deg2rad(lat))
        return w / w.mean()

    w_lr = area_weights(lat_lr)
    w_hr = area_weights(lat_hr)

    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    def global_mean_bld(ds, weights):
        out = []
        for m in months:
            if m not in ds.time.values:
                out.append(np.nan); continue
            bld_m = ds["bld"].sel(time=m).values
            bld_m = np.where(bld_m > 0, bld_m, np.nan)
            w2d   = np.broadcast_to(weights[:, None], bld_m.shape)
            valid  = np.isfinite(bld_m)
            if valid.sum() == 0:
                out.append(np.nan); continue
            out.append(float(np.nansum(bld_m[valid] * w2d[valid]) /
                             np.nansum(w2d[valid])))
        return out

    gm_lr = global_mean_bld(ds_lr, w_lr)
    gm_hr = global_mean_bld(ds_hr, w_hr)

    # Coverage: % ocean grid cells with BLD > 0
    def coverage(ds):
        out = []
        for m in months:
            if m not in ds.time.values:
                out.append(np.nan); continue
            bld_m = ds["bld"].sel(time=m).values
            fin   = np.isfinite(bld_m)
            if fin.sum() == 0:
                out.append(np.nan); continue
            out.append(100.0 * (bld_m[fin] > 0).sum() / fin.sum())
        return out

    cov_lr = coverage(ds_lr)
    cov_hr = coverage(ds_hr)

    # ── Plot ──
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.97,
                           top=0.91, bottom=0.08,
                           hspace=0.35, wspace=0.32)

    x = np.arange(12)

    # --- Global mean BLD ---
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(x, gm_lr, "o-", color="#2171b5", lw=2, ms=6, label="LR 0.25°")
    ax0.plot(x, gm_hr, "s-", color="#e6550d", lw=2, ms=6, label="HR 0.083°")
    ax0.set_xticks(x); ax0.set_xticklabels(month_labels, fontsize=9)
    ax0.set_ylabel("Global mean BLD (m)", fontsize=9)
    ax0.set_title("Monthly global mean BLD  —  2024", fontsize=10)
    ax0.legend(fontsize=9)
    ax0.grid(True, lw=0.4, alpha=0.5)
    ax0.set_xlim(-0.5, 11.5)

    # Fill difference between curves
    gm_lr_a = np.array(gm_lr, dtype=float)
    gm_hr_a = np.array(gm_hr, dtype=float)
    ax0.fill_between(x, gm_lr_a, gm_hr_a, alpha=0.15, color="purple",
                     label="LR−HR spread")

    # --- BLD difference HR−LR ---
    ax1 = fig.add_subplot(gs[1, 0])
    diff_mean = gm_hr_a - gm_lr_a
    cols = ["#d73027" if v > 0 else "#4575b4" for v in diff_mean]
    ax1.bar(x, diff_mean, color=cols, width=0.6)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(month_labels, fontsize=8)
    ax1.set_ylabel("ΔBLD  HR − LR (m)", fontsize=9)
    ax1.set_title("Monthly BLD difference (HR − LR)", fontsize=9)
    ax1.grid(True, axis="y", lw=0.4, alpha=0.5)
    ax1.set_xlim(-0.5, 11.5)

    # --- Coverage ---
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(x, cov_lr, "o-", color="#2171b5", lw=2, ms=6, label="LR 0.25°")
    ax2.plot(x, cov_hr, "s-", color="#e6550d", lw=2, ms=6, label="HR 0.083°")
    ax2.set_xticks(x); ax2.set_xticklabels(month_labels, fontsize=8)
    ax2.set_ylabel("% ocean grid cells with BLD > 0", fontsize=9)
    ax2.set_title("Barrier layer detection coverage", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, lw=0.4, alpha=0.5)
    ax2.set_xlim(-0.5, 11.5)

    # Stats annotation
    valid_both = np.isfinite(gm_lr_a) & np.isfinite(gm_hr_a)
    ann = (
        f"2024 Annual Mean BLD:\n"
        f"  LR 0.25°  : {np.nanmean(gm_lr_a):.1f} m\n"
        f"  HR 0.083° : {np.nanmean(gm_hr_a):.1f} m\n"
        f"  ΔBLD (HR−LR): {np.nanmean(gm_hr_a - gm_lr_a):+.1f} m"
    )
    ax0.text(0.02, 0.97, ann, transform=ax0.transAxes,
             fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#cccccc", alpha=0.85))

    fig.suptitle(
        "LR 0.25° vs HR 0.083°  —  Monthly BLD statistics  —  2024",
        fontsize=12, y=0.97,
    )
    out = OUT_DIR / "compare_lr_hr_monthly_stats.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {out.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log("Loading LR 0.25° data (2024) ...")
    ds_lr = load_lr_2024()
    log(f"  LR: {dict(ds_lr.dims)}")

    log("Loading HR 0.083° data (2024) ...")
    ds_hr = load_hr_2024()
    log(f"  HR: {dict(ds_hr.dims)}")

    fig_annual_mean(ds_lr, ds_hr)
    fig_zoom_regions(ds_lr, ds_hr)
    fig_monthly_stats(ds_lr, ds_hr)

    log("All figures done.")
    log(f"  displays/compare_lr_hr_annual_mean.png")
    log(f"  displays/compare_lr_hr_zoom_regions.png")
    log(f"  displays/compare_lr_hr_monthly_stats.png")
