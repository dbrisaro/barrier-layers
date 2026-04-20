"""
regen_zoom_regions.py

Regenerates compare_lr_hr_zoom_regions.png with 2 rows × 4 columns layout:
  Row 0 = LR 0.25°  for each region
  Row 1 = HR 0.083° for each region
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
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT    = Path(__file__).resolve().parent.parent
LR_PATH = ROOT / "data" / "025deg" / "bld_025deg_monthly_1993_2024.zarr"
HR_PATH = ROOT / "data" / "083deg" / "bld_083deg_monthly_2024.zarr"
OUT_DIR = ROOT / "displays"
OUT_DIR.mkdir(exist_ok=True)

BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues",
    ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_lr_2024():
    ds = xr.open_zarr(LR_PATH, consolidated=False, decode_times=False)
    t  = ds.time.values
    units = ds.time.attrs["units"]
    cal   = ds.time.attrs["calendar"]
    dates = num2date(t, units=units, calendar=cal)
    idx   = [i for i, d in enumerate(dates) if d.year == 2024]
    months = [dates[i].month for i in idx]
    ds24  = ds.isel(time=idx)
    ds24  = ds24.assign_coords(time=months)
    return ds24


def load_hr_2024():
    ds = xr.open_zarr(HR_PATH, consolidated=False, decode_times=False)
    t  = ds.time.values
    units = ds.time.attrs.get("units", "seconds since 1950-01-01")
    cal   = ds.time.attrs.get("calendar", "gregorian")
    try:
        dates  = num2date(t, units=units, calendar=cal)
        months = [d.month for d in dates]
    except Exception:
        months = list(range(1, len(t) + 1))
    ds = ds.assign_coords(time=months)
    return ds


def annual_mean(ds):
    return ds.mean(dim="time")


def mask_bld(data):
    return np.ma.masked_where(data <= 0, data)


def add_bld_panel(ax, lon, lat, data, vmax, title, fig, extent=None):
    pc = ccrs.PlateCarree()
    data_m = mask_bld(data)
    im = ax.pcolormesh(
        lon, lat, data_m,
        vmin=0, vmax=vmax,
        cmap=BLD_CMAP,
        transform=pc,
        rasterized=True,
    )
    ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=4)
    ax.gridlines(linewidth=0.2, color="gray", alpha=0.4,
                 draw_labels=False, zorder=5)
    if extent:
        ax.set_extent(extent, crs=pc)
    ax.set_title(title, fontsize=8, pad=3)
    return im


ZOOM_REGIONS = {
    "W. Pacific Warm Pool":  [-5,  30, 130, 175],
    "Bay of Bengal":         [  5,  25,  78, 100],
    "Amazon Plume":          [ -8,  15, -62, -30],
    "Arabian Sea":           [  8,  28,  52,  78],
}


log("Loading data ...")
ds_lr = load_lr_2024()
ds_hr = load_hr_2024()
log(f"  LR: {dict(ds_lr.dims)}")
log(f"  HR: {dict(ds_hr.dims)}")

am_lr = annual_mean(ds_lr)
am_hr = annual_mean(ds_hr)

lon_lr = ds_lr.longitude.values
lat_lr = ds_lr.latitude.values
lon_hr = ds_hr.longitude.values
lat_hr = ds_hr.latitude.values

data_lr = am_lr["bld"].values
data_hr = am_hr["bld"].values

pc  = ccrs.PlateCarree()
n   = len(ZOOM_REGIONS)   # 4 regions

# 2 rows (LR/HR) × 4 columns (one per region)
fig = plt.figure(figsize=(4.5 * n, 9))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(
    2, n,
    left=0.03, right=0.97, top=0.90, bottom=0.04,
    hspace=0.18, wspace=0.06,
)

# Row labels
for row_i, lab in enumerate(["LR 0.25°", "HR 0.083°"]):
    y_pos = 0.70 - row_i * 0.48
    fig.text(0.005, y_pos, lab,
             ha="left", va="center", fontsize=11, fontweight="bold",
             rotation=90)

region_names = list(ZOOM_REGIONS.keys())

for col, (rname, (lat_min, lat_max, lon_min, lon_max)) in enumerate(ZOOM_REGIONS.items()):
    extent = [lon_min - 3, lon_max + 3, lat_min - 3, lat_max + 3]

    lon_mask = (lon_lr >= lon_min) & (lon_lr <= lon_max)
    lat_mask = (lat_lr >= lat_min) & (lat_lr <= lat_max)
    region_data = data_lr[np.ix_(lat_mask, lon_mask)]
    vmax_reg = float(np.nanpercentile(region_data[region_data > 0], 97)) \
               if np.any(region_data > 0) else 80
    vmax_reg = max(vmax_reg, 20)

    ax_lr = fig.add_subplot(gs[0, col], projection=pc)
    ax_hr = fig.add_subplot(gs[1, col], projection=pc)

    im0 = add_bld_panel(ax_lr, lon_lr, lat_lr, data_lr,
                        vmax=vmax_reg,
                        title=f"{rname} — LR 0.25°",
                        fig=fig, extent=extent)

    im1 = add_bld_panel(ax_hr, lon_hr, lat_hr, data_hr,
                        vmax=vmax_reg,
                        title=f"{rname} — HR 0.083°",
                        fig=fig, extent=extent)

    # Colorbar on rightmost column only (spans both rows)
    if col == n - 1:
        pos_lr = ax_lr.get_position()
        pos_hr = ax_hr.get_position()
        cb_ax = fig.add_axes([
            pos_lr.x1 + 0.006,
            pos_hr.y0,
            0.010,
            pos_lr.y1 - pos_hr.y0,
        ])
        cb = fig.colorbar(im1, cax=cb_ax, orientation="vertical")
        cb.set_label("BLD (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Red bounding box on both panels
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
    fontsize=11, y=0.97,
)

out = OUT_DIR / "compare_lr_hr_zoom_regions.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out.relative_to(ROOT)}")
