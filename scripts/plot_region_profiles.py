"""
plot_region_profiles.py

Monthly mean T/S profiles for selected ocean regions, showing interannual
spread (individual years as thin gray lines) + climatological mean + MLD/ILD
markers for all 6 detection methods.

Aesthetic follows sensitivity_one_point.py:
  left panel  — T (salmon) + S (black) on twin top axis
  right panel — dT/dz gradient
  locator map — right column, showing region bounding box

Regions defined in REGIONS dict. Add or modify freely.

Usage:
    python scripts/plot_region_profiles.py                   # all regions
    python scripts/plot_region_profiles.py bay_of_bengal     # single region

Outputs (displays/):
    profiles_amazon_plume.png
    profiles_bay_of_bengal.png
    profiles_southern_ocean.png
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

CACHE_DIR = ROOT / "data" / "025deg"

# ── Region definitions ────────────────────────────────────────────────────────
REGIONS = {
    "amazon_plume": dict(
        label       = "Amazon Plume",
        lat_min=-5,  lat_max=10,
        lon_min=-55, lon_max=-30,
        depth_max   = 300,
        map_extent  = [-85, 20, -35, 35],
        map_label   = "Amazon\nPlume",
        map_lpos    = (-42.5, 2.5),   # text anchor inside box
    ),
    "bay_of_bengal": dict(
        label       = "Bay of Bengal",
        lat_min=5,  lat_max=25,
        lon_min=80, lon_max=100,
        depth_max   = 300,
        map_extent  = [55, 120, -5, 35],
        map_label   = "Bay of\nBengal",
        map_lpos    = (90, 15),
    ),
    "southern_ocean": dict(
        label       = "Southern Ocean — Indian sector",
        lat_min=-65, lat_max=-45,
        lon_min=20,  lon_max=100,
        depth_max   = 600,            # deeper ML in SO
        map_extent  = [-10, 180, -75, -15],
        map_label   = "Southern\nOcean",
        map_lpos    = (60, -55),
    ),
}

CONFIGS = [
    ("Gradient −0.015°C/m", "gradient",   -0.015, "#1a6faf", "--"),
    ("Gradient −0.025°C/m", "gradient",   -0.025, "#4dacd6", "--"),
    ("Gradient −0.1°C/m",   "gradient",   -0.1,   "#91d4f5", "--"),
    ("Diff 0.2°C",          "difference",  0.2,   "#2a7a3b", "-."),
    ("Diff 0.5°C",          "difference",  0.5,   "#5dbf6e", "-."),
    ("Diff 0.8°C",          "difference",  0.8,   "#a8dbb0", "-."),
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Core routine ──────────────────────────────────────────────────────────────

def run_region(key, cfg):
    lat_min   = cfg["lat_min"];  lat_max  = cfg["lat_max"]
    lon_min   = cfg["lon_min"];  lon_max  = cfg["lon_max"]
    depth_max = cfg["depth_max"]
    label     = cfg["label"]

    cache = CACHE_DIR / f"profiles_{key}_1993_2024.zarr"

    # ── 1. Download / load cache ──────────────────────────────────────────────
    if not cache.exists():
        print(f"[{key}] Downloading from CMEMS ...")
        try:
            ds_r = cm.open_dataset(
                dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
                variables=["thetao_glor", "so_glor", "mlotst_glor"],
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                minimum_latitude=lat_min,  maximum_latitude=lat_max,
                minimum_depth=0.5057600140571594,
                maximum_depth=5727.9169921875,
                coordinates_selection_method="strict-inside",
            )
            has_sal = True
        except Exception:
            ds_r = cm.open_dataset(
                dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
                variables=["thetao_glor", "mlotst_glor"],
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                minimum_latitude=lat_min,  maximum_latitude=lat_max,
                minimum_depth=0.5057600140571594,
                maximum_depth=5727.9169921875,
                coordinates_selection_method="strict-inside",
            )
            has_sal = False
            print(f"  [warning] so_glor not available — temperature only")

        print(f"  Computing spatial mean over box ...")
        out_vars = {
            "temp": ds_r["thetao_glor"].mean(dim=["latitude", "longitude"]),
            "mld":  ds_r["mlotst_glor"].mean(dim=["latitude", "longitude"]),
        }
        if has_sal:
            out_vars["sal"] = ds_r["so_glor"].mean(dim=["latitude", "longitude"])

        ds_out = xr.Dataset(out_vars).compute()
        ds_out.attrs["has_sal"] = int(has_sal)
        ds_out.to_zarr(cache, mode="w")
        print(f"  Cached → {cache.name}")
    else:
        print(f"[{key}] Loading cache: {cache.name}")

    ds      = xr.open_zarr(cache, consolidated=False)
    temp_da = ds["temp"]
    mld_da  = ds["mld"]
    depth   = ds["depth"].values
    times   = pd.DatetimeIndex(ds["time"].values)
    HAS_SAL = "sal" in ds.data_vars
    sal_da  = ds["sal"] if HAS_SAL else None

    dmask = depth <= depth_max
    d_plt = depth[dmask]

    # ── 2. Group by calendar month + compute ILD ──────────────────────────────
    print(f"[{key}] Computing monthly statistics and ILD ...")
    month_temp = {}
    month_sal  = {}
    month_mld  = {}
    month_ild  = {cfg_item[0]: {} for cfg_item in CONFIGS}

    for m in range(1, 13):
        mask  = times.month == m
        t_arr = temp_da.isel(time=mask).values[:, dmask]
        m_arr = mld_da.isel(time=mask).values
        month_temp[m] = t_arr
        month_mld[m]  = m_arr
        if HAS_SAL:
            month_sal[m] = sal_da.isel(time=mask).values[:, dmask]

        t_full = temp_da.isel(time=mask).values
        for lbl, method, thr, color, ls in CONFIGS:
            ilds = [
                ild_from_temp_profile(t_full[yr], depth, method=method,
                                      threshold=thr)
                for yr in range(t_full.shape[0])
            ]
            month_ild[lbl][m] = np.array(ilds)

    # ── 3. Figure ─────────────────────────────────────────────────────────────
    print(f"[{key}] Plotting ...")

    ncol         = 3
    width_ratios = [3, 1, 0.12] * ncol

    fig = plt.figure(figsize=(26, 30))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        4, len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.42, wspace=0.0,
        left=0.04, right=0.78,
        top=0.93, bottom=0.06,
    )

    def col_indices(i):
        g = i % ncol
        return g * 3, g * 3 + 1

    for i, m in enumerate(range(1, 13)):
        row = i // ncol
        ts_col, gr_col = col_indices(i)

        ax_ts = fig.add_subplot(gs[row, ts_col])
        ax_gr = fig.add_subplot(gs[row, gr_col])

        t_arr   = month_temp[m]
        t_mean  = np.nanmean(t_arr, axis=0)
        mld_arr = month_mld[m]
        mld_m   = float(np.nanmean(mld_arr))
        mld_s   = float(np.nanstd(mld_arr))

        # Individual year profiles (gray)
        for t_yr in t_arr:
            ax_ts.plot(t_yr, -d_plt, color="#c8c8c8", lw=0.55, alpha=0.7, zorder=1)
        # Mean temperature
        ax_ts.plot(t_mean, -d_plt, color="salmon", lw=2.0, zorder=3)

        # Salinity
        if HAS_SAL:
            ax_s  = ax_ts.twiny()
            s_arr  = month_sal[m]
            s_mean = np.nanmean(s_arr, axis=0)
            for s_yr in s_arr:
                ax_s.plot(s_yr, -d_plt, color="#d8d8d8", lw=0.45, alpha=0.45, zorder=1)
            ax_s.plot(s_mean, -d_plt, color="black", lw=2.0, zorder=3)
            ax_s.xaxis.set_ticks_position("top")
            ax_s.xaxis.set_label_position("top")
            ax_s.set_xlabel("Salinity (psu)", color="black", fontsize=7)
            ax_s.tick_params(axis="x", colors="black", labelsize=6)
            ax_s.spines["bottom"].set_visible(False)
            ax_s.spines["right"].set_visible(False)
            ax_s.spines["top"].set_position(("outward", 28))
            ax_s.spines["top"].set_color("black")
            ax_s.set_ylim([-depth_max, 0])

        # MLD
        ax_ts.axhline(-mld_m, color="orange", lw=1.8, zorder=5)
        ax_ts.axhspan(-mld_m - mld_s, -mld_m + mld_s,
                      color="orange", alpha=0.13, zorder=2)

        # 6 ILD lines
        for lbl, method, thr, color, ls in CONFIGS:
            ild_arr = month_ild[lbl][m]
            ild_m   = float(np.nanmean(ild_arr))
            ild_s   = float(np.nanstd(ild_arr))
            if not np.isnan(ild_m):
                ax_ts.axhline(-ild_m, color=color, lw=1.4, ls=ls, zorder=4)
                ax_ts.axhspan(-ild_m - ild_s, -ild_m + ild_s,
                              color=color, alpha=0.09, zorder=2)

        # T/S styling
        ax_ts.set_ylim([-depth_max, 0])
        ax_ts.xaxis.set_ticks_position("top")
        ax_ts.xaxis.set_label_position("top")
        ax_ts.set_xlabel("Temperature (°C)", color="salmon", fontsize=7)
        ax_ts.tick_params(axis="x", colors="salmon", labelsize=6)
        ax_ts.tick_params(axis="y", labelsize=6)
        ax_ts.spines["bottom"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)
        ax_ts.spines["top"].set_color("salmon")
        ax_ts.set_ylabel("Depth (m)", fontsize=7)
        pad_top = 42 if HAS_SAL else 14
        ax_ts.set_title(MONTH_NAMES[i], fontsize=10, fontweight="bold",
                        pad=pad_top)

        # Gradient panel
        t_full_mean = np.nanmean(
            temp_da.isel(time=(times.month == m)).values, axis=0)
        valid = ~np.isnan(t_full_mean)
        d_v   = depth[valid]; t_v = t_full_mean[valid]
        grad  = np.diff(t_v) / np.diff(d_v)
        d_mid = (d_v[:-1] + d_v[1:]) / 2.0
        gm    = d_mid <= depth_max

        ax_gr.plot(grad[gm], -d_mid[gm], ".", color="darkgray",
                   markersize=2.5, lw=0.5, ls="-", zorder=2)
        ax_gr.axhline(-mld_m, color="orange", lw=1.4, zorder=5)
        for lbl, method, thr, color, ls in CONFIGS:
            if method == "gradient":
                ax_gr.axvline(thr, color=color, lw=1.0, ls=":", zorder=3, alpha=0.9)
            ild_val = float(np.nanmean(month_ild[lbl][m]))
            if not np.isnan(ild_val):
                ax_gr.axhline(-ild_val, color=color, lw=1.0, ls=ls, zorder=4)

        ax_gr.set_ylim([-depth_max, 0])
        ax_gr.set_xlim([-0.36, 0.06])
        ax_gr.xaxis.set_ticks_position("top")
        ax_gr.xaxis.set_label_position("top")
        ax_gr.set_xlabel("dT/dz (°C/m)", fontsize=6)
        ax_gr.tick_params(axis="both", labelsize=5)
        ax_gr.tick_params(axis="x", rotation=45)
        ax_gr.spines["bottom"].set_visible(False)
        ax_gr.spines["right"].set_visible(False)
        ax_gr.yaxis.set_ticklabels([])

    # ── 4. Locator map ────────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.80, 0.10, 0.18, 0.80],
                          projection=ccrs.PlateCarree())
    ax_map.set_extent(cfg["map_extent"], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax_map.add_feature(cfeature.LAND,      facecolor="#e8e8e8", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5,        zorder=2)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3,        zorder=2,
                       linestyle=":")
    ax_map.add_feature(cfeature.RIVERS,    linewidth=0.4,        zorder=2,
                       edgecolor="#90b4d4")

    # Box fill + outline
    for fill, alpha in [(True, 0.30), (False, 1.0)]:
        ax_map.add_patch(mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2.0, edgecolor="#c0392b",
            facecolor="#e74c3c" if fill else "none",
            alpha=alpha if fill else 1.0,
            transform=ccrs.PlateCarree(), zorder=3 + int(not fill),
        ))

    # Equator
    ax_map.plot(
        [cfg["map_extent"][0], cfg["map_extent"][1]], [0, 0],
        color="gray", lw=0.6, ls="--",
        transform=ccrs.PlateCarree(), zorder=3,
    )

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax_map.set_title("Study region", fontsize=9, fontweight="bold", pad=6)
    ax_map.text(
        cfg["map_lpos"][0], cfg["map_lpos"][1], cfg["map_label"],
        fontsize=7.5, color="#7b241c", ha="center", va="center",
        fontweight="bold", transform=ccrs.PlateCarree(), zorder=5,
    )

    # ── 5. Legend ─────────────────────────────────────────────────────────────
    lx = fig.add_axes([0.04, 0.015, 0.74, 0.038])
    lx.set_axis_off()
    x0 = 0.0
    lx.plot([x0, x0+0.035], [0.55, 0.55], color="orange", lw=2.0,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.040, 0.55, "MLD", fontsize=8, color="orange",
            va="center", transform=lx.transAxes)
    x0 += 0.13
    for lbl, method, thr, color, ls in CONFIGS:
        lx.plot([x0, x0+0.035], [0.55, 0.55], color=color, lw=1.8, ls=ls,
                transform=lx.transAxes, clip_on=False)
        lx.text(x0+0.040, 0.55, lbl, fontsize=7.5, color=color,
                va="center", transform=lx.transAxes)
        x0 += 0.155
    lx.plot([x0, x0+0.035], [0.55, 0.55], color="#c8c8c8", lw=1.5,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.040, 0.55, "Individual years (1993–2024)",
            fontsize=7.5, color="gray", va="center", transform=lx.transAxes)
    lx.text(0.85, 0.05, "Shaded bands = ±1 std across years",
            fontsize=7, color="gray", va="bottom", transform=lx.transAxes)

    # Title
    lat_label = (f"{abs(lat_min)}°{'S' if lat_min<0 else 'N'}–"
                 f"{abs(lat_max)}°{'S' if lat_max<0 else 'N'}")
    lon_label = (f"{abs(lon_min)}°{'W' if lon_min<0 else 'E'}–"
                 f"{abs(lon_max)}°{'W' if lon_max<0 else 'E'}")
    fig.suptitle(
        f"Monthly T/S profiles — {label}  ({lat_label}, {lon_label})  |  1993–2024",
        fontsize=13, y=0.965,
    )

    out = ROOT / "displays" / f"profiles_{key}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{key}] Saved → displays/{out.name}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(REGIONS.keys())
    for k in keys:
        if k not in REGIONS:
            print(f"Unknown region '{k}'. Available: {list(REGIONS.keys())}")
            continue
        run_region(k, REGIONS[k])
