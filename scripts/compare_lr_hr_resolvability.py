"""
compare_lr_hr_resolvability.py

For each region, produces two outputs:

1. displays/resolvability_{region}.png
   Heat-map grid (6 ILD methods × 12 months) showing whether the
   LR vs HR ILD difference is larger than the coarsest level spacing
   at that depth — i.e. whether it is physically distinguishable.

   Cell colour:
     green  →  ΔILD > max(spacing_LR, spacing_HR)   — resolvable
     gray   →  ΔILD ≤ max(spacing_LR, spacing_HR)   — within grid quantisation
   Number in each cell = ΔILD / max_spacing ratio.

2. displays/compare_profiles_{region}_{method}_with_uncertainty.png
   Same profile overlay as before but with ± half-spacing shaded bands
   around each ILD line so the "quantisation uncertainty" is visible.
   Only generated for the 3 most common methods to keep output manageable.

Usage:
    python scripts/compare_lr_hr_resolvability.py               # all regions
    python scripts/compare_lr_hr_resolvability.py bay_of_bengal
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

LR_CACHE_DIR = ROOT / "data" / "025deg"
HR_CACHE_DIR = ROOT / "data" / "083deg"

MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

ILD_CONFIGS = [
    dict(name="gradient_-0.015",  label="Gradient\n−0.015 °C/m",
         method="gradient",   threshold=-0.015, color="#1a6faf"),
    dict(name="gradient_-0.025",  label="Gradient\n−0.025 °C/m",
         method="gradient",   threshold=-0.025, color="#4dacd6"),
    dict(name="gradient_-0.1",    label="Gradient\n−0.1 °C/m",
         method="gradient",   threshold=-0.1,   color="#2171b5"),
    dict(name="difference_0.2",   label="Diff.\nΔT=0.2 °C",
         method="difference", threshold=0.2,    color="#2a7a3b"),
    dict(name="difference_0.5",   label="Diff.\nΔT=0.5 °C",
         method="difference", threshold=0.5,    color="#5dbf6e"),
    dict(name="difference_0.8",   label="Diff.\nΔT=0.8 °C",
         method="difference", threshold=0.8,    color="#74c476"),
]

REGIONS = {
    "amazon_plume": dict(
        label    = "Amazon Plume",
        lat_min=-5, lat_max=10, lon_min=-55, lon_max=-30,
        depth_max=300,
        lr_cache = "amazon_plume_monthly_1993_2024.zarr",
        map_extent=[-80, 15, -35, 30],
    ),
    "bay_of_bengal": dict(
        label    = "Bay of Bengal",
        lat_min=5, lat_max=25, lon_min=80, lon_max=100,
        depth_max=300,
        lr_cache = "profiles_bay_of_bengal_1993_2024.zarr",
        map_extent=[60, 115, -5, 35],
    ),
    "wpwp": dict(
        label    = "W. Pacific Warm Pool",
        lat_min=0, lat_max=20, lon_min=130, lon_max=170,
        depth_max=300,
        lr_cache = None,
        map_extent=[110, 190, -10, 30],
    ),
}

# Methods for which to generate the full profile + uncertainty figure
PROFILE_METHODS = ["gradient_-0.1", "difference_0.2", "difference_0.5"]


# ── helpers ───────────────────────────────────────────────────────────────────

def spacing_at_depth(val, depth_arr):
    """Vertical level spacing of the interval that contains val."""
    if np.isnan(val):
        return np.nan
    idx = np.searchsorted(depth_arr, val)
    idx = np.clip(idx, 1, len(depth_arr) - 1)
    return float(depth_arr[idx] - depth_arr[idx - 1])


def load_lr(key, cfg):
    lr_cache_name = cfg.get("lr_cache")
    if lr_cache_name and (LR_CACHE_DIR / lr_cache_name).exists():
        ds   = xr.open_zarr(LR_CACHE_DIR / lr_cache_name, consolidated=False)
        t    = pd.DatetimeIndex(ds["time"].values)
        mask = t.year == 2024
        return (ds["depth"].values,
                ds["temp"].isel(time=mask).values,
                ds["sal"].isel(time=mask).values if "sal" in ds else None,
                ds["mld"].isel(time=mask).values)

    fresh = LR_CACHE_DIR / f"profiles_{key}_lr_2024.zarr"
    ds    = xr.open_zarr(fresh, consolidated=False)
    return (ds["depth"].values, ds["temp"].values,
            ds["sal"].values if "sal" in ds else None, ds["mld"].values)


def load_hr(key):
    hr_cache = HR_CACHE_DIR / f"profiles_{key}_hr_2024.zarr"
    ds = xr.open_zarr(hr_cache, consolidated=False)
    return (ds["depth"].values, ds["temp"].values,
            ds["sal"].values if "sal" in ds else None, ds["mld"].values)


# ── Figure 1: resolvability heat-map ─────────────────────────────────────────

def fig_resolvability(key, reg_cfg, depth_lr, temp_lr, depth_hr, temp_hr):
    nrow = len(ILD_CONFIGS)
    ncol = 12

    # Build data arrays
    ild_lr_all  = np.full((nrow, ncol), np.nan)
    ild_hr_all  = np.full((nrow, ncol), np.nan)
    sp_lr_all   = np.full((nrow, ncol), np.nan)
    sp_hr_all   = np.full((nrow, ncol), np.nan)
    delta_all   = np.full((nrow, ncol), np.nan)
    ratio_all   = np.full((nrow, ncol), np.nan)

    for r, ild in enumerate(ILD_CONFIGS):
        for m in range(12):
            il_lr = ild_from_temp_profile(temp_lr[m], depth_lr,
                                          method=ild["method"],
                                          threshold=ild["threshold"])
            il_hr = ild_from_temp_profile(temp_hr[m], depth_hr,
                                          method=ild["method"],
                                          threshold=ild["threshold"])
            ild_lr_all[r, m] = il_lr
            ild_hr_all[r, m] = il_hr
            sp_lr = spacing_at_depth(il_lr, depth_lr)
            sp_hr = spacing_at_depth(il_hr, depth_hr)
            sp_lr_all[r, m] = sp_lr
            sp_hr_all[r, m] = sp_hr
            if not (np.isnan(il_lr) or np.isnan(il_hr)):
                delta = abs(il_hr - il_lr)
                delta_all[r, m] = delta
                max_sp = max(sp_lr, sp_hr)
                ratio_all[r, m] = delta / max_sp

    # ── plot ──
    # 3 rows of subplots: top half = cells, bottom strip = month labels
    fig = plt.figure(figsize=(16, nrow * 1.3 + 1.8))
    fig.patch.set_facecolor("white")

    gs_outer = gridspec.GridSpec(
        nrow + 1, 1,
        height_ratios=[1.0] * nrow + [0.42],
        hspace=0.06,
        left=0.16, right=0.98,
        top=0.91, bottom=0.07,
    )
    axes = [fig.add_subplot(gs_outer[r, 0]) for r in range(nrow + 1)]

    for r, ild in enumerate(ILD_CONFIGS):
        ax = axes[r]
        for m in range(12):
            ratio = ratio_all[r, m]
            delta = delta_all[r, m]
            sp_lr = sp_lr_all[r, m]
            sp_hr = sp_hr_all[r, m]
            il_lr = ild_lr_all[r, m]
            il_hr = ild_hr_all[r, m]
            max_sp = max(sp_lr, sp_hr) if not (np.isnan(sp_lr) or np.isnan(sp_hr)) else np.nan

            if np.isnan(ratio):
                facecolor = "#eeeeee"
            elif ratio > 1.0:
                intensity  = min(ratio / 3.0, 1.0)
                g = 0.55 + 0.35 * (1 - intensity)
                facecolor  = (0.1, g, 0.15)
            else:
                g = 0.82 + 0.12 * ratio
                facecolor  = (g, g, g)

            rect = mpatches.FancyBboxPatch(
                (m + 0.04, 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.03",
                facecolor=facecolor, edgecolor="white", lw=1.2,
                transform=ax.transData,
            )
            ax.add_patch(rect)

            txt_color = "white" if (not np.isnan(ratio) and ratio > 1) else "#333333"

            # Top line: ratio
            ratio_str = f"×{ratio:.1f}" if not np.isnan(ratio) else "n/a"
            ax.text(m + 0.5, 0.72, ratio_str,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=txt_color)

            # Middle line: Δ m / max_sp m
            if not np.isnan(delta):
                ax.text(m + 0.5, 0.44,
                        f"Δ{delta:.0f}/{max_sp:.0f}m",
                        ha="center", va="center", fontsize=6.2,
                        color=txt_color)

            # Bottom line: LR / HR ILD values
            if not (np.isnan(il_lr) or np.isnan(il_hr)):
                ax.text(m + 0.5, 0.17,
                        f"{il_lr:.0f}|{il_hr:.0f}m",
                        ha="center", va="center", fontsize=5.5,
                        color=txt_color, alpha=0.85)

        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(ild["label"], fontsize=8.5, rotation=0,
                      ha="right", va="center", labelpad=10)
        ax.spines[:].set_visible(False)

    # Month labels row
    ax_leg = axes[-1]
    ax_leg.set_xlim(0, 12)
    ax_leg.set_ylim(0, 1)
    for m, mn in enumerate(MONTH_NAMES):
        ax_leg.text(m + 0.5, 0.65, mn, ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax_leg.spines[:].set_visible(False)
    ax_leg.set_yticks([]); ax_leg.set_xticks([])

    # Legend patches
    for x0, fc, label in [
        (0.17, (0.1, 0.75, 0.15), "ΔILD > max grid spacing  →  resolvable"),
        (0.52, (0.88, 0.88, 0.88), "ΔILD ≤ max grid spacing  →  within grid quantisation"),
    ]:
        fig.add_artist(mpatches.FancyBboxPatch(
            (x0, 0.005), 0.016, 0.025,
            boxstyle="round,pad=0.002",
            facecolor=fc, edgecolor="#aaaaaa", lw=0.8,
            transform=fig.transFigure, clip_on=False,
        ))
        fig.text(x0 + 0.021, 0.017, label, va="center", fontsize=8.5)

    # Small note below legend
    fig.text(0.17, 0.002,
             "Cell: ×(ΔILD/max_spacing)  |  Δ=|ILD_HR−ILD_LR| m / max_spacing m  |  LR value | HR value (m)",
             fontsize=7, color="#666666", va="bottom")

    fig.suptitle(
        f"LR 0.25° vs HR 0.083°  —  ILD difference resolvability  —  2024\n"
        f"{reg_cfg['label']}    "
        f"Cell value = ΔILD / max(spacing_LR, spacing_HR)",
        fontsize=11, y=0.97,
    )

    out = ROOT / "displays" / f"resolvability_{key}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → displays/{out.name}")


# ── Figure 2: profiles with uncertainty bands ─────────────────────────────────

def fig_profiles_with_uncertainty(key, reg_cfg, ild_cfg,
                                   depth_lr, temp_lr, sal_lr, mld_lr,
                                   depth_hr, temp_hr, sal_hr, mld_hr):
    depth_max = reg_cfg["depth_max"]
    dmask_lr  = depth_lr <= depth_max
    dmask_hr  = depth_hr <= depth_max
    d_lr      = depth_lr[dmask_lr]
    d_hr      = depth_hr[dmask_hr]

    ild_method = ild_cfg["method"]
    ild_thr    = ild_cfg["threshold"]
    ild_label  = ild_cfg["label"]
    ild_clr    = ild_cfg["color"]

    ncol         = 3
    width_ratios = [3, 1, 0.08] * ncol
    fig = plt.figure(figsize=(26, 30))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        4, len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.42, wspace=0.0,
        left=0.04, right=0.78,
        top=0.93, bottom=0.06,
    )
    has_sal = (sal_lr is not None) and (sal_hr is not None)

    for i, m in enumerate(range(12)):
        row = i // ncol
        ts_col = (i % ncol) * 3
        gr_col = ts_col + 1

        ax_ts = fig.add_subplot(gs[row, ts_col])
        ax_gr = fig.add_subplot(gs[row, gr_col])

        t_lr_m   = temp_lr[m][dmask_lr]
        t_hr_m   = temp_hr[m][dmask_hr]
        mld_lr_m = float(mld_lr[m])
        mld_hr_m = float(mld_hr[m])

        ild_lr_m = ild_from_temp_profile(temp_lr[m], depth_lr,
                                         method=ild_method, threshold=ild_thr)
        ild_hr_m = ild_from_temp_profile(temp_hr[m], depth_hr,
                                         method=ild_method, threshold=ild_thr)

        sp_lr = spacing_at_depth(ild_lr_m, depth_lr)
        sp_hr = spacing_at_depth(ild_hr_m, depth_hr)

        # Is the difference resolvable?
        if not (np.isnan(ild_lr_m) or np.isnan(ild_hr_m)):
            delta  = abs(ild_hr_m - ild_lr_m)
            max_sp = max(sp_lr, sp_hr)
            resolvable = delta > max_sp
        else:
            resolvable = False
            delta  = np.nan
            max_sp = np.nan

        # ── T profiles ──
        ax_ts.plot(t_lr_m, -d_lr, color="salmon",  lw=2.2, zorder=3)
        ax_ts.plot(t_hr_m, -d_hr, color="#c0392b", lw=1.6, ls="--",
                   zorder=4, alpha=0.85)

        # ── S profiles ──
        if has_sal:
            ax_s = ax_ts.twiny()
            ax_s.plot(sal_lr[m][dmask_lr], -d_lr, color="black",   lw=2.0, zorder=3)
            ax_s.plot(sal_hr[m][dmask_hr], -d_hr, color="#555555", lw=1.5,
                      ls="--", zorder=4, alpha=0.85)
            ax_s.xaxis.set_ticks_position("top")
            ax_s.xaxis.set_label_position("top")
            ax_s.set_xlabel("Salinity (psu)", color="black", fontsize=7)
            ax_s.tick_params(axis="x", colors="black", labelsize=6)
            ax_s.spines["bottom"].set_visible(False)
            ax_s.spines["right"].set_visible(False)
            ax_s.spines["top"].set_position(("outward", 28))
            ax_s.spines["top"].set_color("black")
            ax_s.set_ylim([-depth_max, 0])

        # ── MLD ──
        ax_ts.axhline(-mld_lr_m, color="orange",  lw=2.0, ls="-",  zorder=5)
        ax_ts.axhline(-mld_hr_m, color="#e74c3c", lw=1.6, ls="--", zorder=5)

        # ── ILD with uncertainty bands ──
        # LR ILD: solid line + ± half-spacing shaded band
        if not np.isnan(ild_lr_m):
            ax_ts.axhline(-ild_lr_m, color=ild_clr, lw=1.8, ls="-", zorder=4)
            ax_ts.axhspan(-(ild_lr_m + sp_lr/2), -(ild_lr_m - sp_lr/2),
                          color=ild_clr, alpha=0.12, zorder=2)

        # HR ILD: dashed line + ± half-spacing shaded band (slightly different alpha)
        if not np.isnan(ild_hr_m):
            ax_ts.axhline(-ild_hr_m, color=ild_clr, lw=1.4, ls="--",
                          zorder=4, alpha=0.85)
            ax_ts.axhspan(-(ild_hr_m + sp_hr/2), -(ild_hr_m - sp_hr/2),
                          color=ild_clr, alpha=0.08, zorder=2,
                          hatch="///", edgecolor=ild_clr, linewidth=0.0)

        # ── Background tint by resolvability ──
        bg_color = "#eafaed" if resolvable else "#f5f5f5"
        ax_ts.set_facecolor(bg_color)

        # ── Annotation ──
        if not np.isnan(delta):
            res_str = f"ΔILD={delta:.0f} m   max_sp={max_sp:.0f} m"
            flag    = "✓ resolvable" if resolvable else "✗ within grid spacing"
            color_f = "#1a7a2a" if resolvable else "#888888"
        else:
            res_str = "ILD not detected"
            flag    = ""
            color_f = "#888888"

        ax_ts.text(0.02, 0.02,
                   f"LR  MLD={mld_lr_m:.0f} ILD={ild_lr_m:.0f} m\n"
                   f"HR  MLD={mld_hr_m:.0f} ILD={ild_hr_m:.0f} m\n"
                   f"{res_str}\n{flag}",
                   transform=ax_ts.transAxes, fontsize=5.5, va="bottom",
                   color=color_f,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor="#cccccc", alpha=0.85))

        # ── Styling ──
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
        ax_ts.set_title(MONTH_NAMES[i], fontsize=10, fontweight="bold",
                        pad=42 if has_sal else 14)

        # ── Gradient panel ──
        for temp_arr, depth_arr, col, ms in [
            (temp_lr[m], depth_lr, "#2171b5", 3.5),
            (temp_hr[m], depth_hr, "#e6550d", 2.0),
        ]:
            valid = ~np.isnan(temp_arr)
            dv = depth_arr[valid]; tv = temp_arr[valid]
            if len(dv) > 1:
                grad  = np.diff(tv) / np.diff(dv)
                d_mid = (dv[:-1] + dv[1:]) / 2.0
                gm    = d_mid <= depth_max
                ax_gr.plot(grad[gm], -d_mid[gm], "o",
                           color=col, ms=ms, lw=0, alpha=0.8, zorder=2)

        if ild_method == "gradient":
            ax_gr.axvline(ild_thr, color=ild_clr, lw=1.0, ls=":", alpha=0.8)

        ax_gr.axhline(-mld_lr_m, color="orange",  lw=1.4, ls="-",  zorder=5)
        ax_gr.axhline(-mld_hr_m, color="#e74c3c", lw=1.1, ls="--", zorder=5)
        if not np.isnan(ild_lr_m):
            ax_gr.axhline(-ild_lr_m, color=ild_clr, lw=1.4, ls="-",  zorder=4)
            ax_gr.axhspan(-(ild_lr_m + sp_lr/2), -(ild_lr_m - sp_lr/2),
                          color=ild_clr, alpha=0.12, zorder=2)
        if not np.isnan(ild_hr_m):
            ax_gr.axhline(-ild_hr_m, color=ild_clr, lw=1.1, ls="--",
                          zorder=4, alpha=0.85)
            ax_gr.axhspan(-(ild_hr_m + sp_hr/2), -(ild_hr_m - sp_hr/2),
                          color=ild_clr, alpha=0.08, zorder=2)

        ax_gr.set_facecolor(bg_color)
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

    # ── Locator map ──
    ax_map = fig.add_axes([0.80, 0.10, 0.18, 0.80],
                          projection=ccrs.PlateCarree())
    ax_map.set_extent(reg_cfg["map_extent"], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax_map.add_feature(cfeature.LAND,      facecolor="#e8e8e8", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2, linestyle=":")
    ax_map.add_patch(mpatches.Rectangle(
        (reg_cfg["lon_min"], reg_cfg["lat_min"]),
        reg_cfg["lon_max"] - reg_cfg["lon_min"],
        reg_cfg["lat_max"] - reg_cfg["lat_min"],
        lw=2, edgecolor="#c0392b", facecolor="#e74c3c", alpha=0.25,
        transform=ccrs.PlateCarree(), zorder=3,
    ))
    ax_map.add_patch(mpatches.Rectangle(
        (reg_cfg["lon_min"], reg_cfg["lat_min"]),
        reg_cfg["lon_max"] - reg_cfg["lon_min"],
        reg_cfg["lat_max"] - reg_cfg["lat_min"],
        lw=2, edgecolor="#c0392b", facecolor="none",
        transform=ccrs.PlateCarree(), zorder=4,
    ))
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}
    ax_map.set_title("Study region", fontsize=9, fontweight="bold", pad=6)

    # ── Legend ──
    lx = fig.add_axes([0.04, 0.015, 0.74, 0.040])
    lx.set_axis_off()
    items = [
        ("T  LR",    "salmon",  "-",  1.8),
        ("T  HR",    "#c0392b", "--", 1.5),
        ("S  LR",    "black",   "-",  1.8),
        ("S  HR",    "#555555", "--", 1.5),
        ("MLD LR",   "orange",  "-",  2.0),
        ("MLD HR",   "#e74c3c", "--", 1.5),
        (f"ILD LR  ({ild_label.replace(chr(10),' ')})", ild_clr, "-",  1.8),
        (f"ILD HR  ({ild_label.replace(chr(10),' ')})", ild_clr, "--", 1.5),
    ]
    x0, step = 0.0, 0.125
    for label, color, ls, lw in items:
        lx.plot([x0, x0+0.038], [0.55, 0.55], color=color, lw=lw, ls=ls,
                transform=lx.transAxes, clip_on=False)
        lx.text(x0+0.043, 0.55, label, fontsize=7, color="black",
                va="center", transform=lx.transAxes)
        x0 += step
    lx.text(0.0, 0.08,
            "Shaded bands = ± ½ grid spacing at ILD depth  |  "
            "Green background = ΔILD > max grid spacing (resolvable)  |  "
            "Gray background = within grid quantisation",
            fontsize=7, color="#555555", va="bottom",
            transform=lx.transAxes)

    lat_l = (f"{abs(reg_cfg['lat_min'])}°{'S' if reg_cfg['lat_min']<0 else 'N'}–"
             f"{abs(reg_cfg['lat_max'])}°{'S' if reg_cfg['lat_max']<0 else 'N'}")
    lon_l = (f"{abs(reg_cfg['lon_min'])}°{'W' if reg_cfg['lon_min']<0 else 'E'}–"
             f"{abs(reg_cfg['lon_max'])}°{'W' if reg_cfg['lon_max']<0 else 'E'}")
    fig.suptitle(
        f"LR 0.25° vs HR 0.083°  —  T/S profiles with vertical resolution uncertainty  —  2024\n"
        f"{reg_cfg['label']}  ({lat_l}, {lon_l})    ILD: {ild_label.replace(chr(10),' ')}",
        fontsize=12, y=0.965,
    )

    out = ROOT / "displays" / \
          f"compare_profiles_{key}_{ild_cfg['name']}_with_uncertainty.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → displays/{out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(REGIONS.keys())

    ild_map = {c["name"]: c for c in ILD_CONFIGS}

    for key in keys:
        if key not in REGIONS:
            print(f"Unknown region '{key}'."); continue
        reg_cfg = REGIONS[key]
        print(f"\n{'='*60}\nRegion: {reg_cfg['label']}\n{'='*60}")

        depth_lr, temp_lr, sal_lr, mld_lr = load_lr(key, reg_cfg)
        depth_hr, temp_hr, sal_hr, mld_hr = load_hr(key)

        print("  [1] Resolvability heat-map ...")
        fig_resolvability(key, reg_cfg, depth_lr, temp_lr, depth_hr, temp_hr)

        print("  [2] Profile figures with uncertainty bands ...")
        for mname in PROFILE_METHODS:
            fig_profiles_with_uncertainty(
                key, reg_cfg, ild_map[mname],
                depth_lr, temp_lr, sal_lr, mld_lr,
                depth_hr, temp_hr, sal_hr, mld_hr,
            )
