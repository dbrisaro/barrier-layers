"""
compare_lr_hr_profiles.py

Overlay LR (0.25°) and HR (0.083°) spatially-averaged T/S profiles for
key ocean regions in 2024.  Runs over multiple ILD methods so you can
see whether the BLD difference between the two products comes from the
raw data or from how the ILD threshold interacts with the finer HR grid.

ILD method in the filename:  compare_profiles_{region}_{method_name}.png
  e.g.  compare_profiles_bay_of_bengal_gradient_-0.1.png
        compare_profiles_amazon_plume_difference_0.2.png

Layout per figure: 12 months × (T+S panel | dT/dz gradient panel)
  LR 0.25° : T = salmon solid  | S = black solid    | grad = steel-blue dots
  HR 0.083°: T = coral dashed  | S = dark-gray dashed | grad = orange dots
  MLD  : LR = orange solid  | HR = tomato dashed
  ILD  : LR = steel-blue solid | HR = orange dashed

HR profile data cached to  data/083deg/profiles_{key}_hr_2024.zarr
LR fresh downloads cached to  data/025deg/profiles_{key}_lr_2024.zarr

Usage:
    python scripts/compare_lr_hr_profiles.py                          # all regions × all methods
    python scripts/compare_lr_hr_profiles.py bay_of_bengal            # one region × all methods
    python scripts/compare_lr_hr_profiles.py bay_of_bengal wpwp       # two regions × all methods

    # Limit to specific ILD methods via --methods flag (comma-separated names):
    python scripts/compare_lr_hr_profiles.py --methods gradient_-0.1,difference_0.2
    python scripts/compare_lr_hr_profiles.py bay_of_bengal --methods difference_0.2
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

LR_CACHE_DIR = ROOT / "data" / "025deg"
HR_CACHE_DIR = ROOT / "data" / "083deg"
HR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── ILD method catalogue ───────────────────────────────────────────────────────
# Each entry: (name_for_filename, display_label, method, threshold, ild_color)
ILD_CONFIGS = [
    dict(name="gradient_-0.015",  label="Gradient  −0.015 °C/m",
         method="gradient",   threshold=-0.015, color="#1a6faf"),
    dict(name="gradient_-0.025",  label="Gradient  −0.025 °C/m",
         method="gradient",   threshold=-0.025, color="#4dacd6"),
    dict(name="gradient_-0.1",    label="Gradient  −0.1 °C/m",
         method="gradient",   threshold=-0.1,   color="#2171b5"),
    dict(name="difference_0.2",   label="Temp. diff.  ΔT = 0.2 °C",
         method="difference", threshold=0.2,    color="#2a7a3b"),
    dict(name="difference_0.5",   label="Temp. diff.  ΔT = 0.5 °C",
         method="difference", threshold=0.5,    color="#5dbf6e"),
    dict(name="difference_0.8",   label="Temp. diff.  ΔT = 0.8 °C",
         method="difference", threshold=0.8,    color="#74c476"),
]
ILD_CONFIGS_BY_NAME = {c["name"]: c for c in ILD_CONFIGS}

# ── Region definitions ─────────────────────────────────────────────────────────
REGIONS = {
    "amazon_plume": dict(
        label       = "Amazon Plume",
        lat_min=-5,   lat_max=10,
        lon_min=-55,  lon_max=-30,
        depth_max   = 300,
        lr_cache    = "amazon_plume_monthly_1993_2024.zarr",  # pre-existing (1993–2024)
        map_extent  = [-80, 15, -35, 30],
        map_lpos    = (-42.5, 2.5),
    ),
    "bay_of_bengal": dict(
        label       = "Bay of Bengal",
        lat_min=5,   lat_max=25,
        lon_min=80,  lon_max=100,
        depth_max   = 300,
        lr_cache    = "profiles_bay_of_bengal_1993_2024.zarr",
        map_extent  = [60, 115, -5, 35],
        map_lpos    = (90, 15),
    ),
    "wpwp": dict(
        label       = "W. Pacific Warm Pool",
        lat_min=0,   lat_max=20,
        lon_min=130, lon_max=170,
        depth_max   = 300,
        lr_cache    = None,   # no pre-existing cache; fresh 2024 download
        map_extent  = [110, 190, -10, 30],
        map_lpos    = (150, 10),
    ),
}


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_lr_2024(key, cfg):
    """Return (depth, temp[12,z], sal[12,z] or None, mld[12]) for LR 2024."""
    lr_cache_name = cfg.get("lr_cache")

    # 1. Use pre-existing full cache (1993–2024) and slice to 2024
    if lr_cache_name and (LR_CACHE_DIR / lr_cache_name).exists():
        print(f"  [LR {key}] from {lr_cache_name}")
        ds = xr.open_zarr(LR_CACHE_DIR / lr_cache_name, consolidated=False)
        t  = pd.DatetimeIndex(ds["time"].values)
        mask = t.year == 2024
        depth = ds["depth"].values
        temp  = ds["temp"].isel(time=mask).values
        mld   = ds["mld"].isel(time=mask).values
        sal   = ds["sal"].isel(time=mask).values if "sal" in ds else None
        return depth, temp, sal, mld

    # 2. Use fresh 2024 download (cached per region)
    fresh = LR_CACHE_DIR / f"profiles_{key}_lr_2024.zarr"
    if not fresh.exists():
        print(f"  [LR {key}] Downloading 2024 from CMEMS ...")
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
            variables=["thetao_glor", "so_glor", "mlotst_glor"],
            minimum_longitude=cfg["lon_min"], maximum_longitude=cfg["lon_max"],
            minimum_latitude=cfg["lat_min"],  maximum_latitude=cfg["lat_max"],
            start_datetime="2024-01-01T00:00:00",
            end_datetime="2024-12-01T00:00:00",
            minimum_depth=0.5057600140571594,
            maximum_depth=508.639892578125,
            coordinates_selection_method="strict-inside",
        )
        ds_out = xr.Dataset({
            "temp": ds_r["thetao_glor"].mean(dim=["latitude","longitude"]),
            "sal":  ds_r["so_glor"].mean(dim=["latitude","longitude"]),
            "mld":  ds_r["mlotst_glor"].mean(dim=["latitude","longitude"]),
        }).compute()
        ds_out.to_zarr(fresh, mode="w")
        print(f"  Cached → {fresh.name}")
    else:
        print(f"  [LR {key}] from {fresh.name}")

    ds    = xr.open_zarr(fresh, consolidated=False)
    depth = ds["depth"].values
    temp  = ds["temp"].values
    sal   = ds["sal"].values if "sal" in ds else None
    mld   = ds["mld"].values
    return depth, temp, sal, mld


def load_hr_2024(key, cfg):
    """Return (depth, temp[12,z], sal[12,z] or None, mld[12]) for HR 2024."""
    hr_cache = HR_CACHE_DIR / f"profiles_{key}_hr_2024.zarr"

    if not hr_cache.exists():
        print(f"  [HR {key}] Downloading 2024 from CMEMS ...")
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
            variables=["thetao", "so", "mlotst"],
            minimum_longitude=cfg["lon_min"], maximum_longitude=cfg["lon_max"],
            minimum_latitude=cfg["lat_min"],  maximum_latitude=cfg["lat_max"],
            start_datetime="2024-01-01T00:00:00",
            end_datetime="2024-12-01T00:00:00",
            minimum_depth=0.49402499198913574,
            maximum_depth=541.0889282226562,
            coordinates_selection_method="strict-inside",
        )
        ds_out = xr.Dataset({
            "temp": ds_r["thetao"].mean(dim=["latitude","longitude"]),
            "sal":  ds_r["so"].mean(dim=["latitude","longitude"]),
            "mld":  ds_r["mlotst"].mean(dim=["latitude","longitude"]),
        }).compute()
        ds_out.to_zarr(hr_cache, mode="w")
        print(f"  Cached → {hr_cache.name}")
    else:
        print(f"  [HR {key}] from {hr_cache.name}")

    ds    = xr.open_zarr(hr_cache, consolidated=False)
    depth = ds["depth"].values
    temp  = ds["temp"].values
    sal   = ds["sal"].values if "sal" in ds else None
    mld   = ds["mld"].values
    return depth, temp, sal, mld


# ── Plotting ───────────────────────────────────────────────────────────────────

def run_one(key, reg_cfg, ild_cfg,
            depth_lr, temp_lr, sal_lr, mld_lr,
            depth_hr, temp_hr, sal_hr, mld_hr):
    """Draw one figure: region × ILD method."""

    depth_max = reg_cfg["depth_max"]
    dmask_lr  = depth_lr <= depth_max
    dmask_hr  = depth_hr <= depth_max
    d_lr      = depth_lr[dmask_lr]
    d_hr      = depth_hr[dmask_hr]

    ild_method = ild_cfg["method"]
    ild_thr    = ild_cfg["threshold"]
    ild_label  = ild_cfg["label"]
    ild_clr    = ild_cfg["color"]
    ild_name   = ild_cfg["name"]

    # Compute ILD for this method
    ild_lr = np.array([
        ild_from_temp_profile(temp_lr[m], depth_lr,
                              method=ild_method, threshold=ild_thr)
        for m in range(12)
    ])
    ild_hr = np.array([
        ild_from_temp_profile(temp_hr[m], depth_hr,
                              method=ild_method, threshold=ild_thr)
        for m in range(12)
    ])

    print(f"    ILD LR: {np.round(ild_lr,1)}")
    print(f"    ILD HR: {np.round(ild_hr,1)}")

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

    def col_indices(i):
        g = i % ncol
        return g * 3, g * 3 + 1

    has_sal = (sal_lr is not None) and (sal_hr is not None)

    for i, m in enumerate(range(12)):
        row = i // ncol
        ts_col, gr_col = col_indices(i)

        ax_ts = fig.add_subplot(gs[row, ts_col])
        ax_gr = fig.add_subplot(gs[row, gr_col])

        t_lr_m   = temp_lr[m][dmask_lr]
        t_hr_m   = temp_hr[m][dmask_hr]
        mld_lr_m = float(mld_lr[m])
        mld_hr_m = float(mld_hr[m])

        # ── Temperature ──
        ax_ts.plot(t_lr_m, -d_lr, color="salmon",  lw=2.2, zorder=3)
        ax_ts.plot(t_hr_m, -d_hr, color="#c0392b", lw=1.6, ls="--",
                   zorder=4, alpha=0.85)

        # ── Salinity (twin top x-axis) ──
        if has_sal:
            ax_s = ax_ts.twiny()
            s_lr_m = sal_lr[m][dmask_lr]
            s_hr_m = sal_hr[m][dmask_hr]
            ax_s.plot(s_lr_m, -d_lr, color="black",  lw=2.0, zorder=3)
            ax_s.plot(s_hr_m, -d_hr, color="#555555", lw=1.5, ls="--",
                      zorder=4, alpha=0.85)
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

        # ── ILD (this method) ──
        # LR ILD in the ILD color (solid), HR ILD in the same color (dashed)
        if not np.isnan(ild_lr[m]):
            ax_ts.axhline(-ild_lr[m], color=ild_clr, lw=1.8, ls="-",  zorder=4)
        if not np.isnan(ild_hr[m]):
            ax_ts.axhline(-ild_hr[m], color=ild_clr, lw=1.4, ls="--", zorder=4,
                          alpha=0.8)

        # ── Styling T/S axes ──
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

        # ── BLD annotation ──
        bld_lr_m = ild_lr[m] - mld_lr_m if not np.isnan(ild_lr[m]) else np.nan
        bld_hr_m = ild_hr[m] - mld_hr_m if not np.isnan(ild_hr[m]) else np.nan
        if not (np.isnan(ild_lr[m]) or np.isnan(ild_hr[m])):
            ann = (
                f"LR: MLD={mld_lr_m:.0f}m  ILD={ild_lr[m]:.0f}m  BLD={bld_lr_m:.0f}m\n"
                f"HR: MLD={mld_hr_m:.0f}m  ILD={ild_hr[m]:.0f}m  BLD={bld_hr_m:.0f}m"
            )
        else:
            ann = f"LR MLD={mld_lr_m:.0f}m  HR MLD={mld_hr_m:.0f}m"
        ax_ts.text(0.02, 0.02, ann, transform=ax_ts.transAxes,
                   fontsize=5.5, va="bottom", color="gray",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor="#cccccc", alpha=0.85))

        # ── Gradient panel ──
        # LR gradient (steel-blue circles — coarser spacing)
        valid_lr = ~np.isnan(temp_lr[m])
        d_v_lr = depth_lr[valid_lr]; t_v_lr = temp_lr[m][valid_lr]
        if len(d_v_lr) > 1:
            grad_lr  = np.diff(t_v_lr) / np.diff(d_v_lr)
            d_mid_lr = (d_v_lr[:-1] + d_v_lr[1:]) / 2.0
            gm_lr    = d_mid_lr <= depth_max
            ax_gr.plot(grad_lr[gm_lr], -d_mid_lr[gm_lr], "o",
                       color="#2171b5", ms=3.5, lw=0, alpha=0.85,
                       zorder=3, label="LR")

        # HR gradient (orange dots — finer spacing)
        valid_hr = ~np.isnan(temp_hr[m])
        d_v_hr = depth_hr[valid_hr]; t_v_hr = temp_hr[m][valid_hr]
        if len(d_v_hr) > 1:
            grad_hr  = np.diff(t_v_hr) / np.diff(d_v_hr)
            d_mid_hr = (d_v_hr[:-1] + d_v_hr[1:]) / 2.0
            gm_hr    = d_mid_hr <= depth_max
            ax_gr.plot(grad_hr[gm_hr], -d_mid_hr[gm_hr], ".",
                       color="#e6550d", ms=2.0, lw=0, alpha=0.75,
                       zorder=2, label="HR")

        # Gradient threshold line (only for gradient methods)
        if ild_method == "gradient":
            ax_gr.axvline(ild_thr, color=ild_clr, lw=1.0, ls=":",
                          alpha=0.8, zorder=1)

        # MLD / ILD on gradient panel
        ax_gr.axhline(-mld_lr_m, color="orange",  lw=1.4, ls="-",  zorder=5)
        ax_gr.axhline(-mld_hr_m, color="#e74c3c", lw=1.1, ls="--", zorder=5)
        if not np.isnan(ild_lr[m]):
            ax_gr.axhline(-ild_lr[m], color=ild_clr, lw=1.4, ls="-",  zorder=4)
        if not np.isnan(ild_hr[m]):
            ax_gr.axhline(-ild_hr[m], color=ild_clr, lw=1.1, ls="--",
                          zorder=4, alpha=0.8)

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

    # ── Locator map ────────────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.80, 0.10, 0.18, 0.80],
                          projection=ccrs.PlateCarree())
    ax_map.set_extent(reg_cfg["map_extent"], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax_map.add_feature(cfeature.LAND,      facecolor="#e8e8e8", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2,
                       linestyle=":")
    ax_map.add_feature(cfeature.RIVERS,    linewidth=0.4, zorder=2,
                       edgecolor="#90b4d4")
    for fill, alpha in [(True, 0.25), (False, 1.0)]:
        ax_map.add_patch(mpatches.Rectangle(
            (reg_cfg["lon_min"], reg_cfg["lat_min"]),
            reg_cfg["lon_max"] - reg_cfg["lon_min"],
            reg_cfg["lat_max"] - reg_cfg["lat_min"],
            linewidth=2.0, edgecolor="#c0392b",
            facecolor="#e74c3c" if fill else "none",
            alpha=alpha if fill else 1.0,
            transform=ccrs.PlateCarree(), zorder=3 + int(not fill),
        ))
    ax_map.plot(
        [reg_cfg["map_extent"][0], reg_cfg["map_extent"][1]], [0, 0],
        color="gray", lw=0.6, ls="--", transform=ccrs.PlateCarree(), zorder=3,
    )
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax_map.set_title("Study region", fontsize=9, fontweight="bold", pad=6)

    # ── Legend ─────────────────────────────────────────────────────────────────
    lx = fig.add_axes([0.04, 0.015, 0.74, 0.040])
    lx.set_axis_off()
    items = [
        ("T  LR 0.25°",        "salmon",  "-",  1.8),
        ("T  HR 0.083°",       "#c0392b", "--", 1.5),
        ("S  LR 0.25°",        "black",   "-",  1.8),
        ("S  HR 0.083°",       "#555555", "--", 1.5),
        ("MLD LR",             "orange",  "-",  2.0),
        ("MLD HR",             "#e74c3c", "--", 1.5),
        (f"ILD LR  ({ild_label})", ild_clr, "-",  1.8),
        (f"ILD HR  ({ild_label})", ild_clr, "--", 1.5),
    ]
    x0, step = 0.0, 0.125
    for label, color, ls, lw in items:
        lx.plot([x0, x0+0.038], [0.55, 0.55], color=color, lw=lw, ls=ls,
                transform=lx.transAxes, clip_on=False)
        lx.text(x0+0.043, 0.55, label, fontsize=7, color="black",
                va="center", transform=lx.transAxes)
        x0 += step

    # ── Title ──────────────────────────────────────────────────────────────────
    lat_l = (f"{abs(reg_cfg['lat_min'])}°{'S' if reg_cfg['lat_min']<0 else 'N'}–"
             f"{abs(reg_cfg['lat_max'])}°{'S' if reg_cfg['lat_max']<0 else 'N'}")
    lon_l = (f"{abs(reg_cfg['lon_min'])}°{'W' if reg_cfg['lon_min']<0 else 'E'}–"
             f"{abs(reg_cfg['lon_max'])}°{'W' if reg_cfg['lon_max']<0 else 'E'}")
    fig.suptitle(
        f"LR 0.25° vs HR 0.083°  —  Spatial-mean T/S profiles  —  2024\n"
        f"{reg_cfg['label']}  ({lat_l}, {lon_l})    "
        f"ILD method: {ild_label}",
        fontsize=12, y=0.965,
    )

    out = ROOT / "displays" / f"compare_profiles_{key}_{ild_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → displays/{out.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    """Parse argv.  Returns (region_keys, ild_names)."""
    args = sys.argv[1:]

    # Pull out --methods flag
    ild_names = None
    if "--methods" in args:
        idx = args.index("--methods")
        ild_names = args[idx + 1].split(",")
        args = args[:idx] + args[idx + 2:]

    region_keys = args if args else list(REGIONS.keys())
    return region_keys, ild_names


if __name__ == "__main__":
    region_keys, ild_names = parse_args()

    # Validate
    for k in region_keys:
        if k not in REGIONS:
            print(f"Unknown region '{k}'. Available: {list(REGIONS.keys())}")
            sys.exit(1)
    ild_cfgs = ILD_CONFIGS
    if ild_names:
        ild_cfgs = [ILD_CONFIGS_BY_NAME[n] for n in ild_names
                    if n in ILD_CONFIGS_BY_NAME]
        missing = [n for n in ild_names if n not in ILD_CONFIGS_BY_NAME]
        if missing:
            print(f"Unknown ILD method(s): {missing}")
            print(f"Available: {list(ILD_CONFIGS_BY_NAME.keys())}")

    print(f"Regions : {region_keys}")
    print(f"Methods : {[c['name'] for c in ild_cfgs]}")
    print(f"Total   : {len(region_keys) * len(ild_cfgs)} figures\n")

    # Load data once per region, reuse across all methods
    for key in region_keys:
        reg_cfg = REGIONS[key]
        print(f"\n{'='*60}")
        print(f"Region: {reg_cfg['label']}")
        print(f"{'='*60}")

        depth_lr, temp_lr, sal_lr, mld_lr = load_lr_2024(key, reg_cfg)
        depth_hr, temp_hr, sal_hr, mld_hr = load_hr_2024(key, reg_cfg)

        for ild_cfg in ild_cfgs:
            out_path = ROOT / "displays" / f"compare_profiles_{key}_{ild_cfg['name']}.png"
            if out_path.exists():
                print(f"  [{ild_cfg['name']}] already exists — skipping")
                continue
            print(f"  [{ild_cfg['name']}] plotting ...")
            run_one(key, reg_cfg, ild_cfg,
                    depth_lr, temp_lr, sal_lr, mld_lr,
                    depth_hr, temp_hr, sal_hr, mld_hr)
