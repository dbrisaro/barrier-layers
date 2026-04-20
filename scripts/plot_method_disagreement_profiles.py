"""
plot_method_disagreement_profiles.py  (v3)

Métrica de acuerdo correcta:
  agreement(lat,lon) = media sobre 6 métodos de [fracción de meses con BLD > 0]
  → mapa de 0–100 % en cada pixel

Tiers (percentiles dentro de cada región):
  • Intermedio      : agreement en pct 25–50  (de mayor a menor acuerdo)
  • Intermedio-bajo : agreement en pct 10–25

Para cada región, muestra los perfiles mensuales con la misma estética que
plot_region_profiles.py:
  - Panel T/S : T salmón (abajo) + S negro (arriba twin) + años grises
  - Orange    : MLD media ± 1std
  - 6 ILD     : línea sólida = tier intermedio,  punteada = tier interm.-bajo
  - Panel dT/dz + umbrales de gradiente
  - Mapa localizador con costas (pyshp, sin cartopy)

Salidas:
  displays/method_agreement_map.png
  displays/method_agreement_{region}.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import shapefile as shp

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
PROF_DIR = ROOT / "data" / "025deg"
OUT_DIR  = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

# Natural Earth shapefiles (cartopy cache)
NE_COAST = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"
NE_LAND  = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"

# ── Helpers de mapa sin cartopy ───────────────────────────────────────────────

def draw_coastlines(ax, extent=None, land_color="#e0e8f0", coast_lw=0.5):
    if NE_LAND.exists():
        sf = shp.Reader(str(NE_LAND))
        patches = []
        for shape in sf.shapes():
            if shape.shapeType == 0:
                continue
            pts = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                patches.append(plt.Polygon(seg, closed=True))
        pc = mcollections.PatchCollection(
            patches, facecolor=land_color, edgecolor="none", zorder=2)
        ax.add_collection(pc)
    if NE_COAST.exists():
        sf = shp.Reader(str(NE_COAST))
        for shape in sf.shapes():
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                ax.plot(seg[:, 0], seg[:, 1],
                        lw=coast_lw, color="#555555", zorder=3)
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

# ── Configuraciones ───────────────────────────────────────────────────────────

CONFIGS = [
    ("Gradient −0.015°C/m", "gradient",   -0.015, "#1a6faf", "--"),
    ("Gradient −0.025°C/m", "gradient",   -0.025, "#4dacd6", "--"),
    ("Gradient −0.1°C/m",   "gradient",   -0.1,   "#91d4f5", "--"),
    ("Diff 0.2°C",          "difference",  0.2,   "#2a7a3b", "-."),
    ("Diff 0.5°C",          "difference",  0.5,   "#5dbf6e", "-."),
    ("Diff 0.8°C",          "difference",  0.8,   "#a8dbb0", "-."),
]
CONFIG_NAMES = [
    "gradient_015", "gradient_025", "gradient_100",
    "difference_02", "difference_05", "difference_08",
]

REGIONS = {
    "tropical_atlantic": dict(
        label      = "Tropical Atlantic",
        lat_min=-5,  lat_max=10,
        lon_min=-55, lon_max=-30,
        depth_max  = 300,
        prof_zarr  = "amazon_plume_monthly_1993_2024.zarr",
        map_extent = [-65, -18, -12, 18],
        map_lpos   = (-42.5, 2.5),
    ),
    "bay_of_bengal": dict(
        label      = "Bay of Bengal",
        lat_min=5,  lat_max=25,
        lon_min=80, lon_max=100,
        depth_max  = 300,
        prof_zarr  = "profiles_bay_of_bengal_1993_2024.zarr",
        map_extent = [70, 110, 0, 32],
        map_lpos   = (90, 15),
    ),
    "southern_ocean": dict(
        label      = "Southern Ocean (Indian sector)",
        lat_min=-65, lat_max=-45,
        lon_min=20,  lon_max=100,
        depth_max  = 600,
        prof_zarr  = "profiles_southern_ocean_1993_2024.zarr",
        map_extent = [5, 115, -73, -38],
        map_lpos   = (60, -55),
    ),
    "wpwp": dict(
        label      = "W. Pacific Warm Pool",
        lat_min=-10, lat_max=10,
        lon_min=130, lon_max=180,
        depth_max  = 300,
        prof_zarr  = "profiles_wpwp_lr_2024.zarr",
        map_extent = [120, 190, -18, 18],
        map_lpos   = (155, 0),
    ),
    "arg_sea_north": dict(
        label      = "N. Argentine Shelf",
        lat_min=-45, lat_max=-30,
        lon_min=-65, lon_max=-50,
        depth_max  = 400,
        prof_zarr  = "arg_sea_north_monthly_1993_2024.zarr",
        map_extent = [-72, -40, -58, -20],
        map_lpos   = (-57.5, -37.5),
    ),
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Mapa de acuerdo global
#    agreement(lat,lon) = media de [BLD>0 rate] sobre los 6 métodos
# ══════════════════════════════════════════════════════════════════════════════

AGR_CACHE = ROOT / "data" / "025deg" / "agreement_map_cache.npz"

print("=" * 60, flush=True)

_ref = xr.open_zarr(SENS_DIR / "bld_gradient_025_1993_2024.zarr", consolidated=False)
lat  = _ref["latitude"].values
lon  = _ref["longitude"].values
times_global = pd.DatetimeIndex(_ref["time"].values)
del _ref

if AGR_CACHE.exists():
    print("Paso 1: cargando mapa de acuerdo desde caché …", flush=True)
    cache_data    = np.load(AGR_CACHE)
    agreement_map = cache_data["agreement_map"]
    print(f"  Cargado: {agreement_map.shape}", flush=True)
else:
    print("Paso 1: mapa de acuerdo (fracción BLD>0 × 6 métodos) …", flush=True)
    detect_rates = {}
    for cname, (lbl, method, thr, col, ls) in zip(CONFIG_NAMES, CONFIGS):
        print(f"  {cname} …", end=" ", flush=True)
        bld = xr.open_zarr(SENS_DIR / f"bld_{cname}_1993_2024.zarr",
                           consolidated=False)["bld"]
        detect_rates[cname] = (bld > 0).mean(dim="time").compute().values.astype(np.float32)
        print("ok", flush=True)

    agr_stack     = np.stack(list(detect_rates.values()), axis=0)
    agreement_map = np.nanmean(agr_stack, axis=0) * 100.0
    np.savez(AGR_CACHE, agreement_map=agreement_map)
    print(f"  Caché guardado → {AGR_CACHE.name}", flush=True)

print(f"Agreement global: "
      f"median={np.nanmedian(agreement_map):.1f}%  "
      f"mean={np.nanmean(agreement_map[agreement_map>0]):.1f}%", flush=True)

# ── Figura: mapa global ───────────────────────────────────────────────────────
print("\nPaso 2: mapa global de acuerdo …", flush=True)

region_colors = {
    "tropical_atlantic": "#e74c3c",
    "bay_of_bengal":     "#8e44ad",
    "southern_ocean":    "#2980b9",
    "wpwp":              "#e67e22",
    "arg_sea_north":     "#16a085",
}

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")
img = ax.pcolormesh(lon, lat, agreement_map, vmin=0, vmax=100,
                    cmap="RdYlGn", rasterized=True)
draw_coastlines(ax, extent=[-180, 180, -90, 90])
cb = fig.colorbar(img, ax=ax, pad=0.01, shrink=0.85, aspect=30)
cb.set_label("Method agreement  (% of methods detecting BLD > 0)", fontsize=9)
cb.ax.tick_params(labelsize=8)

for rkey, rcfg in REGIONS.items():
    col = region_colors[rkey]
    ax.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor=col, facecolor="none", zorder=4
    ))
    ax.text((rcfg["lon_min"]+rcfg["lon_max"])/2, rcfg["lat_min"]-2,
            rcfg["label"], fontsize=7, color=col,
            ha="center", va="top", fontweight="bold", zorder=5)

ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
ax.set_aspect("equal")   # 1 deg lat = 1 deg lon → correct plate carrée
ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_title("")
out_agr = OUT_DIR / "method_agreement_map.png"
fig.savefig(out_agr, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {out_agr.name}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def region_slice(rcfg):
    li = np.where((lat >= rcfg["lat_min"]) & (lat <= rcfg["lat_max"]))[0]
    lo = np.where((lon >= rcfg["lon_min"]) & (lon <= rcfg["lon_max"]))[0]
    return li, lo


def tier_masks_agreement(agr_reg):
    """
    Define tiers POR NIVEL DE ACUERDO (de mayor a menor agreement):
      - 'high'          : top quartile (>p75)
      - 'intermediate'  : p50–p75
      - 'intermed_low'  : p25–p50
      - 'low'           : <p25
    Solo considera pixels con agreement > 0.
    """
    valid = agr_reg[(agr_reg > 0) & np.isfinite(agr_reg)].ravel()
    if valid.size == 0:
        return None, None, None, None, {}
    p25 = float(np.percentile(valid, 25))
    p50 = float(np.percentile(valid, 50))
    p75 = float(np.percentile(valid, 75))

    masks = {
        "high":         (agr_reg >= p75),
        "intermediate": (agr_reg >= p50) & (agr_reg < p75),
        "intermed_low": (agr_reg >= p25) & (agr_reg < p50),
        "low":          (agr_reg >  0)   & (agr_reg < p25),
    }
    return p25, p50, p75, valid, masks


def load_ild_regional_monthly(rcfg, cname):
    """
    ILD mensual climatológico solo para la región → (12, nlat_r, nlon_r).
    """
    da   = xr.open_zarr(SENS_DIR / f"bld_{cname}_1993_2024.zarr",
                        consolidated=False)["ild"]
    da_r = da.sel(
        latitude =slice(rcfg["lat_min"], rcfg["lat_max"]),
        longitude=slice(rcfg["lon_min"], rcfg["lon_max"]),
    )
    full = da_r.compute().values   # (384, nlat_r, nlon_r)
    mon_list = []
    for m in range(1, 13):
        mask_m = times_global.month == m
        mc = np.nanmean(full[mask_m], axis=0).astype(np.float32)
        mon_list.append(mc)
    return np.stack(mon_list, axis=0)   # (12, nlat_r, nlon_r)


def mean_ild_for_tier(ild_mon_r, mask_2d, month_idx):
    """Media de ILD sobre los pixels del tier para un mes dado."""
    arr  = ild_mon_r[month_idx]
    vals = arr[mask_2d & np.isfinite(arr)]
    if (vals > 0).any():
        v = vals[vals > 0]
        return float(np.nanmean(v)), float(np.nanstd(v))
    return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 3. Figura por región
#    Estética de plot_region_profiles.py + ILD por tier
# ══════════════════════════════════════════════════════════════════════════════

TIER_PLOT = {
    # tier → (line style, alpha, label)
    "intermediate":  ("-",  1.0, "Acuerdo intermedio  (p50–p75)"),
}

for rkey, rcfg in REGIONS.items():
    print(f"\nPaso 3 [{rcfg['label']}] …", flush=True)

    zp = PROF_DIR / rcfg["prof_zarr"]
    if not zp.exists():
        print(f"  Zarr no encontrado: {rcfg['prof_zarr']}. Saltando.", flush=True)
        continue

    # ── Agreement en la región ────────────────────────────────────────────────
    li, lo = region_slice(rcfg)
    lat_r  = lat[li]; lon_r = lon[lo]
    agr_reg = agreement_map[np.ix_(li, lo)]
    p25, p50, p75, valid_agr, masks_r = tier_masks_agreement(agr_reg)

    if not masks_r:
        print("  Sin pixels válidos. Saltando.", flush=True)
        continue

    print(f"  Agreement: p25={p25:.1f}%  p50={p50:.1f}%  p75={p75:.1f}%", flush=True)
    for t, m in masks_r.items():
        print(f"    {t}: {int(m.sum())} pixels", flush=True)

    # ── ILD por config × tier × mes ──────────────────────────────────────────
    print("  Cargando ILD regional (6 configs) …", flush=True)
    # ild_data[cname][tier][month 1-12] = (mean, std)
    ild_data = {}
    for cname in CONFIG_NAMES:
        print(f"    {cname} …", end=" ", flush=True)
        ild_mon_r = load_ild_regional_monthly(rcfg, cname)
        ild_data[cname] = {}
        for tier in ["intermediate", "intermed_low"]:
            mask_2d = masks_r[tier]
            ild_data[cname][tier] = {}
            for mi in range(12):
                m_val, m_std = mean_ild_for_tier(ild_mon_r, mask_2d, mi)
                ild_data[cname][tier][mi + 1] = (m_val, m_std)
        print("ok", flush=True)

    # ── Perfiles T/S/MLD ─────────────────────────────────────────────────────
    ds_p      = xr.open_zarr(zp, consolidated=False)
    temp_da   = ds_p["temp"]
    sal_da    = ds_p["sal"]  if "sal"  in ds_p.data_vars else None
    mld_da    = ds_p["mld"]  if "mld"  in ds_p.data_vars else None
    depth     = ds_p["depth"].values
    times_p   = pd.DatetimeIndex(ds_p["time"].values)
    HAS_SAL   = sal_da is not None
    depth_max = rcfg["depth_max"]
    dmask     = depth <= depth_max
    d_plt     = depth[dmask]

    note_t = "(solo 2024)" if len(times_p) <= 12 else "(1993–2024)"
    print(f"  Perfiles T+S+MLD: {temp_da.shape}  {note_t}", flush=True)

    # Agrupar por mes
    month_temp = {}; month_sal = {}; month_mld = {}
    for m in range(1, 13):
        mask_m = times_p.month == m
        month_temp[m] = temp_da.isel(time=mask_m).values[:, dmask]
        if HAS_SAL:
            month_sal[m] = sal_da.isel(time=mask_m).values[:, dmask]
        if mld_da is not None:
            month_mld[m] = mld_da.isel(time=mask_m).values

    # Gradient en todos los niveles para el panel dT/dz
    temp_full = temp_da.values  # (time, depth)

    # ── Figura ────────────────────────────────────────────────────────────────
    print("  Generando figura …", flush=True)

    ncol         = 3   # 3 meses por fila (4 filas = 12 meses)
    nrow         = 4
    # Por cada mes: columna T/S (ancha) + columna grad (angosta) + spacer
    width_ratios = [3, 1, 0.10] * ncol

    fig = plt.figure(figsize=(26, 30))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        nrow, len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.45, wspace=0.0,
        left=0.04, right=0.78,
        top=0.93, bottom=0.06,
    )

    for i, m in enumerate(range(1, 13)):
        row   = i // ncol
        g     = i % ncol
        ts_col = g * 3
        gr_col = g * 3 + 1

        ax_ts = fig.add_subplot(gs[row, ts_col])
        ax_gr = fig.add_subplot(gs[row, gr_col])

        t_arr   = month_temp[m]
        t_mean  = np.nanmean(t_arr, axis=0)
        mld_m   = float(np.nanmean(month_mld[m])) if m in month_mld else np.nan

        # ── Panel T/S ─────────────────────────────────────────────────────────
        # Años individuales (gris)
        for t_yr in t_arr:
            ax_ts.plot(t_yr, -d_plt, color="#c8c8c8", lw=0.55, alpha=0.7, zorder=1)
        # Media T
        ax_ts.plot(t_mean, -d_plt, color="salmon", lw=2.0, zorder=3)

        # Salinidad (twin top)
        if HAS_SAL and m in month_sal:
            ax_s   = ax_ts.twiny()
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
            ax_s.spines["top"].set_position(("outward", 38))
            ax_s.spines["top"].set_color("black")
            ax_s.set_ylim([-depth_max, 0])

        # MLD
        if np.isfinite(mld_m):
            ax_ts.axhline(-mld_m, color="orange", lw=1.8, zorder=5)

        # ILD por config × tier
        for cname, (lbl, method, thr, color, ls_orig) in zip(CONFIG_NAMES, CONFIGS):
            for tier, (ls_tier, alpha_tier, _) in TIER_PLOT.items():
                ild_m, _ = ild_data[cname][tier][m]
                if np.isfinite(ild_m) and ild_m > 0:
                    ax_ts.axhline(-ild_m, color=color, lw=1.6,
                                  ls=ls_tier, alpha=alpha_tier, zorder=4)

        # T axis styling — top
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

        # ── Panel dT/dz ───────────────────────────────────────────────────────
        t_full_mean = np.nanmean(
            temp_full[times_p.month == m], axis=0)
        valid = ~np.isnan(t_full_mean)
        if valid.sum() > 2:
            d_v = depth[valid]; t_v = t_full_mean[valid]
            grad  = np.diff(t_v) / np.diff(d_v)
            d_mid = (d_v[:-1] + d_v[1:]) / 2.0
            gm    = d_mid <= depth_max
            ax_gr.plot(grad[gm], -d_mid[gm], ".", color="darkgray",
                       markersize=2.5, lw=0.5, ls="-", zorder=2)

        if np.isfinite(mld_m):
            ax_gr.axhline(-mld_m, color="orange", lw=1.4, zorder=5)

        for cname, (lbl, method, thr, color, ls_orig) in zip(CONFIG_NAMES, CONFIGS):
            if method == "gradient":
                ax_gr.axvline(thr, color=color, lw=1.0, ls=":", zorder=3, alpha=0.9)
            for tier, (ls_tier, alpha_tier, _) in TIER_PLOT.items():
                ild_m_val, _ = ild_data[cname][tier][m]
                if np.isfinite(ild_m_val) and ild_m_val > 0:
                    ax_gr.axhline(-ild_m_val, color=color, lw=1.0,
                                  ls=ls_tier, zorder=4, alpha=alpha_tier)

        ax_gr.set_ylim([-depth_max, 0])
        ax_gr.set_xlim([-0.36, 0.06])
        ax_gr.xaxis.set_ticks_position("bottom")
        ax_gr.xaxis.set_label_position("bottom")
        ax_gr.set_xlabel("dT/dz (°C/m)", fontsize=6)
        ax_gr.tick_params(axis="both", labelsize=5)
        ax_gr.tick_params(axis="x", rotation=45)
        ax_gr.spines["top"].set_visible(False)
        ax_gr.spines["right"].set_visible(False)
        ax_gr.yaxis.set_ticklabels([])

    # ── Mapa localizador (pyshp) ──────────────────────────────────────────────
    ax_map = fig.add_axes([0.80, 0.10, 0.18, 0.80])
    me = rcfg["map_extent"]
    draw_coastlines(ax_map, extent=[me[0], me[1], me[2], me[3]])

    # Agreement overlay en el mapa
    lon2d, lat2d = np.meshgrid(lon_r, lat_r)
    img_agr = ax_map.pcolormesh(lon2d, lat2d, agr_reg,
                                vmin=0, vmax=100, cmap="RdYlGn",
                                rasterized=True, zorder=0, alpha=0.85)

    # Overlay tier intermedio
    tv = np.where(masks_r["intermediate"], 1.0, np.nan)
    ax_map.pcolormesh(lon2d, lat2d, tv, vmin=0, vmax=1,
                      cmap=mcolors.ListedColormap(["#2980b9"]),
                      alpha=0.55, rasterized=True, zorder=1)

    # Caja de la región
    ax_map.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor="#c0392b", facecolor="none", zorder=5
    ))

    lat_ctr = (rcfg["lat_min"] + rcfg["lat_max"]) / 2.0
    ax_map.set_aspect(1 / np.cos(np.deg2rad(lat_ctr)))
    ax_map.tick_params(labelsize=6)
    ax_map.set_title("Study region\n(agreement %)", fontsize=8,
                     fontweight="bold", pad=5)
    cb_m = fig.colorbar(img_agr, ax=ax_map, orientation="horizontal",
                        pad=0.08, shrink=0.9, aspect=20)
    cb_m.set_label("Agreement (%)", fontsize=7)
    cb_m.ax.tick_params(labelsize=6)

    leg_p = [
        mpatches.Patch(color="#2980b9", alpha=0.6,
                       label=f"Intermedio\n({p50:.0f}–{p75:.0f}%)"),
    ]
    ax_map.legend(handles=leg_p, fontsize=6, loc="lower left",
                  framealpha=0.85)

    # ── Leyenda inferior ──────────────────────────────────────────────────────
    lx = fig.add_axes([0.04, 0.013, 0.74, 0.038])
    lx.set_axis_off()
    x0 = 0.0
    lx.plot([x0, x0+0.032], [0.55, 0.55], color="orange", lw=2.0,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.036, 0.55, "MLD", fontsize=8, color="orange",
            va="center", transform=lx.transAxes)
    x0 += 0.09

    for cname, (lbl, method, thr, color, ls) in zip(CONFIG_NAMES, CONFIGS):
        lx.plot([x0, x0+0.032], [0.55, 0.55], color=color, lw=1.8, ls="-",
                transform=lx.transAxes, clip_on=False)
        lx.text(x0+0.036, 0.55, lbl, fontsize=7, color=color,
                va="center", transform=lx.transAxes)
        x0 += 0.155

    lx.plot([x0, x0+0.032], [0.55, 0.55], color="#c8c8c8", lw=1.5,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.036, 0.55, "Individual years",
            fontsize=7, color="gray", va="center", transform=lx.transAxes)

    # Suptitle
    lat_l = (f"{abs(rcfg['lat_min'])}°{'S' if rcfg['lat_min']<0 else 'N'}–"
             f"{abs(rcfg['lat_max'])}°{'S' if rcfg['lat_max']<0 else 'N'}")
    lon_l = (f"{abs(rcfg['lon_min'])}°{'W' if rcfg['lon_min']<0 else 'E'}–"
             f"{abs(rcfg['lon_max'])}°{'W' if rcfg['lon_max']<0 else 'E'}")

    fig.suptitle(
        f"T/S profiles — {rcfg['label']}  ({lat_l}, {lon_l})  |  {note_t}\n"
        f"ILD promedio sobre píxeles de acuerdo intermedio  (p50–p75,  {p50:.0f}–{p75:.0f}%)",
        fontsize=11, y=0.968,
    )

    out_fig = OUT_DIR / f"method_agreement_{rkey}.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_fig.name}", flush=True)

print("\n¡Listo! Archivos generados:", flush=True)
for f in sorted(OUT_DIR.glob("method_agreement_*.png")):
    print(f"  displays/{f.name}", flush=True)
