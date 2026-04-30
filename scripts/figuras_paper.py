"""
figuras_paper.py

Genera las figuras para el paper sobre Barrier Layers.

Figuras:
  Fig 1 — Mean BLD, ensemble spread y method agreement (3 paneles)
           → displays/fig1_bld_mean_spread_agreement.png

Uso:
    python scripts/figuras_paper.py
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import zarr as zarr_lib
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SENS_DIR  = ROOT / "data/025deg/sensitivity"
PT_DIR    = ROOT / "data/025deg"
OUT_DIR   = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colormap BLD: blanco en 0, azules crecientes ──────────────────────────────
BLD_CMAP = LinearSegmentedColormap.from_list(
    "bld_blues", ["white", "#c6dbef", "#6baed6", "#2171b5", "#08306b"], N=256)

# ── Configs de detección ──────────────────────────────────────────────────────
CONFIGS = [
    ("gradient_015",  "Gradient −0.015°C/m"),
    ("gradient_025",  "Gradient −0.025°C/m"),
    ("gradient_100",  "Gradient −0.1°C/m"),
    ("difference_02", "Diff 0.2°C"),
    ("difference_05", "Diff 0.5°C"),
    ("difference_08", "Diff 0.8°C"),
]

# ── Helpers de visualización ──────────────────────────────────────────────────
def _base_map(ax, fig, img, title, cbar_label, levels=None):
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.set_global()
    ax.set_title(title, fontsize=10, pad=6, loc="left")
    cb = fig.colorbar(img, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.7, aspect=35)
    cb.set_label(cbar_label, fontsize=8)
    if levels is not None:
        cb.set_ticks(levels)
    cb.ax.tick_params(labelsize=7)

def add_bld_map(ax, fig, lon, lat, data, title,
                levels=(0, 5, 10, 15, 20, 30, 40, 50, 75, 100)):
    """BLD map con colorbar discreta."""
    levels = np.array(levels)
    cmap   = plt.get_cmap("Blues", len(levels) - 1)
    norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    img = ax.pcolormesh(lon, lat, np.ma.masked_where(data <= 0, data),
                        norm=norm, cmap=cmap,
                        transform=ccrs.PlateCarree(), rasterized=True)
    _base_map(ax, fig, img, title, "BLD (m)", levels=levels)

def add_spread_map(ax, fig, lon, lat, data, title,
                   levels=(0, 2, 4, 6, 8, 10, 15, 20, 30)):
    """Spread map con colorbar discreta."""
    levels = np.array(levels)
    cmap   = plt.get_cmap("YlOrRd", len(levels) - 1)
    norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    img = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap,
                        transform=ccrs.PlateCarree(), rasterized=True)
    _base_map(ax, fig, img, title, "σ (m)", levels=levels)

def add_agreement_map(ax, fig, lon, lat, data, title,
                      levels=(0, 20, 40, 60, 80, 100)):
    """Agreement map con colorbar discreta."""
    levels = np.array(levels)
    cmap   = plt.get_cmap("RdYlGn", len(levels) - 1)
    norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    img = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap,
                        transform=ccrs.PlateCarree(), rasterized=True)
    _base_map(ax, fig, img, title, "% configs with BLD > 0", levels=levels)


# ══════════════════════════════════════════════════════════════════════════════
# Figura 1 — Mean BLD, ensemble spread y method agreement
# ══════════════════════════════════════════════════════════════════════════════
def fig1_bld_mean_spread_agreement():
    print("=== Figura 1: Mean BLD, spread y agreement ===")

    # Coordenadas de referencia
    ref  = xr.open_zarr(SENS_DIR / "bld_gradient_015_1993_2024.zarr", consolidated=False)
    lat  = ref["latitude"].values
    lon  = ref["longitude"].values

    # Cargar media temporal y tasa de detección por config
    mean_maps, detect_maps = {}, {}
    for name, label in CONFIGS:
        print(f"  {name}...", end=" ", flush=True)
        bld = xr.open_zarr(SENS_DIR / f"bld_{name}_1993_2024.zarr",
                           consolidated=False)["bld"]
        mean_maps[name]   = bld.mean(dim="time").compute().values
        detect_maps[name] = (bld > 0).mean(dim="time").compute().values
        print("ok")

    # Estadísticas del ensemble
    stack        = np.stack(list(mean_maps.values()), axis=0)   # (6, lat, lon)
    ens_mean     = np.nanmean(stack, axis=0)
    ens_std      = np.nanstd(stack,  axis=0)
    detect_agree = np.mean(
        np.stack(list(detect_maps.values()), axis=0), axis=0) * 100

    # Figura — 3 paneles en una fila
    proj = ccrs.Robinson()
    fig = plt.figure(figsize=(10, 8.75))
    fig.patch.set_facecolor("white")

    # GridSpec 2×4: a=[0,0:2], b=[0,2:4], c=[1,1:3] → los 3 mapas del mismo tamaño
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           left=0.01, right=0.99, top=0.97, bottom=0.04,
                           hspace=0.2, wspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0:2], projection=proj)
    ax_b = fig.add_subplot(gs[0, 2:4], projection=proj)
    ax_c = fig.add_subplot(gs[1, 1:3], projection=proj)

    add_bld_map(ax_a, fig, lon, lat, ens_mean, "a)  Mean BLD")
    add_spread_map(ax_b, fig, lon, lat, ens_std, "b)  Ensemble spread,  σ(BLD)",
                   levels=np.arange(0, 55, 5))
    add_agreement_map(ax_c, fig, lon, lat, detect_agree, "c)  Detection agreement",
                      levels=np.arange(0, 110, 10))

    out = OUT_DIR / "fig1_bld_mean_spread_agreement.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 2a — Seasonal ensemble mean BLD (4 paneles)
# ══════════════════════════════════════════════════════════════════════════════
SEASONS      = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABEL = {"DJF": "Dec–Jan–Feb", "MAM": "Mar–Apr–May",
                "JJA": "Jun–Jul–Aug", "SON": "Sep–Oct–Nov"}

def _load_seasonal_maps():
    """Carga media estacional (lat, lon) por config. Devuelve dict config → dict season → array."""
    ref = xr.open_zarr(SENS_DIR / "bld_gradient_015_1993_2024.zarr", consolidated=False)
    lat = ref["latitude"].values
    lon = ref["longitude"].values

    seas_maps = {}
    for name, label in CONFIGS:
        print(f"  {name}...", end=" ", flush=True)
        bld = xr.open_zarr(SENS_DIR / f"bld_{name}_1993_2024.zarr",
                           consolidated=False)["bld"]
        bld_pos = bld.where(bld > 0)
        seas_maps[name] = {
            s: bld_pos.sel(time=bld_pos.time.dt.season == s).mean(dim="time").compute().values
            for s in SEASONS
        }
        print("ok")
    return lat, lon, seas_maps


def fig2a_seasonal_ensemble():
    print("=== Figura 2a: Seasonal ensemble mean BLD ===")
    lat, lon, seas_maps = _load_seasonal_maps()

    # Media del ensemble por estación
    ens_seas = {}
    for s in SEASONS:
        stack = np.stack([seas_maps[name][s] for name, _ in CONFIGS], axis=0)
        ens_seas[s] = np.nanmean(stack, axis=0)

    levels = np.arange(0, 110, 10)

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.5),
                             subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97,
                        bottom=0.1, hspace=0.08, wspace=0.02)

    cmap_d = plt.get_cmap("Blues", len(levels) - 1)
    norm_d = BoundaryNorm(levels, ncolors=cmap_d.N, clip=True)

    panel_labels = ["a)", "b)", "c)", "d)"]
    for ax, s, lbl in zip(axes.flat, SEASONS, panel_labels):
        data = np.ma.masked_where(~np.isfinite(ens_seas[s]) | (ens_seas[s] <= 0), ens_seas[s])
        img  = ax.pcolormesh(lon, lat, data, norm=norm_d, cmap=cmap_d,
                             transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
        ax.set_global()
        ax.set_title(f"{lbl}  {s}  ({SEASON_LABEL[s]})", fontsize=10, pad=4, loc="left")

    # Colorbar compartida
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("BLD (m)", fontsize=8)
    cb.set_ticks(levels)
    cb.ax.tick_params(labelsize=7)

    out = OUT_DIR / "fig2a_seasonal_ensemble.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")
    return lat, lon, seas_maps  # reutilizable por fig2b


def fig2b_seasonal_per_method(lat=None, lon=None, seas_maps=None):
    print("=== Figura 2b: Seasonal BLD per method ===")
    if seas_maps is None:
        lat, lon, seas_maps = _load_seasonal_maps()

    # Escala compartida entre todos los paneles
    levels = np.arange(0, 110, 10)
    cmap_d = plt.get_cmap("Blues", len(levels) - 1)
    norm_d = BoundaryNorm(levels, ncolors=cmap_d.N, clip=True)

    proj = ccrs.Robinson()
    pc   = ccrs.PlateCarree()

    # 6 filas (métodos) × 4 columnas (estaciones)
    fig, axes = plt.subplots(6, 4, figsize=(18, 16),
                             subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97,
                        bottom=0.05, hspace=0.15, wspace=0.04)

    for ri, (name, label) in enumerate(CONFIGS):
        for ci, s in enumerate(SEASONS):
            ax   = axes[ri, ci]
            data = np.ma.masked_where(
                ~np.isfinite(seas_maps[name][s]) | (seas_maps[name][s] <= 0),
                seas_maps[name][s])
            ax.pcolormesh(lon, lat, data, norm=norm_d, cmap=cmap_d,
                          transform=pc, rasterized=True)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
            ax.set_global()
            # Título: estación en fila 0, método en columna 0
            if ri == 0:
                ax.set_title(f"{s}\n{SEASON_LABEL[s]}", fontsize=9, pad=4, loc="center")
            if ci == 0:
                ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                        fontsize=8, va="center", ha="right", rotation=90)

    # Colorbar compartida al fondo
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("BLD (m)", fontsize=9)
    cb.set_ticks(levels)
    cb.ax.tick_params(labelsize=8)

    out = OUT_DIR / "fig2b_seasonal_per_method.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 4 — Trends en ILD, MLD y BLD
# ══════════════════════════════════════════════════════════════════════════════
TREND_CACHE = ROOT / "data/025deg/trend_cache.npz"
YEARS       = np.arange(32, dtype=float)   # 1993–2024

def _vect_linregress(arr_tyx, x):
    """Regresión OLS vectorizada → slope (m/yr), p-value."""
    xm  = x.mean(); Sxx = np.sum((x - xm) ** 2)
    ym  = np.nanmean(arr_tyx, axis=0)
    Sxy = np.nansum((arr_tyx - ym[None]) * (x - xm)[:, None, None], axis=0)
    slope = Sxy / Sxx
    inter = ym - slope * xm
    yhat  = inter[None] + slope[None] * x[:, None, None]
    resid = arr_tyx - yhat
    n     = np.sum(np.isfinite(arr_tyx), axis=0).astype(float)
    SSres = np.nansum(resid ** 2, axis=0)
    se    = np.sqrt(SSres / np.where(n > 2, n - 2, np.nan) / Sxx)
    tstat = slope / np.where(se > 0, se, np.nan)
    df    = np.where(n > 2, n - 2, 1).astype(float)
    pval  = 2 * t_dist.sf(np.abs(tstat), df=df)
    slope[n < 5] = np.nan; pval[n < 5] = np.nan
    return slope, pval


def _load_trend_maps():
    """Carga o computa trends per-pixel para todos los configs y variables.
    Devuelve dict: config → var → (trend m/decade, pval)
    También devuelve lat, lon.
    """
    if TREND_CACHE.exists():
        print("  Cargando trends desde caché...", flush=True)
        cache = np.load(TREND_CACHE, allow_pickle=True)
        lat   = cache["lat"]
        lon   = cache["lon"]
        results = cache["results"].item()
        return lat, lon, results

    print("  Computando trends (puede tardar)...", flush=True)
    ref = zarr_lib.open(SENS_DIR / "bld_gradient_015_1993_2024.zarr", mode="r")
    lat = ref["latitude"][:]
    lon = ref["longitude"][:]

    results = {}
    for name, label in CONFIGS:
        print(f"    {name}...", flush=True)
        results[name] = {}
        z = zarr_lib.open(SENS_DIR / f"bld_{name}_1993_2024.zarr", mode="r")
        nt   = z["bld"].shape[0]
        nlat = z["bld"].shape[1]
        nlon = z["bld"].shape[2]
        for varname, pos_only in [("ild", True), ("mld", False), ("bld", True)]:
            ann = np.full((nt // 12, nlat, nlon), np.nan, dtype=np.float32)
            for yi in range(nt // 12):
                chunk = z[varname][yi * 12:(yi + 1) * 12][:]
                if pos_only:
                    chunk = np.where(chunk > 0, chunk, np.nan)
                ann[yi] = np.nanmean(chunk, axis=0)
            slope, pval = _vect_linregress(ann.astype(np.float64), YEARS)
            results[name][varname] = (slope * 10, pval)   # m/decade
        print(f"    {name} ok")

    np.savez(TREND_CACHE, lat=lat, lon=lon, results=np.array(results, dtype=object))
    print("  Caché guardado.")
    return lat, lon, results


def _trend_map(ax, fig, lon, lat, trend, pval, norm, cmap, title, row0=False, col0=False, label_row=""):
    """Dibuja un mapa de trend con stippling y colorbar."""
    ax.pcolormesh(lon, lat, trend, norm=norm, cmap=cmap,
                  transform=ccrs.PlateCarree(), rasterized=True)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
    ax.set_global()
    # Stippling: puntos negros donde p >= 0.05
    insig_i, insig_j = np.where((pval >= 0.05) & np.isfinite(trend))
    step = max(1, insig_i.size // 3000)
    if insig_i.size > 0:
        ax.scatter(lon[insig_j[::step]], lat[insig_i[::step]],
                   s=0.3, color="k", alpha=0.25, transform=ccrs.PlateCarree(), zorder=4)
    if row0:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4, loc="center")
    if col0 and label_row:
        ax.text(-0.02, 0.5, label_row, transform=ax.transAxes,
                fontsize=7.5, va="center", ha="right", rotation=90)


def fig4a_trend_ensemble():
    print("=== Figura 4a: Trend ensemble ILD / MLD / BLD ===")
    lat, lon, results = _load_trend_maps()

    # Ensemble mean de trends
    ens = {}
    for var in ["ild", "mld", "bld"]:
        stack_t = np.stack([results[n][var][0] for n, _ in CONFIGS], axis=0)
        stack_p = np.stack([results[n][var][1] for n, _ in CONFIGS], axis=0)
        ens[var] = (np.nanmean(stack_t, axis=0),
                    np.nanmean(stack_p, axis=0))   # media del pval como proxy

    # Escalas de color
    def sym_norm(arr):
        v = np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98)
        return mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 10),
                             subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96,
                        bottom=0.06, hspace=0.22)

    panels = [
        ("ild", "a)  ILD trend", "RdBu_r",  "ILD trend (m decade⁻¹)"),
        ("mld", "b)  MLD trend", "RdBu_r",  "MLD trend (m decade⁻¹)"),
        ("bld", "c)  BLD trend", "RdBu_r",  "BLD trend (m decade⁻¹)"),
    ]
    for ax, (var, title, cmap, cblabel) in zip(axes, panels):
        t, p   = ens[var]
        norm   = sym_norm(t)
        img    = ax.pcolormesh(lon, lat, t, norm=norm, cmap=cmap,
                               transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=2)
        ax.set_global()
        insig_i, insig_j = np.where((p >= 0.05) & np.isfinite(t))
        step = max(1, insig_i.size // 3000)
        if insig_i.size > 0:
            ax.scatter(lon[insig_j[::step]], lat[insig_i[::step]],
                       s=0.3, color="k", alpha=0.25,
                       transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(title, fontsize=10, pad=4, loc="left")
        cb = fig.colorbar(img, ax=ax, orientation="horizontal",
                          pad=0.04, shrink=0.7, aspect=35)
        cb.set_label(cblabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    out = OUT_DIR / "fig4a_trend_ensemble.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")
    return lat, lon, results


def fig4b_trend_per_method(lat=None, lon=None, results=None):
    print("=== Figura 4b: Trend por método (6×3) ===")
    if results is None:
        lat, lon, results = _load_trend_maps()

    VARS = [
        ("ild", "ILD trend (m decade⁻¹)", "RdBu_r"),
        ("mld", "MLD trend (m decade⁻¹)", "RdBu_r"),
        ("bld", "BLD trend (m decade⁻¹)", "RdBu_r"),
    ]

    # Norma compartida por variable (entre todos los métodos)
    norms = {}
    for var, *_ in VARS:
        all_vals = np.concatenate([
            results[n][var][0][np.isfinite(results[n][var][0])]
            for n, _ in CONFIGS
        ])
        v = np.nanpercentile(np.abs(all_vals), 98)
        norms[var] = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(6, 3, figsize=(14, 18),
                             subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96,
                        bottom=0.06, hspace=0.1, wspace=0.04)

    imgs = {var: None for var, *_ in VARS}

    for ri, (name, label) in enumerate(CONFIGS):
        for ci, (var, cblabel, cmap) in enumerate(VARS):
            ax   = axes[ri, ci]
            t, p = results[name][var]
            norm = norms[var]
            img  = ax.pcolormesh(lon, lat, t, norm=norm, cmap=cmap,
                                 transform=ccrs.PlateCarree(), rasterized=True)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=2)
            ax.set_global()
            insig_i, insig_j = np.where((p >= 0.05) & np.isfinite(t))
            step = max(1, insig_i.size // 2000)
            if insig_i.size > 0:
                ax.scatter(lon[insig_j[::step]], lat[insig_i[::step]],
                           s=0.2, color="k", alpha=0.2,
                           transform=ccrs.PlateCarree(), zorder=4)
            if ri == 0:
                ax.set_title(var.upper(), fontsize=10, fontweight="bold",
                             pad=4, loc="center")
            if ci == 0:
                ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                        fontsize=8, va="center", ha="right", rotation=90)
            imgs[var] = img

    # Colorbar por columna
    for ci, (var, cblabel, cmap) in enumerate(VARS):
        cbar_ax = fig.add_axes([0.07 + ci * 0.31, 0.03, 0.28, 0.012])
        cb = fig.colorbar(imgs[var], cax=cbar_ax, orientation="horizontal")
        cb.set_label(cblabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    out = OUT_DIR / "fig4b_trend_per_method.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 3 — Perfiles regionales y series temporales
# ══════════════════════════════════════════════════════════════════════════════
from src.barrier_layers import ild_from_temp_profile

# 3 puntos: nombre, lat, lon, color identificador, profundidad máxima del perfil
FIG3_POINTS = [
    ("Southern Ocean",      -54.75,  60.25, "#1a3a6b", 300),
    ("Amazon Plume",          4.25, -41.75, "#7b3a00", 200),
    ("W. Pacific Warm Pool",  5.00, 155.00, "#8b1a1a", 250),
]

FIG3_NC = {
    "Southern Ocean":       PT_DIR / "single_point_southern_ocean.nc",
    "Amazon Plume":         PT_DIR / "single_point_tropical_atlantic.nc",
    "W. Pacific Warm Pool": PT_DIR / "single_point_warm_pool.nc",
}

# Colores / estilos para las 6 configs ILD (igual que en los demás scripts)
ILD_STYLES = [
    ("Gradient −0.015°C/m", "gradient",   -0.015, "#1a6faf", "--"),
    ("Gradient −0.025°C/m", "gradient",   -0.025, "#4dacd6", "--"),
    ("Gradient −0.1°C/m",   "gradient",   -0.1,   "#91d4f5", "--"),
    ("Diff 0.2°C",          "difference",  0.2,   "#2a7a3b", "-."),
    ("Diff 0.5°C",          "difference",  0.5,   "#5dbf6e", "-."),
    ("Diff 0.8°C",          "difference",  0.8,   "#a8dbb0", "-."),
]


def _download_warm_pool():
    """Descarga perfil mensual 1993-2024 en el Pacific Warm Pool (5°N, 155°E) si no existe."""
    nc_path = FIG3_NC["W. Pacific Warm Pool"]
    if nc_path.exists():
        return
    print("  Descargando W. Pacific Warm Pool desde CMEMS (1993–2024) ...")
    import copernicusmarine as cm
    ds = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
        variables=["thetao_glor", "so_glor", "mlotst_glor"],
        minimum_longitude=155.0, maximum_longitude=155.0,
        minimum_latitude=5.0,   maximum_latitude=5.0,
        minimum_depth=0.5057600140571594,
        maximum_depth=508.639892578125,
        coordinates_selection_method="nearest",
    )
    pt = ds.squeeze().drop_vars(["latitude", "longitude"], errors="ignore")
    nc_out = xr.Dataset(
        {
            "temp": xr.DataArray(pt["thetao_glor"].values, dims=["time", "depth"],
                                 coords={"time": pt.time, "depth": pt.depth}),
            "sal":  xr.DataArray(pt["so_glor"].values,     dims=["time", "depth"],
                                 coords={"time": pt.time, "depth": pt.depth}),
            "mld":  xr.DataArray(pt["mlotst_glor"].values, dims=["time"],
                                 coords={"time": pt.time}),
        },
        attrs={"latitude": 5.0, "longitude": 155.0},
    )
    nc_out.to_netcdf(nc_path)
    print(f"  Guardado → {nc_path.name}")


def _load_point_data(nc_path):
    ds    = xr.open_dataset(nc_path)
    depth = ds["depth"].values.astype(float)
    temp  = ds["temp"].values.astype(float)   # (384, ndepth)
    sal   = ds["sal"].values.astype(float)    # (384, ndepth)
    mld   = ds["mld"].values.astype(float)    # (384,)
    times = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    return depth, temp, sal, mld, times


def _compute_ilds_timeseries(temp, depth):
    """Calcula ILD para cada mes y cada config. Devuelve (6, nt)."""
    nt   = temp.shape[0]
    ilds = np.full((6, nt), np.nan)
    for ci, (_, method, thr, color, ls) in enumerate(ILD_STYLES):
        for ti in range(nt):
            ilds[ci, ti] = ild_from_temp_profile(
                temp[ti], depth, method=method, threshold=thr, use_abs=True
            )
    return ilds


def fig3_regional_profiles_timeseries():
    print("=== Figura 3: Perfiles regionales y series temporales ===")
    _download_warm_pool()

    # Cargar datos y calcular ILD
    rows = []
    for pname, plat, plon, pcolor, dmax in FIG3_POINTS:
        print(f"  {pname}: cargando...", end=" ", flush=True)
        depth, temp, sal, mld, times = _load_point_data(FIG3_NC[pname])
        ilds = _compute_ilds_timeseries(temp, depth)
        rows.append(dict(name=pname, lat=plat, lon=plon, color=pcolor, dmax=dmax,
                         depth=depth, temp=temp, sal=sal, mld=mld, times=times,
                         ilds=ilds))
        print("ok")

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.10,
                        hspace=0.40, wspace=0.40)

    yr_unique = np.arange(1993, 2025)
    yr_x      = yr_unique.astype(float) - 1993.0

    for ri, row in enumerate(rows):
        pname   = row["name"]
        pcolor  = row["color"]
        depth   = row["depth"]
        temp    = row["temp"]
        sal     = row["sal"]
        mld_arr = row["mld"]
        times   = row["times"]
        ilds    = row["ilds"]
        dmax    = row["dmax"]
        dmask   = depth <= dmax

        months  = times.month.values

        # ── Perfiles climatológicos por mes ─────────────────────────────────
        ax_T = axes[ri, 0]
        ax_S = axes[ri, 1]
        for m in range(1, 13):
            mm = months == m
            ax_T.plot(np.nanmean(temp[mm][:, dmask], axis=0),
                      depth[dmask], lw=0.6, color="gray", alpha=0.45)
            ax_S.plot(np.nanmean(sal[mm][:, dmask], axis=0),
                      depth[dmask], lw=0.6, color="gray", alpha=0.45)

        # Mean anual
        t_mean = np.nanmean(temp[:, dmask], axis=0)
        s_mean = np.nanmean(sal[:, dmask],  axis=0)
        ax_T.plot(t_mean, depth[dmask], lw=2.0, color=pcolor)
        ax_S.plot(s_mean, depth[dmask], lw=2.0, color=pcolor)

        # MLD media
        mld_mean = float(np.nanmean(mld_arr))
        for ax in (ax_T, ax_S):
            ax.axhline(mld_mean, color="darkorange", lw=1.6, ls="-", zorder=5)

        # ILD desde perfil medio
        for _, method, thr, color, ls in ILD_STYLES:
            ild_c = ild_from_temp_profile(np.nanmean(temp[:, :], axis=0),
                                          depth, method=method, threshold=thr,
                                          use_abs=True)
            if np.isfinite(ild_c) and ild_c <= dmax:
                for ax in (ax_T, ax_S):
                    ax.axhline(ild_c, color=color, lw=1.2, ls=ls, zorder=4)

        # Estilos eje T
        lat_s = f"{abs(row['lat']):.2f}{'°S' if row['lat'] < 0 else '°N'}"
        lon_s = f"{abs(row['lon']):.2f}{'°W' if row['lon'] < 0 else '°E'}"
        ax_T.set_ylabel(f"{pname}\n({lat_s}, {lon_s})\n\nDepth (m)", fontsize=8)
        ax_T.set_xlabel("Temperature (°C)", fontsize=8)
        ax_S.set_xlabel("Salinity (psu)", fontsize=8)
        for ax in (ax_T, ax_S):
            ax.set_ylim(dmax, 0)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="both", lw=0.3, alpha=0.3)

        # ── Serie temporal anual ─────────────────────────────────────────────
        ax_ts = axes[ri, 2]
        years = times.year.values

        mld_ann     = np.full(len(yr_unique), np.nan)
        ild_ann     = np.full((6, len(yr_unique)), np.nan)
        for yi, yr in enumerate(yr_unique):
            mask = years == yr
            mld_ann[yi] = np.nanmean(mld_arr[mask])
            for ci in range(6):
                v = ilds[ci, mask]
                pos = v[v > 0]
                if pos.size > 0:
                    ild_ann[ci, yi] = np.nanmean(pos)

        # Relleno entre mín/máx ILD (spread)
        ild_ens  = np.nanmean(ild_ann, axis=0)
        ild_lo   = np.nanmin(ild_ann,  axis=0)
        ild_hi   = np.nanmax(ild_ann,  axis=0)
        bld_ens  = ild_ens - mld_ann

        # Métodos individuales (líneas finas)
        for ci, (_, method, thr, color, ls) in enumerate(ILD_STYLES):
            ax_ts.plot(yr_unique, ild_ann[ci], color=color, lw=0.9,
                       alpha=0.55, ls=ls)

        # Relleno spread ILD
        valid_fill = np.isfinite(ild_lo) & np.isfinite(ild_hi)
        if valid_fill.any():
            ax_ts.fill_between(yr_unique[valid_fill],
                               ild_lo[valid_fill], ild_hi[valid_fill],
                               color="#4dacd6", alpha=0.15)

        # BLD ensemble mean (solo cuando positivo)
        bld_ens_pos = np.where(bld_ens > 0, bld_ens, np.nan)

        # MLD, ensemble ILD y BLD (líneas gruesas)
        ax_ts.plot(yr_unique, mld_ann,     color="darkorange", lw=2.0,
                   label="MLD", zorder=5)
        ax_ts.plot(yr_unique, ild_ens,     color="#1a3a6b",    lw=2.0,
                   label="ILD (ens. mean)", zorder=5)
        ax_ts.plot(yr_unique, bld_ens_pos, color="#7b2d8b",    lw=2.0,
                   label="BLD (ens. mean)", zorder=5)

        # Líneas de tendencia (OLS)
        for arr, col in [(mld_ann, "darkorange"), (ild_ens, "#1a3a6b"),
                         (bld_ens_pos, "#7b2d8b")]:
            v = np.isfinite(arr)
            if v.sum() > 5:
                m_, b_ = np.polyfit(yr_x[v], arr[v], 1)
                ax_ts.plot(yr_unique[v], m_ * yr_x[v] + b_,
                           color=col, lw=1.4, ls="--", zorder=6)

        ax_ts.set_ylabel("Depth (m)", fontsize=8)
        ax_ts.set_xlabel("Year", fontsize=8)
        ax_ts.tick_params(labelsize=7)
        ax_ts.invert_yaxis()
        ax_ts.spines["top"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)
        ax_ts.grid(axis="y", lw=0.3, alpha=0.35)

    # ── Títulos de columna (solo fila 0) ────────────────────────────────────
    for ci, ttl in enumerate(["a)  Temperature profiles",
                               "b)  Salinity profiles",
                               "c)  ILD & MLD time series"]):
        axes[0, ci].set_title(ttl, fontsize=10, fontweight="bold", pad=5, loc="left")

    # ── Leyenda compartida ───────────────────────────────────────────────────
    handles = [Line2D([0], [0], color="darkorange", lw=2.0, label="MLD (mean)"),
               Line2D([0], [0], color="#1a3a6b",    lw=2.0, label="ILD (ensemble mean)"),
               Line2D([0], [0], color="#7b2d8b",    lw=2.0, label="BLD (ensemble mean)")]
    for lbl, method, thr, color, ls in ILD_STYLES:
        handles.append(Line2D([0], [0], color=color, lw=1.2, ls=ls, label=f"ILD {lbl}"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7.5, framealpha=0.7, bbox_to_anchor=(0.5, -0.01))

    out = OUT_DIR / "fig3_regional_profiles_timeseries.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_bld_mean_spread_agreement()
    lat, lon, seas_maps = fig2a_seasonal_ensemble()
    fig2b_seasonal_per_method(lat, lon, seas_maps)
    fig3_regional_profiles_timeseries()
    lat, lon, results = fig4a_trend_ensemble()
    fig4b_trend_per_method(lat, lon, results)
    print("Done.")
