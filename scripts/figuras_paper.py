"""
figuras_paper.py

Genera las figuras para el paper sobre Barrier Layers.

Figuras:
  Fig 1 — Mean BLD, ensemble spread y method agreement (3 paneles)
           → displays/fig1_bld_mean_spread_agreement.png

Uso:
    python scripts/figuras_paper.py
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data/025deg/sensitivity"
OUT_DIR  = ROOT / "displays"
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
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_bld_mean_spread_agreement()
    lat, lon, seas_maps = fig2a_seasonal_ensemble()
    fig2b_seasonal_per_method(lat, lon, seas_maps)
    print("Done.")
