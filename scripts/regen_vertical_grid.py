"""
regen_vertical_grid.py
Grillas verticales LR vs HR — dos paneles:
  Izq : 0–6000 m  (LR: 75 niveles; HR: 50 niveles)
  Der : 0–300 m   (zoom en el rango relevante para BLD)
Cada panel muestra dos columnas de agua con líneas en cada nivel.
Depth levels sourced directly from the CMEMS product catalogue
(copernicusmarine.describe), not from downloaded data.
Sin spines bottom ni right.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "displays"

# ── Actual product depth levels (from CMEMS catalogue) ────────────────────────
# LR: cmems_mod_glo_phy-all_my_0.25deg_P1M-m  — 75 levels, 0.506–5902 m
DLR = np.array([
    0.5058, 1.5559, 2.6677, 3.8563, 5.1404, 6.543, 8.0925, 9.8228,
    11.7737, 13.991, 16.5253, 19.4298, 22.7576, 26.5583, 30.8746, 35.7402,
    41.18, 47.2119, 53.8506, 61.1128, 69.0217, 77.6112, 86.9294, 97.0413,
    108.0303, 120.0, 133.0758, 147.4062, 163.1645, 180.5499, 199.79,
    221.1412, 244.8906, 271.3564, 300.8875, 333.8628, 370.6885, 411.7939,
    457.6256, 508.6399, 565.2923, 628.026, 697.2587, 773.3683, 856.679,
    947.4479, 1045.8542, 1151.9912, 1265.8615, 1387.377, 1516.3636,
    1652.5685, 1795.6708, 1945.2955, 2101.0266, 2262.4216, 2429.0251,
    2600.3804, 2776.0393, 2955.5703, 3138.5649, 3324.6409, 3513.4456,
    3704.6567, 3897.9819, 4093.1587, 4289.9526, 4488.1548, 4687.5811,
    4888.0698, 5089.4785, 5291.6831, 5494.5752, 5698.0605, 5902.0576,
], dtype=np.float32)

# HR: cmems_mod_glo_phy_my_0.083deg_P1M-m  — 50 levels, 0.494–5727.9 m
DHR = np.array([
    0.4940, 1.5414, 2.6457, 3.8195, 5.0782, 6.4406, 7.9296, 9.573,
    11.405, 13.4671, 15.81, 18.4956, 21.5988, 25.2114, 29.4447, 34.4342,
    40.3441, 47.3737, 55.7643, 65.8073, 77.8539, 92.3261, 109.7293,
    130.666, 155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.213,
    453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1062.44, 1245.291,
    1452.251, 1684.284, 1941.893, 2225.078, 2533.336, 2865.703, 3220.82,
    3597.032, 3992.484, 4405.224, 4833.291, 5274.784, 5727.917,
], dtype=np.float32)

CLR = "#2980b9"   # LR blue
CHR = "#e74c3c"   # HR red

# ── Figure: 1 row × 2 panels ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=False)
fig.patch.set_facecolor("white")

panels = [
    ("0–6000 m  (columna completa)", 6000),
    ("0–300 m  (zoom relevante para BLD)", 300),
]

for ax, (title, dmax) in zip(axes, panels):

    lr_vis = DLR[DLR <= dmax]          # LR levels in range
    hr_vis = DHR[DHR <= dmax]          # HR levels in range

    # ── Draw level lines as horizontal segments ──────────────────────────────
    # LR: left column  x = 0.1–0.45
    for d in lr_vis:
        ax.hlines(d, 0.10, 0.45, colors=CLR, lw=1.4, alpha=0.85)

    # HR: right column  x = 0.55–0.90
    for d in hr_vis:
        ax.hlines(d, 0.55, 0.90, colors=CHR, lw=1.4, alpha=0.85)

    # Column spines (vertical lines)
    ax.vlines(0.275, 0, dmax, colors=CLR, lw=0.8, alpha=0.3)
    ax.vlines(0.725, 0, dmax, colors=CHR, lw=0.8, alpha=0.3)

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=CLR, lw=1.4,
               label=f"LR  0.25°  ({len(lr_vis)} niveles)"),
        Line2D([0], [0], color=CHR, lw=1.4,
               label=f"HR  0.083°  ({len(hr_vis)} niveles)"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=9,
              frameon=False, handlelength=1.5,
              prop={"weight": "normal", "size": 9})

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_ylim(dmax, 0)          # depth increases downward
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    ax.set_ylabel("Profundidad (m)", fontsize=10, fontweight="normal")
    # no ax.set_title — info goes in the slide caption
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()

out = OUT_DIR / "vertical_grid_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
