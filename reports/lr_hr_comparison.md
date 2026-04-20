# LR vs HR Comparison Report

**Products compared:** CMEMS 0.25° (LR) vs CMEMS 0.083° (HR) monthly reanalysis
**Period:** 2024 (12 months)
**Generated:** 2026-03-12

> **Central question:** Do the two products capture different barrier layer physics, or do apparent differences arise from grid quantization and slight data offsets?

---

## 1. Vertical Grid Structure

The two products use different vertical grids. In the BLD-relevant zone (0–300 m):

| Grid | Levels (0–300 m) | Shallowest | Deepest |
|---|---|---|---|
| LR 0.25° | 34 | 0.51 m | 271 m |
| HR 0.083° | 28 | 0.49 m | 266 m |

Level spacing at key depths:

| Depth | LR spacing (m) | HR spacing (m) | Finer grid |
|---|---|---|---|
| ~10 m | 2.0 | 1.8 | HR |
| ~20 m | 3.3 | 3.1 | HR |
| ~30 m | 4.3 | 5.0 | **LR** |
| ~50 m | 6.6 | 8.4 | **LR** |
| ~75 m | 8.6 | 12.0 | **LR** |
| ~100 m | 11.0 | 17.4 | **LR** |
| ~150 m | 15.8 | 25.2 | **LR** |
| ~200 m | 21.4 | 36.3 | **LR** |

**Key point:** Near the surface (0–25 m), the grids are nearly identical. Below ~30 m, LR is actually *finer* than HR. The common assumption that HR resolves the thermocline better in the vertical is not supported for these two specific products. See `displays/vertical_grid_comparison.png`.

---

## 2. Raw T/S Profile Comparison

Spatially-averaged profiles were compared for three key regions — Amazon Plume (western Atlantic), Bay of Bengal (Indian Ocean), W. Pacific Warm Pool (Pacific) — using all 12 months of 2024. LR profiles were interpolated to the HR depth grid before comparison.

### Temperature (0–300 m, 12 months × depth levels)

| Region | Bias HR−LR (°C) | RMSE (°C) | R | n points |
|---|---|---|---|---|
| Amazon Plume | +0.25 | 0.46 | 0.998 | 324 |
| Bay of Bengal | +0.33 | 0.49 | 0.998 | 324 |
| W. Pacific Warm Pool | +0.24 | 0.37 | 0.999 | 324 |

### Salinity (0–300 m, 12 months × depth levels)

| Region | Bias HR−LR (PSU) | RMSE (PSU) | R | n points |
|---|---|---|---|---|
| Amazon Plume | −0.06 | 0.14 | 0.979 | 324 |
| Bay of Bengal | −0.21 | 0.25 | 0.998 | 324 |
| W. Pacific Warm Pool | −0.05 | 0.06 | 0.987 | 324 |

**Key point:** Both products show near-perfect agreement in raw T/S (R > 0.979 everywhere). HR is consistently slightly warmer (+0.24 to +0.33 °C) and fresher (−0.05 to −0.21 PSU) than LR. There is no strong depth-dependent breakdown. **The BLD difference between the two products does not originate from large differences in the raw T/S data.** See `displays/scatter_profiles_lr_hr.png`.

---

## 3. Global MLD / ILD / BLD Scatter

2024 annual-mean MLD, ILD, and BLD were compared globally. HR was regridded to the LR 0.25° grid using nearest-neighbour interpolation. Only grid cells where both products have valid values are included.

### Global statistics

| Variable | Bias HR−LR (m) | RMSE (m) | R | n cells |
|---|---|---|---|---|
| MLD | +3.0 | 13.4 | 0.905 | 678 442 |
| ILD | +2.9 | 20.1 | 0.863 | 405 414 |
| BLD | +4.8 | 19.5 | 0.839 | 405 414 |

### By ocean basin

**MLD**

| Basin | Bias (m) | RMSE (m) | R |
|---|---|---|---|
| Atlantic | +1.7 | 10.4 | 0.939 |
| Indian | +0.4 | 7.9 | 0.959 |
| Pacific | +1.4 | 5.1 | 0.942 |
| Southern | +8.8 | 24.2 | 0.778 |

**ILD**

| Basin | Bias (m) | RMSE (m) | R |
|---|---|---|---|
| Atlantic | +1.8 | 19.8 | 0.793 |
| Indian | +1.6 | 11.6 | 0.871 |
| Pacific | +4.8 | 24.2 | 0.882 |
| Southern | +1.8 | 10.1 | 0.792 |

**BLD**

| Basin | Bias (m) | RMSE (m) | R |
|---|---|---|---|
| Atlantic | +3.8 | 18.9 | 0.705 |
| Indian | +3.2 | 9.9 | 0.883 |
| Pacific | +6.6 | 24.0 | 0.863 |
| Southern | +3.6 | 7.9 | 0.565 |

**Key points:**
- HR MLD and ILD are slightly deeper than LR on average (bias ~+3 m), consistent with the slight warm bias in temperature profiles.
- HR BLD is slightly larger than LR (bias +4.8 m globally), partly because the MLD bias and ILD bias do not cancel perfectly.
- The Southern Ocean shows the poorest agreement (MLD R = 0.78, BLD R = 0.56), consistent with greater dynamical variability and the coarser representation of eddies in LR.
- The Indian Ocean shows the best agreement (R > 0.88 for all three variables).

See `displays/scatter_bld_lr_hr.png`.

---

## 4. ILD Difference Resolvability

The ILD detection algorithm snaps to discrete depth levels. If |ILD_HR − ILD_LR| ≤ max(spacing_LR, spacing_HR) at that depth, the difference is within grid quantization and **cannot** be attributed to different physical signals. Only differences exceeding the maximum local spacing are considered *genuinely resolvable*.

Resolvability ratio = Δ ILD / max(spacing_LR at ILD_LR, spacing_HR at ILD_HR). Ratio > 1 → resolvable.

### Resolvable months per region and ILD method (out of 12)

| Method | Amazon Plume | Bay of Bengal | W. Pacific Warm Pool |
|---|---|---|---|
| Gradient −0.015 °C/m | 10/12 | 2/12 | 0/12 |
| Gradient −0.025 °C/m | 5/12 | 4/12 | 1/12 |
| Gradient −0.1 °C/m | 3/12 | 10/12 | 5/12 |
| Diff ΔT = 0.2 °C | 4/12 | 3/12 | 2/12 |
| Diff ΔT = 0.5 °C | 3/12 | 4/12 | 0/12 |
| Diff ΔT = 0.8 °C | 6/12 | 4/12 | 2/12 |

### Months where difference is resolvable (ratio > 1)

**Amazon Plume:**
- Gradient −0.015: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Oct, Dec
- Gradient −0.025: Jan, Apr, May, Aug, Oct
- Gradient −0.1: Aug, Sep, Oct
- Diff 0.2 °C: Jan, Feb, Oct, Dec
- Diff 0.5 °C: Jun, Jul, Aug
- Diff 0.8 °C: Jan, Apr, May, Jun, Aug, Oct

**Bay of Bengal:**
- Gradient −0.015: Apr, May
- Gradient −0.025: Jan, Apr, May, Aug
- Gradient −0.1: Jan, Feb, Mar, Apr, Jun, Jul, Aug, Sep, Oct, Nov
- Diff 0.2 °C: Mar, Jun, Jul
- Diff 0.5 °C: Jan, Feb, Aug, Dec
- Diff 0.8 °C: Feb, Jul, Aug, Nov

**W. Pacific Warm Pool:**
- Gradient −0.015: none
- Gradient −0.025: May
- Gradient −0.1: Aug, Sep, Oct, Nov, Dec
- Diff 0.2 °C: Oct, Nov
- Diff 0.5 °C: none
- Diff 0.8 °C: Mar, Dec

**Key points:**
- The resolvability is highly method- and region-dependent. No single pattern holds universally.
- For the strictest gradient threshold (−0.015 °C/m), the Amazon Plume shows 10/12 resolvable months, but the WPWP shows 0/12.
- For the commonly used gradient −0.1 °C/m, the Bay of Bengal has 10/12 resolvable months, while the Amazon Plume has only 3/12.
- For the difference methods, resolvability is generally moderate (2–6/12), with no basin consistently better than others.
- **When differences are not resolvable, the two products should be considered equivalent within the vertical grid uncertainty.** Interpreting the ILD difference as a physical signal in those months is not warranted.

See `displays/resolvability_{region}.png` and `displays/compare_profiles_{region}_{method}_with_uncertainty.png`.

---

## 5. Summary

| Question | Answer |
|---|---|
| Do the raw T/S profiles differ significantly? | No — R > 0.979, biases < 0.5 °C and < 0.25 PSU |
| Is HR vertically finer in the thermocline? | No — LR is actually finer below ~30 m for these specific products |
| Do MLD/ILD/BLD agree globally? | Reasonably (R ~ 0.84–0.91), with HR slightly deeper overall |
| Where do they disagree most? | Southern Ocean (R = 0.57 for BLD) |
| Are ILD differences physically resolvable? | Only in some month/region/method combinations — see Section 4 |
| Main source of BLD difference? | Grid quantization of ILD detection, not raw data differences |

---

## 6. Output Figures

| File | Description |
|---|---|
| `displays/vertical_grid_comparison.png` | LR vs HR depth level spacing, full column and 0–200 m zoom |
| `displays/scatter_profiles_lr_hr.png` | T/S scatter per region, all months × depth levels |
| `displays/scatter_bld_lr_hr.png` | Global MLD/ILD/BLD scatter per ocean basin |
| `displays/compare_lr_hr_annual_mean.png` | Annual mean maps: LR, HR, and LR−HR difference |
| `displays/compare_lr_hr_zoom_regions.png` | Regional zoom maps for 4 key areas |
| `displays/compare_lr_hr_monthly_stats.png` | Monthly global mean BLD and coverage time series |
| `displays/compare_profiles_{region}_{method}.png` | T/S profile overlays, 3 regions × 6 methods (18 figures) |
| `displays/compare_profiles_{region}_{method}_with_uncertainty.png` | Same with ±½ grid spacing uncertainty bands (9 figures) |
| `displays/resolvability_{region}.png` | Resolvability heat-map: 6 methods × 12 months (3 figures) |
