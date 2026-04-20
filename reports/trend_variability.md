# Barrier Layer Depth — Trend and Variability Report

**Period:** 1993–2024
**Data:** CMEMS 0.25° monthly reanalysis, sensitivity ensemble (6 ILD configs)
**Generated:** 2026-03-16

---

## 1. Global Mean BLD Trend (1993–2024)

Linear trend in the globally-averaged annual mean BLD, computed per ILD configuration.

| Config | Trend (m decade⁻¹) | R | p-value | Significant? |
|---|---:|---:|---:|---|
| Gradient −0.015°C/m | +0.80 | 0.618 | 0.0002 | ** |
| Gradient −0.025°C/m | +1.26 | 0.709 | <0.0001 | ** |
| Gradient −0.1°C/m | +1.75 | 0.705 | <0.0001 | ** |
| Diff ΔT = 0.2°C | +0.43 | 0.278 | 0.123 | — |
| Diff ΔT = 0.5°C | +1.08 | 0.521 | 0.002 | ** |
| Diff ΔT = 0.8°C | +1.31 | 0.574 | 0.001 | ** |

**Key points:**
- 5 out of 6 configurations show a statistically significant positive trend (p < 0.05).
- The exception is **Diff 0.2°C**, the strictest difference threshold, which shows the weakest trend and no significance — this config detects barrier layers in ~48% of the ocean, the largest coverage, and is more dominated by noisy detections.
- Trend magnitude ranges from +0.8 to +1.75 m/decade depending on method. The **gradient methods** show the strongest trends.
- The positive trend is a robust physical signal: it appears across independent method families (gradient and difference), which minimises the chance of it being a methodological artefact.

---

## 2. Per-Pixel Trend Map (default config: Gradient −0.1°C/m)

| Metric | Value |
|---|---|
| Ocean pixels with significant trend (p < 0.05) | 19.9% |
| Of those, positive trend | 90.0% |
| Of those, negative trend | 10.0% |

**Interpretation:** While only ~20% of the ocean shows a statistically significant trend at the local level (expected given 32-year record and high spatial variability), the strong directional bias (9:1 positive to negative) confirms the global increase is not dominated by isolated regions.

Negative trends are concentrated in the **subtropical gyres** (notably the eastern subtropics) where barrier layers are thin and infrequent. Positive trends are most pronounced in:
- **W. Pacific Warm Pool** — strongest signal (+6.9 m/decade)
- **Tropical Indian Ocean** — strong and highly significant (+2.4 m/decade)
- **Bay of Bengal** — present but not globally significant

See `displays/bld_trend_map.png` and `displays/bld_trend_ensemble.png`.

---

## 3. Regional Trends (default config: Gradient −0.1°C/m)

| Region | Trend (m decade⁻¹) | R | p-value | Significant? |
|---|---:|---:|---:|---|
| Tropical Indian Ocean | +2.39 | 0.766 | <0.0001 | ** |
| Bay of Bengal | +0.76 | 0.143 | 0.436 | — |
| W. Pacific Warm Pool | +6.94 | 0.468 | 0.007 | ** |
| Amazon Plume | +1.05 | 0.412 | 0.019 | * |

**Key points:**
- **W. Pacific Warm Pool** shows by far the largest trend (+6.94 m/decade), consistent with the known warming and freshening of the western tropical Pacific. The correlation (R=0.47) is moderate — there is substantial interannual variability superimposed on the trend.
- **Tropical Indian Ocean** shows the strongest correlation (R=0.77), meaning the trend there is the cleanest and most monotonic.
- **Bay of Bengal** trend is not statistically significant (p=0.44) despite the strong seasonal cycle. High interannual variability driven by monsoon-related salinity stratification masks any long-term signal at this timescale.
- **Amazon Plume** shows a significant positive trend (p=0.02), consistent with increased freshwater export from the Amazon basin.

See `displays/bld_trend_regions.png`.

---

## 4. Spatial Variability Statistics (1993–2024)

All values are spatial means over the global ocean, default config (Gradient −0.1°C/m), positive BLD only.

| Metric | Value |
|---|---|
| Global mean total temporal std | 12.4 m |
| Global mean seasonal amplitude (max clim − min clim) | 29.3 m |
| Global mean interannual std (deseasonalised annual means) | 5.9 m |

**Key points:**
- **Seasonal variability dominates**: the seasonal amplitude (29.3 m) is ~5× larger than the interannual std (5.9 m). Most of the temporal variance in BLD is seasonal, not interannual.
- **Total std (12.4 m)** integrates both contributions. Regions of highest total std are the Bay of Bengal (strong monsoon cycle) and the W. Pacific Warm Pool (both seasonal and ENSO-driven interannual variability).
- **Interannual std (5.9 m)** is the relevant metric for detecting climate change signals above the noise.

See `displays/bld_variability_maps.png`.

---

## 5. Annual Anomaly Time Series (1993–2024)

Global mean BLD anomaly relative to the 1993–2024 mean (deseasonalised annual means, gradient −0.1°C/m).

| Year | Global anomaly (m) |
|---|---|
| 1998 | −1.73 ← minimum |
| 2022 | +4.81 ← maximum |

**1998** is the minimum anomaly year, coinciding with the strong **1997–98 El Niño** (warm SST, weakened salinity stratification in the W. Pacific). **2022** is the maximum, coinciding with the **2021–23 La Niña** sequence (anomalous freshwater accumulation in the tropical western Pacific and Indian Ocean sectors).

This ENSO fingerprint is consistent with the known physics: El Niño years deepen and weaken the thermocline in the western Pacific, reducing barrier layer thickness; La Niña years do the opposite.

See `displays/bld_anomaly_maps.png`.

---

## 6. Output Figures

| File | Description |
|------|-------------|
| `displays/bld_trend_map.png` | Per-pixel BLD trend (m/decade), gradient −0.1°C/m; stippling = p ≥ 0.05 |
| `displays/bld_trend_regions.png` | Annual mean BLD + trend line for 4 key regions |
| `displays/bld_trend_ensemble.png` | Ensemble mean trend and spread (std across 6 configs) |
| `displays/bld_variability_maps.png` | Total std, seasonal amplitude, interannual std — global maps |
| `displays/bld_anomaly_maps.png` | Annual BLD anomaly maps, every 5th year + extreme years (1998, 2022) |
