import numpy as np
import xarray as xr


def ild_from_temp_profile(temp_1d, depth_1d, method="gradient", threshold=-0.1,
                          ref_depth=10.0, use_abs=False):
    """
    Estimate the Isothermal Layer Depth (ILD) from a 1-D temperature profile.

    Parameters
    ----------
    method : "gradient" | "difference"
        gradient  — first depth below 5 m where dT/dz <= threshold (threshold < 0).
        difference — first depth below ref_depth where T(z) departs from T(ref_depth)
                    by more than threshold (threshold > 0).
    threshold : float
        gradient  : negative value, e.g. -0.015, -0.025, -0.1 (°C/m)
        difference: positive value, e.g.  0.2,   0.5,   0.8  (°C)
    ref_depth : float
        Reference depth for the difference method (default 10 m).
    use_abs : bool
        If True, use absolute values to handle temperature inversions:
          gradient  : |dT/dz| >= |threshold|  (detects both cooling and warming layers)
          difference: |T(z) - T(ref)| > threshold  (detects both warm and cold anomalies)
        Default False preserves the original signed behaviour.
    """
    temp  = np.asarray(temp_1d,  dtype=float)
    depth = np.asarray(depth_1d, dtype=float)

    if temp.size < 2 or depth.size < 2:
        return np.nan

    if method == "gradient":
        mask = depth > 5.0
        if np.sum(mask) < 2:
            return np.nan
        t, d = temp[mask], depth[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.diff(t) / np.diff(d)
        depth_mid = (d[:-1] + d[1:]) / 2.0
        if use_abs:
            below = np.abs(grad) >= np.abs(threshold)
        else:
            below = grad <= threshold        # original: only cooling
        return float(depth_mid[np.argmax(below)]) if np.any(below) else np.nan

    elif method == "difference":
        if depth[0] > ref_depth or depth[-1] < ref_depth:
            return np.nan
        T_ref     = float(np.interp(ref_depth, depth, temp))
        below_ref = depth > ref_depth
        if not np.any(below_ref):
            return np.nan
        t_sub, d_sub = temp[below_ref], depth[below_ref]
        if use_abs:
            crossed = np.abs(t_sub - T_ref) > threshold   # any departure
        else:
            crossed = t_sub < T_ref - threshold            # original: only cooling
        return float(d_sub[np.argmax(crossed)]) if np.any(crossed) else np.nan

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'gradient' or 'difference'.")


def compute_segment_gradient(temp, depth):
    temp = np.asarray(temp, dtype=float)
    depth = np.asarray(depth, dtype=float)

    if temp.ndim != 1 or depth.ndim != 1 or temp.size != depth.size or temp.size < 2:
        raise ValueError("temp and depth must be 1D arrays of equal length with at least 2 points.")
    if np.any(np.diff(depth) == 0):
        raise ValueError("depth cannot contain repeated values.")

    gradient = (temp[1:] - temp[:-1]) / (depth[1:] - depth[:-1])
    depth_mid = (depth[1:] + depth[:-1]) / 2.0

    return depth_mid, gradient


def first_depth_below_threshold(gradient, depth_mid, threshold=-0.1):
    depth_mask = depth_mid > 5.0

    if not np.any(depth_mask):
        return None, None

    gradient_filtered = gradient[depth_mask]
    depth_mid_filtered = depth_mid[depth_mask]

    below = gradient_filtered <= threshold
    if np.any(below):
        idx = np.argmax(below)
        return depth_mid_filtered[idx], gradient_filtered[idx]
    return None, None


def compute_global_bld(ds, temp_var, mld_var, method="gradient", threshold=-0.1,
                       ref_depth=10.0, use_abs=False):
    ild = xr.apply_ufunc(
        ild_from_temp_profile,
        ds[temp_var],
        ds["depth"],
        input_core_dims=[["depth"], ["depth"]],
        output_dtypes=[np.float32],
        vectorize=True,
        dask="parallelized",
        kwargs={"method": method, "threshold": threshold,
                "ref_depth": ref_depth, "use_abs": use_abs},
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).rename("ild").astype("float32")

    mld = ds[mld_var].astype("float32").rename("mld")
    bld = (ild - mld).astype("float32").rename("bld")

    return xr.Dataset(
        {"mld": mld, "ild": ild, "bld": bld},
        coords={"time": ds["time"], "latitude": ds["latitude"], "longitude": ds["longitude"]},
    )
