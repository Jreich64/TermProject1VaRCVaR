"""Fama-French 5-factor model for backfilling missing equity-fund prefixes.

Used by `brownian_bridge_helper` in DataLoader.py: when a fund (`_E` ticker)
is missing data at the start of the requested window, we fit a 5-factor model
on its available history, then simulate returns for the missing period using
the *actual* historical factor values plus a bootstrapped residual draw.

This anchors the synthetic prefix to real historical market regimes (2008,
COVID, etc.) instead of a free-running random walk.

Falls back gracefully to GBM if:
  - the fund is not equity (`_E`),
  - we don't have enough overlap (default 252 trading days = ~1 year) to fit,
  - the FF5 data file isn't available and pandas_datareader isn't installed,
  - the missing period falls outside the factor data's coverage.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:  # allow standalone use / testing
    _HAS_STREAMLIT = False

FF5_LOCAL_PATH = Path("data2") / "F-F_Research_Data_5_Factors_2x3_daily.csv"
FF5_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FF5_DATASET_NAME = "F-F_Research_Data_5_Factors_2x3_daily"


# ---------- Data loading ----------

def _read_local_ff5(path: Path) -> pd.DataFrame:
    """Read either our cached clean CSV or Ken French's raw downloaded CSV."""
    # First try: our previously-cached clean format (DatetimeIndex, decimal values)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(FF5_FACTOR_COLS + ["RF"]).issubset(df.columns):
            return df.astype(np.float64)
    except Exception:
        pass

    # Fall back: raw Ken French format (3 header rows, YYYYMMDD index, percent values)
    df = pd.read_csv(path, skiprows=3, index_col=0)
    # Drop trailing junk rows ("Annual Factors:", blank lines, etc.)
    idx_numeric = pd.to_numeric(df.index.astype(str), errors="coerce")
    df = df.loc[idx_numeric.notna()]
    df.index = pd.to_datetime(df.index.astype(int).astype(str), format="%Y%m%d")
    df = df.astype(np.float64) / 100.0
    return df


def _download_ff5() -> pd.DataFrame:
    """Fetch the daily FF5 dataset via pandas_datareader."""
    try:
        from pandas_datareader.famafrench import FamaFrenchReader
    except ImportError as exc:
        raise RuntimeError(
            "Fama-French data not bundled and pandas_datareader is not installed. "
            "Either add it to requirements.txt or place the CSV at "
            f"{FF5_LOCAL_PATH}."
        ) from exc

    reader = FamaFrenchReader(FF5_DATASET_NAME, start="1990")
    raw = reader.read()
    df = raw[0].copy()           # daily key is 0; raw[1] is the annual aggregation
    df = df.astype(np.float64) / 100.0
    df.index = pd.to_datetime(df.index)
    return df


def _load_ff5_impl() -> pd.DataFrame:
    """Inner loader used by both Streamlit-cached and standalone paths."""
    if FF5_LOCAL_PATH.exists():
        return _read_local_ff5(FF5_LOCAL_PATH)

    df = _download_ff5()
    # Best-effort write-through cache so future container starts don't re-download
    try:
        FF5_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FF5_LOCAL_PATH)
    except OSError:
        pass
    return df


if _HAS_STREAMLIT:
    @st.cache_resource(show_spinner="Loading Fama-French 5 factors")
    def load_ff5_daily() -> pd.DataFrame:
        return _load_ff5_impl()
else:
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def load_ff5_daily() -> pd.DataFrame:
        return _load_ff5_impl()


# ---------- Fitting ----------

def fit_ff5(fund_log_returns: pd.Series, factors: pd.DataFrame, min_obs: int = 252):
    """OLS regression of fund excess (log) returns on the 5 factors.

    Returns a dict with `alpha`, `betas` (length-5 array), `residuals` array,
    or None if the overlap with the factor data is below `min_obs`.

    Note: FF factors are simple returns; we treat them as log returns. For daily
    data the approximation error is O(r^2), well below the residual noise.
    """
    fr = fund_log_returns.dropna()
    overlap = fr.index.intersection(factors.index)
    if len(overlap) < min_obs:
        return None

    fr = fr.loc[overlap].astype(np.float64)
    fct = factors.loc[overlap]
    excess = (fr - fct["RF"]).values

    X = fct[FF5_FACTOR_COLS].values.astype(np.float64)
    X = np.column_stack([np.ones(len(X)), X])

    coef, *_ = np.linalg.lstsq(X, excess, rcond=None)
    alpha = float(coef[0])
    betas = coef[1:].astype(np.float64)

    fitted = X @ coef
    residuals = (excess - fitted).astype(np.float64)

    return {
        "alpha": alpha,
        "betas": betas,
        "residuals": residuals,
        "n_obs": int(len(overlap)),
    }


# ---------- Simulation ----------

def ff5_simulate_log_returns(target_dates: pd.DatetimeIndex,
                             fit: dict,
                             factors: pd.DataFrame,
                             rng: np.random.Generator):
    """Simulate daily log returns for the given dates from a fitted FF5 model.

    Returns a 1-D float64 array aligned to `target_dates`, or None if the
    factor data doesn't cover the requested range (even after small-gap fill).
    """
    factors_subset = factors.reindex(target_dates)
    if factors_subset.isna().any().any():
        # Pad small gaps (weekends/holidays the factor file may skip differently)
        factors_subset = factors_subset.ffill().bfill()
        if factors_subset.isna().any().any():
            return None

    X = factors_subset[FF5_FACTOR_COLS].values.astype(np.float64)
    rf = factors_subset["RF"].values.astype(np.float64)

    factor_part = fit["alpha"] + X @ fit["betas"] + rf
    eps = rng.choice(fit["residuals"], size=len(factor_part), replace=True)
    return factor_part + eps


# ---------- Bulk fits for display ----------

FF5_FIT_COLUMNS = [
    "alpha",
    "beta_Mkt-RF",
    "beta_SMB",
    "beta_HML",
    "beta_RMW",
    "beta_CMA",
    "resid_std",
    "r_squared",
    "n_obs",
]


def _compute_all_ff5_fits_impl(adj_close_df: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """Fit FF5 for every equity (`_E`) column in `adj_close_df`.

    Returns a DataFrame indexed by fund ticker with columns:
      alpha, beta_<factor> (5), resid_std, r_squared, n_obs.
    Funds with insufficient overlap (<min_obs days) get NaN coefficients
    and `n_obs` reports the actual overlap count (or 0 if no overlap).
    """
    try:
        factors = load_ff5_daily()
    except Exception:
        # No factor data → return empty frame so caller can skip the section
        return pd.DataFrame(columns=FF5_FIT_COLUMNS)

    rows = {}
    for col in adj_close_df.columns:
        if not str(col).endswith("_E"):
            continue
        prices = adj_close_df[col].dropna()
        if len(prices) < min_obs + 1:
            rows[col] = {c: np.nan for c in FF5_FIT_COLUMNS}
            rows[col]["n_obs"] = int(len(prices))
            continue

        log_returns = np.log(prices / prices.shift(1)).dropna()
        fit = fit_ff5(log_returns, factors, min_obs=min_obs)
        if fit is None:
            overlap = log_returns.index.intersection(factors.index)
            rows[col] = {c: np.nan for c in FF5_FIT_COLUMNS}
            rows[col]["n_obs"] = int(len(overlap))
            continue

        # R² = 1 - SS_resid / SS_tot, where the denominator uses excess returns
        fr = log_returns.loc[log_returns.index.intersection(factors.index)].astype(np.float64)
        excess = (fr - factors.loc[fr.index, "RF"]).values
        ss_tot = float(np.sum((excess - excess.mean()) ** 2))
        ss_res = float(np.sum(fit["residuals"] ** 2))
        r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

        rows[col] = {
            "alpha": fit["alpha"],
            "beta_Mkt-RF": float(fit["betas"][0]),
            "beta_SMB": float(fit["betas"][1]),
            "beta_HML": float(fit["betas"][2]),
            "beta_RMW": float(fit["betas"][3]),
            "beta_CMA": float(fit["betas"][4]),
            "resid_std": float(np.std(fit["residuals"], ddof=1)) if len(fit["residuals"]) > 1 else float("nan"),
            "r_squared": r2,
            "n_obs": fit["n_obs"],
        }

    if not rows:
        return pd.DataFrame(columns=FF5_FIT_COLUMNS)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[FF5_FIT_COLUMNS]
    df.index.name = "Fund"
    return df.sort_index()


if _HAS_STREAMLIT:
    @st.cache_resource(show_spinner="Fitting Fama-French 5 factors per fund")
    def compute_all_ff5_fits(_adj_close_df: pd.DataFrame, fits_id: int, min_obs: int = 252) -> pd.DataFrame:
        # Underscore on `_adj_close_df` skips Streamlit's hashing (DataFrames are
        # expensive to hash); `fits_id` is the actual cache key.
        return _compute_all_ff5_fits_impl(_adj_close_df, min_obs=min_obs)
else:
    def compute_all_ff5_fits(_adj_close_df: pd.DataFrame, fits_id: int = 0, min_obs: int = 252) -> pd.DataFrame:
        return _compute_all_ff5_fits_impl(_adj_close_df, min_obs=min_obs)
