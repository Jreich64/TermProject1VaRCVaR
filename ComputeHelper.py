from PortfolioReturns import MyPortfolioReturns
import streamlit as st
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor


def _safe_workers():
    """Return a sensible thread-pool size for the current container.
    Reads cgroup CPU quota when available so we don't oversubscribe on Railway.
    """
    quota = os.cpu_count() or 2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            limit, period = f.read().split()
            if limit != "max":
                quota = max(1, int(int(limit) // int(period)))
    except (FileNotFoundError, ValueError, OSError):
        pass
    return max(1, min(4, quota))


@st.cache_resource
def _get_thread_pool():
    """One pool per server process, reused across Streamlit reruns."""
    return ThreadPoolExecutor(max_workers=_safe_workers())


@st.cache_resource(show_spinner="Loading Portfolio")
def load_portfolio(sigma, seed, use_fund_sigmas=True):
    return MyPortfolioReturns(sigma, seed, use_fund_sigmas)


def _compute_var_cvar_one(args):
    """Compute VaR and CVaR for all quantiles for a single (tau, delta) series.
    Pure numpy; releases the GIL so threads run in parallel.
    """
    (tau, delta), series, quantile_range, normalize_returns = args
    norm = float(np.sqrt(365.0 / tau)) if normalize_returns else 1.0

    # Convert once to a contiguous float32 numpy array
    arr = np.ascontiguousarray(series.values, dtype=np.float32)
    arr_sorted = np.sort(arr)                               # releases GIL
    cumsum = np.cumsum(arr_sorted, dtype=np.float64)        # accumulate in float64 for accuracy

    var_vals = np.quantile(arr_sorted, quantile_range)      # vectorized over all quantiles
    cutoff = np.searchsorted(arr_sorted, var_vals, side='right')
    safe_idx = np.maximum(cutoff, 1)
    cvar_vals = np.where(
        cutoff > 0,
        cumsum[safe_idx - 1] / safe_idx,
        arr_sorted[0],
    )

    scale = norm * -1_000_000.0
    return (tau, delta), var_vals * scale, cvar_vals * scale


@st.cache_data(show_spinner="Computing VaR and CVaR", max_entries=2, ttl=3600)
def load_regular_var_and_cvar(returns_dict, min_quantile, max_quantile, normalize_returns):
    quantile_range = np.round(np.arange(min_quantile, max_quantile + 0.01, 0.01), 2)

    work = [
        ((tau, delta), df, quantile_range, normalize_returns)
        for (tau, delta), df in returns_dict.items()
    ]

    pool = _get_thread_pool()
    results = list(pool.map(_compute_var_cvar_one, work))

    var_records, cvar_records = {}, {}
    for key, var_vals, cvar_vals in results:
        for q, v, c in zip(quantile_range, var_vals, cvar_vals):
            var_records.setdefault(float(q), {})[key] = float(v)
            cvar_records.setdefault(float(q), {})[key] = float(c)

    var_dict = {q: pd.Series(v).unstack() for q, v in var_records.items()}
    cvar_dict = {q: pd.Series(v).unstack() for q, v in cvar_records.items()}
    for quantile in var_dict.keys():
        var_dict[quantile].index.name = "Tau"
        var_dict[quantile].columns.name = "Delta"
        cvar_dict[quantile].index.name = "Tau"
        cvar_dict[quantile].columns.name = "Delta"
    return var_dict, cvar_dict


def _shock_one(args):
    key, regular, sector_weights, selected_sectors, selected_devalue = args
    return key, regular - (sector_weights[selected_sectors].sum(axis=1) * selected_devalue)


@st.cache_data(show_spinner="Computing Shocked Returns", max_entries=2, ttl=3600)
def shock_returns(returns_dict, sectors_dict, selected_sectors, selected_devalue):
    work = [
        (key, returns_dict[key], sectors_dict[key], selected_sectors, selected_devalue)
        for key in returns_dict.keys()
    ]
    pool = _get_thread_pool()
    return dict(pool.map(_shock_one, work))
