from PortfolioReturns import MyPortfolioReturns
import streamlit as st
import numpy as np
import pandas as pd


@st.cache_resource(show_spinner="Loading Portfolio")
def load_portfolio(sigma, seed):
    return MyPortfolioReturns(sigma, seed)


@st.cache_data(show_spinner="Computing VaR and CVaR")
def load_regular_var_and_cvar(returns_dict, min_quantile, max_quantile, normalize_returns):
    var_records = {}
    cvar_records = {}
    quantile_range = np.round(np.arange(min_quantile, max_quantile + 0.01, 0.01), 2)
    for (tau, delta), df in returns_dict.items():
        normalization_factor = 1.0
        if normalize_returns:
            normalization_factor = np.sqrt(365.0 / tau)
        sorted_vals = df.sort_values().values
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        var_data = df.quantile(quantile_range)
        for quantile in quantile_range:
            curr_var = var_data[quantile]
            var_records.setdefault(float(quantile), {})[(tau, delta)] = curr_var * normalization_factor * -1_000_000
            cutoff_idx = np.searchsorted(sorted_vals, curr_var, side='right')
            if cutoff_idx > 0:
                cvar_val = cumsum[cutoff_idx - 1] / cutoff_idx
            else:
                cvar_val = sorted_vals[0]
            cvar_records.setdefault(float(quantile), {})[(tau, delta)] = cvar_val * normalization_factor * -1_000_000

    var_dict = {q: pd.Series(v).unstack() for q, v in var_records.items()}
    cvar_dict = {q: pd.Series(v).unstack() for q, v in cvar_records.items()}
    for quantile in var_dict.keys():
        var_dict[quantile].index.name = "Tau"
        var_dict[quantile].columns.name = "Delta"
        cvar_dict[quantile].index.name = "Tau"
        cvar_dict[quantile].columns.name = "Delta"
    return var_dict, cvar_dict


@st.cache_data(show_spinner="Computing Shocked Returns")
def shock_returns(returns_dict, sectors_dict, selected_sectors, selected_devalue):
    shocked_returns = {}
    for key in returns_dict.keys():
        curr_regular_returns = returns_dict[key]
        curr_sector_weights = sectors_dict[key]
        shocked_returns[key] = curr_regular_returns - (curr_sector_weights[selected_sectors].sum(axis=1) * selected_devalue)
    return shocked_returns
