from ErrorLog import ErrorList
import streamlit as st
import datetime as dt

import numpy as np
import pandas as pd

from FundNames import Fund_Names as All_Fund_Names
from SectorNames import AllowedSectorNames
from PlotHelper2D import plot_var_cvar_2d
from ComputeHelper import load_portfolio, load_regular_var_and_cvar, shock_returns


def main():
    st.title("Portfolio Var/CVaR Calculator")
    st.sidebar.header("Model Params Input")
    use_bridge = st.sidebar.checkbox("Use Brownian Bridge", value=False)
    normalize_returns = st.sidebar.checkbox("Normalize Var and CVAR to Tau = 365", value=True)
    sigma = st.sidebar.number_input("Brownian Bridge Sigma", value=1.0, step=0.1)
    seed = st.sidebar.number_input("Seed", min_value=0, value=0, step=1)
    start_date = st.sidebar.date_input("Start Date", value=dt.datetime(2005, 12, 31), min_value=dt.datetime(2005, 12, 31), max_value=dt.datetime(2024, 12, 31))
    end_date = st.sidebar.date_input("End Date", value=dt.datetime(2024, 12, 31), min_value=dt.datetime(2005, 12, 31), max_value=dt.datetime(2024, 12, 31))
    min_tau = st.sidebar.number_input("Minimum Tau", min_value=1, max_value=366, value=1, step=1)
    max_tau = st.sidebar.number_input("Maximum Tau", min_value=1, max_value=366, value=366, step=1)
    min_delta = st.sidebar.number_input("Minimum Delta", min_value=1, max_value=31, value=1, step=1)
    max_delta = st.sidebar.number_input("Maximum Delta", min_value=1, max_value=31, value=31, step=1)
    min_quantile = st.sidebar.number_input("Minimum Quantile", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
    max_quantile = st.sidebar.number_input("Maximum Quantile", min_value=0.01, max_value=0.1, value=0.1, step=0.01)
    min_sector_devalue = st.sidebar.number_input("Minimum Sector Devalue Percent", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    max_sector_devalue = st.sidebar.number_input("Maximum Sector Devalue Percent", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    st.subheader("Select Funds")
    if '_saved_funds' not in st.session_state:
        st.session_state['_saved_funds'] = []
    if '_saved_sectors_input' not in st.session_state:
        st.session_state['_saved_sectors_input'] = []
    selected_funds = st.multiselect("Selected Funds", options=All_Fund_Names, default=st.session_state['_saved_funds'], max_selections=50)
    selected_sectors = st.multiselect("Selected Sectors To Shock", options=AllowedSectorNames, default=st.session_state['_saved_sectors_input'])

    st.subheader("Select point calculation and 2D plot parameters")
    selected_tau = st.slider("Tau for point calculation and 2D plot", min_value=min_tau, max_value=max_tau, value=(max_tau+min_tau)//2, step=1)
    selected_delta = st.slider("Delta for point calculation and 2D plot", min_value=min_delta, max_value=max_delta, value=(max_delta+min_delta)//2, step=1)
    selected_quantile = st.slider("Quantile for point calculation and 2D plot", min_value=min_quantile, max_value=max_quantile, value=np.round((min_quantile+max_quantile)/2, 2), step=0.01)
    selected_devalue_percent = st.slider("Sector Shock Devalue Percent for point calculation and 2D plot", min_value=min_sector_devalue, max_value=max_sector_devalue, value=(min_sector_devalue+max_sector_devalue)/2, step=0.01)
    run = st.button("Run")

    if run:
        st.session_state['_saved_funds'] = selected_funds
        st.session_state['_saved_sectors_input'] = selected_sectors
        curr_portfolio_returns = load_portfolio(sigma, seed)
        try:
            returns_df_dict, sectors_df_dict = curr_portfolio_returns.portfolio_returns(pd.Timestamp(start_date), pd.Timestamp(end_date), min_tau=min_tau, max_tau=max_tau,
            min_delta=min_delta, max_delta=max_delta, n=None, fund_names=selected_funds, use_bridge=use_bridge)
            regular_var, regular_cvar = load_regular_var_and_cvar(returns_df_dict, min_quantile, max_quantile, normalize_returns)
            shocked_returns = shock_returns(returns_df_dict, sectors_df_dict, selected_sectors, selected_devalue_percent)
            shocked_var, shocked_cvar = load_regular_var_and_cvar(shocked_returns, min_quantile, max_quantile, normalize_returns)
            st.session_state['selected_tau'] = selected_tau
            st.session_state['selected_delta'] = selected_delta
            st.session_state['selected_quantile'] = selected_quantile
            st.session_state['sector_df_dict'] = sectors_df_dict
            st.session_state['returns'] = returns_df_dict
            st.session_state["returns_data"] = returns_df_dict[selected_tau, selected_delta]
            st.session_state["regular_var"] = regular_var
            st.session_state["regular_cvar"] = regular_cvar
            st.session_state['selected_devalue'] = selected_devalue_percent
            st.session_state['selected_sectors'] = selected_sectors
            st.session_state['shocked_returns'] = shocked_returns
            st.session_state['shocked_var'] = shocked_var
            st.session_state['shocked_cvar'] = shocked_cvar
            # Clear stale 3D cache so it recomputes with new data
            st.session_state.pop("_3d_shock_cache_key", None)
            st.session_state.pop("_3d_shocked_var_frames", None)
            st.session_state.pop("_3d_shocked_cvar_frames", None)

        except Exception as e:
            st.error(e)

    if st.session_state.get("regular_var") is not None:
        returns_dict = st.session_state['returns']
        regular_var = st.session_state["regular_var"]
        regular_cvar = st.session_state["regular_cvar"]
        shocked_returns = shock_returns(returns_dict, st.session_state['sector_df_dict'], selected_sectors, selected_devalue_percent)
        st.session_state['shocked_returns'] = shocked_returns
        shocked_var, shocked_cvar = load_regular_var_and_cvar(shocked_returns, min_quantile, max_quantile, normalize_returns)
        st.session_state['shocked_var'] = shocked_var
        st.session_state['shocked_cvar'] = shocked_cvar
        st.session_state['selected_tau'] = selected_tau
        st.session_state['selected_delta'] = selected_delta
        st.session_state['selected_quantile'] = selected_quantile
        st.session_state['selected_devalue'] = selected_devalue_percent
        st.session_state['selected_sectors'] = selected_sectors
        st.session_state['normalize_returns'] = normalize_returns

        # Guard against selected_tau/delta not existing in returns dict
        if (selected_tau, selected_delta) in returns_dict:
            st.session_state["returns_data"] = returns_dict[selected_tau, selected_delta]
        else:
            st.warning(f"Tau={selected_tau}, Delta={selected_delta} not in computed results. Adjust sliders or click Run again.")
            return

        q = round(selected_quantile, 2)
        if q not in regular_var:
            st.warning(f"Quantile={q} not in computed results. Adjust slider or click Run again.")
            return

        var_df = regular_var[q]
        cvar_df = regular_cvar[q]
        shocked_var_df = shocked_var[q]
        shocked_cvar_df = shocked_cvar[q]

        point_data = {
            'Tau': selected_tau,
            'Delta': selected_delta,
            'Quantile': selected_quantile,
            'Devalue Percent': selected_devalue_percent,
            'VaR': var_df.loc[selected_tau, selected_delta],
            'CVaR': cvar_df.loc[selected_tau, selected_delta],
            'Shocked VaR': shocked_var_df.loc[selected_tau, selected_delta],
            'Shocked CVaR': shocked_cvar_df.loc[selected_tau, selected_delta]
        }
        st.write(f"Sectors Shocked: {selected_sectors}")
        st.dataframe(pd.DataFrame([point_data]))

        fig_tau, html_tau = plot_var_cvar_2d(var_df, cvar_df, selected_tau, selected_delta, sweep="tau",
                                             selected_quantile=q, shocked_var_df=shocked_var_df, shocked_cvar_df=shocked_cvar_df)
        st.plotly_chart(fig_tau, use_container_width=True)
        st.download_button("Download Tau Sweep", html_tau, "var_cvar_tau_sweep.html", "text/html")

        fig_delta, html_delta = plot_var_cvar_2d(var_df, cvar_df, selected_tau, selected_delta, sweep="delta",
                                                 selected_quantile=q, shocked_var_df=shocked_var_df, shocked_cvar_df=shocked_cvar_df)
        st.plotly_chart(fig_delta, use_container_width=True)
        st.download_button("Download Delta Sweep", html_delta, "var_cvar_delta_sweep.html", "text/html")

        fig_q, html_q = plot_var_cvar_2d(var_df, cvar_df, selected_tau, selected_delta, sweep="quantile",
                                         var_dict=regular_var, cvar_dict=regular_cvar,
                                         shocked_var_dict=shocked_var, shocked_cvar_dict=shocked_cvar)
        st.plotly_chart(fig_q, use_container_width=True)
        st.download_button("Download Quantile Sweep", html_q, "var_cvar_quantile_sweep.html", "text/html")


if __name__ == "__main__":
    main()
