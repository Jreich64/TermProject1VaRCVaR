import streamlit as st
import numpy as np

st.title("VaR / CVaR Results")

if "regular_var" not in st.session_state:
    st.warning("No results yet. Run the model on the main page first.")
else:
    var_dict = st.session_state["regular_var"]
    cvar_dict = st.session_state["regular_cvar"]
    selected_tau = st.session_state['selected_tau']
    selected_delta = st.session_state['selected_delta']
    selected_quantile = st.session_state['selected_quantile']
    selected_quantile_key = np.round(selected_quantile, 2)
    selected_returns = st.session_state["returns_data"]
    selected_devalue_percent = st.session_state['selected_devalue']
    selected_sectors = st.session_state['selected_sectors']
    shocked_var_dict = st.session_state['shocked_var']
    shocked_cvar_dict = st.session_state['shocked_cvar']
    shocked_returns = st.session_state['shocked_returns']
    sector_df_dict = st.session_state['sector_df_dict']

    st.subheader("Params from Point Calculation/2D Plot")
    st.write(f"Tau: {selected_tau}, Delta: {selected_delta}, Quantile: {selected_quantile}, Devalue Percent: {selected_devalue_percent}")
    st.write(f"Selected Sectors to shock: {selected_sectors}")

    st.subheader(f"Returns with Tau: {selected_tau}, and Delta: {selected_delta}")
    st.dataframe(selected_returns)

    st.subheader(f"Shocked Returns with Tau: {selected_tau}, and Delta: {selected_delta}")
    st.dataframe(shocked_returns[selected_tau, selected_delta])

    st.subheader(f"VaR at quantile {selected_quantile}")
    st.dataframe(var_dict[selected_quantile_key])

    st.subheader(f"CVaR at quantile {selected_quantile}")
    st.dataframe(cvar_dict[selected_quantile_key])

    st.subheader(f"Sector weights with Tau: {selected_tau}, and Delta: {selected_delta}")
    st.dataframe(sector_df_dict[selected_tau, selected_delta])

    st.subheader(f"Shocked VaR at quantile {selected_quantile}")
    st.dataframe(shocked_var_dict[selected_quantile_key])

    st.subheader(f"Shocked CVaR at quantile {selected_quantile}")
    st.dataframe(shocked_cvar_dict[selected_quantile_key])

    if "fund_sigmas" in st.session_state:
        st.subheader("Per-Fund Brownian Bridge Sigma (Daily Log-Return Std Dev)")
        fund_sigmas = st.session_state["fund_sigmas"]
        st.dataframe(fund_sigmas.rename("Sigma").to_frame())

    ff5_fits = st.session_state.get("ff5_fits")
    if ff5_fits is not None and not ff5_fits.empty:
        st.subheader("Fama-French 5 Factor Coefficients (Equity Funds)")
        st.write(
            "OLS regression of each `_E` fund's daily excess log return on the "
            "Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA). `alpha` is the "
            "daily intercept, `beta_*` are factor loadings, `resid_std` is the "
            "daily residual standard deviation, `r_squared` is the in-sample fit, "
            "and `n_obs` is the number of overlapping daily observations used."
        )
        # Format with reasonable precision; keep n_obs as int.
        styled = ff5_fits.style.format({
            "alpha": "{:.6f}",
            "beta_Mkt-RF": "{:.4f}",
            "beta_SMB": "{:.4f}",
            "beta_HML": "{:.4f}",
            "beta_RMW": "{:.4f}",
            "beta_CMA": "{:.4f}",
            "resid_std": "{:.6f}",
            "r_squared": "{:.4f}",
            "n_obs": "{:d}",
        })
        st.dataframe(styled, use_container_width=True)
        st.download_button(
            "Download FF5 Fits (CSV)",
            ff5_fits.to_csv().encode("utf-8"),
            "ff5_fits.csv",
            "text/csv",
        )
