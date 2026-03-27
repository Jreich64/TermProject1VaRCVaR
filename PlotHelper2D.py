import plotly.graph_objects as go


def plot_var_cvar_2d(var_df, cvar_df, selected_tau, selected_delta, sweep="tau",
                     selected_quantile=None, var_dict=None, cvar_dict=None,
                     shocked_var_df=None, shocked_cvar_df=None,
                     shocked_var_dict=None, shocked_cvar_dict=None):
    fig = go.Figure()
    shocked_var_y = None
    shocked_cvar_y = None

    if sweep == "tau":
        x = var_df.index
        var_y = var_df[selected_delta]
        cvar_y = cvar_df[selected_delta]
        x_label = "Tau"
        title = f"VaR & CVaR vs Tau (Delta={selected_delta}, Quantile={selected_quantile})"
        if shocked_var_df is not None and shocked_cvar_df is not None:
            shocked_var_y = shocked_var_df[selected_delta]
            shocked_cvar_y = shocked_cvar_df[selected_delta]
    elif sweep == "delta":
        x = var_df.columns
        var_y = var_df.loc[selected_tau]
        cvar_y = cvar_df.loc[selected_tau]
        x_label = "Delta"
        title = f"VaR & CVaR vs Delta (Tau={selected_tau}, Quantile={selected_quantile})"
        if shocked_var_df is not None and shocked_cvar_df is not None:
            shocked_var_y = shocked_var_df.loc[selected_tau]
            shocked_cvar_y = shocked_cvar_df.loc[selected_tau]
    elif sweep == "quantile":
        if var_dict is None or cvar_dict is None:
            raise ValueError("var_dict and cvar_dict required for sweep='quantile'")
        quantiles = sorted(var_dict.keys())
        var_y = [var_dict[q].loc[selected_tau, selected_delta] for q in quantiles]
        cvar_y = [cvar_dict[q].loc[selected_tau, selected_delta] for q in quantiles]
        x = quantiles
        x_label = "Quantile"
        title = f"VaR & CVaR vs Quantile (Tau={selected_tau}, Delta={selected_delta})"
        if shocked_var_dict is not None and shocked_cvar_dict is not None:
            shocked_var_y = [shocked_var_dict[q].loc[selected_tau, selected_delta] for q in quantiles]
            shocked_cvar_y = [shocked_cvar_dict[q].loc[selected_tau, selected_delta] for q in quantiles]
    else:
        raise ValueError("sweep must be 'tau', 'delta', or 'quantile'")

    fig.add_trace(go.Scatter(x=x, y=var_y, mode='lines+markers', name='VaR', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=cvar_y, mode='lines+markers', name='CVaR', line=dict(color='red')))
    if shocked_var_y is not None and shocked_cvar_y is not None:
        fig.add_trace(go.Scatter(x=x, y=shocked_var_y, mode='lines+markers', name='Shocked VaR', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=x, y=shocked_cvar_y, mode='lines+markers', name='Shocked CVaR', line=dict(color='purple', dash='dash')))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Value",
        template="plotly_white"
    )

    html_str = fig.to_html(include_plotlyjs=True, full_html=True)
    return fig, html_str
