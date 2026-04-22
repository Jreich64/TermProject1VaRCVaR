import numpy as np
import pandas as pd


def calculate_fund_sigmas(adj_close_df):
    """Calculate daily log-return standard deviation for each fund from available price data."""
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    sigmas = log_returns.std()
    # Funds with too few valid prices produce NaN std → fall back to the median
    # daily vol across all funds. The old fallback of 1.0 (= 100% daily vol)
    # caused exp() overflow when extrapolating across multi-year gaps.
    fallback = sigmas.median()
    if not np.isfinite(fallback):
        fallback = 0.02  # ~2% daily vol, sane default if even the median is NaN
    sigmas = sigmas.fillna(fallback)
    return sigmas
