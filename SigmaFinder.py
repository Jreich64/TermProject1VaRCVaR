import numpy as np
import pandas as pd


def calculate_fund_sigmas(adj_close_df):
    """Calculate daily log-return standard deviation for each fund from available price data."""
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    sigmas = log_returns.std().fillna(1.0)
    return sigmas
