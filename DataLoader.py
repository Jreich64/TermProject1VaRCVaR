import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
from SigmaFinder import calculate_fund_sigmas


def _bridge_safe_workers():
    """Cgroup-aware worker count so we don't oversubscribe Railway containers."""
    quota = os.cpu_count() or 2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            limit, period = f.read().split()
            if limit != "max":
                quota = max(1, int(int(limit) // int(period)))
    except (FileNotFoundError, ValueError, OSError):
        pass
    return max(1, min(4, quota))

def download_main_data():
    local_folder = Path.cwd()
    folder_path = os.path.join(local_folder, "data2")
    if not os.path.exists(folder_path):
        gdown.download_folder(id='1B4tnad6AijTH9rZUfy1bYwD6VjAO7bur', resume=True, output=folder_path, verify=False)
    return folder_path

def collect_adj_close_helper(adj_close_path, suffix):
    df = pd.read_csv(adj_close_path)
    df['as_of'] = pd.to_datetime(df['as_of'])
    df.rename({'as_of': 'Date'}, axis=1, inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    df = df.rename(columns={col: f"{col}{suffix}" for col in df.columns})
    return df

def collect_meta_data_helper(adj_close_path, suffix):
    df = pd.read_csv(adj_close_path)
    df['as_of'] = pd.to_datetime(df['as_of'])
    df.rename({'as_of': 'Date'}, axis=1, inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    df['ask_id'] = df['ask_id'] + suffix
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    return df

def collect_adj_close(folder_path):
    bond_path = os.path.join(folder_path, 'us_bond_intermediate_core_adj_close.csv')
    equity_path = os.path.join(folder_path, 'us_equity_adj_close.csv')
    bond_df = collect_adj_close_helper(bond_path, '_B')
    equity_df = collect_adj_close_helper(equity_path, '_E')
    combined_df = bond_df.join(equity_df, how='outer')
    combined_df = combined_df.astype(np.float32)
    return combined_df

def foward_and_back_fill_where_valid(df):
    df = df.copy()
    columns = list(df.columns)
    for column in columns:
        first_valid_index = df[column].first_valid_index()
        last_valid_index = df[column].last_valid_index()
        if (first_valid_index is not None) and (last_valid_index is not None):
            df.loc[first_valid_index:last_valid_index, column] = (df.loc[first_valid_index:last_valid_index, column]).ffill().bfill()
    return df


def collect_fund_metadata(folder_path, adj_close_df):
    bond_csv_paths = ['us_bond_intermediate_core_credit_quality.csv', 
    'us_bond_intermediate_fixed_income_duration_yield.csv', 
    'us_bond_intermediate_fixed_income_primary_sector.csv'
    ]
    equity_csv_paths = ['us_equity_economic_region_exposure.csv', 'us_equity_sectors.csv', 'us_equity_styles.csv']
    bond_csv_paths = [os.path.join(folder_path, curr_path) for curr_path in bond_csv_paths]
    equity_csv_paths = [os.path.join(folder_path, curr_path) for curr_path in equity_csv_paths]
    bond_dfs = [collect_meta_data_helper(path, "_B") for path in bond_csv_paths]
    equity_dfs = [collect_meta_data_helper(path, "_E") for path in equity_csv_paths]
    bond_combined = reduce(lambda left, right: left.merge(right, on=['Date', 'ask_id'], how='outer'), bond_dfs)
    equity_combined = reduce(lambda left, right: left.merge(right, on=['Date', 'ask_id'], how='outer'), equity_dfs)
    df = pd.concat([bond_combined, equity_combined])
    fund_sector_dict = {fund: group.drop(columns='ask_id').reindex(adj_close_df.index) for fund, group in df.groupby('ask_id')}
    for key in fund_sector_dict.keys():
        fund_sector_dict[key] = foward_and_back_fill_where_valid(fund_sector_dict[key]).fillna(0).astype(np.float32)
    return fund_sector_dict

def find_nan_ranges(df_column):
    nan_ranges = []
    is_nan = df_column.isna().to_numpy()
    in_nan_block = False
    start = None
    for i, val in enumerate(is_nan):
        if val and not in_nan_block:
            start = i
            in_nan_block = True
        elif not val and in_nan_block:
            nan_ranges.append((start, i-1))
            in_nan_block = False
    if in_nan_block:
        nan_ranges.append((start, len(df_column) - 1))
    return nan_ranges

def brownian_bridge_helper(df_column, nan_ranges, sigma=1.0, seed=0):
    my_rng_gen = np.random.default_rng(seed=seed)
    df_column = df_column.copy()
    len_df = len(df_column)
    dates = df_column.index.to_numpy(dtype='datetime64[D]')
    dt = np.diff(dates).astype(float)
    for start, end in nan_ranges:
        start -= 1
        end +=1
        if start < 0:
            # NaN block at the very beginning: walk BACKWARD from the first known price.
            x0 = df_column.iloc[end]
            curr_dt = np.flip(dt[0:end])  # time gaps in backward order
            increments = my_rng_gen.normal(0.0, sigma * np.sqrt(curr_dt))
            log_returns = np.cumsum(increments)
            # log_returns[i] = noise after (i+1) backward steps from anchor at `end`,
            # which lands at position end-1-i. Reverse before assigning so position 0
            # gets the most-walked value and position end-1 gets the least-walked.
            df_column.iloc[0:end] = (x0 * np.exp(-log_returns))[::-1].astype(df_column.dtype)
        elif end >= len_df:
            # NaN block at the very end: walk FORWARD from the last known price.
            x0 = df_column.iloc[start]
            curr_dt = dt[start:]
            increments = my_rng_gen.normal(0.0, sigma * np.sqrt(curr_dt))
            log_returns = np.cumsum(increments)
            df_column.iloc[start + 1:] = (x0 * np.exp(log_returns)).astype(df_column.dtype)
        else:
            # Geometric Brownian bridge between two known anchors. Doing this in
            # log-space (then exponentiating) keeps the path strictly positive
            # and consistent with how `sigma` was estimated (std of log returns).
            curr_dt = dt[start:end]
            x0 = df_column.iloc[start]
            xt = df_column.iloc[end]
            increments = my_rng_gen.normal(0.0, sigma * np.sqrt(curr_dt))
            W = np.concatenate(([0.0], np.cumsum(increments)))
            t = (dates[start:end+1] - dates[start]).astype(float)
            T = t[-1]
            log_x0, log_xt = np.log(x0), np.log(xt)
            log_bridge = log_x0 + (log_xt - log_x0) * (t / T) + (W - (t / T) * W[-1])
            bridge = np.exp(log_bridge)
            df_column.iloc[start+1:end] = bridge[1:-1].astype(df_column.dtype)
    return df_column


def brownian_bridge(df, sigma, seed, fund_sigmas=None):
    df = df.copy()
    columns = list(df.columns)

    # IMPORTANT: pandas DataFrame __getitem__ is NOT thread-safe (it mutates an
    # internal block-manager cache). Extract every column as a standalone Series
    # in the main thread BEFORE handing work off to the pool.
    column_series = {col: df[col].copy() for col in columns}

    def _process_column(column):
        col_sigma = float(fund_sigmas[column]) if fund_sigmas is not None else sigma
        series = column_series[column]
        nan_ranges = find_nan_ranges(series)
        if not nan_ranges:
            return column, None
        # Per-column seed keeps results reproducible AND avoids RNG contention across threads
        col_seed = (seed + (hash(str(column)) & 0x7FFFFFFF)) & 0x7FFFFFFF
        filled = brownian_bridge_helper(series, nan_ranges, col_sigma, col_seed)
        return column, filled

    with ThreadPoolExecutor(max_workers=_bridge_safe_workers()) as ex:
        results = list(ex.map(_process_column, columns))

    # Write results back single-threaded (DataFrame __setitem__ also isn't thread-safe)
    for column, filled in results:
        if filled is not None:
            df[column] = filled
    return df


_data_cache = {}

def get_all_data(sigma, seed, use_fund_sigmas=True):
    key = (sigma, seed, use_fund_sigmas)
    if key in _data_cache:
        return _data_cache[key]
    folder_path = download_main_data()
    adj_close_df = collect_adj_close(folder_path)
    fund_sector_df_dict = collect_fund_metadata(folder_path, adj_close_df)
    fund_sigmas = calculate_fund_sigmas(adj_close_df)
    if use_fund_sigmas:
        adj_close_with_brownian_bridge = brownian_bridge(adj_close_df, sigma, seed, fund_sigmas=fund_sigmas)
    else:
        adj_close_with_brownian_bridge = brownian_bridge(adj_close_df, sigma, seed)
    result = (adj_close_df, adj_close_with_brownian_bridge, fund_sector_df_dict, fund_sigmas)
    _data_cache[key] = result
    return result

if __name__ == "__main__":
    sigma = 1.0
    seed = 0
    adj_close_df, adj_close_with_brownian_bridge, fund_sector_df_dict, fund_sigmas = get_all_data(sigma, seed)
    print(adj_close_with_brownian_bridge)
