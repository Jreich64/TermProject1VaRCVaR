import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path
from functools import reduce

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
        fund_sector_dict[key] = foward_and_back_fill_where_valid(fund_sector_dict[key]).fillna(0)
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
            x0 = df_column.iloc[end]
            curr_dt = np.flip(dt[0:end])
            increments = my_rng_gen.normal(0.0, sigma * np.sqrt(curr_dt))
            log_returns = np.cumsum(increments)
            df_column.iloc[0:end] = x0 * np.exp(-log_returns)
        elif end >= len_df:
            x0 = df_column.iloc[start]
            curr_dt = dt[start:]
            increments = my_rng_gen.normal(0.0, sigma * np.sqrt(curr_dt))
            log_returns = np.cumsum(increments)
            df_column.iloc[start + 1:] = x0 * np.exp(log_returns)
        else:
            curr_dt = dt[start:end]
            x0 = df_column.iloc[start]
            xt = df_column.iloc[end]
            num_points = end - start + 1
            increments = my_rng_gen.normal(0.0, sigma*curr_dt)
            W = np.concatenate(([0.0], np.cumsum(increments)))
            t = (dates[start:end+1] - dates[start]).astype(float)
            T = t[-1]
            bridge = x0 + (xt-x0)*(t/T) + (W - ((t/T)*W[-1]))
            df_column.iloc[start+1:end] = bridge[1:-1]
    return df_column


def brownian_bridge(df, sigma, seed):
    df = df.copy()
    columns = df.columns
    for column in columns:
        nan_ranges = find_nan_ranges(df[column])
        if len(nan_ranges) > 0:
            df[column] = brownian_bridge_helper(df[column], nan_ranges, sigma, seed)
    return df


_data_cache = {}

def get_all_data(sigma, seed):
    key = (sigma, seed)
    if key in _data_cache:
        return _data_cache[key]
    folder_path = download_main_data()
    adj_close_df = collect_adj_close(folder_path)
    fund_sector_df_dict = collect_fund_metadata(folder_path, adj_close_df)
    adj_close_with_brownian_bridge = brownian_bridge(adj_close_df, sigma, seed)
    result = (adj_close_df, adj_close_with_brownian_bridge, fund_sector_df_dict)
    _data_cache[key] = result
    return result

if __name__ == "__main__":
    sigma = 1.0
    seed = 0
    adj_close_df, adj_close_with_brownian_bridge, fund_sector_df_dict = get_all_data(sigma, seed)
    print(adj_close_with_brownian_bridge)
