from ErrorLog import ErrorList
from PortfolioPrices import MyClosePrices

import numpy as np
import pandas as pd

class MyPortfolioReturns:
    def __init__(self, sigma, seed):
        self.sigma = sigma
        self.seed = seed
        self.close_pricer = MyClosePrices(sigma=sigma, seed=seed)

    def portfolio_prices(self, start_date, end_date, n=None, fund_names=None, use_bridge=False):
        curr_returns, curr_sector_df_dict = self.close_pricer.fund_prices(start_date, end_date, n, fund_names, use_bridge)
        num_funds = curr_returns.shape[1]
        curr_portfolio_returns = curr_returns
        #curr_sector_weights = list(curr_sector_df_dict.values())
        #curr_sector_weights_index, curr_sector_weights_columns = curr_sector_weights[0].index, curr_sector_weights[0].columns
        #curr_sector_weights = sum([fund_sector_df.values for fund_sector_df in curr_sector_weights]) / num_funds
        #curr_sector_weights = pd.DataFrame(curr_sector_weights, columns=curr_sector_weights_columns, index=curr_sector_weights_index)
        curr_sector_weights = pd.concat(curr_sector_df_dict.values()).fillna(0).groupby(level=0).sum() / num_funds
        return curr_portfolio_returns, curr_sector_weights

    def portfolio_returns(self, start_date, end_date, min_tau=1, max_tau=3, min_delta=1, max_delta=3, n=None, fund_names=None, use_bridge=False):
        returns_dict = {}
        sectors_dict = {}
        all_prices, all_sector_weights = self.portfolio_prices(start_date, end_date, n, fund_names, use_bridge)
        all_sector_weights = all_sector_weights.astype(np.float32)
        for tau in range(min_tau, max_tau + 1):
            tau_returns = (((all_prices.shift(-tau) - all_prices) / all_prices).mean(axis=1)).dropna().astype(np.float32)
            for delta in range(min_delta, max_delta + 1):
                sampled_returns = tau_returns.iloc[::delta]
                returns_dict[(tau, delta)] = sampled_returns
                sectors_dict[(tau, delta)] = all_sector_weights.loc[sampled_returns.index]

        return returns_dict, sectors_dict





if __name__ == "__main__":
    sigma = 1.0
    seed = 0
    my_returns = MyPortfolioReturns(sigma, seed)
    start_date = "2005-12-31"
    end_date = "2024-12-31"
    curr_returns, curr_sector_weights = my_returns.portfolio_returns(start_date, end_date)
    print(curr_returns)
    print(curr_sector_weights)


