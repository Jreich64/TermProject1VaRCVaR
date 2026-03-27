from DataLoader import get_all_data
from ErrorLog import ErrorList

import numpy as np
import pandas as pd

class MyClosePrices:
    def __init__(self, sigma=1.0, seed=0):
        self.sigma = sigma
        self.seed = seed
        self.my_rng_gen = np.random.default_rng(seed)
        self.adj_close, self.adj_close_with_bridge, self.fund_sector_df_dict = get_all_data(sigma, seed)
        self.min_start_date = pd.to_datetime('2005-12-31')
        self.max_start_date = pd.to_datetime('2024-12-31')
        self.max_num_funds = 50

    def fund_names_allowed_in_period(self, start_date, end_date, use_bridge):
        if use_bridge:
            curr_data = self.adj_close_with_bridge
        else:
            curr_data = self.adj_close
        curr_data = pd.DataFrame(curr_data[start_date:end_date]).dropna(axis=1, how='any')
        return set(curr_data.columns)
    
    def fund_prices(self, start_date, end_date, n=None, fund_names=None, use_bridge=False):
        if (fund_names is not None) and (n is None):
            n = len(fund_names)
        start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        #Date Checks
        if start_date < self.min_start_date:
            error_string = f"Error the inputted start date {start_date.strftime('%Y-%m-%d')} is before the minimum of {self.min_start_date.strftime('%Y-%m-%d')}"
            ErrorList.append(error_string)
            print(error_string)
            start_date = self.min_start_date
        if end_date > self.max_start_date:
            error_string = f"Error the inputted end date {end_date.strftime('%Y-%m-%d')} is after the maximum of {self.max_start_date.strftime('%Y-%m-%d')}"
            ErrorList.append(error_string)
            print(error_string)
            end_date = self.max_start_date
        #Set source data
        if use_bridge:
            curr_all_prices = self.adj_close_with_bridge
        else:
            curr_all_prices = self.adj_close

        #Fund names allowed during time period
        curr_allowed_fund_names = self.fund_names_allowed_in_period(start_date, end_date, use_bridge)
        if (n is None) and (fund_names is None):
            n = min(len(curr_allowed_fund_names), 50)

        if n > 50:
            error_string = f"Error n: {n} is greater than 50"
            ErrorList.append(error_string)
            print(error_string)
            raise ValueError(error_string)

        if n > len(curr_allowed_fund_names):
            error_string = f"""
            Error the desired numbers of funds n: {n} is greater than the available funds: {len(curr_allowed_fund_names)}
            during the time range of the selected start date {start_date.strftime('%Y-%m-%d')} to the end date of {end_date.strftime('%Y-%m-%d')}
            """
            ErrorList.append(error_string)
            print(error_string)
            n = len(curr_allowed_fund_names)

        
        if fund_names is None:
            fund_names = self.my_rng_gen.choice(list(curr_allowed_fund_names), size=n, replace=False).tolist()

        if len(fund_names) != n:
            error_string = f"Error n: {n} does not equal {len(fund_names)}, the length of fund names: {fund_names}"
            ErrorList.append(error_string)
            print(error_string)
            fund_names = fund_names[:n]

        if not (set(fund_names) <= curr_allowed_fund_names):
            missing_names = set(fund_names) - curr_allowed_fund_names
            error_string = f"Error the funds: {missing_names} are not available during {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ErrorList.append(error_string)
            print(error_string)
            raise ValueError(error_string)

        return_prices = pd.DataFrame(curr_all_prices.loc[start_date:end_date, fund_names])
        return_sector_dict = {}
        for fund_name in fund_names:
            if fund_name in self.fund_sector_df_dict.keys():
                curr_sector_df = self.fund_sector_df_dict[fund_name].loc[start_date:end_date].copy()
                return_sector_dict[fund_name] = curr_sector_df
        template = (list(return_sector_dict.values())[0]).copy()
        template.iloc[:] = 0
        for fund_name in fund_names:
            if not fund_name in self.fund_sector_df_dict.keys():
                return_sector_dict[fund_name] = template.copy()
        return return_prices, return_sector_dict
        



if __name__ == "__main__":
    sigma = 1.0
    seed = 0
    my_prices = MyClosePrices(sigma=sigma, seed=seed)
    start_date = "2005-12-31"
    end_date = "2024-12-31"
    n = 30
    my_prices_result, my_sector_dict_result = my_prices.fund_prices(start_date=start_date, end_date=end_date, n=n, use_bridge=False)
    print(my_prices_result)
    for key, val in my_sector_dict_result.items():
        print()
        print(key)
        print(val)
