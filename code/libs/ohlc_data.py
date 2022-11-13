import os
import datetime as dt
import pandas as pd
import numpy as np

import torch
from libs.utilities import get_funding_rate

MAX_PRICE = 100000

class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class CoinDataset:
    def __init__(self, df_dir, ohlc_mode, date_before='2022-09-01 00:00:00', days=1010): #'../data/'
        self.date_before = date_before
        sdate  = dt.datetime.strptime(date_before.split()[0],'%Y-%m-%d')
        delta = dt.timedelta(days=-days)
        self.date_after  = sdate + delta
        self.days = days
        self.ohlc_mode = ohlc_mode

        self.periods = [5, 20, 50]
        self.columns = ['Open', 'High', 'Low', 'Close']
        self.columns += ['MA_' + str(p) for p in self.periods]

        self.df_2h = pd.read_csv(df_dir + '2h_ohlcv.csv', index_col=0)
        self.df_30min = pd.read_csv(df_dir + '30m_ohlcv.csv', index_col=0)
        self.df_15min = pd.read_csv(df_dir + '15m_ohlcv.csv', index_col=0)
        self.df_fr = pd.read_csv(df_dir + 'Funding_Rate.csv', index_col=0)

        self.df_2h = self._add_MA(self.df_2h)
        self.df_30min = self._add_MA(self.df_30min)
        self.df_15min = self._add_MA(self.df_15min)

        df_15min = self.df_15min.iloc[self.df_15min.index<date_before]
        df_15min = self.df_15min.truncate(before=str(self.date_after))
        self.indice = df_15min.index

    def _add_MA(self, df):
        assert isinstance(df, pd.DataFrame)
        for period in self.periods:
            name = 'MA_' + str(period) 
            df[name] = df['Close'].rolling(period).mean()
        return df

    def _get_subset(self, df, index_date, counts):
        sub_df = df.iloc[df.index<=index_date].tail(counts)
        sub_df = sub_df.loc[:, self.columns]
        return sub_df

    def _get_prices(self, df, index_date):
        sub_df = df.loc[index_date:].head(2)
        prices = sub_df['Close'].to_list()
        return prices
 
    def __len__(self):
        return len(self.indice)
    
    def __getitem__(self, index):
        index_date = self.indice[index]
        if self.ohlc_mode == 'multi':
            sub_15m_df = self._get_subset(self.df_15min, index_date, 24) / MAX_PRICE
            sub_30m_df = self._get_subset(self.df_30min, index_date, 12) / MAX_PRICE
            sub_2h_df = self._get_subset(self.df_2h, index_date, 12) / MAX_PRICE

            ohlc = np.concatenate((sub_15m_df.to_numpy(), sub_30m_df.to_numpy(), sub_2h_df.to_numpy()), axis=0)
        else:
            ohlc = (self._get_subset(self.df_15min, index_date, 1)/MAX_PRICE).to_numpy()

        ohlc = torch.tensor(ohlc.flatten(), dtype=torch.float32)
        fund_rate = get_funding_rate(self.df_fr, index_date)  
        prices = self._get_prices(self.df_15min, index_date)
        return ohlc, prices, fund_rate, index_date

    def get_date_index(self, date):
        date = date + ' ' + '00:00:00'
        return self.indice.to_list().index(date)
 
def add_days(date, days):
    start_date  = dt.datetime.strptime(date,'%Y-%m-%d')
    delta = dt.timedelta(days=days)
    end_date  = start_date + delta
    return str(end_date).split()[0]

def get_folder_dataset(count, data_path, ohlc_mode, hyperparams, date_from, learn=True):
    """
    date_from format example '2020-01-01'
    """
    datasets = {}
    dataset = CoinDataset(data_path, ohlc_mode)
    train_days = hyperparams['train_days'] 
    valid_days = hyperparams['valid_days'] 
    test_days = hyperparams['test_days'] 
    roll_offset = hyperparams['roll_offset'] 

    for i in range(count):
        offset_days = roll_offset * i # roll up with the days of (validation + roll_offset)

        date_start = add_days(date_from, offset_days)
        train_start = dataset.get_date_index(date_start) 
        date_end_trian = add_days(date_start, train_days)
        train_end = dataset.get_date_index(date_end_trian) 
        indices = list(range(train_start, train_end))
        train_set = Subset(dataset, indices)

        date_end_valid = add_days(date_end_trian, valid_days)
        valid_end = dataset.get_date_index(date_end_valid) 
        indices = list(range(train_end, valid_end))
        valid_set = Subset(dataset, indices)

        datasets[i] = [train_set, valid_set]   

    if learn:
        return datasets
    else:
        date_finish = add_days(date_end_valid, test_days)
        test_end = dataset.get_date_index(date_finish) 
        indices = list(range(valid_end, test_end))
        test_set = Subset(dataset, indices)
        return test_set    

def get_test_dataset(data_path, ohlc_mode, date_from, days=15):
    dataset = CoinDataset(data_path, ohlc_mode)
    test_days = days

    test_start = dataset.get_date_index(date_from)
    date_to = add_days(date_from, test_days)
    test_end = dataset.get_date_index(date_to)
    indices = list(range(test_start, test_end))
    test_set = Subset(dataset, indices)
    return test_set 