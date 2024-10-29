from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas as pd
import pandas_ta
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
# Some data cleaning as some of the symbols contain '.' and AMTM, CTAS stock possibly delisted
sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')
symbol_list = sp500['Symbol'].unique().tolist()
for s in ['AMTM', 'CTAS']:
    symbol_list.remove(s)
end_date = '2024-09-30'
interval_size = 20 # in years 
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*interval_size)

df = yf.download(tickers=symbol_list, start=start_date, end=end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

df = df.tz_localize(None, level='date') # remove timezone from under the date

df['garman-klass'] = ((np.log(df['high'])-np.log(df['low']))**2)/2 - (2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open'])))**2
# level 1 is the ticker level; RSI is the only indicator we won't normalise
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# Use log normalisation when calculating Bollinger bands
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])
# Pandas TA's ATR takes 3 columns as input so we can't pass it into transform()
# We thus compute ATR using a custom function
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], 
                        low=stock_data['low'], 
                        close=stock_data['close'], 
                        length=14)
    return atr.sub(atr.mean()).div(atr.std()) # returns ATR indicator normalised for each stock

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
# We also compute MACD using a custom function
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std()) # again, normalise before returning

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
# Dollar volume liquidity per million
df['dollar_volume'] = (df['adj close']*df['volume'])*1e-6

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
           df.unstack()[last_cols].resample('M').last().stack('ticker')],
         axis=1)).dropna()

data['dollar_volume'] = (data['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())
data['dollar_volume_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_volume_rank']<150].drop(['dollar_volume', 'dollar_volume_rank'], axis=1)

def compute_returns(df):
    outlier_cutoff = 0.005 # 99.5 percentile is our outlier cutoff
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                               .pct_change(lag)
                               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                               .add(1).pow(1/lag).sub(1))
    return df


data = data.groupby(level=1, group_keys=False).apply(compute_returns).dropna()

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()

observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop('adj close', axis=1)

data = data.dropna()