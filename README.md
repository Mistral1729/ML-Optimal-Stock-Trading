# ML-Optimal-Stock-Trading
A machine learning model for optimal stock trading.

## Packages used:

- pandas,
- pandas_ta,
- pandas_datareader,
- numpy,
- matplotlib,
- statsmodels,
- datetime,
- yfinance,
- scikit-learn,
- PyPortfolioOpt

## Datasets and API used:
- [List of current S&P500 stocks from Wikipedia (as of October 2024)](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## Methodology 

### Aim: An unsupervised ML model that can eventually outperform the SPY S&P500 ETF in the long run in terms of returns.

Features and technical indicators to calculate for each stock:
- ### Garman-Klass volatility

  Defined as $$r_t = \frac{(\ln{H_t} - \ln{L_t})^2}{2} - 2(\ln{2}-1)(\ln{C_t} - \ln{O_t})^2,$$
  where $O_t, C_t, H_t, L_t$ denote the opening, closing, high, low prices of day $t$ respectively.
- ### Relative strength index (RSI)

  Relative strength: $$\mathrm{RS} = \frac{\text{Average gain}}{\text{Average loss}}.$$
  Relative strength index: $$\mathrm{RSI} = 100 - \frac{100}{1+\mathrm{RS}}.$$ 
- ### Bollinger bands

  $$\text{Upper band} = \mathrm{MA}(n) + k\sigma(n),$$
  $$\text{Middle band} = \mathrm{MA}(n),$$
  $$\text{Lower band} = \mathrm{MA}(n) - k\sigma(n),$$
  where $\mathrm{MA}(n)$ is the moving average over $n$ periods and $\sigma(n)$ is the standard deviation over $n$ periods. Usually we take $n=20$ and $k=2$.
- ### Average true range (ATR)

  Given by $$\frac{1}{n}\sum_{t=1}^n TR(t),$$ where $n$ is the number of periods, and $TR(t)$ is the true range of day $t$ defined by $$TR(t)=\max((H_t-L_t), |H_t - C_{t-1}|, |L_t - C_{t-1}|).$$
- ### Moving average convergence divergence (MACD)

  Given by $\mathrm{MACD} = \mathrm{EMA}_{12} - \mathrm{EMA}_{26},$ where $\mathrm{EMA}_t$ denotes the exponential moving average over $t$ periods.
- ### Dollar volume liquidity

  Dollar volume liquidity is the price of a stock multiplied by its daily trading volume

To reduce the model's training time we aggregate the data on a monthly level, filtering the top 150 most liquid stocks per month. We calculate 5-year moving average of dollar volume of each stock before filtering. We then compute monthly returns for different time horizons as features. We download the five Fama-French factors, namely
1. market risk,
2. size,
3. value,
4. operating profitability, and
5. investment

which have been empirically proven for asset returns and are used to assess the risk/return profile of portfolios. It is natural to include past factor exposures as financial features in our model.

We filter out stocks with < 10 months of data, then calculate rolling factor betas and join it with the main features dataframe. Finally, we use an unsupervised $K$-Means clustering algorithm to group similar assets based on their features into 4 clusters. 
We have used the normalised values for all indicators except RSI, because we want to target specific RSI values. In fact our strategy is built on the hypothesis that momentum is persistent and that stocks clustered around RSI 70 centroid should continue to outperform in the following month - thus we select stocks corresponding to cluster 3.

For each month we select assets based on the cluster and form a portfolio based on efficient frontier maximum Sharpe ratio optimisation.

* First we will filter only stocks corresponding to the cluster we choose based on our hypothesis.

* As per our strategy hypothesis we select stocks corresponding to cluster 3.

* We will define a function which optimises portfolio weights using PyPortfolioOpt package and EfficientFrontier optimiser to maximise the Sharpe ratio.

* To optimise the weights of a given portfolio we would need to supply last year's prices to the function.

* Apply single stock weight bounds constraint for diversification (minimum half of equally weighted and maximum 10% of portfolio).

After that we:

* Calculate daily returns for each stock which could land up in our portfolio.

* Then loop over each month's start, select the stocks for the month and calculate their weights for the next month.

* If the maximum Sharpe ratio optimization fails for a given month, apply equally-weighted weights.

* Calculate daily portfolio returns.

Finally we visualise portfolio returns and compare to S&P 500 returns.

![Strategy Returns vs SPY Buy&Hold](https://github.com/user-attachments/assets/144cada6-1acb-471b-942f-e52d349bed29)
![Adjusted Returns](https://github.com/user-attachments/assets/efb554eb-7ccc-4e40-b70e-4106e805ece1)

