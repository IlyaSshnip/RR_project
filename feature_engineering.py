#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# ## Calculating Log Returns

# In[278]:


df_assets['SP500_Log_Return'] = np.log(df_assets['SP500_Price'] / df_assets['SP500_Price'].shift(1))
df_assets['BTC_Log_Return'] = np.log(df_assets['BTC_Price'] / df_assets['BTC_Price'].shift(1))
df_assets['TMUBMUSD01Y_Log_Return'] = np.log(df_assets['TMUBMUSD01Y_Price'] / df_assets['TMUBMUSD01Y_Price'].shift(1))


# ## Calculating Lagged Returns

# In[279]:


df_assets['SP500_Lagged_Return'] = df_assets['SP500_Return'].shift(1)
df_assets['BTC_Lagged_Return'] = df_assets['BTC_Return'].shift(1)
df_assets['TMUBMUSD01Y_Lagged_Return'] = df_assets['TMUBMUSD01Y_Return'].shift(1)


# ## Rolling Statistics

# #### Calculate and plot rolling std for all portfolios with 21-day window

# In[280]:


# Reference: Portfolio_Performance repository
# File: whale_analysis.ipynb
# URL: https://github.com/lrb924/Portfolio_Performance/blob/main/whale_analysis.ipynb

window = 21

df_assets['SP500_Rolling_Mean'] = df_assets['SP500_Log_Return'].rolling(window).mean()
df_assets['SP500_Rolling_Std'] = df_assets['SP500_Log_Return'].rolling(window).std()

df_assets['BTC_Rolling_Mean'] = df_assets['BTC_Log_Return'].rolling(window).mean()
df_assets['BTC_Rolling_Std'] = df_assets['BTC_Log_Return'].rolling(window).std()

df_assets['TMUBMUSD01Y_Rolling_Mean'] = df_assets['TMUBMUSD01Y_Log_Return'].rolling(window).mean()
df_assets['TMUBMUSD01Y_Rolling_Std'] = df_assets['TMUBMUSD01Y_Log_Return'].rolling(window).std()


# In[281]:


plt.figure(figsize=(20, 20))

plt.subplot(3, 1, 1)
plt.plot(df_assets.index, df_assets['SP500_Log_Return'], label='S&P 500 Log Returns')
plt.plot(df_assets.index, df_assets['SP500_Rolling_Mean'], label='Rolling Mean')
plt.plot(df_assets.index, df_assets['SP500_Rolling_Std'], label='Rolling Std')
plt.title('S&P 500 Rolling Statistics')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df_assets.index, df_assets['BTC_Log_Return'], label='Bitcoin Log Returns')
plt.plot(df_assets.index, df_assets['BTC_Rolling_Mean'], label='Rolling Mean')
plt.plot(df_assets.index, df_assets['BTC_Rolling_Std'], label='Rolling Std')
plt.title('Bitcoin Rolling Statistics')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df_assets.index, df_assets['TMUBMUSD01Y_Log_Return'], label='1-Year Treasury Log Returns')
plt.plot(df_assets.index, df_assets['TMUBMUSD01Y_Rolling_Mean'], label='Rolling Mean')
plt.plot(df_assets.index, df_assets['TMUBMUSD01Y_Rolling_Std'], label='Rolling Std')
plt.title('1-Year Treasury Rolling Statistics')
plt.legend()

plt.show()


# ## Exponential Moving averages

# In[282]:


df_assets['SP500_EMA'] = df_assets['SP500_Log_Return'].ewm(span=21, adjust=False).mean()
df_assets['BTC_EMA'] = df_assets['BTC_Log_Return'].ewm(span=21, adjust=False).mean()
df_assets['TMUBMUSD01Y_EMA'] = df_assets['TMUBMUSD01Y_Log_Return'].ewm(span=21, adjust=False).mean()


# ## Interaction Features

# In[283]:


# Interaction between Log Returns
df_assets['SP500_BTC_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['BTC_Log_Return']
df_assets['SP500_Treasury_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['TMUBMUSD01Y_Log_Return']
df_assets['BTC_Treasury_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['TMUBMUSD01Y_Log_Return']

# Interaction between SP500 Log Returns and macroeconomic indicators
df_assets['SP500_UNRATE_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['UNRATE']
df_assets['SP500_GDP_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['A191RL1Q225SBEA']
df_assets['SP500_CPI_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['CPIAUCSL']
df_assets['SP500_VIX_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['VIXCLS']
df_assets['SP500_Treasury_Yield_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['DGS10']
df_assets['SP500_Fed_Funds_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['FEDFUNDS']
df_assets['SP500_Credit_Spread_Interaction'] = df_assets['SP500_Log_Return'] * df_assets['BAMLC0A4CBBB']

# Interaction between Bitcoin Log Returns and macroeconomic indicators
df_assets['BTC_UNRATE_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['UNRATE']
df_assets['BTC_GDP_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['A191RL1Q225SBEA']
df_assets['BTC_CPI_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['CPIAUCSL']
df_assets['BTC_VIX_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['VIXCLS']
df_assets['BTC_Treasury_Yield_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['DGS10']
df_assets['BTC_Fed_Funds_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['FEDFUNDS']
df_assets['BTC_Credit_Spread_Interaction'] = df_assets['BTC_Log_Return'] * df_assets['BAMLC0A4CBBB']

# Interaction between 1-Year Treasury Log Returns and macroeconomic indicators
df_assets['Treasury_UNRATE_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['UNRATE']
df_assets['Treasury_GDP_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['A191RL1Q225SBEA']
df_assets['Treasury_CPI_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['CPIAUCSL']
df_assets['Treasury_VIX_Interaction_Log'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['VIXCLS']
df_assets['Treasury_Treasury_Yield_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['DGS10']
df_assets['Treasury_Fed_Funds_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['FEDFUNDS']
df_assets['Treasury_Credit_Spread_Interaction'] = df_assets['TMUBMUSD01Y_Log_Return'] * df_assets['BAMLC0A4CBBB']


# ## Let's check how the data looks at this point:

# In[284]:


df_assets.dropna()

