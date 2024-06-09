#!/usr/bin/env python
# coding: utf-8

# ## Returns calculation (Daily)

# In[244]:


# Returns will be just the difference of prices between 2 consecutive dates
df_SP500['Return'] = df_SP500['Price'].pct_change() 
df_BTC['Return'] = df_BTC['Price'].pct_change() 
df_TMUBMUSD01Y['Return'] = df_TMUBMUSD01Y['Price'].pct_change() 


# In[245]:


# print(df_SP500.head())
# print(df_TMUBMUSD01Y.head())
# print(df_BTC.head())


# # Exploratory Data Analysis

# ## Summary Statistics

# In[246]:


# Basic Statistics
print(df_SP500.describe())
print(df_BTC.describe())
print(df_TMUBMUSD01Y.describe())


# ## Check for Missing Values

# In[247]:


print(df_SP500.isnull().sum())
print(df_BTC.isnull().sum())
print(df_TMUBMUSD01Y.isnull().sum())
# for return the first column will be none (no previous price)


# ## Price Changes of the S&P 500 Index, Bitcoin, and the 1-Year U.S. Treasury Bond

# In[248]:


plt.figure(figsize=(20, 10))

plt.plot(df_SP500.index, df_SP500['Price'], label='S&P 500')
plt.plot(df_BTC.index, df_BTC['Price'], label='Bitcoin')
plt.plot(df_TMUBMUSD01Y.index, df_TMUBMUSD01Y['Price'], label='1-Year U.S. Treasury Bond')

plt.title('Price Changes of the S&P 500 Index, Bitcoin, and the 1-Year U.S. Treasury Bond')
plt.legend()
plt.show()


# ### S&P 500 Index

# In[249]:


plt.figure(figsize=(20, 5))
plt.plot(df_SP500.index, df_SP500['Price'], label='S&P 500')
plt.title('S&P 500')
plt.legend()
plt.grid(False)
plt.show()


# ### Bitcoin

# In[250]:


plt.figure(figsize=(20, 5))
plt.plot(df_BTC.index, df_BTC['Price'], label='Bitcoin', color='orange')
plt.title('Bitcoin')
plt.legend()
plt.grid(False)
plt.show()


# ### 1-Year U.S. Treasury Bond

# In[251]:


plt.figure(figsize=(20, 5))
plt.plot(df_TMUBMUSD01Y.index, df_TMUBMUSD01Y['Price'], label='1-Year U.S. Treasury Bond', color='green')
plt.title('1-Year U.S. Treasury Bond')
plt.legend()
plt.grid(False)
plt.show()


# ## Daily Returns

# In[252]:


df_combined_returns = pd.concat([df_SP500[['Return']], df_BTC[['Return']], df_TMUBMUSD01Y[['Return']]], axis=1, join='inner')
df_combined_returns.columns = ['SP500_Return', 'BTC_Return', 'TMUBMUSD01Y_Return']


# In[253]:


df_combined_returns


# In[254]:


def percentage_formatter(x, pos):
    return f'{x:.1f}%'

plt.figure(figsize=(20, 10))

plt.subplot(3, 1, 1)
plt.plot(df_SP500.index, df_SP500['Return'], label='S&P 500 Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df_BTC.index, df_BTC['Return'], label='Bitcoin Daily Returns', color='orange')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df_TMUBMUSD01Y.index, df_TMUBMUSD01Y['Return'], label='1-Year U.S. Treasury Bond Returns', color='green')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.legend()

plt.tight_layout()
plt.show()


# ## Forward-filling

# In[255]:


""" Forward-filling missing values ensures consistency gaps 
by filling gaps with the last available observation, which is essential 
when aligning datasets with different trading schedules
, such as traditional markets and Bitcoin (Open on Weekends)."""

# This piece of code was taken from ChapGPT 
# Recommended using Forward-filling for S&P and 1Y US Treasury Bond

df_SP500 = df_SP500.asfreq('D').fillna(method='ffill')
df_TMUBMUSD01Y = df_TMUBMUSD01Y.asfreq('D').fillna(method='ffill')

# Linear Interpolation
df_SP500['Price'] = df_SP500['Price'].interpolate(method='linear')
df_TMUBMUSD01Y['Price'] = df_TMUBMUSD01Y['Price'].interpolate(method='linear')


# ## Merging Files

# In[256]:


# Merge files & Rename Columns
df_assets = pd.merge(df_SP500, df_BTC, on='Date', how='outer')
df_assets = pd.merge(df_assets, df_TMUBMUSD01Y, on='Date', how='outer')

df_assets.columns = ['SP500_Price', 'SP500_Return', 'BTC_Price', 'BTC_Return', 'TMUBMUSD01Y_Price', 'TMUBMUSD01Y_Return']

# Drop NaN
df_assets.dropna(subset=['SP500_Return', 'BTC_Return', 'TMUBMUSD01Y_Return'], inplace=True)

# print(df_assets.head())


# ### + Macroeconomic Indicators

# In[257]:


# Forward-filling
macro_data = pd.concat([UN_Rate, Real_GDP, CPI, VIX, Ten_Year_Treasury_Yield, Federal_Funds_Rate, BBB_Credit_Spread], axis=1).ffill()

# Merge with the assets
df_assets = df_assets.join(macro_data, how='left')

# Ffill and Bfill
df_assets = df_assets.ffill().bfill()

#print(df_assets.head())


# In[258]:


correlation_matrix = df_assets[['SP500_Return', 'BTC_Return', 'TMUBMUSD01Y_Return']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Returns')
plt.show()


# In[259]:


plt.figure(figsize=(14, 10))

correlation_matrix = df_assets.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Correlation between Returns and Macroeconomic Indicators', fontsize=14)

plt.show()


# ## Statistical Analysis

# ### Summary

# In[260]:


print(df_combined_returns.describe())


# In[261]:


print(df_combined_returns.skew())


# In[262]:


print(df_combined_returns.kurtosis())


# ### ADF

# In[263]:


def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Reject the null hypothesis - stationary")
    else:
        print("Fail to reject the null hypothesis - non-stationary")

adf_test(df_combined_returns['SP500_Return'])
adf_test(df_combined_returns['BTC_Return'])
adf_test(df_combined_returns['TMUBMUSD01Y_Return'])


# ### KDE

# In[264]:


sns.histplot(df_combined_returns['SP500_Return'], kde=True)
plt.title('Distribution of S&P 500 Returns')
plt.show()


# In[265]:


sns.histplot(df_combined_returns['BTC_Return'], kde=True)
plt.title('Distribution of Bitcoin Returns')
plt.show()


# In[266]:


sns.histplot(df_combined_returns['TMUBMUSD01Y_Return'], kde=True)
plt.title('Distribution of 1-Year Treasury Bond Returns')
plt.show()


# ### Shapiro-Wilk Test

# In[267]:


def shapiro_test(series):
    stat, p_value = shapiro(series.dropna())
    print(f'Shapiro-Wilk Test: Statistics={stat}, p-value={p_value}')
    if p_value > 0.05:
        print("Fail to reject the null hypothesis - data is normally distributed")
    else:
        print("Reject the null hypothesis - data is not normally distributed")


# In[268]:


print("SP500 Returns:")
shapiro_test(df_combined_returns['SP500_Return'])

print("\nBTC Returns:")
shapiro_test(df_combined_returns['BTC_Return'])

print("\n1-Year Treasury Bond Returns:")
shapiro_test(df_combined_returns['TMUBMUSD01Y_Return'])


# ### Plots

# In[269]:


plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
sns.boxplot(x=df_combined_returns['SP500_Return'])
plt.title('S&P 500 Returns')

plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 2)
sns.boxplot(x=df_combined_returns['BTC_Return'])
plt.title('BTC_Return')

plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 3)
sns.boxplot(x=df_combined_returns['TMUBMUSD01Y_Return'])
plt.title('1-Year U.S. Treasury Bond Returns')

plt.show()


# ### Outliers

# In[270]:


def extreme_values(df, column, percentile=0.01):
    threshold_high = df[column].quantile(1 - percentile)
    threshold_low = df[column].quantile(percentile)
    high = df[df[column] >= threshold_high]
    low = df[df[column] <= threshold_low]
    return high, low


# In[271]:


high_sp500, low_sp500 = extreme_values(df_combined_returns, 'SP500_Return')
high_btc, low_btc = extreme_values(df_combined_returns, 'BTC_Return')
high_treasury, low_treasury = extreme_values(df_combined_returns, 'TMUBMUSD01Y_Return')


# In[272]:


extreme_sp500_dates = high_sp500.index.union(low_sp500.index)
print(extreme_sp500_dates)


# In[273]:


extreme_btc_dates = high_btc.index.union(low_btc.index)
print(extreme_btc_dates)


# In[274]:


extreme_treasury_dates = high_treasury.index.union(low_treasury.index)
print(extreme_treasury_dates)


# ## ACF & PACF

# In[275]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['SP500_Return'].dropna(), lags=40)
plt.title('ACF of S&P 500 Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['SP500_Return'].dropna(), lags=40)
plt.title('PACF of S&P 500 Returns')
plt.show()


# In[276]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['BTC_Return'].dropna(), lags=40)
plt.title('ACF of Bitcoin Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['BTC_Return'].dropna(), lags=40)
plt.title('PACF of Bitcoin Returns')
plt.show()


# In[277]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['TMUBMUSD01Y_Return'].dropna(), lags=40)
plt.title('ACF of 1-Year Treasury Bond Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['TMUBMUSD01Y_Return'].dropna(), lags=40)
plt.title('PACF of 1-Year Treasury Bond Returns')
plt.show()

