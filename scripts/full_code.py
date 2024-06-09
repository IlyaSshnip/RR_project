#!/usr/bin/env python
# coding: utf-8

# # Project Description:
# In this project, we set out to evaluate the market risk of three key financial assets with very different risk profiles: the **S&P 500, Bitcoin,** and the **1-Year U.S. Treasury Bond**. 
# 
# To do this, we utilized different Value at Risk (VaR) metrics. We used **Historical VaR, Parametric VaR, MonteCarlo VaR and Expected Shortfall (ES) / Conditional VaR**.
# 
# Our analysis compreends a decade, from May 1, 2014, to May 1, 2024, which helps to understand both typical market conditions and periods of extreme market stress ( Covid ).

# --------------------------------------------------------------------------------------

# # Dataset Description
# 
# The final dataset has 52 columns, which include a variety of financial and macroeconomic variables from May 1, 2014, to May 1, 2024.
# 
# It primarily focuses on three main financial assets: the S&P 500, Bitcoin, and the 1-Year U.S. Treasury Bond. Alongside these, it also incorporates several macroeconomic indicators and features to improve our model. Here’s a detailed look at what each column represents:
# 
# - **Asset Data**: Provides the historical performance and returns of the S&P 500, Bitcoin, and 1-Year U.S. Treasury Bond.
# 
# - **Macroeconomic Indicators**: Offer insights into the broader economic environment affecting asset prices and returns. We used them for correlation and to create the interaction terms, that we decided to drop off, due to overestimation results. They still play an important role. We can use them separately with VaR calculations to estimate the market risk, as they still offer valuable insights.
# 
# - **Features (Used in VaR Calculation)**: Used for MC VaR calculations to predict future price paths and assess market risk. The log returns will have 95% weight for the calculations and the other features 5%, this is to avoid overestimation of the VaR threshold.
# 
# - **Interaction Features**: They capture the combined effects of asset returns and macroeconomic factors. They won’t be used in our final models, because even after scaling and normalizing appropriately, they still made our model overestimate our VaR.
# 
# # Columns Description
# 
# ## Baseline Data
# 
# ### Asset Data:
# 1. **SP500_Price**: The closing price of the S&P 500 index.
# 
# 2. **BTC_Price**: The closing price of Bitcoin.
# 
# 3. **TMUBMUSD01Y_Price**: The closing price of the 1-Year U.S. Treasury Bond.
# 
# ### Macroeconomic Indicators:
# 4. **UNRATE**: Unemployment Rate.
# 
# 5. **A191RL1Q225SBEA**: Real GDP.
# 
# 6. **CPIAUCSL**: Consumer Price Index.
# 
# 7. **VIXCLS**: Volatility Index (VIX).
# 
# 8. **DGS10**: 10-Year Treasury Yield.
# 
# 9. **FEDFUNDS**: Federal Funds Rate.
# 
# 10. **BAMLC0A4CBBB**: BBB Credit Spread.
# 
# ## Created Data - Financial Transformations on Baseline Data
# 
# ### Additional Features used for Data Visualizations and Statistics:
# 11. **SP500_Return**: The daily percentage change in the closing price of the S&P 500.
# 
# 12. **BTC_Return**: The daily percentage change in the closing price of Bitcoin.
# 
# 13. **TMUBMUSD01Y_Return**: The daily percentage change in the closing price of the 1-Year U.S. Treasury Bond.
# 
# ### Features (Used in VaR Calculation):
# 14. **SP500_Log_Return**: Log return of S&P 500 prices.
# 
# 15. **BTC_Log_Return**: Log return of Bitcoin prices.
# 
# 16. **TMUBMUSD01Y_Log_Return**: Log return of 1-Year Treasury Bond prices.
# 
# 17. **SP500_Lagged_Return**: Previous day’s return of S&P 500.
# 
# 18. **BTC_Lagged_Return**: Previous day’s return of Bitcoin.
# 
# 19. **TMUBMUSD01Y_Lagged_Return**: Previous day’s return of 1-Year Treasury Bond.
# 
# 20. **SP500_Rolling_Mean**: 21-day rolling mean of S&P 500 log returns.
# 
# 21. **SP500_Rolling_Std**: 21-day rolling standard deviation of S&P 500 log returns.
# 
# 22. **BTC_Rolling_Mean**: 21-day rolling mean of Bitcoin log returns.
# 
# 23. **BTC_Rolling_Std**: 21-day rolling standard deviation of Bitcoin log returns.
# 
# 24. **TMUBMUSD01Y_Rolling_Mean**: 21-day rolling mean of 1-Year Treasury Bond log returns.
# 
# 25. **TMUBMUSD01Y_Rolling_Std**: 21-day rolling standard deviation of 1-Year Treasury Bond log returns.
# 
# 26. **SP500_EMA**: 21-day exponential moving average of S&P 500 log returns.
# 
# 27. **BTC_EMA**: 21-day exponential moving average of Bitcoin log returns.
# 
# 28. **TMUBMUSD01Y_EMA**: 21-day exponential moving average of 1-Year Treasury Bond log returns.
# 
# ### Interaction Features:
# 29. **SP500_BTC_Interaction**: Interaction term between S&P 500 Log Returns and Bitcoin Log Returns.
# 
# 30. **SP500_Treasury_Interaction**: Interaction term between S&P 500 Log Returns and 1-Year Treasury Log Returns.
# 
# 31. **BTC_Treasury_Interaction**: Interaction term between Bitcoin Log Returns and 1-Year Treasury Log Returns.
# 
# 32. **SP500_UNRATE_Interaction**: Interaction term between S&P 500 Log Returns and Unemployment Rate.
# 
# 33. **SP500_GDP_Interaction**: Interaction term between S&P 500 Log Returns and Real GDP.
# 
# 34. **SP500_CPI_Interaction**: Interaction term between S&P 500 Log Returns and Consumer Price Index.
# 
# 35. **SP500_VIX_Interaction**: Interaction term between S&P 500 Log Returns and Volatility Index.
# 
# 36. **SP500_Treasury_Yield_Interaction**: Interaction term between S&P 500 Log Returns and 10-Year Treasury Yield.
# 
# 37. **SP500_Fed_Funds_Interaction**: Interaction term between S&P 500 Log Returns and Federal Funds Rate.
# 
# 38. **SP500_Credit_Spread_Interaction**: Interaction term between S&P 500 Log Returns and BBB Credit Spread.
# 
# 39. **BTC_UNRATE_Interaction**: Interaction term between Bitcoin Log Returns and Unemployment Rate.
# 
# 40. **BTC_GDP_Interaction**: Interaction term between Bitcoin Log Returns and Real GDP.
# 
# 41. **BTC_CPI_Interaction**: Interaction term between Bitcoin Log Returns and Consumer Price Index.
# 
# 42. **BTC_VIX_Interaction**: Interaction term between Bitcoin Log Returns and Volatility Index.
# 
# 43. **BTC_Treasury_Yield_Interaction**: Interaction term between Bitcoin Log Returns and 10-Year Treasury Yield.
# 
# 44. **BTC_Fed_Funds_Interaction**: Interaction term between Bitcoin Log Returns and Federal Funds Rate.
# 
# 45. **BTC_Credit_Spread_Interaction**: Interaction term between Bitcoin Log Returns and BBB Credit Spread.
# 
# 46. **Treasury_UNRATE_Interaction**: Interaction term between 1-Year Treasury Log Returns and Unemployment Rate.
# 
# 47. **Treasury_GDP_Interaction**: Interaction term between 1-Year Treasury Log Returns and Real GDP.
# 
# 48. **Treasury_CPI_Interaction**: Interaction term between 1-Year Treasury Log Returns and Consumer Price Index.
# 
# 49. **Treasury_VIX_Interaction_Log**: Interaction term between 1-Year Treasury Log Returns and Volatility Index.
# 
# 50. **Treasury_Treasury_Yield_Interaction**: Interaction term between 1-Year Treasury Log Returns and 10-Year Treasury Yield.
# 
# 51. **Treasury_Fed_Funds_Interaction**: Interaction term between 1-Year Treasury Log Returns and Federal Funds Rate.
# 
# 52. **Treasury_Credit_Spread_Interaction**: Interaction term between 1-Year Treasury Log Returns and BBB Credit Spread.

# --------------------------------------------------------------------------------------

# # Key Problems
# 
# ## What we decided to exclude:
# 
# ### Volatility Modeling  
# 
# We tried to improve our Value at Risk (VaR) predictions using advanced volatility models like GARCH (1,1), eGARCH (1,1), tGARCH (1,1), and apARCH (1,1). Our idea was to estimate conditional volatility and integrate it into our feature engineering. Inspired by some research projects, we wanted to create additional synthetic log returns using our engineered features plus a stochastic part with the standard deviation equal to the conditional volatility.
# 
# However, things didn’t go as planned. We faced several issues:
# 
# 1. **Model Parameters**: We tried different parameter settings in the GARCH models.
# 2. **Distributions**: We tried various distributions to see if they improved performance (T, SkewdT, GED)
# 3. **Winsorizing**: We applied winsorizing to reduce the impact of extreme values.
# 
# Despite all these efforts, our VaR predictions didn’t improve. After many attempts, we realized that this advanced stochastic model wasn’t working for our project and decided to drop it.
# 
# ### Interaction Features Challenges
# 
# We also had issues with the interaction features we created to capture the combined effects of asset returns and macroeconomic factors. Even after reducing their weight in the model, scalling and normalizing they still had a huge impact on our estimations.
# 
# Key problems included:
# 
# - **Overestimation**: The interaction features consistently led to overestimation in our VaR model, even after scaling and normalizing them appropriately.
# - **Weight Adjustment**: Reducing the weight of these features as we did for the other ones still didn’t help as much as we thought.
# 
# Because of these issues, we decided not to use the interaction features in our final model.
# 
# ## Additional Challenges
# 
# Besides these major challenges, we also faced smaller problems with the code and debugging. Since this project was mainly our own, we were following different research papers and trying to combine ideas from various sources. This made it tough to make everything uniform and ensure it all made sense together.
# 
# ### Specific Issues Included:
# 
# - **Trial and Error**: We spent a lot of time trying different approaches and removing those that didn’t work.
# - **Data Processing**: Figuring out the right preprocessing steps for our assets was challenging. Each asset had different characteristics, making it hard to decide what to keep or remove, especially regarding distribution and outliers, at the end we kept our outliers because they related to real market scenarios and not some issue in the data. Of course, it affected our results, but they still pretty acceptable.
# 
#  *Note: For the 1Y Treasury bond, the main issue was having such a spike related to COVID. This is the result we are in general less satisfied about.*
# 
# ### Conclusion
# 
# Working with advanced volatility models and interaction features was an important learning experience, even though it didn’t enhance our VaR predictions as we hoped. These challenges reminded us of the importance of model simplicity and carefully considering feature impacts in predictive modeling. Plus, it was a real eye-opener on how crucial it is to ensure uniformity when integrating ideas from different sources and understanding the right data processing steps for various assets.

# --------------------------------------------------------------------------------------

# # Structure
# Most of the structure and code ideas for this project up to the feature engineering stage, were adapted from a machine learning project we worked on in the first semester. This original project was designed by our professor and had a completely different objective.
# 
# https://github.com/michaelwozniak/ML-in-Finance-I-case-study-forecasting-tax-avoidance-rates/blob/main/notebooks/01.project-description-%26-data-preparation-%26-EDA.ipynb
# 
# ## For the feature engineering part:
# Returns, Log Returns and lagged returns, we used for different projects for other disciplines. They are widely used for many different purposes in finance.
# 
# #### Rolling Statistics and Exponential Moving averages were taken from:
# 
# Reference: Portfolio_Performance repository
# File: whale_analysis.ipynb
# URL: https://github.com/lrb924/Portfolio_Performance/blob/main/whale_analysis.ipynb
# 
# *Interactions: just simple calculations.*

# # Dependencies loading

# In[240]:


import pandas as pd # Standard
import numpy as np # Standard
from datetime import datetime # Date and Time
from pandas_datareader import data as pdr # Import data from FRED

import matplotlib.pyplot as plt # Data Visualization
from matplotlib.ticker import FuncFormatter # Data Visualization
import seaborn as sns # Data Visualization

from scipy.stats import shapiro # Statistics
from statsmodels.tsa.stattools import adfuller # Statistics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Statistics


# In[241]:


import warnings
warnings.filterwarnings("ignore")


# # Data preparation

# ## Data loading

# In[242]:


BTC_url = 'https://raw.githubusercontent.com/IlyaSshnip/RR_project/main/Bitcoin_daily.csv'
SP500_url = 'https://raw.githubusercontent.com/IlyaSshnip/RR_project/main/S&P500.csv'
TMUBMUSD01Y_url = 'https://raw.githubusercontent.com/IlyaSshnip/RR_project/main/TMUBMUSD01Y.csv'

df_BTC = pd.read_csv(BTC_url)
df_BTC.columns = df_BTC.columns.str.strip()

df_SP500 = pd.read_csv(SP500_url)
df_SP500.columns = df_SP500.columns.str.strip()

df_TMUBMUSD01Y = pd.read_csv(TMUBMUSD01Y_url)
df_TMUBMUSD01Y.columns = df_TMUBMUSD01Y.columns.str.strip()

start=datetime(2014, 5, 1)
end=datetime(2024, 5, 1)

# Macroeconomic Indicators (from FRED):

UN_Rate = pdr.get_data_fred('UNRATE', start, end) # Unemployment Rate
Real_GDP = pdr.get_data_fred('A191RL1Q225SBEA', start, end) # Real GDP
CPI = pdr.get_data_fred('CPIAUCSL', start, end) # Consumer Price Index
VIX = pdr.get_data_fred('VIXCLS', start, end) # Volatility Index
Ten_Year_Treasury_Yield = pdr.get_data_fred('DGS10', start, end) # 10-Year Treasury Yield
Federal_Funds_Rate = pdr.get_data_fred('FEDFUNDS', start, end) # Federal Funds Rate
BBB_Credit_Spread = pdr.get_data_fred('BAMLC0A4CBBB', start, end) # BBB Credit Spread

# print(df_BTC.head())
# print(df_SP500.head())
# print(df_TMUBMUSD01Y.head())
# print(UN_Rate.head())
# print(Real_GDP.head())
# print(CPI.head())
# print(VIX.head())
# print(Ten_Year_Treasury_Yield.head())
# print(Federal_Funds_Rate.head())
# print(BBB_Credit_Spread.head())


# ## Dataset adjustment

# In[243]:


# Select and rename the columns
df_SP500 = df_SP500[['Date', 'Close']].rename(columns={'Close': 'Price'})
df_BTC = df_BTC[['Date', 'Price']]
df_TMUBMUSD01Y = df_TMUBMUSD01Y[['Date', 'Close']].rename(columns={'Close': 'Price'})

########################################################################################

# Define Price as 'numeric'

df_SP500['Price'] = df_SP500['Price'].astype(str)
df_BTC['Price'] = df_BTC['Price'].astype(str)
df_TMUBMUSD01Y['Price'] = df_TMUBMUSD01Y['Price'].astype(str)

df_SP500['Price'] = pd.to_numeric(df_SP500['Price'].str.replace(',', ''), errors='coerce')
df_BTC['Price'] = pd.to_numeric(df_BTC['Price'].str.replace(',', ''), errors='coerce')
df_TMUBMUSD01Y['Price'] = pd.to_numeric(df_TMUBMUSD01Y['Price'].str.replace(',', ''), errors='coerce')

########################################################################################

# Date
df_SP500['Date'] = pd.to_datetime(df_SP500['Date'])
df_SP500.set_index('Date', inplace=True)

df_BTC['Date'] = pd.to_datetime(df_BTC['Date'])
df_BTC.set_index('Date', inplace=True)

df_TMUBMUSD01Y['Date'] = pd.to_datetime(df_TMUBMUSD01Y['Date'])
df_TMUBMUSD01Y.set_index('Date', inplace=True)

# Sorting by Date (because bitcoin starts with last price, while S&P and Bond start with first price)
df_SP500.sort_index(inplace=True)
df_BTC.sort_index(inplace=True)
df_TMUBMUSD01Y.sort_index(inplace=True)

df_SP500 = df_SP500.loc[start:end]
df_BTC = df_BTC.loc[start:end] # Cryptocurrency market is open on weekends
df_TMUBMUSD01Y = df_TMUBMUSD01Y.loc[start:end]

########################################################################################

# print(df_SP500.head())
# print(df_SP500.tail()) 
# print(df_TMUBMUSD01Y.head())
# print(df_TMUBMUSD01Y.tail())
# print(df_BTC.tail())


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


# As expected, the returns for Bitcoin over the last 10 years have been huge, significantly outpacing those of the S&P 500. The S&P 500 shows steady growth, while the 1-Year U.S. Treasury Bond prices have remained stable with little increase or decrease throughout their existence.
# 

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


# This plot shows the daily returns for the S&P 500, Bitcoin, and the 1-Year U.S. Treasury Bond over the last 10 years. We can see that COVID-19 had a big impact, especially on the S&P 500 and Bitcoin returns, causing large spikes in volatility. The 1-Year U.S. Treasury Bond returns stayed mostly stable, with one significant spike during the pandemic.

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


# In our opinion, this approach approximates reality the best. The last price on Friday will be the price for the entire weekend, which mirrors real-life market behavior. The same logic applies to holidays.

# ## Merging Files

# Let's merge it all together with our macroeconomic factors:

# In[256]:


# Merge files & Rename Columns
df_assets = pd.merge(df_SP500, df_BTC, on='Date', how='outer')
df_assets = pd.merge(df_assets, df_TMUBMUSD01Y, on='Date', how='outer')

df_assets.columns = ['SP500_Price', 'SP500_Return', 'BTC_Price', 'BTC_Return', 'TMUBMUSD01Y_Price', 'TMUBMUSD01Y_Return']

# Drop NaN
df_assets.dropna(subset=['SP500_Return', 'BTC_Return', 'TMUBMUSD01Y_Return'], inplace=True)

# print(df_assets.head())


# ### + Macroeconomic Indicators

# The same approach applies to macroeconomic factors, where we use forward and backward filling to avoid any NaN values. This ensures data consistency and completeness:

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


# The correlation matrix shows that S&P 500 and Bitcoin returns have a weak positive correlation (0.17), while both have minimal correlation with 1-Year U.S. Treasury returns.

# In[259]:


plt.figure(figsize=(14, 10))

correlation_matrix = df_assets.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Correlation between Returns and Macroeconomic Indicators', fontsize=14)

plt.show()


# #### Federal Funds Rate (0.32 with S&P 500, 0.32 with Bitcoin, 0.32 with 1-Year Treasury):
# - Federal Funds Rates are positively correlated with returns for all three assets;
# 
# #### BBB Credit Spread (0.45 with S&P 500, 0.53 with Bitcoin, 0.22 with 1-Year Treasury):
# - Moderate positive correlations with S&P 500 and Bitcoin returns indicate that larger credit spreads (risk premiums) are associated with higher returns for these riskier assets;
# 
# - Positive correlation with 1-Year Treasury returns shows that during times of increased credit risk, investors seek the safety of Treasuries;
# 
# #### Unemployment Rate (-0.2 with S&P 500, -0.2 with Bitcoin, -0.4 with 1-Year Treasury):
# - Higher unemployment rates are negatively correlated with S&P 500 and Bitcoin returns;
# - Negative correlation with 1-Year Treasury returns shows that higher unemployment rates drive investors to seek safer investments;
# 
# #### Real GDP (0.06 with S&P 500, -0.2 with Bitcoin, -0.22 with 1-Year Treasury):
# - Higher GDP growth has a small positive impact on stock market returns;
# - Negative correlation with Bitcoin and 1-Year Treasury returns implies that higher GDP growth leads to lower returns for these assets;
# 
# #### Consumer Price Index (0.19 with S&P 500, 0.19 with Bitcoin, -0.2 with 1-Year Treasury):
# - Positive correlations with S&P 500 and Bitcoin returns indicate that higher inflation (CPI) corresponds to higher returns;
# - Negative correlation with 1-Year Treasury returns shows that higher inflation reduces Treasury returns;
# 
# #### Volatility Index (VIX) (0.32 with S&P 500, 0.32 with Bitcoin, 0.32 with 1-Year Treasury):
# - Positive correlations across all three assets indicate that higher market volatility is associated with higher returns;
# 
# #### 10-Year Treasury Yield (0.32 with S&P 500, 0.32 with Bitcoin, 0.32 with 1-Year Treasury):
# - Moderate positive correlations with all three assets, meaning higher yields may coincide with higher returns, reflecting economic growth and higher interest rates.

# ## Statistical Analysis

# Alright, let's dive into the statistical analysis. This part is super important because it helps us understand the behavior of our data. By looking at things like summary statistics, skewness, kurtosis, and stationarity, we get a clearer picture of how the S&P 500, Bitcoin, and the 1-Year U.S. Treasury Bond have performed over time.
# 
# We’ll also run some tests and visualize the data to see if there are any patterns or anomalies. This step is crucial because it sets the foundation for more complex modeling later on. Essentially, it’s like getting to know our data really well before making any predictions or decisions.
# 
# So, let's get started and see what our data has to tell us!

# ### Summary

# In[260]:


print(df_combined_returns.describe())


# In[261]:


print(df_combined_returns.skew())


# In[262]:


print(df_combined_returns.kurtosis())


# S&P 500 and Bitcoin returns have small average returns, Bitcoin is more volatile. 
# S&P 500 returns are slightly left skewed, while Bitcoin is nearly symmetrical. Both have high kurtosis, meaning there are extreme values (outliers). 
# The 1-Year Treasury Bond returns show extreme positive skew and high kurtosis due to a few very large returns related to COVID. These results indicate non-normal distributions, but let's proceed with further analysis!

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


# Data is already stationary, there is no need to use differencing.

# ### KDE

# The KDE plots show that the distributions of S&P 500 and Bitcoin returns are centered around zero with heavy tails, suggesting high volatility. In contrast, the 1-Year U.S. Treasury Bond returns display positive skew with extreme values.

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

# We can see from the plots and summary statistics that our data doesn't follow a normal distribution. However, to confirm our observations, we'll apply a statistical test, specifically the Shapiro-Wilk test. This will give us a more accurate understanding of the data's distribution.

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

# The box plots show that both the S&P 500 and Bitcoin returns have many outliers, meaning there are some extreme values. The 1-Year U.S. Treasury Bond returns also have some very high values.
# 
# We should look into these extreme values to see if they are linked to specific market events.

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


# ### Conclusion

# COVID-19 pandemic was the major driver of extreme market movements across these different asset classes.

# ## ACF & PACF

# Let's look at the ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) tests. These tests show how current values relate to past values in our data.
# 
# The ACF tells us if there's a pattern over time by checking the correlation between different time points. High values mean the data points are related.
# 
# The PACF shows the direct correlation between current and past values, removing the effect of values in between. This helps pinpoint where the data is most strongly correlated.

# In[275]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['SP500_Return'].dropna(), lags=40)
plt.title('ACF of S&P 500 Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['SP500_Return'].dropna(), lags=40)
plt.title('PACF of S&P 500 Returns')
plt.show()


# The ACF and PACF plots show a big spike at lag 1, meaning the S&P 500 returns are influenced by their immediate past values. After that, the correlations drop off quickly, so there's not much long-term pattern.

# In[276]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['BTC_Return'].dropna(), lags=40)
plt.title('ACF of Bitcoin Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['BTC_Return'].dropna(), lags=40)
plt.title('PACF of Bitcoin Returns')
plt.show()


# The ACF and PACF plots show a big spike at lag 1, meaning Bitcoin returns are influenced by their immediate past values. After that, the correlations drop off quickly, so there's not much long-term pattern.

# In[277]:


plt.figure(figsize=(12, 6))
plot_acf(df_combined_returns['TMUBMUSD01Y_Return'].dropna(), lags=40)
plt.title('ACF of 1-Year Treasury Bond Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_combined_returns['TMUBMUSD01Y_Return'].dropna(), lags=40)
plt.title('PACF of 1-Year Treasury Bond Returns')
plt.show()


# The ACF and PACF plots for the 1-Year Treasury Bond show a big spike at lag 1, indicating that returns are influenced by their immediate past values. There's another noticeable spike at lag 2, but overall, the long-term correlations are low.

# # Feature Engineering

# ## Calculating Log Returns

# In[278]:


df_assets['SP500_Log_Return'] = np.log(df_assets['SP500_Price'] / df_assets['SP500_Price'].shift(1))
df_assets['BTC_Log_Return'] = np.log(df_assets['BTC_Price'] / df_assets['BTC_Price'].shift(1))
df_assets['TMUBMUSD01Y_Log_Return'] = np.log(df_assets['TMUBMUSD01Y_Price'] / df_assets['TMUBMUSD01Y_Price'].shift(1))


# ### We used log returns instead of simple returns for VaR metrics for a few reasons:
# 1. Log returns add up over time, which makes them easier to work with.
# 2. They tend to be more normally distributed, which aligns with our statistical results above.
# 3. Log returns treat ups and downs more symmetrically, so they're less skewed by big jumps.
# 4. They assume continuous compounding, which matches how most financial instruments are priced.

# ## Calculating Lagged Returns

# In[279]:


df_assets['SP500_Lagged_Return'] = df_assets['SP500_Return'].shift(1)
df_assets['BTC_Lagged_Return'] = df_assets['BTC_Return'].shift(1)
df_assets['TMUBMUSD01Y_Lagged_Return'] = df_assets['TMUBMUSD01Y_Return'].shift(1)


# Lagged returns are just the returns from the previous day. We use them to capture how today's returns are influenced by yesterday's returns. Our PACF results showed big spikes at lag 1 for all the assets, meaning there's a strong immediate correlation. So, it makes sense to include lagged returns in our model.Lagged Log Returns: Calculated to capture temporal dependencies and previous returns impact.

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


# We used a 21-day rolling window to calculate the rolling mean and standard deviation for each asset in our dataset. The rolling mean provides an average return over the past 21 days, while the rolling standard deviation measures the volatility over the same period. These rolling statistics help us track how the average returns and volatility change over time, smoothing out short-term fluctuations and providing a better sense of long-term trends.
# 
# Incorporating these rolling statistics into our Monte Carlo VaR model is essential for making the simulations more realistic. By including the rolling mean and standard deviation, we ensure that our model takes into account recent trends and volatility. This makes our risk assessments more accurate, reflecting the actual market conditions more closely.
# 
# The idea to use rolling statistics came from a project in the Portfolio_Performance repository from the file "whale_analysis.ipynb". This approach inspired us to integrate similar calculations into our model to improve the robustness of our risk analysis.

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


# The EMA gives us a smoothed average return, putting more weight on recent data points, which helps capture the latest market trends. Unlike a simple moving average, the EMA responds faster to recent price changes.
# 
# Incorporating the EMA into our Monte Carlo VaR model is crucial for accurately reflecting current market conditions. By including the EMA, our model stays up-to-date with recent trends, making our risk assessments more responsive to market dynamics.

# ## Interaction Features
# 
# Let's make a compreensive use of the macroeconomic indicators and add interactions between them and between them and macroeconomic indicators.

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


# Interaction Terms: Created between log returns and macroeconomic indicators to capture combined effects.

# ## Let's check how the data looks at this point:

# In[284]:


df_assets.dropna()


# We are good to move to the VaR metrics!!

# # Historical Method

# In[285]:


# Function to calculate VaR
def calculate_var(returns, days, confidence_level):
    sorted_returns = np.sort(returns)
    index = int(np.floor((1 - confidence_level) * len(sorted_returns) * days))
    var = sorted_returns[index]
    return var

# Calculate rolling 10-day returns for each asset
df_assets['SP500_Rolling_10'] = df_assets['SP500_Return'].rolling(window=10).sum()
df_assets['BTC_Rolling_10'] = df_assets['BTC_Return'].rolling(window=10).sum()
df_assets['TMUB_Rolling_10'] = df_assets['TMUBMUSD01Y_Return'].rolling(window=10).sum()

# Calculate 1-day and 10-day VaR for each asset using a 95% confidence level
confidence_level = 0.95
var_1day_sp500 = calculate_var(df_assets['SP500_Return'], 1, confidence_level)
var_10day_sp500 = calculate_var(df_assets['SP500_Rolling_10'].dropna(), 1, confidence_level)
var_1day_btc = calculate_var(df_assets['BTC_Return'], 1, confidence_level)
var_10day_btc = calculate_var(df_assets['BTC_Rolling_10'].dropna(), 1, confidence_level)
var_1day_tmub = calculate_var(df_assets['TMUBMUSD01Y_Return'], 1, confidence_level)
var_10day_tmub = calculate_var(df_assets['TMUB_Rolling_10'].dropna(), 1, confidence_level)


# In[286]:


# Function to perform backtesting of VaR
def backtest_var(returns, var):
    exceedances = returns < var
    return exceedances.sum(), exceedances.sum() / len(returns) * 100
# Backtesting 1-day and 10-day VaR for each asset
backtest_1day_sp500 = backtest_var(df_assets['SP500_Return'], var_1day_sp500)
backtest_10day_sp500 = backtest_var(df_assets['SP500_Rolling_10'].dropna(), var_10day_sp500)
backtest_1day_btc = backtest_var(df_assets['BTC_Return'], var_1day_btc)
backtest_10day_btc = backtest_var(df_assets['BTC_Rolling_10'].dropna(), var_10day_btc)
backtest_1day_tmub = backtest_var(df_assets['TMUBMUSD01Y_Return'], var_1day_tmub)
backtest_10day_tmub = backtest_var(df_assets['TMUB_Rolling_10'].dropna(), var_10day_tmub)


# In[287]:


# Print the numerical results
print("VaR and Backtesting Results:")
print(f"SP500 - 1-day VaR: {var_1day_sp500:.2%}, Exceedances: {backtest_1day_sp500}")
print(f"SP500 - 10-day VaR: {var_10day_sp500:.2%}, Exceedances: {backtest_10day_sp500}")
print(f"BTC - 1-day VaR: {var_1day_btc:.2%}, Exceedances: {backtest_1day_btc}")
print(f"BTC - 10-day VaR: {var_10day_btc:.2%}, Exceedances: {backtest_10day_btc}")
print(f"TMUB - 1-day VaR: {var_1day_tmub:.2%}, Exceedances: {backtest_1day_tmub}")
print(f"TMUB - 10-day VaR: {var_10day_tmub:.2%}, Exceedances: {backtest_10day_tmub}")


# In[288]:


# Visualization
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(df_assets['SP500_Return'], label='SP500 Daily Returns')
plt.axhline(y=var_1day_sp500, color='r', linestyle='-', label='1-day VaR')
plt.axhline(y=var_10day_sp500, color='g', linestyle='-', label='10-day VaR')
plt.title('SP500 Returns and VaR')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df_assets['BTC_Return'], label='BTC Daily Returns')
plt.axhline(y=var_1day_btc, color='r', linestyle='-', label='1-day VaR')
plt.axhline(y=var_10day_btc, color='g', linestyle='-', label='10-day VaR')
plt.title('BTC Returns and VaR')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df_assets['TMUBMUSD01Y_Return'], label='TMUB Daily Returns')
plt.axhline(y=var_1day_tmub, color='r', linestyle='-', label='1-day VaR')
plt.axhline(y=var_10day_tmub, color='g', linestyle='-', label='10-day VaR')
plt.title('TMUB Returns and VaR')
plt.legend()

plt.tight_layout()
plt.show()


# # Parametric method to calculate VaR

# In[289]:


df_log_returns = df_assets[['SP500_Log_Return', 'BTC_Log_Return', 'TMUBMUSD01Y_Log_Return']]
df_log_returns.dropna(inplace=True)

#print(df_log_returns)


# In[290]:


# We will calculate maximum loss of each security seperately:

returns_percent = df_log_returns * 100

# Calculating mean and standard deviation of our returns
mean_returns = returns_percent.mean()
print (mean_returns)

std_dev_returns = returns_percent.std()
print (std_dev_returns)

# We set confidence level at 95%
z_score = 1.645  # Z-score for 95% confidence level (one-sided)


# In[291]:


#Calculating VaR of the securities:
VaR_BTC = (mean_returns['BTC_Log_Return'] + z_score * std_dev_returns['BTC_Log_Return']) * -1
print(f"Value at Risk (VaR) for BTC at 95% confidence level is {VaR_BTC:.2f}%")

VaR_SP500 = (mean_returns['SP500_Log_Return'] + z_score * std_dev_returns['SP500_Log_Return']) * -1
print(f"Value at Risk (VaR) for S&P500 at 95% confidence level is {VaR_SP500:.2f}%")

VaR_TBONDS = (mean_returns['TMUBMUSD01Y_Log_Return'] + z_score * std_dev_returns['TMUBMUSD01Y_Log_Return']) * -1
print(f"Value at Risk (VaR) for T-bonds at 95% confidence level is {VaR_TBONDS:.2f}%")


# This means that, with 95% confidence, our maximum loss for Bitcoin, S&P500 and Treasury Bonds are not expected to exceed 6.26%, 1.56% and 16.58% in one day. On the other hand, there is 5% chance that our portfolio will lose more than 6.80%, 1.89% and 16.58% of initial investment in each security over one trading day. It is clearly seen that the VaR for T-Bonds is higher, which does not make much sense as Bonds are considered relativley safe investment instruments.
# 
# The primary reason for this is that Bonds are more sensitive to interest rates, meaning that the interest rates have to be factored into the model. In addition, the characteristics of a bond change everyday since the maturity changes everyday. Therefore, the VaR calculation approach for bonds is more complicated compared to stocks, i.e. incorporates Duration and Convexity.

# # Theoretical portfolio VaR with Variance-Covariance matrix

# Just for the purpose of demonstration on how to apply the matrix, we assume that our "portfolio" consists of the three securities :) (https://medium.com/@akjha22/quantitative-finance-using-python-8-value-at-risk-d9e280439435)

# In[292]:


# We assign equal weights to our portfolio 
weights = np.array([0.33, 0.33, 0.33])

cov_matrix = df_log_returns.cov()
print(cov_matrix)

# Calculating portfolio mean and standard deviation
portfolio_mean = np.dot(weights, mean_returns)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_std_dev = np.sqrt(portfolio_variance)


# In[293]:


#Calculating VaR with 95% confidence level:
VaR = z_score * portfolio_std_dev

print(f"Portfolio Value at Risk (VaR) at 95% confidence level is {VaR}")


# Hence, our theoretical portfolio would lose a maximum of 6% given a 95% confidence level in a single day.

# # Expanding the Time Horizon

# In[294]:


## Now we will calculate the maximum loss over 10 days.
## To do that, we simply multiply our daily VaR by the square root of assumed number of days 
 
n_days = 10
VaR_10_day_BTC = VaR_BTC * np.sqrt(n_days)
print(f"Value at Risk (VaR) over {n_days} days for BTC at 95% confidence level: {VaR_10_day_BTC:.2f}%")

VaR_10_day_SP500 = VaR_SP500 * np.sqrt(n_days)
print(f"Value at Risk (VaR) over {n_days} days for S&P500 at 95% confidence level: {VaR_10_day_SP500:.2f}%")

VaR_10_day_TBONDS = VaR_TBONDS * np.sqrt(n_days)
print(f"Value at Risk (VaR) over {n_days} days for Treasury Bonds at 95% confidence level: {VaR_10_day_TBONDS:.2f}%")

# For theoretical portfolio:
VaR_10_day_Portfolio = -VaR * np.sqrt(n_days) * 100
print(f"Value at Risk (VaR) over {n_days} days for the portfolio at 95% confidence level: {VaR_10_day_Portfolio:.2f}%")


# The results show that in 10 days, our "portfolio" would lose" 18.69% of initial investment.

# In[295]:


# we set the backtesting period for 10 years 
backtesting_period = 2520

# The initial variables of violations:
num_violations_BTC = 0
num_violations_SP500 = 0
num_violations_TBONDS = 0

# Now to iterate over the backtesting period:
for i in range(len(returns_percent) - backtesting_period, len(returns_percent)):
     
    actual_return_BTC = returns_percent['BTC_Log_Return'].iloc[i]
    actual_return_SP500 = returns_percent['SP500_Log_Return'].iloc[i]
    actual_return_TBONDS = returns_percent['TMUBMUSD01Y_Log_Return'].iloc[i]

    
    # Checking if actual loss exceeds VaR for each asset
    if actual_return_BTC < VaR_BTC:
        num_violations_BTC += 1
    if actual_return_SP500 < VaR_SP500:
        num_violations_SP500 += 1
    if actual_return_TBONDS < VaR_TBONDS:
        num_violations_TBONDS += 1


# Now we compare the results with the expected # of valuations:

# In[296]:


# The expected number of violations based on the confidence level
expected_violations_BTC = backtesting_period * (1 - 0.95)
expected_violations_SP500 = backtesting_period * (1 - 0.95)
expected_violations_TBONDS = backtesting_period * (1 - 0.95)

# Finally, we compare actual vs expected
print(f"Actual violations for BTC: {num_violations_BTC}")
print(f"Expected violations for BTC: {expected_violations_BTC}")

print(f"Actual violations for SP500: {num_violations_SP500}")
print(f"Expected violations for SP500: {expected_violations_SP500}")

print(f"Actual violations for T-bonds: {num_violations_TBONDS}")
print(f"Expected violations for T-bonds: {expected_violations_TBONDS}")


# Based on the backtesting and binomial testing results, we observe that the number of actual violations are for each security is slightly lower than expected ones. This means that the model is relatively close to expected performance but slightly underestimates the risk. For bonds, however, the differences in violations is pretty high, indicating that the Parametric VaR is overestiating risk for T-bonds.

# In[297]:


# We store data for plotting
actual_returns_BTC = []
actual_returns_SP500 = []
actual_returns_TBONDS = []
VaR_violations_BTC = []
VaR_violations_SP500 = []
VaR_violations_TBONDS = []


for i in range(len(returns_percent) - backtesting_period, len(returns_percent)):
    
    actual_return_BTC = returns_percent['BTC_Log_Return'].iloc[i]
    actual_return_SP500 = returns_percent['SP500_Log_Return'].iloc[i]
    actual_return_TBONDS = returns_percent['TMUBMUSD01Y_Log_Return'].iloc[i]

    # Append actual returns to lists
    actual_returns_BTC.append(actual_return_BTC)
    actual_returns_SP500.append(actual_return_SP500)
    actual_returns_TBONDS.append(actual_return_TBONDS)

    # Check if actual loss exceeds VaR for each asset and store violations
    if actual_return_BTC < VaR_BTC:
        VaR_violations_BTC.append(actual_return_BTC)
    else:
        VaR_violations_BTC.append(None)  # No violation

    if actual_return_SP500 < VaR_SP500:
        VaR_violations_SP500.append(actual_return_SP500)
    else:
        VaR_violations_SP500.append(None)  # No violation

    if actual_return_TBONDS < VaR_TBONDS:
        VaR_violations_TBONDS.append(actual_return_TBONDS)
    else:
        VaR_violations_TBONDS.append(None)  # No violation

# setting date range for the x-axis
date_range = returns_percent.index[-backtesting_period:]

# finally, the plot looks as follows:
def plot_VaR(actual_returns, VaR_violations, VaR_value, title):
    plt.figure(figsize=(14, 7))
    plt.plot(date_range, actual_returns, label='Actual Returns')
    plt.axhline(y=VaR_value, color='red', linestyle='--', label='VaR Threshold')
    plt.scatter(date_range, VaR_violations, color='red', label='VaR Violations', zorder=5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.show()


plot_VaR(actual_returns_BTC, VaR_violations_BTC, VaR_BTC, 'BTC Actual Returns vs VaR Violations')
plot_VaR(actual_returns_SP500, VaR_violations_SP500, VaR_SP500, 'SP500 Actual Returns vs VaR Violations')
plot_VaR(actual_returns_TBONDS, VaR_violations_TBONDS, VaR_TBONDS, 'T-bonds Actual Returns vs VaR Violations')


# In[298]:


def calculate_conditional_var(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) of a return series.
    
    :param returns: pd.Series or np.array, the return series of the asset
    :param confidence_level: float, the confidence level for VaR, default is 0.95
    :return: float, the CVaR value
    """
    # Drop any NaN values in the return series
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("Return series is empty after removing NaN values.")
    
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var_threshold].mean()
    return cvar

# Assuming 'returns' is the return series
confidence_level = 0.95

# Calculate Conditional VaR for Bitcoin, S&P 500, and Treasury Bonds
try:
    cvar_btc = calculate_conditional_var(df_assets['BTC_Log_Return'], confidence_level)
    print(f"Conditional Value at Risk (CVaR) at {confidence_level*100}% confidence level for BTC: {cvar_btc}")
except ValueError as e:
    print(f"Error calculating CVaR for BTC: {e}")

try:
    cvar_sp500 = calculate_conditional_var(df_assets['SP500_Log_Return'], confidence_level)
    print(f"Conditional Value at Risk (CVaR) at {confidence_level*100}% confidence level for S&P 500: {cvar_sp500}")
except ValueError as e:
    print(f"Error calculating CVaR for S&P 500: {e}")

try:
    cvar_tbonds = calculate_conditional_var(df_assets['TMUBMUSD01Y_Log_Return'], confidence_level)
    print(f"Conditional Value at Risk (CVaR) at {confidence_level*100}% confidence level for Treasury Bonds: {cvar_tbonds}")
except ValueError as e:
    print(f"Error calculating CVaR for Treasury Bonds: {e}")

# Define weights if not defined
weights = np.array([0.33, 0.33, 0.33])

# For the theoretical portfolio
try:
    portfolio_returns = df_log_returns.dot(weights)
    cvar_portfolio = calculate_conditional_var(portfolio_returns, confidence_level)
    print(f"Conditional Value at Risk (CVaR) at {confidence_level*100}% confidence level for the portfolio: {cvar_portfolio}")
except ValueError as e:
    print(f"Error calculating CVaR for the portfolio: {e}")


# # MonteCarlo VaR

# ## 1. Parameters:

# - **Assets**: S&P 500, Bitcoin, 1-Year Treasury Bond
# - **Number of simulations**: 10000
# - **Number of days**: 10
# - **Alpha**: 5%
# - **Confidence Level**: \(1 - $\alpha$ = 95\%\)

# In[299]:


assets = ['SP500', 'BTC', 'TMUBMUSD01Y']
n_simulations = 10000
n_days = 10
confidence_level = 0.95
feature_weight = 0.25  # Weight for feature adjustments


# ## 2. Function to generate future scenarios:

# - **last_price:** last known price of each asset.
# - **last_features**: last row of features.
# - **simulated_paths**: it's basically a matrix that stores simulated price paths.

# In[300]:


def simulate_price_paths(scaled_df, asset, n_simulations, n_days, feature_weight):
    last_price = df_assets[f'{asset}_Price'].iloc[-1]
    last_features = scaled_df.iloc[-1].copy()
    simulated_paths = np.zeros((n_days, n_simulations))

    for i in range(n_simulations):
        simulated_returns = []
        for j in range(n_days):
            base_return = np.random.choice(df_assets[f'{asset}_Log_Return'].dropna())
            feature_adjustment = feature_weight * (last_features[f'{asset}_Lagged_Return'] +
                                                   last_features[f'{asset}_Rolling_Mean'] +
                                                   last_features[f'{asset}_EMA'])
            adjusted_return = base_return + feature_adjustment
            simulated_returns.append(adjusted_return)
            last_features[f'{asset}_Log_Return'] = adjusted_return
            last_features[f'{asset}_Lagged_Return'] = adjusted_return
            last_features[f'{asset}_Rolling_Mean'] = np.mean(simulated_returns)
            last_features[f'{asset}_Rolling_Std'] = np.std(simulated_returns)
            last_features[f'{asset}_EMA'] = last_features[f'{asset}_EMA'] * 0.95 + adjusted_return * 0.5

        simulated_paths[:, i] = last_price * np.exp(np.cumsum(simulated_returns))
    return simulated_paths


# #### Outer Loop:
# The loop **`for i in range(n_simulations)`** runs a specified number of simulations (10000) to predict the future prices of an asset over a given number of days. For each simulation:
# - **`simulated_returns`** is initialized as an empty list to store the simulated log returns for each day within the current simulation.
# 
# #### Inner Loop:
# The loop **`for j in range(n_days)`** runs a specified number of simulations (10000) to predict the future prices of an asset over a given number of days (10);
# 
# #### Base return:
# We are selecting any historical log return randomly for our simulation:
# 
# **`np.random.choice(df_assets[f'{asset}_Log_Return'].dropna())`**
# 
# #### Feature Adjustment:
# **`feature_adjustment = feature_weight * ( last_features[f'{asset}_Lagged_Return'] +     last_features[f'{asset}_Rolling_Mean'] + last_features[f'{asset}_EMA'])`**
#  
#  **`adjusted_return = base_return + feature_adjustment`**
# 
# Our return will be adjusted by 25% * Features (calculated in feature engineering section)
#             
# #### This section of the code updates the simulation's running list of returns and dynamically adjusts feature values based on these returns:
# 
# `simulated_returns.append(adjusted_return)`
# 
# `last_features[f'{asset}_Log_Return'] = adjusted_return
# last_features[f'{asset}_Lagged_Return'] = adjusted_return
# last_features[f'{asset}_Rolling_Mean'] = np.mean(simulated_returns)
# last_features[f'{asset}_Rolling_Std'] = np.std(simulated_returns)`
# 
# To give more weight to the recent returns:
# 
# `last_features[f'{asset}_EMA'] = last_features[f'{asset}_EMA'] * 0.95 + adjusted_return * 0.5`
# 
# #### Cumulative returns:
# `simulated_paths[:, i] = last_price * np.exp(np.cumsum(simulated_returns))`

# ## 3. Function to calculate VaR:

# The `calculate_VaR` function calculates the VaR for each asset using simulated price paths. 
# It computes returns from the simulations and then finds the value at 95th percentile, representing the maximum expected loss at a given confidence level. This VaR value corresponds to the worst-case loss that is not expected to be exceeded with the chosen confidence level.

# In[301]:


def calculate_VaR(simulated_paths, confidence_level):
    simulated_returns = simulated_paths[-1, :] / simulated_paths[0, :] - 1
    VaR = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return VaR


# #### Store Results:
# `VaR_results = {}`
# 
# #### Loop Over Assets:
# - Iterates through each asset in the assets list ('SP500', 'BTC', 'TMUBMUSD01Y') to perform the simulation and VaR calculation individually for each one.
# 
# `for asset in assets:`
# 
# #### Simulate Price Paths:
# - Calls the simulate_price_paths function, which generates future price paths for the asset based on historical log returns and feature adjustments. 
# 
# `simulated_paths = simulate_price_paths(df_assets, asset, n_simulations=10000, n_days=7, feature_weight=0.05)`
# 
# #### Calculate VaR:
# 
# - Calls the calculate_VaR function to compute the VaR for the asset from the simulated price paths.
# 
# `VaR = calculate_VaR(simulated_paths, confidence_level=0.95)`
# 
# #### Store VaR
# 
# `VaR_results[asset] = VaR`

# In[302]:


VaR_results = {}

for asset in assets:
    simulated_paths = simulate_price_paths(df_assets, asset, n_simulations=10000, n_days=10, feature_weight=0.05)
    VaR = calculate_VaR(simulated_paths, confidence_level=0.95)
    VaR_results[asset] = VaR
    print(f"{asset}_Log_Return VaR at 95% confidence level: {VaR:.2%}")


# These results tell us the maximum expected losses for each asset over the specified period, with 95% confidence. The S&P 500 is expected to have the smallest potential loss, while the 1-Year U.S. Treasury Bond has the highest potential loss, according to the simulations. Although in general, the 1Y treasury bond is considered to be the safest asset of the three, we are taking in account a period of extreme market conditions that will influence this results. 

# ## 4. Backtesting our results

# The backtest_VaR function identifies and returns the moments where the assets returns fall below our VaR threshold, indicating violations.
# 
# `def backtest_VaR(series, VaR):`
# 
# `violations = series[series < VaR]`
#      
# `return violations`
# 
# This code loops through each asset, retrieves its VaR, and finds moments where actual returns fall below VaR. It then plots these returns, the VaR line, and violations, and prints the number of violations. 
# This helps us visually understand how often the VaR threshold was breached:
# 
# `for asset in assets:`
# 
# `VaR = VaR_results[asset]`
# 
# `violations = backtest_VaR(df_assets[f'{asset}_Log_Return'], VaR)`
# 
# `plot_backtest(df_assets, asset, VaR, violations)`
# 
# `print(f"Number of violations for {asset}: {len(violations)}\n")`
# 
# 

# In[303]:


def backtest_VaR(series, VaR):
    violations = series[series < VaR]
    return violations

def plot_backtest(df_assets, asset, VaR, violations):
    plt.figure(figsize=(10, 5))
    plt.plot(df_assets.index, df_assets[f'{asset}_Log_Return'], label=f'{asset}_Log_Return')
    plt.axhline(y=VaR, color='r', linestyle='--', label=f'VaR (95%)')
    plt.scatter(violations.index, violations, color='red', label='Violations')
    plt.title(f'Backtesting VaR for {asset}')
    plt.legend()
    plt.show()

for asset in assets:
    VaR = VaR_results[asset]
    violations = backtest_VaR(df_assets[f'{asset}_Log_Return'], VaR)
    plot_backtest(df_assets, asset, VaR, violations)
    print(f"Number of violations for {asset}: {len(violations)}\n")


# **S&P 500:**
# 
# **Expected Losses:** = (5% * 2520)/10 = +/- 12.5
# 
# **Number of Violations:** 13
# 
# - **What Happened:** There were 13 times when the S&P 500's actual returns dropped below our VaR threshold.
# 
# - **What It Means:** Considering we looked at 10 years of data, 13 violations show that our VaR model is quite accurate. We expected about 12.5 days (5% of the time) on 10 days horizon to go below the threshold, so having just 13, means our model is pretty reliable.
# 
# **Bitcoin:**
# 
# **Number of Violations:** 12
# 
# - **What Happened:** Bitcoin had 12 times when its returns were worse than what our VaR model predicted.
# 
# - **What It Means:** Given Bitcoin's wild price swings, 12 violations over 10 years indicate that our model handles Bitcoin's volatility pretty well. It's slightly conservative but does a good job in predicting extreme losses.
# 
# **1-Year U.S. Treasury Bond:**
# 
# **Number of Violations:** 14
# 
# - **What Happened:** For the Treasury Bond, there were 14 moments where actual returns fell below our VaR threshold, with most drops during stressful times like the COVID-19 pandemic.
# 
# - **What It Means:** The results are pretty similar to the other 2 assets, slightly more, as the volatility in the bond, apart from COVID pandemic, is pretty stable.

# # Conditional VAR

# ## 1. Conditional VAR for 1 day only

# In[304]:


# ChatGPT used
def calculate_cvar(returns, confidence_level=0.95):
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var_threshold].mean()
    return cvar

df_assets = df_assets.dropna(subset=['SP500_Log_Return', 'BTC_Log_Return', 'TMUBMUSD01Y_Log_Return'])

# Calculate 1-day CVaR for each asset using log returns
cvar_1day_sp500_log = calculate_cvar(df_assets['SP500_Log_Return'], confidence_level=0.95)
cvar_1day_btc_log = calculate_cvar(df_assets['BTC_Log_Return'], confidence_level=0.95)
cvar_1day_tmub_log = calculate_cvar(df_assets['TMUBMUSD01Y_Log_Return'], confidence_level=0.95)

print("1-day 95% CVaR for SP500 (Log Returns): ", cvar_1day_sp500_log)
print("1-day 95% CVaR for BTC (Log Returns): ", cvar_1day_btc_log)
print("1-day 95% CVaR for TMUB (Log Returns): ", cvar_1day_tmub_log)


# SP500 CVaR: the average loss is approximately 2.67%. BTC CVaR: the average loss is about 8.61%. TMUB CVaR: the average loss is about 13.26%.

# ## 2. Conditional VAR for 10 days

# In[305]:


# Calculate 10-day log returns, making sure to drop NaN values after the rolling sum
df_assets['SP500_Log_Return_10day'] = df_assets['SP500_Log_Return'].rolling(window=10).sum()
df_assets['BTC_Log_Return_10day'] = df_assets['BTC_Log_Return'].rolling(window=10).sum()
df_assets['TMUBMUSD01Y_Log_Return_10day'] = df_assets['TMUBMUSD01Y_Log_Return'].rolling(window=10).sum()

# Drop NaN values after calculating the rolling sums
df_assets = df_assets.dropna(subset=['SP500_Log_Return_10day', 'BTC_Log_Return_10day', 'TMUBMUSD01Y_Log_Return_10day'])

# Calculate CVaR (Conditional Value at Risk)
def calculate_cvar(returns, confidence_level=0.95):
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var_threshold].mean()
    return cvar

# Calculate 10-day CVaR for each asset using log returns
cvar_10day_sp500_log = calculate_cvar(df_assets['SP500_Log_Return_10day'], confidence_level=0.95)
cvar_10day_btc_log = calculate_cvar(df_assets['BTC_Log_Return_10day'], confidence_level=0.95)
cvar_10day_tmub_log = calculate_cvar(df_assets['TMUBMUSD01Y_Log_Return_10day'], confidence_level=0.95)

print("10-day 95% CVaR for SP500 (Log Returns): ", cvar_10day_sp500_log)
print("10-day 95% CVaR for BTC (Log Returns): ", cvar_10day_btc_log)
print("10-day 95% CVaR for TMUB (Log Returns): ", cvar_10day_tmub_log)


# SP500 CVaR: the average loss is approximately 6.92%. BTC CVaR: the average loss is about 25.93%. TMUB CVaR: the average loss is about 35.84%.

# # 3. Backtesting CVAR results

# In[307]:


#ChatGPT used

# Function to backtest CVaR
def backtest_CVaR(series, CVaR):
    violations = series[series < CVaR]
    return violations

# Function to plot backtest results
def plot_backtest(df_assets, asset, CVaR, violations):
    plt.figure(figsize=(10, 5))
    plt.plot(df_assets.index, df_assets[f'{asset}_Log_Return_10day'], label=f'{asset} 10-day Log Return')
    plt.axhline(y=CVaR, color='r', linestyle='--', label=f'10-day CVaR (95%)')
    plt.scatter(violations.index, violations, color='red', label='Violations')
    plt.title(f'Backtesting 10-day CVaR for {asset}')
    plt.legend()
    plt.show()

# 10-day CVaR results from log returns calculation
CVaR_log_results = {
    'SP500': cvar_10day_sp500_log,
    'BTC': cvar_10day_btc_log,
    'TMUBMUSD01Y': cvar_10day_tmub_log
}

# List of assets to backtest
assets = ['SP500', 'BTC', 'TMUBMUSD01Y']

# Backtest CVaR for each asset and plot the results
for asset in assets:
    CVaR = CVaR_log_results[asset]
    violations = backtest_CVaR(df_assets[f'{asset}_Log_Return_10day'], CVaR)
    plot_backtest(df_assets, asset, CVaR, violations)
    print(f"Number of violations for {asset}: {len(violations)}\n")


# # Models Comparison:

# In[5]:


data = {
    "Asset": ["S&P 500", "Bitcoin", "1-Year U.S. Treasury Bond"],
    "Historical VaR (10-day)": ["-7.26%", "-15.78%", "-25.19%"],
    "Parametric VaR (10-day)": ["-4.93%", "-19.79%", "-52.44%"],
    "Monte Carlo VaR (10-day)": ["-4.01%", "-14.21%", "-22.29%"],
    "Conditional VaR (10-day)": ["-2.37%", "-9.09%", "-13.79%"]
}

df_var = pd.DataFrame(data)
df_var


# ## S&P 500

# For the S&P 500, the Historical VaR shows a 7.26% potential loss, which makes sense since it captures big market drops we've seen in the past. The Parametric and Monte Carlo VaR methods show lower risks at 4.93% and 4.01%, reflecting a more stable market. The Conditional VaR is the lowest at 2.37%, showing that, on average, the worst losses aren't as bad as the absolute worst-case scenario.

# ## Bitcoin

# For Bitcoin, the Historical VaR shows a 15.78% potential loss, highlighting its high volatility. The Parametric VaR shows the highest risk at 19.79%, which makes sense given Bitcoin's volatility. The Monte Carlo and Conditional VaR methods show lower risks at 14.21% and 9.09%, giving a more balanced view of potential losses.

# ## 1-Year U.S. Treasury Bond

# For the 1-Year U.S. Treasury Bond, the Historical VaR is high at 25.19%, which reflects its sensitivity to extreme market conditions like the COVID-19 pandemic. The Parametric VaR is surprisingly high at 52.44%, likely overestimating the risk due to those extreme events. The Monte Carlo and Conditional VaR results at 22.29% and 13.79% are more reasonable and provide a more realistic view of potential losses.

# # Summary

# Overall, the results make sense. For the S&P 500, they show how it generally performs steadily but can drop significantly during market crashes. For Bitcoin, they reflect its extreme volatility and potential for big losses. For the Treasury Bond, while the Parametric VaR seems to overestimate the risk, the other methods give more realistic estimates. 
