#!/usr/bin/env python
# coding: utf-8

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


# # Theoretical portfolio VaR with Variance-Covariance matrix

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

# In[299]:


assets = ['SP500', 'BTC', 'TMUBMUSD01Y']
n_simulations = 10000
n_days = 10
confidence_level = 0.95
feature_weight = 0.25  # Weight for feature adjustments


# ## 2. Function to generate future scenarios:

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


# ## 3. Function to calculate VaR:

# In[301]:


def calculate_VaR(simulated_paths, confidence_level):
    simulated_returns = simulated_paths[-1, :] / simulated_paths[0, :] - 1
    VaR = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return VaR


# In[302]:


VaR_results = {}

for asset in assets:
    simulated_paths = simulate_price_paths(df_assets, asset, n_simulations=10000, n_days=10, feature_weight=0.05)
    VaR = calculate_VaR(simulated_paths, confidence_level=0.95)
    VaR_results[asset] = VaR
    print(f"{asset}_Log_Return VaR at 95% confidence level: {VaR:.2%}")


# ## 4. Backtesting our results

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

