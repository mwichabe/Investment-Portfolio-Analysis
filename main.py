import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Reading the dataset
data = pd.read_csv('Data_All.csv')

# Select relevant columns for analysis
selected_columns = ['Bitcoin-USD',
                    'Bitcoin-CNY',
                    'Gold',
                    'Composite Stock Index',
                    'Energy',
                    'Material',
                    'Industrial',
                    'Consumer discretionary',
                    'Consumer Staple',
                    'Health Care',
                    'Financials',
                    'Information Technology',
                    'Telecommunication',
                    'Utilities',
                    'Commodity',
                    'A-Shares',
                    'B-Shares',
                    'C-Bonds',
                    'T-Bonds',
                    ]

selected_data = data[selected_columns]

# Calculate daily returns for assets
returns = selected_data.pct_change().dropna()

# Calculate the mean and CVaR for each portfolio using the weight allocation data
weight_allocation = pd.DataFrame({
    'Assets': selected_columns,
    'Weights': [0.044, 0.042, 0.0364, 0.42, 0.042, 0.054, 0.076, 0.056, 0.0419, 0.290, 0.041,
                0.0441, 0.076, 0.026, 0.082, 0.0442, 0.074, 0.349, 0.090, ]
})

portfolio_returns = (returns * weight_allocation['Weights']).sum(axis=1)
portfolio_mean = portfolio_returns.mean()
portfolio_cvar = portfolio_returns.quantile(0.05)

# Add Bitcoin to the portfolios
bitcoin_weight = 0.05
weight_allocation['Bitcoin Weight'] = bitcoin_weight
weight_allocation['Adjusted Weights'] = (1 - bitcoin_weight) * weight_allocation['Weights'] / (1 - bitcoin_weight)

# Calculate the mean and CVaR for the portfolios with Bitcoin
portfolio_returns_with_bitcoin = (returns * weight_allocation['Adjusted Weights']).sum(axis=1)
portfolio_mean_with_bitcoin = portfolio_returns_with_bitcoin.mean()
portfolio_cvar_with_bitcoin = portfolio_returns_with_bitcoin.quantile(0.05)

# Apply Ridge Regression to estimate the covariance matrix
ridge = Ridge(alpha=0.1)  # Adjust the value of alpha as needed
ridge.fit(returns, returns['Composite Stock Index'])
cov_matrix = ridge.coef_ * np.cov(returns.values.T)

# Use PCA for dimensionality reduction
pca = PCA(n_components=len(selected_columns))
pca.fit(returns)
principal_components = pca.transform(returns)
# Use the transformed returns for analysis
portfolio_returns = pd.DataFrame(principal_components, columns=['PC{}'.format(i) for i in range(len(selected_columns))])
benchmark_returns = portfolio_returns['PC0'].values.flatten()


# Interpreting the results and performing significance tests
def check_significance(statistic, degrees_of_freedom, alpha):
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
    p_value = 1 - chi2.cdf(statistic, degrees_of_freedom)
    result = np.where(statistic > critical_value, 'Significant', 'Not significant')
    return result, p_value


alpha = 0.05


# Perform spanning tests
def hk_spanning_test(portfolio_returns, benchmark_returns):
    portfolio_mean = np.reshape(pd.DataFrame(portfolio_returns).mean().to_numpy(), (-1, 1))
    benchmark_mean = np.reshape(np.array([benchmark_returns.mean()]), (1, 1))
    wald_statistic = ((portfolio_mean - benchmark_mean) ** 2) * len(portfolio_returns)
    return wald_statistic


def ffk_spanning_test(portfolio_returns, benchmark_returns):
    portfolio_var = np.var(portfolio_returns, ddof=1)
    benchmark_var = np.var(benchmark_returns, ddof=1)
    ffk_statistic = ((portfolio_var - benchmark_var) ** 2) * len(portfolio_returns)
    return ffk_statistic


hk_statistic = hk_spanning_test(portfolio_returns, benchmark_returns)
ffk_statistic = ffk_spanning_test(portfolio_returns, benchmark_returns)
hk_statistic_with_bitcoin = hk_spanning_test(portfolio_returns_with_bitcoin, benchmark_returns)
ffk_statistic_with_bitcoin = ffk_spanning_test(portfolio_returns_with_bitcoin, benchmark_returns)
hk_result, hk_p_value = check_significance(hk_statistic_with_bitcoin - hk_statistic, len(portfolio_returns), alpha)
ffk_result, ffk_p_value = check_significance(ffk_statistic_with_bitcoin - ffk_statistic, len(portfolio_returns), alpha)


print('HK Spanning Test Result:', hk_result)
print('HK Spanning Test p-value:', hk_p_value)
print('FFK Spanning Test Result:', ffk_result)
print('FFK Spanning Test p-value:', ffk_p_value)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_cvar, portfolio_mean, label='Portfolios without Bitcoin')
plt.scatter(portfolio_cvar_with_bitcoin, portfolio_mean_with_bitcoin, label='Portfolios with Bitcoin')
plt.xlabel('CVaR')
plt.ylabel('Mean')
plt.title('Mean-CVaR Trade-offs')
plt.legend()
plt.show()
