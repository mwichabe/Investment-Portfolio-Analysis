# Investment Portfolio Analysis

This project aims to analyze the impact of adding Bitcoin to an investment portfolio. It combines two articles related to weight allocation, mean CVaR calculations, and spanning tests. The analysis is performed using Python.

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Analysis](#analysis)
4. [Results](#results)
5. [Dependencies](#dependencies)
6. [Usage](#usage)
7. [Contributing](#contributing)


## Introduction
In this project, we investigate the impact of incorporating Bitcoin into an investment portfolio. We utilize weight allocation techniques, calculate the mean and CVaR for the portfolio, and perform spanning tests to evaluate the significance of the results.

## Data
The analysis uses the "Data_All.csv" dataset, which includes historical prices of various assets such as Bitcoin, Gold, Composite Stock Index, Energy, and others. The dataset should be placed in the same directory as the Python script.

## Analysis
The analysis involves the following steps:
- Reading and preprocessing the dataset
- Implementing the weight allocation for the investment portfolio
- Calculating the mean and CVaR for the portfolios with and without Bitcoin
- Performing spanning tests (HK and FFK) to evaluate the impact of adding Bitcoin
- Visualizing the trade-offs between mean and CVaR for the portfolios

## Results
The analysis yielded the following results:
- HK Spanning Test Result: Not significant for all portfolios
- HK Spanning Test p-value: 1.0 for all portfolios
- FFK Spanning Test Result: Not significant for all portfolios
- FFK Spanning Test p-value: 1.0 for all portfolios

These results indicate that adding Bitcoin did not have a statistically significant impact on the investment portfolios based on the chosen spanning tests.

## Dependencies
The following Python libraries are required to run the analysis:
- pandas
- numpy
- matplotlib
- scipy.stats
- sklearn.linear_model
- sklearn.decomposition

## Usage
1. Ensure that Python and the required libraries are installed.
2. Place the "Data_All.csv" dataset in the same directory as the Python script.
3. Run the Python script "main.py" to perform the analysis.
4. View the results and visualizations in the console and the generated plots.

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


