#!/usr/bin/env python
# coding: utf-8

# packages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import zipfile
import io
from io import StringIO
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import matplotlib.pyplot as plt


## Log file
timestamp = datetime.today().strftime('%Y%m%d_%H%M')
logfile = f'LEI_mapping_log_{timestamp}.log'
logpath = 'C:/Users/Alexis/Documents/PythonProjects/CarbonEmissionsForecast/Logs/'
log = io.StringIO()

## Upload file
resp = urlopen("https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=csv")
zipfile = ZipFile(BytesIO(resp.read()))
zipfile.namelist()

## The three csv files are stored in a dictonary
names = {}
for name in zipfile.namelist():
    if name == "API_EN.ATM.CO2E.PC_DS2_en_csv_v2_2708833.csv":
        names[name.split('_')[0]+name.split('_')[1]] = pd.read_csv(zipfile.open(name), skiprows=4)
    else:
        names[name.split('_')[0]+name.split('_')[1]] = pd.read_csv(zipfile.open(name))

# Rename the dataframe containing CO2 time series.
CO2_emissions = names['APIEN.ATM.CO2E.PC']
CO2_emissions.drop(CO2_emissions.columns[len(CO2_emissions.columns)-1], axis=1, inplace=True)

# Excluding years 2019 and 2020 missing most data.
CO2_emissions_1960_2018 = CO2_emissions.iloc[:,:len(CO2_emissions.columns)-2]
CO2_emissions_1960_2018.isnull().sum(axis=0)[4:] # We observe a higher data coverage as of 1990

index_1 = CO2_emissions_1960_2018.columns.get_loc("1960")
index_2 = CO2_emissions_1960_2018.columns.get_loc("1990")
CO2_emissions_1990_2018 = CO2_emissions_1960_2018
CO2_emissions_1990_2018.drop(CO2_emissions_1990_2018.columns[index_1:index_2], axis=1, inplace=True)
# We drop the remaining countries with NaN values.
CO2_emissions_1990_2018 = CO2_emissions_1990_2018.dropna()

plt.plot(CO2_emissions_1990_2018.iloc[1,CO2_emissions_1990_2018.columns.get_loc("1990"):])
plt.show()
training_data = CO2_emissions_1990_2018.iloc[1,CO2_emissions_1990_2018.columns.get_loc("1990"):CO2_emissions_1990_2018.columns.get_loc("2017")]
test_data = CO2_emissions_1990_2018.iloc[1,CO2_emissions_1990_2018.columns.get_loc("2017"):]

# Set the index as a date object and values as float64:
from pandas.tseries import offsets
training_data.index = pd.to_datetime(training_data.index, format='%Y') + offsets.YearEnd()
test_data.index = pd.to_datetime(test_data.index, format='%Y') + offsets.YearEnd()
training_data = training_data.astype('float64')
test_data = test_data.astype('float64')
# If a series is not stationary it is possible to detrend it. Either by
# differenciating or by model fitting.

# Manual differenciating:
# Calculate the difference of the time series
CO2_stationary = training_data.diff().dropna()
## Dicky-Fuller test to test for stationarity:
from statsmodels.tsa.stattools import adfuller
DF_test = adfuller(training_data)
# Print the test statistic and the p-value
print('p-value:', DF_test[1]) # The p-value must be below 0.05 to reject the null hypothesis that the series is non-stationary

# Run ADF test on the differenced time series
DF_test_diff = adfuller(CO2_stationary)
print('p-value:', DF_test_diff[1]) # The p-value must be below 0.05 to reject the null hypothesis that the series is non-stationary

### Create an ARMA model:
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(CO2_stationary.values, order=(2,1,2))

# Fit the model
results = model.fit()
print(results.summary())

# Reconstructing the original time series after differenciating
diff_forecast = results.get_forecast(steps=10).predicted_mean
from numpy import cumsum
mean_forecast = cumsum(diff_forecast) + training_data[-1] # We add the last value of the original time series

# plot the training data
plt.plot(training_data.index, training_data.values, label='observed')

# plot the mean predictions
plt.plot(pd.date_range(start='1/1/2016', periods=10, freq='A'), mean_forecast, color='r', label='forecast')
plt.show()

# ARIMA with diff 1:
# Create ARIMA(2,1,2) model
# GROWTH RATE OF POPULATION, GDP GROWTH RATE
from statsmodels.tsa.statespace.sarimax import SARIMAX
arima_model = SARIMAX(training_data.values, order = (2,1,2))

# Fit ARIMA model
arima_results = arima_model.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean
confidence_intervals = arima_results.get_forecast(steps=10).conf_int()
low_int, high_int = zip(*confidence_intervals)

# It is also possible to predict values from the sample:
arima_results.get_prediction(start=-5)
# Print forecast
print(arima_value_forecast)
## Create an ARMAX model:
model = ARMA(df['productivity'], order=(p,q), exog = df['hours_sleep'])

### Forecasting:
# Generate predictions
one_step_forecast = arima_results.get_prediction(start=-5)
# Dynamic forecast
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean


# Print best estimate  predictions
print(mean_forecast)

# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, 
               upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()


