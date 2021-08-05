#!/usr/bin/env python
# coding: utf-8

# packages
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import io
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

## Log file
timestamp = datetime.today().strftime('%Y%m%d_%H%M')
logfile = f'LEI_mapping_log_{timestamp}.log'
logpath = 'C:/Users/Alexis/Documents/PythonProjects/CarbonEmissionsForecast/Logs/'
log = io.StringIO()

## Upload files
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

####################
# Prepare the data #
####################
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

def arima(dataset, AR_lags, diff, MA_lags, train_split_year, forecast_steps):

    # Prediction dates
    pred_dates = pd.date_range(start='1/1/'+str(train_split_year), periods=forecast_steps, freq='A')
    # DataFrames to be filled:
    CO2_emissions_forecast = dataset.iloc[:, 0:2]
    CO2_emissions_forecast["DF_test_p_val"], CO2_emissions_forecast["significant"] = [np.nan, np.nan]
    for t in range(len(pred_dates)-1):
        CO2_emissions_forecast[str(pred_dates.year.values[t])] = np.nan
    
    CO2_emissions_conf_int = dataset.iloc[:, 0:2]
    CO2_emissions_conf_int["Low_int"], CO2_emissions_conf_int["High_int"] = ["Low", "High"]
    #CO2_emissions_conf_int["val"] = np.nan
    CO2_emissions_conf_int = CO2_emissions_conf_int.reset_index()
    CO2_emissions_conf_int = pd.melt(CO2_emissions_conf_int, id_vars=['Country Name', 'Country Code'], value_vars=['Low_int', 'High_int'])
    CO2_emissions_conf_int.pop('value')

    for t in range(len(pred_dates)-1):
        CO2_emissions_conf_int[str(pred_dates.year.values[t])] = np.nan

    for i in range(len(dataset)-1):
        training_data = dataset.iloc[i,dataset.columns.get_loc("1990"):dataset.columns.get_loc(str(train_split_year+1))]
        test_data = dataset.iloc[i,dataset.columns.get_loc(str(train_split_year+1)):]

        # Set the index as a date object and values as float64:
        from pandas.tseries import offsets
        training_data.index = pd.to_datetime(training_data.index, format='%Y') + offsets.YearEnd()
        test_data.index = pd.to_datetime(test_data.index, format='%Y') + offsets.YearEnd()
        training_data = training_data.astype('float64')
        test_data = test_data.astype('float64')

        # ARIMA(AR_lags,diff,MA_lags) model
        # GROWTH RATE OF POPULATION, GDP GROWTH RATE
        arima_model = SARIMAX(training_data.values, order = (AR_lags,diff,MA_lags))

        # Fit the ARIMA model
        arima_results = arima_model.fit()
        # Make ARIMA forecast of next 4 values
        arima_value_forecast = arima_results.get_forecast(steps=forecast_steps).predicted_mean
        confidence_intervals = arima_results.get_forecast(steps=forecast_steps).conf_int()
        low_int, high_int = zip(*confidence_intervals)

