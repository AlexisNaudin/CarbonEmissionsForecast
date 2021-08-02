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

## Log file
timestamp = datetime.today().strftime('%Y%m%d_%H%M')
logfile = f'LEI_mapping_log_{timestamp}.log'
logpath = 'C:/Users/Alexis/Documents/PythonProjects/CarbonEmissionsForecast/Logs/'
log = io.StringIO()

## Upload file
resp = urlopen("https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=csv")
zipfile = ZipFile(BytesIO(resp.read()))
zipfile.namelist()

names = {}
for name in zipfile.namelist():
    if name == "API_EN.ATM.CO2E.PC_DS2_en_csv_v2_2708833.csv":
        names[name.split('_')[0]+name.split('_')[1]] = pd.read_csv(zipfile.open(name), skiprows=4)
    else:
        names[name.split('_')[0]+name.split('_')[1]] = pd.read_csv(zipfile.open(name))

