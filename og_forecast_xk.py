# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:55:23 2017

@author: Xianhui
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import operator

#%%
# https://www.quandl.com/data/BKRHUGHES-Baker-Hughes-Investor-Relations?page=4
import quandl
quandl_code = ["COM/WLD_CRUDE_WTI", 'COM/PNGASUS_USD']
quandl.get(quandl_code, authtoken="5bf_EsY4XUaFM4jsegu2").plot()
quandl_code_rig = ['BKRHUGHES/RIGS_BY_STATE_TOTALUS_LAND', 
                   'BKRHUGHES/RIGS_BY_STATE_TEXAS_LAND']
quandl.get(quandl_code_rig, authtoken="5bf_EsY4XUaFM4jsegu2").plot()
quandl_code_future = ['EIA/STEO_NYWFUTR_M']
quandl.get(quandl_code_future, authtoken="5bf_EsY4XUaFM4jsegu2").plot()

# https://www.eia.gov/opendata/qb.php?sdid=AEO.2017.REF2017.PRCE_NOMP_TEN_NA_WTI_NA_USA_NDLRPBRL.A
from EIAgov import EIAgov
EIA_API = '996373ca98fc464563a7e2309dd2c951'
eia_future = ['AEO.2017.REF2017.PRCE_NOMP_TEN_NA_WTI_NA_USA_NDLRPBRL.A',
              'AEO.2017.REF2017.PRCE_RLP_TEN_NA_WTI_NA_USA_Y13DLRPBBL.A'] # STEO.NYWSTEO.M
eia = EIAgov(EIA_API, eia_future)
eia.GetData().set_index('Date').sort_index().plot(figsize=(8,3))