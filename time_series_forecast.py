# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:15:38 2021

@author: Yasmine
"""
import pandas as pd
import datetime
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
from functions import resample_missing_weeks,fit_sarimax_model,get_error_metrics,compute_forecasts
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
import itertools
pdq = list(itertools.product(p, d, q))

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y %W %w')
data= pd.read_csv('data.csv',sep=";", parse_dates=['data_type'], index_col='data_type',date_parser=dateparse)
series=data['avg']
# Compute the periods of a season
nobs=len(series)
if (nobs//52 !=0):
    season=12
else:
    season=4
        # Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], season) for x in list(itertools.product(p, d, q))]
# Resample missing weeks
resample_missing_weeks(series,nobs)


minimum_aic, param,param_seasonal, seasonal_order=fit_sarimax_model(series,pdq,seasonal_pdq)
mod = sm.tsa.statespace.SARIMAX(series,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
try:
    results = mod.fit()
    print(results.summary())
    p_value_array=results.test_serial_correlation(method='ljungbox',lags=None)[0][1]
    if all(i >= 0.05 for i in p_value_array):
        results.plot_diagnostics(figsize=(15, 12))
        plt.show()

        split_point=np.floor((1-0.2)*len(series)).astype(int)
        split_date=series.index[split_point]
                
        pred = results.get_prediction(start=pd.to_datetime(split_date), dynamic=False)
        pred_ci = pred.conf_int()
        '''''''''''''''''''''''''''''''''' Compute error measures ''''''''''''''''''''''''''' 
        test=series.where(series.index >=split_date)
        test.dropna(inplace=True)
        
        testing_score, rmse, mae, evs, msle=get_error_metrics(test,pred)  
        
        ax = series['2016':].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='Test Forecast', alpha=.7)
                
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
                
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        plt.show()
               
        ''''''''''''''''''''''''''''' Forecasting a Time Series ''''''''''''''' 
        date_calcul = datetime.datetime(2018, 8, 19)
        forecast=compute_forecasts(series,date_calcul,results)
        ax = series['2016':].plot(label='observed')
        forecast.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        

except ValueError:
    print(ValueError)
    
  
    
    
