# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:27:07 2021

@author: Yasmine
"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
         
def resample_missing_weeks(series,nobs):
    if len(series.resample('W'))!=nobs:
            series=series.resample('W').mean()
            series=series.replace(to_replace=0,value=np.nan)
            for i in range(1,len(series)):
                if (math.isnan(series[i])):
                    series[i]=series[i-1]
                    
def fit_sarimax_model(series,pdq,seasonal_pdq):    
    minimun_aic=np.Infinity
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(series,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=False)
        
                results = mod.fit()
                if results.aic<minimun_aic:
                    minimum_aic=results.aic
                    #order=param
                    seasonal_order=param_seasonal
            
                print('ARIMA{}x{} season - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    return minimum_aic, param,param_seasonal, seasonal_order

def get_error_metrics(test,pred):
    testing_score=r2_score(test,pred.predicted_mean)
                
    rmse = np.sqrt(mean_squared_error(test, pred.predicted_mean))
            
    mae=mean_absolute_error( test, pred.predicted_mean)
            
    evs=explained_variance_score(test, pred.predicted_mean)
            
    msle=mean_squared_log_error(test, pred.predicted_mean)
    
    return testing_score, rmse, mae, evs, msle

def compute_forecasts(s,computation_date,model_fit_result):
    computing_week=computation_date.isocalendar()[1]-1
    last_year_in_series=s.last('1W').index[0].year
    last_week_in_series=s.last('1W').index[0].isocalendar()[1]
    if (last_year_in_series==computation_date.year)&(last_week_in_series==computing_week):
        all_forecast=model_fit_result.forecast(steps=14)
    else:
        total_nb_weeks=52*(computation_date.year-last_year_in_series)-last_week_in_series+computing_week+14
        all_forecast=model_fit_result.forecast(steps=total_nb_weeks)
    total_forecast_weeks=[all_forecast.index[i].isocalendar()[1] for i in range(len(all_forecast))]
    f=all_forecast[pd.Index(total_forecast_weeks).get_loc(computing_week+1)::]
    return f 
