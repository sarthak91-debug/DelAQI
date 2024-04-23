import math
import glob
import pandas as pd
import matplotlib.pyplot as plt

import sktime 
import numpy as np
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series

from sktime.regression.dummy import *
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.plotting import plot_series

from darts import TimeSeries
from darts.datasets import WeatherDataset
from darts.models import NBEATSModel, NaiveDrift, NaiveMean

from darts.metrics import mape


from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error


from sarthak_eda import resample

DATA_NEW="../data/delhi_aqi_new.csv" 


def sktimemodel(train_size:int=0.8):
    """df=pd.read_csv(DATA_NEW)

    df['date']=pd.to_datetime(arg=df['date'],format="%Y-%m-%d %H:%M:%S")
    df.set_index('date',inplace=True)
    
    df=df.resample('D').agg({'pm10':np.mean,'nh3':np.mean,
                                  'pm2_5':np.mean,'co':np.mean,'no':np.mean,
                                  'no2':np.mean,'o3':np.mean,'so2':np.mean})
    """

    _,df=resample("D")

    df.fillna(df.mean(),inplace=True)

    #print(df.isnull().sum())
    df.set_index('date',inplace=True)
    count_df=round((df.shape[0]*train_size)+3)
    train_df=df.iloc[:count_df,:]
    test_df=df.iloc[count_df:,:]    

    # Naive Regressor model
    naive_reg=NaiveForecaster(strategy='mean')
    naive_reg.fit(train_df['pm2_5'])

    y_pred =naive_reg.predict(fh=np.arange(1,157))

    print(y_pred)


def modelDartsNbeats(sample_freq:str,train_size:int=0.8):

    _,df=resample(sample_freq)

    df.fillna(df.mean(),inplace=True)

    df.set_index('date',inplace=True)
    count_df=round((df.shape[0]*train_size)+3)
    train_df=df.iloc[:count_df,:]
    test_df=df.iloc[count_df:,:]


    if sample_freq=='H':
        series= TimeSeries.from_series(df['pm2_5'],fill_missing_dates=True,freq='H')
    else:
        series= TimeSeries.from_series(df['pm2_5'])

    series_train=series[:count_df]
    series_test=series[count_df:]

    if 'nbeats_model.pt' in glob.glob('*.pt'):
        model=NBEATSModel.load(path='nbeats_model.pt')
    else:
        model=NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=25,
            activation='ReLU')

        model.fit(series_train)
        model.save("nbeats_model.pt")

    pred=model.predict(len(series_test))

    #actual=np.array([[x] for x in test_df['pm2_5'].values])
    actual=series_test.values()

    

    series.plot()
    pred.plot(label='prediction')
    plt.show()
    return actual,pred,'nbeats'


def modelDartsNaive(sample_freq:str,train_size:int=0.8):

    _,df=resample(sample_freq)

    df.fillna(df.mean(),inplace=True)

    df.set_index('date',inplace=True)
    count_df=round((df.shape[0]*train_size)+3)
    train_df=df.iloc[:count_df,:]
    test_df=df.iloc[count_df:,:]


    if sample_freq=='H':
        series= TimeSeries.from_series(df['pm2_5'],fill_missing_dates=True,freq='H')
    else:
        series= TimeSeries.from_series(df['pm2_5'])

    series_train=series[:count_df]
    print(len(series))
    series_test=series[count_df:]

    if 'naive_model.pkl' in glob.glob('*.pkl'):
        model=NaiveDrift.load(path='naive_model.pkl')
    else:
        model=NaiveDrift()

        model.fit(series_train)
        model.save("naive_model.pkl")

    pred=model.predict(len(series_test))

    #actual=np.array([[x] for x in test_df['pm2_5'].values])
    actual=series_test.values()

    

    series.plot()
    pred.plot(label='prediction')
    plt.show()
    return actual,pred,'naive'


def modelDartsNaiveMean(sample_freq:str,train_size:int=0.8):

    _,df=resample(sample_freq)

    df.fillna(df.mean(),inplace=True)

    df.set_index('date',inplace=True)
    count_df=round((df.shape[0]*train_size)+3)
    train_df=df.iloc[:count_df,:]
    test_df=df.iloc[count_df:,:]


    if sample_freq=='H':
        series= TimeSeries.from_series(df['pm2_5'],fill_missing_dates=True,freq='H')
    else:
        series= TimeSeries.from_series(df['pm2_5'])

    series_train=series[:count_df]
    series_test=series[count_df:]

    if 'naivemean_model.pkl' in glob.glob('*.pkl'):
        model=NaiveMean.load(path='naivemean_model.pkl')
    else:
        model=NaiveMean()

        model.fit(series_train)
        model.save("naivemean_model.pkl")

    pred=model.predict(len(series_test))

    #actual=np.array([[x] for x in test_df['pm2_5'].values])
    actual=series_test.values()

    

    series.plot()
    pred.plot(label='prediction')
    plt.show()
    return actual,pred,'naive_mean'



def metrics(modelFunc):

    m={}
    
    actual,pred,mname=modelFunc
    model=[mname]
    error={'mape':mean_absolute_percentage_error,
           'rmse':mean_squared_error}
    #print("Mean absolute percentage error is {:.2f}%."\
    #      .format(mean_absolute_percentage_error(actual, pred.values())))
    
    
    for mod in model:
        e_dict={}
        for i,e in enumerate(error.items()):

            if e[0]=='rmse':
                e_dict[e[0]]=math.sqrt(e[1](actual, pred.values()))
            else:
                e_dict[e[0]]=e[1](actual, pred.values())*100

        m[mod]=e_dict

    df=pd.DataFrame(m)
    print(df)



    


if __name__=="__main__":
    metrics(modelDartsNaive('D'))
