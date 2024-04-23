import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf,adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FEAT_OFF=0 # Value if you want to calculate features (do not need to change as already caclulatted and uploaded on github)
DATA_NEW="../data/delhi_aqi_new.csv" # change with your path
DATA_OLD="../data/delhi_aqi.csv" # change with your path

"""
Utility Functions
"""

def day_part(hour):
    if hour in [4,5]:
        return "dawn"
    elif hour in [6,7]:
        return "early morning"
    elif hour in [8,9,10]:
        return "late morning"
    elif hour in [11,12,13]:
        return "noon"
    elif hour in [14,15,16]:
        return "afternoon"
    elif hour in [17, 18,19]:
        return "evening"
    elif hour in [20, 21, 22]:
        return "night"
    elif hour in [23,24,1,2,3]:
        return "midnight"
    

"""
Calculate Features
"""
def features():
    df=resample('H')

    for col in df.columns:
        if col!='date':
            df[col+'_diff_mean']=df[col].apply(lambda x:\
                x-np.mean(df[col].values))


    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day
    df['hour']=df['date'].dt.hour

    df['day_of_week']=df['date'].dt.day_of_week
    df['day_of_year']=df['date'].dt.day_of_year

    df['day_part']=df['hour'].apply(day_part)

    # mean of every pollutant
    mpo=df[['pm10','pm2_5','nh3','so2','o3','no2','no','co']].apply(np.mean,axis=0)

    if FEAT_OFF:
        df.to_csv('../data/delhi_aqi_new.csv',index=False)


"""
Resample Time Series, Returns dataframe with only the numeric columns
"""

def resample(sample:str='M'):
    df=pd.read_csv(DATA_NEW)

    df_nums=df.copy()

    df_nums['date']=pd.to_datetime(arg=df_nums['date'],format="%Y-%m-%d %H:%M:%S")
    df_nums.set_index('date',inplace=True)
    if sample=='M':
        df_nums=df_nums.resample('M').agg({'pm10':np.mean,'nh3':np.mean,
                                  'pm2_5':np.mean,'co':np.mean,'no':np.mean,
                                  'no2':np.mean,'o3':np.mean,'so2':np.mean})
        
    elif sample=='D':
        df_nums=df_nums.resample('D').agg({'pm10':np.mean,'nh3':np.mean,
                                  'pm2_5':np.mean,'co':np.mean,'no':np.mean,
                                  'no2':np.mean,'o3':np.mean,'so2':np.mean})

    else:
        None
    

    # In case multiple values for the same date
    df_nums=df_nums.groupby(by='date').agg({'pm10':np.mean,'nh3':np.mean,
                                  'pm2_5':np.mean,'co':np.mean,'no':np.mean,
                                  'no2':np.mean,'o3':np.mean,'so2':np.mean}).reset_index()

    
    return df,df_nums


"""
Plots for Pollution 
"""

def Polplots(sample:str,dtformat:str='manual'):
    
    df,df_nums=resample(sample)

    # Line Plot for pm10 and pm2_5

    if sample!='H':
        tit='Monthly mean pm10 and pm 2.5 pollutant levels'

    else:tit='Hourly mean pm10 pollutant levels'

    ax=plt.subplot(1,1,1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_title(tit)
    ax.set_ylabel('Pollutant Values')

    ax.set_xlabel('Date')


    ax.grid(True)
    ax.plot(df_nums['date'].values,df_nums['pm10'].values,'-b')
    ax.plot(df_nums['date'].values,df_nums['pm2_5'].values,'-r')
    ax.legend(['pm10','pm 2.5'])


    if dtformat!='manual':
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))

    ax.tick_params(axis='x',direction='in',
                   rotation=40,labelsize='small')

    plt.show()

    # violin plots of pollutants levels based on day of the week

    plt.figure(figsize=(12,7))
    sns.violinplot(data=df, x="pm10", y="day_part",orient='h',hue='day_part')
    plt.title("Distribution of pm10 pollutants based on part of day")
    plt.xticks(rotation=20)
    plt.show()

    sns.stripplot(data=df, x="co", y="day_part",orient='h',hue='day_part')
    plt.title("CO pollutants levels based on part of day")
    plt.xticks(rotation=20)
    plt.show()


def decompose(toplot:bool):
    _,df=resample('D')
    df.fillna(df.mean(),inplace=True)

    if toplot:
        mdec=seasonal_decompose(df['pm2_5'],model='multiplicative',period=12)
        adec=seasonal_decompose(df['pm2_5'],model='additive',period=12)
        
        mdec.plot()
        plt.suptitle('Mul')
        adec.plot()
        plt.suptitle('Add')
        plt.show()

        # Draw Plot

        # The acf and pacf plots suggest that there might be a trend and seasonal component involved in our data
        fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
        plot_acf(df['pm2_5'].tolist(), lags=10, ax=axes[0])
        plot_pacf(df['pm2_5'].tolist(), lags=10, ax=axes[1])
        plt.show()

    adf=adfuller(x=df['pm2_5'])
    print(adf)

    if adf[1]<0.05:
        print("We reject the null hypothesis and say the series is stationary")

if __name__=="__main__":
    Polplots('M')
    #decompose(False)
    #decompose(True)
   #_,df_nums=resample('D')
   #print(df_nums.dtypes)
    