from darts import TimeSeries
from darts.datasets import WeatherDataset
from darts.models import NBEATSModel, NaiveDrift, NaiveMean
from darts.metrics import mape

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle

from sarthak_eda import resample,Polplots

st.title("AQI Predict")
st.header("Predicting pm 2.5 levels of Delhi to keep the people informed and safe")



with st.sidebar:
    options=st.selectbox(
        "Select the time series model",
        ('No Model','NBEATS','Naive Mean','SARIMAX')
    )

    if options=='No Model':
        st.write(f"{options} selected for predictions, Select a model to visualize the results")
        
        
    duration=st.slider(
        'Select the next number of days for which you want to forecast',
        1,100,1
    )


    plot_cols=st.multiselect(
        "Select columns for the correlation plot",
        ('pm2_5','co','pm10','no','no2','o3'),
        default=('pm2_5','co')
    )


    plt_hue=st.selectbox(
        "Select column for hue",
        ('day_part','hour')
    )
    


@st.cache_resource
def load_model(*args):
    option=args[0]

    if option=='Naive Mean':
        if 'naivemean_model.pkl' in glob.glob('*.pkl'):
            model=NaiveMean.load(path='naivemean_model.pkl')



    elif option=='NBEATS':
        if 'nbeats_model.pt' in glob.glob('*.pt'):
            model=NBEATSModel.load(path='nbeats_model.pt')

    elif option=='SARIMAX':
        if "final_sarimax_model.pkl" in glob.glob('*.pkl'):
            with open('final_sarimax_model.pkl','rb') as f:
                model=pickle.load(f,encoding='utf-8')

    else:
        model=None

    return model
        

#creating a correlation plot with two variables and a hue variable

def interactivePlot():
    
    df_pol,df=resample('D')
    print(df_pol)
    if len(plot_cols)<=1 or len(plt_hue)<1:
        st.header('Select two columns and hue column to see the visualization')

    elif len(plot_cols)>2:
        st.title('You selected more than 2 columns')

    else:
        c1,c2=plot_cols
        p=sns.relplot(data=df_pol,x=c1,y=c2,hue=plt_hue)

        st.pyplot(p)





def showPred():
    _,df=resample('D')
    df.fillna(df.mean(),inplace=True)
    df.set_index('date',inplace=True)
    count_df=round((df.shape[0]*0.8)+3)
    train_df=df.iloc[:count_df,:]
    test_df=df.iloc[count_df:,:]

    if options=='No Model':
        fig,axs=plt.subplots()
        axs.plot(train_df['pm2_5'])
        axs.set_ylabel('pm 2.5 level')
        st.pyplot(fig)    
    
    else:
        model=load_model(options)
        pred=model.predict(duration)

        if options=='NBEATS':
            fig,axs=plt.subplots()
            axs.plot(train_df['pm2_5'])
            pred.plot(label='prediction')
            axs.set_ylabel('pm 2.5 level')
            st.pyplot(fig)

        elif options=='Naive Mean':
            fig,axs=plt.subplots()
            axs.plot(train_df['pm2_5'])
            pred.plot(label='prediction')
            axs.set_ylabel('pm 2.5 level')
            st.pyplot(fig)
        
        elif options=='SARIMAX':
            print(test_df)
            pred=model.forecast(steps=30)
            fig,axs=plt.subplots()
            axs.plot(train_df['pm2_5'])
            plt.plot(pred,label='prediction')
            axs.set_ylabel('pm 2.5 level')
            plt.legend()
            st.pyplot(fig)


if __name__=='__main__':

    showPred()

    st.divider()
    
    st.header("Plots showing insights into our data")
    interactivePlot()
    #st.image('../monthly_mean.png')