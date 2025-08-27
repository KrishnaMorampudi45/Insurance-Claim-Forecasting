# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
from pandas.tseries.offsets import DateOffset
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


# Write directly to the app
st.title(" :robot: Insurance Claim Forecaster ")
st.write(
  """The Results may vary from real-time this is just a prediction from previous data
  """
)

# Get the current credentials
cnx=st.connection('snowflake')
session=cnx.session()
months=st.number_input('Enter the number of months to predict',min_value=6)
df=session.sql('select * from CLAIM_DATA')
df=df.to_pandas()
df['DATE']=pd.to_datetime(df['DATE'])
df.set_index('DATE',inplace=True)
df.index = df.index.date  
#st.write(df)
with session.file.get_stream('@DATA_STAGE/sarimax_model.pkl') as stream:
    results=pickle.load(stream)
    
if st.button('Forecast'):
    st.success('Model loaded sucessfully',icon='✅')
    future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,months+1)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
    future_df=pd.concat([df,future_datest_df])
    #st.write(future_df)
    n=120
    future_df['FORECAST'] = results.predict(start = n, end = n+months, dynamic= True)
    future_df['CLAIM_AMOUNT'] = pd.to_numeric(future_df['CLAIM_AMOUNT'], errors='coerce')
    future_df['FORECAST'] = pd.to_numeric(future_df['FORECAST'],     errors='coerce')
    #st.write(future_df)
    predict_df=future_df.tail(months+72)
    
    st.line_chart(predict_df, y=['FORECAST'], use_container_width=True)
    


