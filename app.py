from operator import index
import streamlit as st
import plotly.express as px
import pandas as pd
import os 
# streamlit run app.py


# import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# ML stuff
from pycaret.classification  import setup, compare_models, pull, save_model, load_model

with st.sidebar:
     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
     st.title("</> AutoStreamML")
     choise = st.radio("Navigation",["Upload","Profiling","Modelling","Download"])
     st.info("Go ahead  knock yourself out guys,do whatever you want")
     st.info("This application allows you to build an automated ML pipeline using Streamlit, ydata-profiling and PyCaret")

# check wether file exist or not
if os.path.exists('sourcedata.csv'):
   df = pd.read_csv('sourcedata.csv',index_col=None)

if choise == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv('sourcedata.csv', index=None)
        st.dataframe(df)


if choise == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    st.write("Read Documentation ---> https://docs.profiling.ydata.ai/latest/ ")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choise == "Modelling":
   st.title("Machine Learning go")
   st.write("Read Documentation ---> https://pycaret.org/")
   chosen_target = st.selectbox('Choose the Target Column', df.columns)
   if st.button('Run Modelling'):
      setup(df, target=chosen_target)
      setup_df = pull()
      st.dataframe(setup_df)
      best_model = compare_models()
      compare_df = pull()
      st.dataframe(compare_df)
      save_model(best_model, 'best_model')

if choise == "Download":
   with open('best_model.pkl','rb') as f:      
        st.download_button('Download Model',f,file_name="best_model.pkl")

