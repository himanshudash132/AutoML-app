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
    with st.expander('About this app'):
      st.markdown('**What can this app do?**')
      st.info('This application allows you to build an automated ML pipeline using Streamlit, ydata-profiling and PyCaret')
      st.markdown('**how to use the AutoStreamML?**')
      st.warning("""Upload Your Dataset: Begin by uploading your dataset using the "Upload" option in the sidebar. Choose the file from your local machine and wait for it to be processed.

      Perform Exploratory Data Analysis (EDA): Once the dataset is uploaded, navigate to the "Profiling" section. Here, you'll find automated tools for conducting exploratory data analysis. Review the generated reports to gain insights into your data.

      Build Classification Models: Move to the "Modelling" section to build classification models based on your dataset. Choose the target column and click "Run Modelling" to initiate the process. The application will guide you through model setup, comparison, and selection of the best model.

      Download Your Model: After selecting the best classification model, head to the "Download" section. Here, you can download the trained machine learning model for later use in your applications.""")
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



   
# from operator import index  # Importing index from operator module (unused in provided code)
# import streamlit as st  # Importing Streamlit library and aliasing it as st
# import plotly.express as px  # Importing Plotly Express library and aliasing it as px
# import pandas as pd  # Importing pandas library and aliasing it as pd
# import os  # Importing os module for interacting with the operating system

# # streamlit run app.py

# # Importing profiling libraries
# # from pandas_profiling import ProfileReport  (Commented out, not used in provided code)
# from streamlit_pandas_profiling import st_profile_report  # Importing Streamlit Pandas Profiling module
# from ydata_profiling import ProfileReport  # Importing ydata-profiling module

# # Importing ML modules
# from pycaret.classification import setup, compare_models, pull, save_model, load_model  # Importing PyCaret classification functions

# # Setting up Streamlit sidebar
# with st.sidebar:
#      st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")  # Displaying an image in the sidebar
#      st.title("</> AutoStreamML")  # Setting the title in the sidebar
#      choise = st.radio("Navigation",["Upload","Profiling","Modelling","Download"])  # Creating a radio button group for navigation
#      st.info("Go ahead knock yourself out guys, do whatever you want")  # Displaying an info message in the sidebar
#      st.info("This application allows you to build an automated ML pipeline using Streamlit, ydata-profiling and PyCaret")  # Displaying an info message in the sidebar

# # Checking if the file 'sourcedata.csv' exists
# if os.path.exists('sourcedata.csv'):
#    df = pd.read_csv('sourcedata.csv',index_col=None)  # Reading the CSV file into a DataFrame

# # Handling different choices made by the user
# if choise == "Upload":
#     st.title("Upload Your Dataset")  # Setting the title for the upload section
#     file = st.file_uploader("Upload Your Dataset")  # Creating a file uploader widget
#     if file:
#         df = pd.read_csv(file,index_col=None)  # Reading the uploaded file into a DataFrame
#         df.to_csv('sourcedata.csv', index=None)  # Saving the DataFrame to a CSV file
#         st.dataframe(df)  # Displaying the DataFrame in the app

# if choise == "Profiling":
#     st.title("Automated Exploratory Data Analysis")  # Setting the title for the profiling section
#     st.write("Read Documentation ---> https://docs.profiling.ydata.ai/latest/")  # Providing a link to documentation
#     profile_df = df.profile_report()  # Generating a profile report for the DataFrame
#     st_profile_report(profile_df)  # Displaying the profile report in the app

# if choise == "Modelling":
#    st.title("Machine Learning go")  # Setting the title for the modeling section
#    st.write("Read Documentation ---> https://pycaret.org/")  # Providing a link to documentation
#    chosen_target = st.selectbox('Choose the Target Column', df.columns)  # Creating a selectbox widget for choosing the target column
#    if st.button('Run Modelling'):  # Creating a button widget for running modeling
#       setup(df, target=chosen_target)  # Setting up the PyCaret environment for modeling
#       setup_df = pull()  # Pulling the processed data after setup
#       st.dataframe(setup_df)  # Displaying the processed data in the app
#       best_model = compare_models()  # Comparing models and selecting the best one
#       compare_df = pull()  # Pulling the comparison results
#       st.dataframe(compare_df)  # Displaying the comparison results in the app
#       save_model(best_model, 'best_model')  # Saving the best model

# if choise == "Download":
#    with open('best_model.pkl','rb') as f:  # Opening the saved model file in binary mode
#         st.download_button('Download Model',f,file_name="best_model.pkl")  # Creating a download button for the model


