import streamlit as st  
from pycaret.classification import load_model


pipline = load_model('best_model_test')

pipline


