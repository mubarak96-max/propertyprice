import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
from flask import Flask, redirect

app = Flask(__name__)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        with open('propertypricepredictor1.pkl', 'rb') as file:
            return pickle.load(file)

rf = load_model()

@app.route('/')
def index():
    # Redirect to the Streamlit app
    return redirect("http://localhost:8501")

if __name__ == '__main__':
    # Start the Streamlit app
    subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    # Run the Flask app
    app.run(debug=True)
