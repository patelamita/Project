

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import traceback  # Add this line to import the traceback module

import xgboost as xgb
print(xgb.__version__)

# Load the DataFrame
df = pd.read_pickle('df.pkl')

# Display title
st.title("Laptop Price Predictor")

try:
    # Load the XGBoost model from file
    pipe = joblib.load('pipe.pkl')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    # Print additional information about the error
    st.error(traceback.format_exc())

    # Log the error to a file for further investigation
    with open('error.log', 'a') as f:
        f.write(f"Error loading model: {e}\n")
        f.write(traceback.format_exc())

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

#cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())
if st.button('Predict Price'):
    try:
        if 'pipe' not in globals():
            st.error("Model not loaded. Please make sure the model is loaded successfully before predicting.")
        else:
            # Preprocess input
            touchscreen = 1 if touchscreen == 'Yes' else 0
            ips = 1 if ips == 'Yes' else 0
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

            # Create query array
            query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

            query = query.reshape(1, 12)
            st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))  
    except Exception as e:
        st.error(f"Error predicting price: {e}")
        # Print additional information about the error
        st.error(traceback.format_exc())
