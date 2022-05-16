# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:00:10 2022

@author: Ahmad Faaiz
"""

# %% Imports & Constants
import os
import pickle
import numpy as np
import streamlit as st
# import tensorflow as tf
from tensorflow.keras.models import load_model

SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'mm_scaler_dl.pkl')
ENCODER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'oh_encoder.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'dl_model.h5')

# %% some optimization (GPU)
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# %% load scaler & encoder
with open(SCALER_PATH, 'rb') as file:
    mm_scaler = pickle.load(file)

with open(ENCODER_PATH, 'rb') as file:
    oh_encoder = pickle.load(file)

# %% Load DL model
model = load_model(MODEL_PATH)

# %% Deployment example
# Glucose & BMI features only
patient_info = np.array([116, 25.6, 31]).reshape(1, -1)

scaled_patient_info = mm_scaler.transform(patient_info)

diabetes_outcome = {0: 'negative', 1: 'positive'}
outcome = np.argmax(model.predict(scaled_patient_info))
# another approach
# outcome = oh_encoder.inverse_transform(model.predict(scaled_patient_info))

print(f'The patient diabetes outcome is {diabetes_outcome[outcome]}')

# %% build app
with st.form('Diabetes Prediction Form'):
    st.header("Patient's info")
    glucose = st.number_input('Glucose')
    bmi = st.number_input('BMI')
    age = int(st.number_input('Age'))
    submitted = st.form_submit_button('Submit')
    if submitted:
        patient_info = np.array([glucose, bmi, age]).reshape(1, -1)
        scaled_patient_info = mm_scaler.transform(patient_info)
        outcome = np.argmax(model.predict(scaled_patient_info))
        st.write(
            f'Diabetes outcome is {diabetes_outcome[outcome]}')
        if outcome:
            st.warning('You are in risk of getting diabetes!'.upper())
        else:
            st.balloons()
            st.success('You are diabetic-free')
