# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:40:10 2022

@author: Ahmad Faaiz
"""

# %% Imports & Constants
import os
import pickle
import numpy as np
import streamlit as st

SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'mm_scaler_ml.pkl')
PIPELINE_PATH = os.path.join(os.getcwd(), 'saved_objects', 'ml_model.pkl')

# %% load scaler
with open(SCALER_PATH, 'rb') as file:
    mm_scaler = pickle.load(file)

# %% Load ML model
with open(PIPELINE_PATH, 'rb') as file:
    model = pickle.load(file)

# %% Deployment example
# Glucose & BMI features only
patient_info = np.array(
    [6, 148, 72, 35, 30, 33.6, 0.627, 50]).reshape(1, -1)

scaled_patient_info = mm_scaler.transform(patient_info)

diabetes_outcome = {0: 'negative', 1: 'positive'}
outcome = int(model.predict(scaled_patient_info))

print(f'The patient diabetes outcome is {diabetes_outcome[outcome]}')

# %% build app
with st.form('Diabetes Prediction Form'):

    st.header("Patient's info")
    PREG = int(st.number_input('Pregnancies'))
    GLU = st.number_input('Glucose')
    BP = st.number_input('Blood Pressure')
    ST = st.number_input('Skin Thickness')
    INS = int(st.number_input('Insulin'))
    BMI = st.number_input('BMI')
    DPF = st.number_input('Diabetes Pedigree Function')
    AGE = int(st.number_input('Age'))

    submitted = st.form_submit_button('Submit')

    if submitted:
        patient_info = np.array(
            [PREG, GLU, BP, ST, INS, BMI, DPF, AGE]).reshape(1, -1)
        scaled_patient_info = mm_scaler.transform(patient_info)
        outcome = int(model.predict(scaled_patient_info))
        st.write(
            f'Diabetes outcome is {diabetes_outcome[outcome]}')
        if outcome:
            st.warning('You are in risk of getting diabetes!'.upper())
        else:
            st.balloons()
            st.success('You are diabetic-free')
