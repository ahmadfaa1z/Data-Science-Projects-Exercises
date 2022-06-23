# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:29:00 2022

@author: Ahmad Faaiz
"""

# %% Imports
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from ha_lib import separate_features_target, evaluate_model
from PIL import Image

SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'ml_model.pkl')
IMAGE_WARNING = os.path.join(os.getcwd(), 'static',
                             'prevention_is_better_than_cure.png')
IMAGE_SUCCESS = os.path.join(os.getcwd(), 'static',
                             'healthy-heart.png')

image_warning = Image.open(IMAGE_WARNING)
image_success = Image.open(IMAGE_SUCCESS)

# %% load scaler
with open(SCALER_PATH, 'rb') as file:
    mm_scaler = pickle.load(file)

# %% Load ML model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# %% new data
test_data = {'age': [65, 61, 45, 40, 48, 41, 36, 45, 57, 69],
             'sex': [1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
             'cp': [3, 0, 1, 1, 2, 0, 2, 0, 0, 2],
             'trtbps': [142, 140, 128, 125, 132, 108, 121, 111, 155, 179],
             'chol': [220, 207, 204, 307, 254, 165, 214, 198, 271, 273],
             'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             'restecg': [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
             'thalachh': [158, 138, 172, 162, 180, 115, 168, 176, 112, 151],
             'exng': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
             'oldpeak': [2.3, 1.9, 1.4, 0, 0, 2, 0, 0, 0.8, 1.6],
             'slp': [1, 2, 2, 2, 2, 1, 2, 2, 2, 1],
             'caa': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             'thall': [1, 3, 2, 2, 2, 3, 2, 2, 3, 3],
             'output': [1, 0, 1, 1, 1, 0, 1, 0, 0, 0]
             }

df_test = pd.DataFrame(test_data)  # create DataFrame from test_data

# %% preprocess new data
# Get features data and target data
x_new_test, y_new_test = separate_features_target(df_test,
                                                  target_name='output')

scaled_x = mm_scaler.transform(x_new_test)  # scale the data

# %% Make predictions
heart_attack_chance = {0: 'less chance of heart attack',
                       1: 'more chance of heart attack'}

new_pred = model.predict(scaled_x)

# %% Model Evaluation
evaluate_model(model, y_true=y_new_test, y_pred=new_pred)

# %% Streamlit app
st.header('Some test data here...')
st.dataframe(df_test)

with st.form('Heart Attack Prediction Form'):

    st.header("Patient's info")
    AGE = int(st.slider('Age', 0, 150))
    SEX = st.radio('Sex', ['Male', 'Female'])
    CP = st.radio('Chest Pain type', ['Typical angina',
                                      'Atypical angina',
                                      'Non-anginal pain',
                                      'Asymptomatic'])
    TRTBPS = int(st.number_input('Resting Blood Pressure (in mm Hg)', 0))
    CHOL = int(st.number_input('Cholestoral in mg/dl', 0))
    FBS = st.checkbox('Is fasting blood sugar > 120 mg/dl')
    RESTECG = st.radio(
        'Resting electrocardiographic results',
        ['Normal', 'Having ST-T wave abnormality',
         'showing probable/definite Left Ventricular Hypertrophy'])
    THALACHH = int(st.number_input('Maximum heart rate achieved', 0))
    EXNG = st.radio('Exercise induced angina', ['Yes', 'No'])
    OLDPEAK = st.number_input('Previous peak', 0.0)
    SLP = int(st.number_input('Slope of peak exercise ST segment', 0))
    CAA = int(st.number_input('Number of major vessels', 0, 3))
    THALL = int(st.number_input('Thalium Stress Test result', 0, 3))

    # %%% Process data for prediction
    SEX = 1 if SEX == 'Male' else 0

    if CP == 'Typical angina':
        CP = 0
    elif CP == 'Atypical angina':
        CP = 1
    elif CP == 'Non-anginal pain':
        CP = 2
    elif CP == 'Asymptomatic':
        CP = 3

    if RESTECG == 'Normal':
        RESTECG = 0
    elif RESTECG == 'Having ST-T wave abnormality':
        RESTECG = 1
    else:
        RESTECG = 2

    EXNG = 1 if EXNG == 'Yes' else 0

    # %%% Submit & make prediction
    submitted = st.form_submit_button('Submit')

    if submitted:
        data = np.array(
            [AGE, SEX, CP, TRTBPS, CHOL, FBS, RESTECG,
             THALACHH, EXNG, OLDPEAK, SLP, CAA, THALL]).reshape(1, -1)
        scaled_data = mm_scaler.transform(data)

        outcome = int(model.predict(scaled_data))

        st.subheader(f'The patient have {heart_attack_chance[outcome]}')
        if outcome:
            st.warning(
                '!! PREVENTION IS BETTER THAN CURE. TAKE CARE OF YOUR HEALTH')
            st.image(image_warning, caption='Take care of your heart always')
        else:
            st.success('KEEP UP THE GOOD HEALTH')
            st.image(image_success, caption='Be good to your heart always')
