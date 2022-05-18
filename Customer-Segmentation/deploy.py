# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:27:26 2022

@author: Ahmad Faaiz
"""
# %% Imports
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from cs_lib import encoder_cat_features, separate_features_target
from cs_lib import return_saved_objects

# %% Paths
TEST_DATASET_PATH = os.path.join(os.getcwd(), 'database', 'new_customers.csv')
SAVE_RESULT = os.path.join(os.getcwd(), 'database',
                           'new_customers_results.csv')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'oh_encoder.pkl')
IMPUTER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'knn_imputer.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'dl_model.h5')
TARGET_ENCODER_PATH = os.path.join(os.getcwd(), 'saved_objects',
                                   'encoder-Segmentation.pkl')

# %% Load objects
mm_scaler = return_saved_objects(SCALER_PATH)
oh_encoder = return_saved_objects(ENCODER_PATH)
imputer = return_saved_objects(IMPUTER_PATH)
segmentation_encoder = return_saved_objects(TARGET_ENCODER_PATH)

# %% Load DL model
model = load_model(MODEL_PATH)

# %% Load test data
df_test = pd.read_csv(TEST_DATASET_PATH)
num_cols = list(df_test.columns[
    (df_test.dtypes == 'int64') | (df_test.dtypes == 'float64')
])
cat_cols = list(df_test.columns[(df_test.dtypes == 'object')])
cat_features = cat_cols[:-1]

# %%% Data Cleaning
df_clean = df_test.copy()  # create dataframe copy for cleaning

# %%%% encode categorical data & change to numerical
for col_name in cat_cols:
    encoded = encoder_cat_features(
        df_clean, col_name, save_encoder=True)
    df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')

# %%%% impute nan values
imputed = imputer.transform(df_clean)
df_clean = pd.DataFrame(np.round(imputed), columns=df_clean.columns)

# %%% Data Preprocessing
X, y = separate_features_target(df_clean, 'Segmentation')
X_scaled = mm_scaler.transform(X)

# %% Model prediction
y_pred = model.predict(X_scaled).argmax(1)

# get A B C D labels by using inverse_transform
label_pred = segmentation_encoder.inverse_transform(y_pred)

# save segmentation as a result dataset
df_test['Segmentation'] = label_pred
df_test.to_csv(SAVE_RESULT, index=False)
