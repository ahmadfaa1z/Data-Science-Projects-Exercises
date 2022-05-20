# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:30:07 2022

@author: Ahmad Faaiz
"""
# %% Imports
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ncmodule import impute_time_series, create_train_test, plot_pred_actual
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# %% PATH
TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'database',
                                  'cases_malaysia_train.csv')
TEST_DATASET_PATH = os.path.join(os.getcwd(), 'database',
                                 'cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'static', 'model-architecture.png')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'model.h5')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'mm_scaler.pkl')

# %% EDA
# %%% Load data
df_train = pd.read_csv(TRAIN_DATASET_PATH,
                       parse_dates=['date'], index_col='date')
df_test = pd.read_csv(TEST_DATASET_PATH,
                      parse_dates=['date'], index_col='date')

# %%% Inspect data
df_train.info()
"""
    cases_new column is object type, some data are not numeric
    cluster_...columns have missing values in 2020
"""
df_test.info()
# cases_new column in test data have one missing value

# %%% Data Cleaning

# make copy to reference between the cleaned data and original data
clean_train = df_train.copy()
clean_test = df_test.copy()

# Convert to numerical data for cases_new column in train dataset
clean_train['cases_new'] = pd.to_numeric(clean_train['cases_new'],
                                         errors='coerce')

# impute missing values using interpolation
impute_time_series(clean_train['cases_new'], method='linear')
impute_time_series(clean_test['cases_new'], method='linear')

# get train and test data
train_data = clean_train['cases_new'].values
test_data = clean_test['cases_new'].values

# %%% Visualize train and test data
for i in [train_data, test_data]:
    plt.figure()
    plt.plot(i)
    plt.show()

# %%% Preprocess data
mm_scaler = MinMaxScaler()
scaled_x_train = mm_scaler.fit_transform(train_data.reshape(-1, 1))
scaled_x_test = mm_scaler.transform(test_data.reshape(-1, 1))
pickle.dump(mm_scaler, open(SCALER_PATH, 'wb'))  # save scaler object

x_train, x_test, y_train, y_test = create_train_test(30, scaled_x_train,
                                                     scaled_x_test)

# %% Deep Learning Model
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True,
               input_shape=x_train.shape[1:]))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mse', metrics='mse')  # wrap/compile

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

# %%% Model Training
hist = model.fit(x_train, y_train, epochs=30, batch_size=128,
                 callbacks=[tensorboard_callback, early_stopping_callback])

# %%% Save model and plot
plot_model(model, to_file=PLOT_MODEL_PATH)
model.save(MODEL_PATH)

# %% Model prediction
predicted = model.predict(x_test)  # make prediction on x_test

# %% Get original scale
inversed_y_pred = mm_scaler.inverse_transform(predicted)
inversed_y_true = mm_scaler.inverse_transform(y_test.reshape(-1, 1))

# %% Plot predicted value and actual value

# Plot predicted and actual with min-max scale
plot_pred_actual(predicted, y_test, to_file='pred-actual-minmax-scale.png')

# Plot predicted and actual with original scale
plot_pred_actual(inversed_y_pred, inversed_y_true,
                 to_file='pred-actual-original-scale.png')

# %% Performance Evaluation
print("The mean absolute percentage error for this model is {:.2f}%".format(
    (mean_absolute_error(y_test, predicted)/sum(abs(y_test)))*100))

"""
    Reports:
        The MAPE for this model ranges from 0.16% to 0.27%
        Even with different imputation method use for the missing values,
        the MAPE is still below 1%.
"""
