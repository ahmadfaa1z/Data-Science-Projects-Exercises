# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:03:43 2022

@author: Ahmad Faaiz
"""

# %% Imports
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
from cs_lib import plot_missing_val, plot_features_target, encoder_cat_features
from cs_lib import check_unique_values, separate_features_target
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# %% Paths
TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'database', 'train.csv')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'oh_encoder.pkl')
IMPUTER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'knn_imputer.pkl')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'dl_model.h5')
PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'static',
                               'model-architecture.png')
FIGURES_PATH = os.path.join(os.getcwd(), 'static')

# %% EDA
# %%% 1) Load data
df = pd.read_csv(TRAIN_DATASET_PATH)

# %%% 2) Inspect data
df.info()  # overall info
df.describe()  # check numerical statistics
df.isna().sum()  # check missing values count
df[df.duplicated(keep=False)]  # check for all duplicates
# --> No duplicates

# get numerical column names and categorical column names
num_cols = list(df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')])
cat_cols = list(df.columns[(df.dtypes == 'object')])
cat_features = cat_cols[:-1]

# check unique values for categorical columns including nan
for col_name in cat_cols:
    check_unique_values(df, col_name)

# %%%% Visualization
plt.figure()
df.boxplot()

# to visualize the missing numbers
plot_missing_val(df, to_save=True, f_name='plot-nan-heatmap-train-dataset')

# %%%%% Relationship between features & target excluding missing values
for col_name in cat_features+num_cols[2:]:
    plot_features_target(df, 'categorical', col_name, save=True)
plot_features_target(df, 'continuous', 'Age', save=True)

# %%% 3) Data Cleaning
df_clean = df.copy()  # create dataframe copy for cleaning

# %%%% encode categorical data & change to numerical
for col_name in cat_cols:
    encoded = encoder_cat_features(
        df_clean, col_name, save_encoder=True)
    df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')

# %%%% impute nan values
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
imputed = imputer.fit_transform(df_clean)
pickle.dump(imputer, open(IMPUTER_PATH, 'wb'))
df_clean = pd.DataFrame(np.round(imputed), columns=df_clean.columns)

# check again unique values whether missing values is still exist or not
for col_name in df_clean.columns:
    check_unique_values(df_clean, col_name)

# or visualize the missing numbers using heatmap visualization
plot_missing_val(df_clean, to_save=True, f_name='heatmap-no-missing-values')
# --> no more missing values

# %%% 4) Features Selection
plt.figure(figsize=(12, 12))
sns.heatmap(df_clean.corr(), annot=True, cmap='GnBu')
plt.show()

# %%% 5) Data Preprocessing
# select all features and the target data
X, y = separate_features_target(df_clean, 'Segmentation')

# %%%% minmax scalling
scaler = StandardScaler()
X = scaler.fit_transform(X)
pickle.dump(scaler, open(SCALER_PATH, 'wb'))

# %%%% one hot encoding
oh_encoder = OneHotEncoder(sparse=False)
y = oh_encoder.fit_transform(y.reshape(-1, 1))
pickle.dump(oh_encoder, open(ENCODER_PATH, 'wb'))

# %%%% train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15,
                                                    random_state=20)

# %% Deep Learning
model = Sequential()
model.add(Input(shape=10))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))
model.summary()

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=6)

# %%% Compile
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics='acc')


# %%% Model Training
hist = model.fit(X_train, y_train, epochs=100,
                 validation_data=(X_test, y_test),
                 callbacks=[early_stopping_cb, tb_callback])

# %%% save architecture and model
plot_model(model, to_file=PLOT_MODEL_PATH)
model.save(MODEL_PATH)

# %%% Model Evaluation
y_pred = model.predict(X_test)

print(classification_report(y_test.argmax(1), y_pred.argmax(1)))
acc_score = round(accuracy_score(y_test.argmax(1), y_pred.argmax(1))*100, 2)
print(f'The accuracy score is {acc_score}%')

cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
disp = ConfusionMatrixDisplay(cm, display_labels=list(range(4)))
plt.figure()
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(FIGURES_PATH, 'confusion-matrix-display.png'))
plt.show()

# =============================================================================
# Suggestion on improving the model
# =============================================================================
"""
    From the accuracy score that we get from this model, the model seems to be
    underfitting. What we can do to improve this is probably add more neurons
    or layers which is simply make a bigger and deeper neural network model.
    Other things that we could do is by increasing the epoch and also maybe
    decrease the learning rate
"""
