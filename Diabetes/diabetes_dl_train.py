# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:13:40 2022

@author: Ahmad Faaiz
"""

# %% Imports & Constants
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'mm_scaler_dl.pkl')
ENCODER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'oh_encoder.pkl')
DATASET_PATH = os.path.join(os.getcwd(), 'database', 'diabetes.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_objects', 'dl_model.h5')

# %% EDA
# %%% Load data
df = pd.read_csv(DATASET_PATH)

# %%% Inspect data
df.info()
df.isna().sum()

df.describe().T['min']
# --> 6 features have zero as their min

# %%% Clean data
# %%%% Drop duplicates
df.drop_duplicates(inplace=True)

# %%%% impute nan values
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed, columns=df.columns)

# %%%% impute zero values
# check frequency of zeros in the 6 features
for feature in df.columns[:6]:
    zero_count = df_imputed[df_imputed[feature] == 0][feature].count()
    print('{}: have {} zeros, about {:.2f}% frequency'
          .format(feature, zero_count, 100*zero_count/len(df_imputed)))

# replace zeros to respective mean of the 6 features except pregnancies
for feature in df.columns[1:6]:
    feature_median = df_imputed[feature].median()
    if df[feature].dtype == 'int64':
        df_imputed[feature] = df_imputed[feature]\
            .replace(0, round(feature_median))
    else:
        df_imputed[feature] = df_imputed[feature]\
            .replace(0, feature_median)

# %%% Features selection
# %%%% heatmap
plt.figure()
sns.heatmap(df_imputed.corr(method='kendall'), annot=True, cmap=plt.cm.Reds)
plt.show()

# -> Glucose, Age, BMI

# %%% Data preprocessing
X = df_imputed[['Glucose', 'BMI', 'Age']].to_numpy()
y = df_imputed['Outcome'].values

# %%%% minmax scalling
mm_scaler = MinMaxScaler()
X = mm_scaler.fit_transform(X)
pickle.dump(mm_scaler, open(SCALER_PATH, 'wb'))

# %%%% one hot encoding
oh_encoder = OneHotEncoder(sparse=False)
y = oh_encoder.fit_transform(y.reshape(-1, 1))
pickle.dump(oh_encoder, open(ENCODER_PATH, 'wb'))

# %%%% train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=20)

# %% Deep Learning
model = Sequential()
model.add(Input(shape=3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

# %%% model compile
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics='acc')

# %%% model training
hist = model.fit(X_train, y_train, epochs=30,
                 validation_data=(X_test, y_test))

# %%% model save
model.save(MODEL_PATH)

# %%% Model Evaluation
predicted = model.predict(X_test)

y_pred = np.argmax(predicted, 1)
y_true = np.argmax(y_test, 1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
# about (70-75)+ % with DL model
