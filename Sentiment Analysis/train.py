# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:01:53 2022

@author: Ahmad Faaiz
"""

# %% Imports & Constants
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sentiment_analysis import EDA, DeepLearningModel
from sentiment_analysis import std_data_vectorization, get_pad_sequences
from tensorflow.keras.callbacks import TensorBoard

LOG_PATH = os.path.join(os.getcwd(), 'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

# %% EDA
eda = EDA()
# %%% Load data
df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

# %%% Clean data
review_clean = eda.clean_data(review)

# %%% Data preprocessing
encoder = OneHotEncoder(sparse=False)
sentiment_encoded = encoder.fit_transform(sentiment.values.reshape(-1, 1))
# positive = [0, 1], negative = [1, 0]

# %% Data Vectorization for reviews
tokenizer = std_data_vectorization(review_clean)

# %%% to observe the number of words
word_index = tokenizer.word_index
print(dict(list(word_index.items())[:10]))

# %%% get pad_sequences
padd_seq = get_pad_sequences(tokenizer, review_clean)

# %% train_test_split
X_train, X_test, y_train, y_test = train_test_split(padd_seq,
                                                    sentiment_encoded,
                                                    test_size=0.3,
                                                    random_state=123)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# %% Model Creation
dlm = DeepLearningModel()
model = dlm.std_lstm_model(n_words=10000)

# %% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# %% Model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# %% Model fit
hist = model.fit(X_train, y_train, epochs=3,
                 validation_data=(X_test, y_test),
                 callbacks=[tb_callback])

# %% Model Evaluation
# %%% preallocation of memory approach
# predicted_advanced = np.empty([len(X_test), 2])
# for i, test in enumerate(X_test):
#     predicted_advanced[i, :] = model.predict(np.expand_dims(test, 0))

# faster method
predicted = model.predict(X_test)

# %% Model Analysis
y_pred = np.argmax(predicted, 1)
y_true = np.argmax(y_test, 1)
dlm.model_analysis(y_true, y_pred)

# %% Model save
model.save(MODEL_SAVE_PATH)
