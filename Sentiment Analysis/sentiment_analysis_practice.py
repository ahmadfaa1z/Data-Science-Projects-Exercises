# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:17:16 2022

@author: Ahmad Faaiz
"""
# %% Import modules
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

LOG_PATH = os.path.join(os.getcwd(), 'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

# %% EDA
# %%% Load data
df = pd.read_csv(URL)

review = df['review']
sentiment = df['sentiment']

# %%% Clean data

# remove <br /> tags
review_wo_br = review.str.replace('<br />', '')

# remove HTML tags
review_wo_html = review.replace(to_replace='<.*?>', value='', regex=True)

# remove numerical, lowercase and split
review_clean = review_wo_html.replace(to_replace='[^a-zA-Z]',
                                      value=' ',
                                      regex=True).str.lower().str.split()

# %%% Data preprocessing
encoder = OneHotEncoder(sparse=False)
sentiment_encoded = encoder.fit_transform(sentiment.values.reshape(-1, 1))
# positive = [0, 1]
# negative = [1, 0]

# %% Data Vectorization for reviews
num_words = 105000
oov_token = '<OOV>'

# vectorize words
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(review_clean)

# save tokenizer for deployment purpose
token_json = tokenizer.to_json()
with open(TOKENIZER_JSON_PATH, 'w') as json_file:
    json.dump(token_json, json_file)

# to observe the number of words
word_index = tokenizer.word_index
print(dict(list(word_index.items())[:10]))

# to vectorize the sequences of text
dummy_seq = tokenizer.texts_to_sequences(review_clean)

# pick suitable maxlen
padd_seq = pad_sequences(dummy_seq, maxlen=300,
                         padding='post',
                         truncating='post')

# %% train_test_split
X_train, X_test, y_train, y_test = train_test_split(padd_seq,
                                                    sentiment_encoded,
                                                    test_size=0.3,
                                                    random_state=123)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# %% Model Creation
model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

# %% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# %% Model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# %% Model fit
hist = model.fit(X_train, y_train, epochs=3,
                 validation_data=(X_test, y_test),
                 callbacks=[tb_callback])

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epochs')
plt.legend(['training loss', 'validation loss'])
plt.show()

# %% Model Evaluation
# %%% preallocation of memory approach
predicted_advanced = np.empty([len(X_test), 2])
for i, test in enumerate(X_test[0:5000]):
    predicted_advanced[i, :] = model.predict(np.expand_dims(test, 0))
for i, test in zip(range(5000, 10000), X_test[5000:10000]):
    predicted_advanced[i, :] = model.predict(np.expand_dims(test, 0))
for i, test in zip(range(10000, 15000), X_test[10000:]):
    predicted_advanced[i, :] = model.predict(np.expand_dims(test, 0))

# %% Model Analysis
y_pred = np.argmax(predicted_advanced, 1)
y_true = np.argmax(y_test, 1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

# %% Model save
model.save(MODEL_SAVE_PATH)
