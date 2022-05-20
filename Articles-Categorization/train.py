# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:56:25 2022

@author: Ahmad Faaiz
"""
# %% Imports
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from acmodule import ModelConfig, plot_training, std_train_test_split
from acmodule import save_reports
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# %% PATH & URL
URL = ('https://raw.githubusercontent.com/susanli2016' +
       '/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
ENCODER_PATH = os.path.join(os.getcwd(), 'saved_models', 'oh_encoder.pkl')
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'saved_models',
                                   'tokenizer.json')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
PERFORMANCE_DATA_PATH = os.path.join(os.getcwd(), 'reports_data')
PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'static', 'model-architecture.png')
FIGURES_PATH = os.path.join(os.getcwd(), 'static')
DATASET_PATH = os.path.join(os.getcwd(), 'database', 'text_dataset.csv')

# %% EDA
# %%% Load data
df = pd.read_csv(URL)
df.to_csv(DATASET_PATH, index=False)  # save original dataset to a csv file
categories = df['category']  # --> get category column data
text_data = df['text']  # --> get text data

# %%% Inspect data
cat_names = list(df['category'].unique())  # check unique categories
n_categories = len(cat_names)
# --> 5 categories

# %%% Data Cleaning
# Remove non-alphabetical characters, then lowercase and split words
cleaned_text = text_data.replace(to_replace='[^a-zA-Z]', value=' ',
                                 regex=True).str.lower().str.split()

# %%% Data Preprocessing
encoder = OneHotEncoder(sparse=False)
categories_encoded = encoder.fit_transform(categories.values.reshape(-1, 1))
pickle.dump(encoder, open(ENCODER_PATH, 'wb'))  # For future deployment project

# %%% Decode guide (category name & encoded label)
encoded_category = encoder.transform(
    np.array(cat_names).reshape(-1, 1)).argmax(1)

category_dict = {}
for category, label in zip(cat_names, encoded_category):
    category_dict.update([(category, label)])

#  display category names & labels (0-4)
sort_category = dict(sorted(category_dict.items(), key=lambda x: x[1]))
print(sort_category)
# --> might save for deployment purposes
"""
    bussiness:       0
    entertainment:   1
    politics:        2
    sport:           3
    tech:            4
"""

# %% Model configuration (Hard-Coded)
config = ModelConfig(3, 28000, 400, 5, 128, 64, 0.2)

# %% Data Vectorization for reviews
tokenizer, word_index = config.std_data_vectorization(cleaned_text)

# to observe a portion of the word index
print(dict(list(word_index.items())[:10]))
print(len(word_index))  # 27907
# Adjust num_words such that num_words >= 27907

# checking average number of words in cleaned_text to pick suitable maxlen
print(np.mean([len(i) for i in cleaned_text]))
# --> choosing starting maxlen = 400
# --> increasing maxlen might increase model performance

# %%% get pad_sequences
padd_seq = config.get_pad_sequences(tokenizer, cleaned_text)

# %% train_test_split
X_train, X_test, y_train, y_test = std_train_test_split(padd_seq,
                                                        categories_encoded)

# %% Deep Learning Model
# Create model with the configurations set in ModelConfig object
model = config.std_lstm_model(n_categories=n_categories)

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=2)

# %%% Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# %%% Model Training
hist = model.fit(X_train, y_train, epochs=config.epochs,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_cb, early_stopping_cb])

# To plot the training/validation loss and metrics
plot_training(hist, to_save=True)

# %%% Model Evaluation
# %%%% Prediction
config.get_prediction(model, X_test, y_test)
config.y_pred  # prediction labels
config.y_true  # True labels from (OneHotEncoder) to (0-4 labels)

# %%%% Reports
# show all reports
config.show_reports(acc_score=(1), plot_confusion_matrix=(1),
                    display_labels=sort_category.keys())

# %%% Save plot & model
plot_model(model, to_file=PLOT_MODEL_PATH)
model.save(MODEL_PATH)

# %% Save model performance
# get the accuracy score and f1_score & settings+performance data
config.get_reports()
pickle.dump(config.performance_dict, open(os.path.join(
    PERFORMANCE_DATA_PATH, f'data_{config.config_num}.pkl'), 'wb'))

# %% Uncomment below to save different reports
save_reports(data_num=3)  # save to a csv file

# %% Discussion
# =============================================================================
# DISCUSSION/ANALYSIS
# =============================================================================
"""
    Looking at the models_performance.csv, maxlen =  400 give better
    performance than maxlen = 800.

    Will add more later...
"""
