# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:37:30 2022

@author: Ahmad Faaiz
"""
# %% Imports & Constants
import os
import json
import numpy as np
from sentiment_analysis import EDA, get_pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

# %% Model loading
sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

# %% tokenizer loading
with open(JSON_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)

# %% Deployment
new_review = [r'I think the first one hour is interesting but \
              the second half of the movie is boring.<br /><br /> \
                  This movie just wasted my precious time and hard \
                      earned money.<br /><br />This movie should be \
                          banned to avoid time being wasted.']

# Input() is problematic in Spyder v5.1.5
# new_review = [input('What is your review on the movie?\n\n')]

eda = EDA()
eda.clean_data_list(new_review)

# %% to vectorize the new review
tokenizer = tokenizer_from_json(loaded_tokenizer)
new_review = get_pad_sequences(tokenizer, new_review)

# %% Model predict
outcome = sentiment_classifier.predict(np.expand_dims(new_review, -1))
sentiment_dict = {0: 'negative', 1: 'positive'}
print('the review is', sentiment_dict[np.argmax(outcome)])
