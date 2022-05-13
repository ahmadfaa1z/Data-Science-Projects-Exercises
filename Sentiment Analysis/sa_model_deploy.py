# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:07:08 2022

@author: Ahmad Faaiz
"""

# %% Import modules
import os
import re
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')

# %% Model loading
sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

# %% tokenizer loading
JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
with open(JSON_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)

# %% Deployment
new_review = ['I think the first one hour is interesting but \
              the second half of the movie is boring.<br /><br /> \
                  This movie just wasted my precious time and hard \
                      earned money.<br /><br />This movie should be \
                          banned to avoid time being wasted.']

for i, text in enumerate(new_review):
    new_review[i] = re.sub('<.*?>', '', text)
for i, text in enumerate(new_review):
    new_review[i] = re.sub('[^a-zA-Z]', ' ', text).lower().split()


# %% to vectorize the new review

# to feed the tokens into integers
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)

# to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(new_review)

# to pad the data to ensure every row of data has equal length
new_review = pad_sequences(new_review, maxlen=300,
                           padding='post',
                           truncating='post')

# %% Model predict
outcome = sentiment_classifier.predict(np.expand_dims(new_review, -1))
sentiment_dict = {0: 'negative', 1: 'positive'}
print('the review is', sentiment_dict[np.argmax(outcome)])
