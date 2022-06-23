# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:53:02 2022

@author: Ahmad Faaiz
"""
# %% Import modules
import os
import re
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

# %% EDA Class


class EDA():

    def clean_data(self, data):
        """
        Clean data for unnecessary strings/characters.

        Parameters
        ----------
        data : pandas Series
            data with the unnecessary strings/characters.

        Returns
        -------
        cleaned_data : pandas Series
            The data without the unnecessary strings/characters.

        """

        # remove HTML tags
        cleaned_data = data.replace(to_replace='<.*?>',
                                    value='', regex=True)

        # remove numerical, lowercase and split
        cleaned_data = cleaned_data.replace(to_replace='[^a-zA-Z]',
                                            value=' ',
                                            regex=True).str.lower().str.split()

        return cleaned_data

    def clean_data_list(self, data_list):
        """
        Clean data from given list.

        Parameters
        ----------
        data_list : list
            Contains sentences with some unnecessary characters.

        Returns
        -------
        None.

        """
        for i, text in enumerate(data_list):
            data_list[i] = re.sub('<.*?>', '', text)
        for i, text in enumerate(data_list):
            data_list[i] = re.sub('[^a-zA-Z]', ' ', text).lower().split()

# %% Deep Learning Class


class DeepLearningModel():

    def std_lstm_model(self, n_words, n_categories=2, embed_output=64,
                       nodes=32, dropout=0.2):
        """
        Create a standard model with Embedding, LSTM,
        bidirectional and Dropout layers.

        Parameters
        ----------
        num_words : int
            DESCRIPTION.
        n_category : int, optional
            DESCRIPTION. The default is 2.
        embed_output : int, optional
            DESCRIPTION. The default is 64.
        nodes : int, optional
            DESCRIPTION. The default is 32.
        dropout : int, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        model : tensorflow Sequential Object
            DESCRIPTION.

        """

        model = Sequential()
        model.add(Embedding(n_words, embed_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(n_categories, activation='softmax'))
        model.summary()

        return model

    def model_analysis(self, y_true, y_pred):
        """
        Display analysis report.

        Parameters
        ----------
        y_true : array
            DESCRIPTION.
        y_pred : array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))

# %% Other functions


def std_data_vectorization(data, num_words=10000, oov_token='<OOV>'):
    # tokenizer object
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(data)

    # save tokenizer for deployment purpose
    token_json = tokenizer.to_json()
    with open(TOKENIZER_JSON_PATH, 'w') as json_file:
        json.dump(token_json, json_file)

    return tokenizer


def get_pad_sequences(tokenizer, data, maxlen=200):
    # to vectorize the sequences of text
    dummy_seq = tokenizer.texts_to_sequences(data)

    # pad the data to ensure every row of data has equal length
    padd_seq = pad_sequences(dummy_seq, maxlen=maxlen,
                             padding='post',
                             truncating='post')

    return padd_seq
