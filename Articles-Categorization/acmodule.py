# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:17:30 2022

@author: Ahmad Faaiz
"""
import os
import csv
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

FIGURES_PATH = os.path.join(os.getcwd(), 'static')
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'saved_models',
                                   'tokenizer.json')
PERFORMANCE_PATH = filename = os.path.join(os.getcwd(), 'report',
                                           'models_performance.csv')
PERFORMANCE_DATA_PATH = os.path.join(os.getcwd(), 'reports_data')


class ModelConfig():

    def __init__(self, config_num, n_words, maxlen,
                 epochs, embed_op, nodes, dropout_val):
        self.config_num = config_num
        self.n_words = n_words
        self.maxlen = maxlen
        self.epochs = epochs
        self.embed_op = embed_op
        self.nodes = nodes
        self.dropout_val = dropout_val

    def std_data_vectorization(self, data, oov_token='<OOV>'):
        tokenizer = Tokenizer(num_words=self.n_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        # save tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        with open(TOKENIZER_JSON_PATH, 'w') as json_file:
            json.dump(token_json, json_file)

        return tokenizer, tokenizer.word_index

    def get_pad_sequences(self, tokenizer, data):
        dummy_seq = tokenizer.texts_to_sequences(data)

        padd_seq = pad_sequences(dummy_seq, maxlen=self.maxlen,
                                 padding='post',
                                 truncating='post')

        return padd_seq

    def std_lstm_model(self, n_categories=2, show_summary=True):
        """
        Create a standard model with Embedding, LSTM,
        Bidirectional, Dropout and Dense layers.

        Parameters
        ----------
        num_words : int
            Number of words.
        n_categories : int, optional
            Number of class to be outputed. The default is 2.
        embed_output : int, optional
            The embedding layer output. The default is 64.
        nodes : int, optional
            The number of nodes for LSTM layer. The default is 32.
        dropout : int, optional
            The dropout value. The default is 0.2.
        show_summary : bool, optional
            The option to show model summary. The default is True.

        Returns
        -------
        model : tensorflow Sequential Object
            The deep learning model created.

        """

        model = Sequential()
        model.add(Embedding(self.n_words, self.embed_op))
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True)))
        model.add(Dropout(self.dropout_val))
        model.add(Bidirectional(LSTM(self.nodes)))
        model.add(Dropout(self.dropout_val))
        model.add(Dense(n_categories, activation='softmax'))
        if show_summary:
            model.summary()

        return model

    def get_reports(self):
        cr = classification_report(self.y_true, self.y_pred,
                                   output_dict=True)
        self.acc_score = cr.get('accuracy')
        self.f1_score = cr.get('weighted avg').get('f1-score')
        self.performance_dict = {'Config_num': self.config_num,
                                 'num_words': self.n_words,
                                 'maxlen': self.maxlen,
                                 'n_epochs': self.epochs,
                                 'embed_output': self.embed_op,
                                 'nodes': self.nodes,
                                 'Accuracy_Score': round(
                                     self.acc_score*100, 2),
                                 'F1_Score': round(
                                     self.f1_score*100, 2)
                                 }

    def get_prediction(self, model, X_test, y_test):
        self.y_pred = model.predict(X_test).argmax(1)
        self.y_true = y_test.argmax(1)

    def show_reports(self, class_report=True, acc_score=False,
                     plot_confusion_matrix=False, display_labels=None):

        if class_report:
            print(classification_report(self.y_true, self.y_pred))

        if acc_score:
            print('The accuracy score is {}%'.format(
                round(accuracy_score(self.y_true, self.y_pred)*100, 2)))

        if plot_confusion_matrix:
            cm = confusion_matrix(self.y_true, self.y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
            plt.figure()
            disp.plot(cmap=plt.cm.Purples, xticks_rotation=25)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(
                FIGURES_PATH, 'confusion-matrix-plot.png'),
                bbox_inches='tight')
            plt.show()


def plot_training(hist, to_save=False):
    loss = list(hist.history.keys())[::2]
    metrics = list(hist.history.keys())[1::2]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for key in loss:
        ax1.plot(hist.history[key])
    ax1.legend(loss)
    for key in metrics:
        ax2.plot(hist.history[key])
    ax2.legend(metrics)

    if to_save:
        plt.savefig(os.path.join(FIGURES_PATH, 'plot-training.png'))
    plt.xlabel('epochs')
    plt.show()


def std_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=20)

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, X_test, y_train, y_test


def save_reports(data_num):
    data_rows = []

    for i in range(data_num):
        with open(os.path.join(
                PERFORMANCE_DATA_PATH, f'data_{i+1}.pkl'), 'rb') as file:
            data_rows.append(pickle.load(file))

    fields = ['Config_num', 'num_words', 'maxlen',
              'n_epochs', 'embed_output', 'nodes',
              'Accuracy_Score', 'F1_Score']

    with open(PERFORMANCE_PATH, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data_rows)


# TODO: add docstring
"""
    There are a few more ideas I wanted to experiment with the
    ModelConfig class but I will try to further experiment it later
    due to time constraint.

    I might have complicated some things here, very sorry if it is not
    quite readable or understandable
"""
