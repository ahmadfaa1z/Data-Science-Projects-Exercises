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
        """
        Create tokenizer object and the word index of the tokenizer

        Parameters
        ----------
        data : array-like
            The text data for creating the tokenizer.
        oov_token : str, optional
            DESCRIPTION. The default is '<OOV>'.

        Returns
        -------
        tokenizer : Tokenizer object
            The tokenizer object created from given data.
        tokenizer.word_index : dict
            A dictionary containing the words and their respective token.

        """
        tokenizer = Tokenizer(num_words=self.n_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        # save tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        with open(TOKENIZER_JSON_PATH, 'w') as json_file:
            json.dump(token_json, json_file)

        return tokenizer, tokenizer.word_index

    def get_pad_sequences(self, tokenizer, data):
        """
        Return padding sequences from given tokenizer and data.

        Parameters
        ----------
        tokenizer : Tokenizer object
            The tokenizer is required to create a dummy sequences.
        data : array-like
            DESCRIPTION.

        Returns
        -------
        padd_seq : array-like
            The padding sequences are created with maximum length
            specified during creation of the class.

        """
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
        """
        Save accuracy and f1 scores, and model performance in a dictionary.

        Returns
        -------
        None.

        """
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
        """
        Make prediction with given test data.

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        X_test : TYPE
            DESCRIPTION.
        y_test : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.y_pred = model.predict(X_test).argmax(1)
        self.y_true = y_test.argmax(1)

    def show_reports(self, class_report=True, acc_score=False,
                     plot_confusion_matrix=False, display_labels=None):
        """
        Display a variery of reports depending on the parameters.

        Parameters
        ----------
        class_report : bool, optional
            The option to show classification report. The default is True.
        acc_score : bool, optional
            The option to show accuracy score. The default is False.
        plot_confusion_matrix : bool, optional
            The option to display confusion matrix. The default is False.
        display_labels : TYPE, optional
            If provided, labels will be shown in confusion matrix.
            The default is None.

        Returns
        -------
        None.

        """

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
    """
    Plot training graph with matplotlib.

    Parameters
    ----------
    hist : TYPE
        DESCRIPTION.
    to_save : bool, optional
        The option to save the created figure. The default is False.

    Returns
    -------
    None.

    """
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
    """
    A standard way of splitting train and test data.

    Parameters
    ----------
    X : TYPE
        Features data.
    y : TYPE
        Target data.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=20)

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, X_test, y_train, y_test


def save_reports(data_num):
    """
    Saving reports inside a csv file.

    Parameters
    ----------
    data_num : int
        The number that reflects the different model configurations used.

    Returns
    -------
    None.

    """
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
