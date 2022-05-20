# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:30:54 2022

@author: Ahmad Faaiz
"""
import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_PATH = os.path.join(os.getcwd(), 'static')


def create_train_test(window_size, scaled_train, scaled_test):
    """
    Create modified train and test data with respect to given window
    size and scaled values from original train and test data.

    Parameters
    ----------
    window_size : int
        The window size to capture for time-series data.
    scaled_train : array
        Scaled values from train data.
    scaled_test : array
        Scaled values from test data.

    Returns
    -------
    x_train : array
        DESCRIPTION.
    x_test : array
        DESCRIPTION.
    y_train : array
        DESCRIPTION.
    y_test : array
        DESCRIPTION.

    """
    x_train = []
    y_train = []

    # training dataset
    for i in range(window_size, len(scaled_train)):
        x_train.append(scaled_train[i-window_size:i, 0])
        y_train.append(scaled_train[i, 0])

    # testing dataset
    total = np.concatenate((scaled_train, scaled_test))
    window_length = window_size + len(scaled_test)
    data = total[-window_length:]

    x_test = []
    y_test = []

    for i in range(window_size, len(data)):
        x_test.append(data[i-window_size:i, 0])
        y_test.append(data[i, 0])

    # list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # expanding dimensions
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test, y_train, y_test


def impute_time_series(data, method='ffill'):
    """
    Impute missing values in a time-series data.

    Parameters
    ----------
    data : pandas.Series
        The data to be imputed.
    method : str, optional
        The method of imputation. The default is 'ffill'.

    Returns
    -------
    None.

    """
    if method == 'ffill':
        data.fillna(method='ffill', inplace=True)
    elif method == 'bfill':
        data.fillna(method='bfill', inplace=True)
    elif method == 'linear':
        data.interpolate(method='linear', inplace=True)


def plot_pred_actual(y_pred, y_true, to_file=None):
    """
    Plot predicted and actual time-series values.

    Parameters
    ----------
    y_pred : array
        Predicted value.
    y_true : array
        True/actual value.
    to_file : str, optional
        The filename for saving purposes. The default is None.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.legend(['Predicted', 'Actual'])
    if to_file is not None:
        plt.savefig(os.path.join(FIGURES_PATH, to_file))
    plt.show()
