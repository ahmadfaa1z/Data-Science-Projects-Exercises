# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:27:54 2022

@author: Ahmad Faaiz
"""

# %% Imports
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# %% Paths
FIGURES_PATH = os.path.join(os.getcwd(), 'static')
OBJECTS_PATH = os.path.join(os.getcwd(), 'saved_objects')

# %% Functions


def plot_missing_val(df, to_save=False, f_name='plot-nan-heatmap'):
    """
    Plotting the missing values as heatmap representation.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data to be plotted.
    to_save : bool, optional
        The option to save the heatmap figure. The default is False.
    f_name : str, optional
        The filename for saving purposes. The default is 'plot-nan-heatmap'.

    Returns
    -------
    None.

    """
    plt.figure()
    sns.heatmap(df.isna(), cmap='binary_r')
    if to_save:
        plt.savefig(os.path.join(FIGURES_PATH, f'{f_name}.png'))


def separate_features_target(df, target_name):
    """
    Separate features columns and target columns from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that contains both features and target.
    target_name : str
        The column name of the target.

    Returns
    -------
    x : array-like
        features data.
    y : array-like
        target data.

    """
    x = df.drop(labels=[target_name], axis=1).to_numpy()
    y = df[target_name].values
    return (x, y)


def plot_features_target(df, data_type, feature_name,
                         target_name='Segmentation', save=False):
    """
    Create a figure showing the relationship between the target and
    the categorical/numerical features.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains all the features and target values.
    data_type : str
        Either 'categorical' or 'continuous' feature.
    feature_name : str
        The column name for feature.
    target_name : str, optional
        The column name for target. The default is 'Segmentation'.
    save : bool, optional
        Option to save the figure created. The default is False.

    Returns
    -------
    None.

    """
    if data_type == 'categorical':
        plt.figure()
        sns.countplot(x=df[feature_name], hue=df[target_name], palette='GnBu')
        plt.xticks(rotation=45)
        plt.title(feature_name.capitalize())
        if save:
            plt.savefig(os.path.join(
                FIGURES_PATH, f'count-{feature_name}-{target_name}.png'))
        plt.show()
    elif data_type == 'continuous':
        plt.figure()
        sns.histplot(x=df[feature_name], hue=df[target_name])
        plt.title(feature_name.capitalize())
        if save:
            plt.savefig(os.path.join(
                FIGURES_PATH, f'histogram-{feature_name}-{target_name}.png'))
        plt.show()


def encoder_cat_features(data, feature_name,
                         save_encoder=False):
    """
    Encode the categorical data in a dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe that contains the categorical data.
    feature_name : str
        The name of the categorical feature to encode.
    save_encoder : bool, optional
        The option to save the encoder. The default is False.

    Returns
    -------
    None.

    """
    features_data = data[feature_name]
    nonulls = np.array(features_data.dropna())
    le = LabelEncoder()
    encoded = le.fit_transform(nonulls)
    if save_encoder:
        pickle.dump(le, open(os.path.join(
            OBJECTS_PATH, f'encoder-{feature_name}.pkl'), 'wb'))
    features_data.loc[features_data.notnull()] = encoded


def check_unique_values(data, col_name):
    """
    For checking unique values in the data by a column

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe that contains the data to be checked.
    col_name : str
        The name of the column to be accessed.

    Returns
    -------
    None.

    """
    print('{}:\n{}'.format(col_name, data[col_name].unique()))


def return_saved_objects(path):
    """
    Return an object from a pickle file.

    Parameters
    ----------
    path : str
        The path to the saved objects.

    Returns
    -------
    object
        Return an object from a pickle file.

    """
    with open(path, 'rb') as file:
        return pickle.load(file)
