# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:03:46 2022

@author: Ahmad Faaiz
"""

# %% Imports
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

FIGURES_PATH = os.path.join(os.getcwd(), 'static')

# %% Functions


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


def plot_features_vs_output(df, data_type, feature_name, target_name='output',
                            save=False):
    """
    Create a figure showing the relationship between the output ant
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
        The column name for target. The default is 'output'.
    save : bool, optional
        Option to save the figure created. The default is False.

    Returns
    -------
    None.

    """
    if data_type == 'categorical':
        plt.figure()
        sns.countplot(x=df[feature_name], hue=df[target_name], palette='GnBu')
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


def evaluate_model(model, y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print('The classifier is {} and the accuracy is {:.2f}%'.format(
        model.steps[0][1], accuracy_score(y_true, y_pred)*100))
