# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:28:36 2022

@author: Ahmad Faaiz
"""

# %% Imports
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ha_lib import separate_features_target, plot_features_target
from ha_lib import evaluate_model
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_objects', 'ml_model.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'scaler.pkl')
DATASET_PATH = os.path.join(os.getcwd(), 'database', 'heart.csv')
FIGURES_PATH = os.path.join(os.getcwd(), 'static')

# %% EDA
# %%% 1) Load data
df = pd.read_csv(DATASET_PATH)

# %%% 2) Inspect data
df.info()
df.describe().T

df.isna().sum()  # check for missing values
# --> No missing values for all columns

df[df.duplicated(keep=False)]  # check for all duplicates
# TODO: --> there is duplicates, proceed to clean at 3) Clean data

# %%%% Visualization
plt.figure()
df.boxplot()
plt.savefig(os.path.join(FIGURES_PATH, 'boxplot_all.png'))
"""
    from the boxplot above, there are columns that contains categorical values
    and other columns that contains continuous values
"""
# save the column names for continuous features and categorical features
con_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg',
            'exng', 'slp', 'caa', 'thall']

# %%%%% Categorical features vs output
for f in cat_cols:
    plot_features_target(df, data_type='categorical',
                         feature_name=f, save=True)

# %%%%% Continuous features vs output
for f in con_cols:
    plot_features_target(df, data_type='continuous',
                         feature_name=f, save=True)

# %%% 3) Clean data
df.drop_duplicates(inplace=True)  # drop duplicates and keep first

# %%% 4) Features Selection
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(method='kendall'), annot=True, cmap='GnBu')
plt.savefig(os.path.join(FIGURES_PATH, 'heatmap.png'))
plt.show()
"""
The heatmap shows high +ve correlation between output and cp, thalachh, slp
while there is also high -ve correlation between output and exng, oldpeak, caa,
thall

Although there are certain features that have high correlation with the target,
all features are used during training for the machine learning model.
"""

# Get features data and target data
X, y = separate_features_target(df, target_name='output')

# %%% 5) Data Preprocessing
mm_scaler = MinMaxScaler()
X = mm_scaler.fit_transform(X)
pickle.dump(mm_scaler, open(SCALER_PATH, 'wb'))

# %%%% train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=20)

# %% Machine Learning

# Create classifiers
steps_tree = [('Tree', DecisionTreeClassifier())]
steps_forest = [('Forest', RandomForestClassifier())]
steps_logistic = [('Logis', LogisticRegression(solver='liblinear'))]
steps_svc = [('SVC', SVC())]
steps_knn = [('KNN', KNeighborsClassifier())]

# Create pipeline from steps
tree_pipeline = Pipeline(steps_tree)
forest_pipeline = Pipeline(steps_forest)
logistic_pipeline = Pipeline(steps_logistic)
svc_pipeline = Pipeline(steps_svc)
knn_pipeline = Pipeline(steps_knn)

# Make list of pipelines
pipelines = [tree_pipeline, forest_pipeline, logistic_pipeline,
             svc_pipeline, knn_pipeline]

for p in pipelines:
    p.fit(x_train, y_train)

max_score = 0

# display the accuracy for each classifier
for i, p in enumerate(pipelines):
    print(p.steps[0][0]+':\t', round(p.score(x_test, y_test)*100, 2))

    # choosing the best model
    if max_score < p.score(x_test, y_test):
        max_score = p.score(x_test, y_test)
        best_pipeline = p

"""
    Most of the time, looking at the scores of these ML models,
    RandomForestClassifier has the best accuracy
"""

# %%% save ML model
pickle.dump(best_pipeline, open(MODEL_SAVE_PATH, 'wb'))

# %%% Model Evaluation
y_pred = best_pipeline.predict(x_test)
y_true = y_test

evaluate_model(best_pipeline, y_true, y_pred)
