# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:02:56 2022

@author: Ahmad Faaiz
"""

# %% Imports & Constants
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SCALER_PATH = os.path.join(os.getcwd(), 'saved_objects', 'mm_scaler_ml.pkl')
PIPELINE_PATH = os.path.join(os.getcwd(), 'saved_objects', 'ml_model.pkl')
DATASET_PATH = os.path.join(os.getcwd(), 'database', 'diabetes.csv')

# %% EDA
# %%% Load data
df = pd.read_csv(DATASET_PATH)

# %%% Inspect data
df.info()
df.isna().sum()

df.describe().T['min']
# --> 6 features have zero as their min

# %%% Clean data
# %%%% Drop duplicates
df.drop_duplicates(inplace=True)

# %%%% impute nan values
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed, columns=df.columns)

# %%%% impute zero values
# check frequency of zeros in the 6 features
for feature in df.columns[:6]:
    zero_count = df_imputed[df_imputed[feature] == 0][feature].count()
    print('{}: have {} zeros, about {:.2f}% frequency'
          .format(feature, zero_count, 100*zero_count/len(df_imputed)))

# replace zeros to respective mean of the 6 features except pregnancies
for feature in df.columns[1:6]:
    feature_median = df_imputed[feature].median()
    if df[feature].dtype == 'int64':
        df_imputed[feature] = df_imputed[feature]\
            .replace(0, round(feature_median))
    else:
        df_imputed[feature] = df_imputed[feature]\
            .replace(0, feature_median)

# %%% Features selection
# %%%% heatmap
plt.figure()
sns.heatmap(df_imputed.corr(method='kendall'), annot=True, cmap=plt.cm.Reds)
plt.show()

# %%% Data preprocessing
X = df_imputed.drop(labels=['Outcome'], axis=1).to_numpy()
y = df_imputed['Outcome'].values

# %%%% minmax scalling
mm_scaler = MinMaxScaler()
X = mm_scaler.fit_transform(X)
pickle.dump(mm_scaler, open(SCALER_PATH, 'wb'))

# %%%% train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=20)

# %% Machine Learning
steps_tree = [('Tree', DecisionTreeClassifier())]
steps_forest = [('Forest', RandomForestClassifier())]
steps_logistic = [('Logis', LogisticRegression(solver='liblinear'))]
steps_svc = [('SVC', SVC())]
steps_knn = [('KNN', KNeighborsClassifier())]

tree_pipeline = Pipeline(steps_tree)
forest_pipeline = Pipeline(steps_forest)
logistic_pipeline = Pipeline(steps_logistic)
svc_pipeline = Pipeline(steps_svc)
knn_pipeline = Pipeline(steps_knn)

pipelines = [tree_pipeline, forest_pipeline, logistic_pipeline,
             svc_pipeline, knn_pipeline]

for p in pipelines:
    p.fit(X_train, y_train)

print('Without PCA:')
for i, p in enumerate(pipelines):
    print(p.steps[0][0]+':\t', p.score(X_test, y_test))

# %%% save ML model
pickle.dump(svc_pipeline, open(PIPELINE_PATH, 'wb'))
