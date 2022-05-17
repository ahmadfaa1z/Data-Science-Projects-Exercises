![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

Dataset retrieved from: [Dataset Link](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

# Heart Attack
- This project is to predict the chance of a person having heart attack.

## Python Scripts
- **train.py**
  - This script trains the machine learning model and save important settings to be used in the deploy script.
- **deploy.py**
  - This script is for predicting new data through streamlit app by accessing the model and scaler created by the train script.
- **ha_lib.py**
  - This file is a library containing functions to be used in the train and deploy scripts.

## Directories
- saved_objects
  - This is where the machine learning model and scaler are saved.
- static
  - This folder contains image files related to this Heart Attack Prediction project.
- database
  - This folder contains the dataset related to heart attack (heart.csv).

# Streamlit App
### Top View
<img src="../Heart Attack/static/app-top-view.png" width=800>

### Output View
<img src="../Heart Attack/static/app-output-view.png" width=800>

# Exploratory Data Analysis
## Missing values
<center><img src="../Heart Attack/static/missing-values-in-dataframe.jpg" width=300></center>

- No missing values in the dataset

## Duplicated samples/rows
- There are two samples with the same data in the dataset.
- One of them are drop during data cleaning

## Relationship between features & target
- Figures below shows the relationship between the each feature and the chance of getting heart attack (output)
### Categorical Features Visualization
<img src="../Heart Attack/static/count-caa-output.png" width=400>
<img src="../Heart Attack/static/count-cp-output.png" width=400>
<img src="../Heart Attack/static/count-exng-output.png" width=400>
<img src="../Heart Attack/static/count-fbs-output.png" width=400>
<img src="../Heart Attack/static/count-restecg-output.png" width=400>
<img src="../Heart Attack/static/count-sex-output.png" width=400>
<img src="../Heart Attack/static/count-slp-output.png" width=400>
<img src="../Heart Attack/static/count-thall-output.png" width=400>

### Continuous numerical features Visualization
<img src="../Heart Attack/static/histogram-age-output.png" width=400>
<img src="../Heart Attack/static/histogram-chol-output.png" width=400>
<img src="../Heart Attack/static/histogram-oldpeak-output.png" width=400>
<img src="../Heart Attack/static/histogram-thalachh-output.png" width=400>
<img src="../Heart Attack/static/histogram-trtbps-output.png" width=400>

#### Heatmap
<center><img src="../Heart Attack/static/heatmap.png" width=800></center>

- Although there are certain features that have high correlation with the target, all features are used during training for the machine learning model.

# Model performance & Reports
<center><img src="../Heart Attack/static/tested-classifiers-scores.png" width=200 ></center>

- Five classifiers are tested
<center><img src="../Heart Attack/static/reports.png" width=450 ></center>

- The accuracy scores of the classifiers might varied due to random hyperparameters during creation of classifier.
- With that said, the best classifier and the max accuracy score could also change.

## During Deployment
<center><img src="../Heart Attack/static/report-deploy.png" width=450 ></center>

- This is the report when validating model with new data