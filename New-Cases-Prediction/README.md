![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# New Cases Prediction
## Description
This project analyse the past cases of 30 days in Malaysia to predict number of new cases using a deep learning model with LSTM neural network.

## Python Scripts
- `train.py`
  - This is the main script for training the deep learning model and analyse the model performance by getting the mean absolute percentage error and plotting the predicted and actual cases with test data.
- `ncmodule.py`
  - This module contains the functions needed for the main script to run.
## Directories
- `database`
  - This is the folder containing the dataset of both train and test data.
- `logs`
  - This folder contains the log files for tensorboard.
- `saved_objects`
  - This folder contains the saved model and the scaler object for future deployment purposes.
- `static`
  - This folder contains the image files related to this project.
## Tensorboard
- How to run tensorboard
  1. Open Anaconda promt
  2. Activate the specific environment
  3. Type `tensorboard --logdir <path>`
    - replace `<path>` with the path to the logs folder
<center><img src="../New-Cases-Prediction/static/tensorboard-view.jpeg" width=600></center>

## EDA
When the train dataset and test dataset are loaded. There are some non-numerical data and missing values that needed to be treated before passing it to the deep learning model.

### EDA steps:
1. Convert non-numerical data to missing values in new_cases column for train dataset.
2. Impute missing values with linear interpolation method
  - Plot the graph to visualize the data (as references)
3. Get train and test data with window size of 30 days to be trained to the deep learning model.

## Deep Learning Model
<center><img src="../New-Cases-Prediction/static/model-architecture.png" width=200></center>

- This is the model architecture used for this project.
- Loss and metrics used for this model are the mean squared error (MSE).
### Model Performance
<center><img src="../New-Cases-Prediction/static/tensorboard-training-graph.png" width=200></center>

- This is the training graph viewed in tensorboard.
<center><img src="../New-Cases-Prediction/static/MAPE.png" width=500></center>

- The mean absolute percentage error for the model created is less than `1%`.

## Results
<center><img src="../New-Cases-Prediction/static/pred-actual-minmax-scale.png" width=400></center>

- The above image shows the predicted and actual cases in minmax scale.
<center><img src="../New-Cases-Prediction/static/pred-actual-original-scale.png" width=400></center>

- The above image shows the predicted and actual cases in the original scale.
## References
Data retrieved from: https://github.com/MoH-Malaysia/covid19-public