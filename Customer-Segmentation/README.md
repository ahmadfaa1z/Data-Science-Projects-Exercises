![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# Customer Segmentation
## Description
- This is a multi-class classification where customers are classified into 4 segments 
(A, B, C, D) based on their customer’s gender, age, profession, spending pattern, and etc.

## Python Scripts
- **train.py**
  - This script trains the machine learning model and save important settings to be used in the deploy script.
- **deploy.py**
  - This script is for predicting new data through streamlit app by accessing the model and scaler created by the train script.
- **ha_lib.py**
  - This file is a library containing functions to be used in the train and deploy scripts.

## Directories
- *saved_objects*
  - This is where the deep learning model and other objects are saved such as the encoders, imputer and scaler.
- *static*
  - This folder contains image files related to this Customer Segmentation project.
- *database*
  - This folder contains the dataset related to the project.
  - **train.csv**
    - This is the data for training the model
  - **new_customers.csv**
    - This is the test data to get the label for customer's segmentation (class label)
  - **new_customers_results.csv**
    - This is the test data with customer's segmentation target label

## How to run Tensorboard
- To run Tensorboard,
  1. Open Anaconda promt
  2. Activate the specific environment
  4. Type `tensorboard --logdir <path>`
    - replace `<path>` with the path to the logs folder

## References
Dataset retrieved from: [Dataset Link](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation)