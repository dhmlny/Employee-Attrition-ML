# Employee Attrition Prediction & Clustering

An end-to-end Machine Learning pipeline built to predict employee attrition and perform workforce behavioral segmentation using a dataset of 15,000 records.

## Repository Contents
*   `Employee_Attrition_Clean.ipynb`: Jupyter Notebook containing data preprocessing, clustering, and predictive modeling steps.

## Pipeline Overview
1. **Preprocessing & Scaling**: Applied one-hot encoding for categorical department attributes and mapped salary levels. Fitted the `StandardScaler` exclusively on the training subset to prevent Data Leakage.
2. **K-Means Clustering & PCA**: Grouped employees into 3 behavioral segments using K-Means (optimized via the Elbow Method) and projected the high-dimensional feature space to 2D using PCA.
3. **Supervised Classification**: Trained and compared a Multi-Layer Perceptron (MLP) Neural Network against a Gradient Boosting Classifier (GBM).

## Results
Evaluation metrics on the 25% test set:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (GBM)** | **97.65%** | **96.32%** | **93.73%** | **95.01%** |
| **MLP (Neural Network)** | **96.85%** | **93.39%** | **93.39%** | **93.39%** |

## How to Run
Open `Employee_Attrition_Clean.ipynb` in Jupyter or Google Colab, upload the dataset CSV when prompted, and execute the cells.

## Tech Stack
Python, Scikit-Learn, Pandas, NumPy, MLP, Gradient Boosting, K-Means, PCA, Matplotlib, Seaborn.
