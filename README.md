# Employee Attrition Prediction & Clustering

An end-to-end Python pipeline to analyze and predict employee attrition using a dataset of 15,000 records. It combines unsupervised clustering for profile analysis and supervised learning for prediction.

## Repository Structure
*   `Employee_Attrition_Clean.ipynb`: Clean Jupyter Notebook implementing the pipeline.
*   `educational_website/`: Interactive web dashboard simulating the models and backpropagation.

## Pipeline Overview
1. **Preprocessing**: Handled categorical encoding and standardized features exclusively on the training split to avoid data leakage.
2. **Clustering & PCA**: Applied K-Means (K=3) to identify high-risk employee profiles (e.g., overworked stars with a 50.1% churn rate) and visualized them using 2D PCA projection.
3. **Modeling**: Trained and compared a Multi-Layer Perceptron (MLP) Neural Network against a Gradient Boosting Classifier (GBM).

## Results
Evaluation metrics on the 25% test set:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (GBM)** | **97.65%** | **96.32%** | **93.73%** | **95.01%** |
| **MLP (Neural Network)** | **96.85%** | **93.39%** | **93.39%** | **93.39%** |

## How to Run
1. **Notebook**: Open `Employee_Attrition_Clean.ipynb` in Jupyter or Google Colab and run the cells.
2. **Dashboard**: Open `educational_website/index.html` in any web browser to view the interactive simulations.

## Tech Stack
Python, Scikit-Learn, Pandas, NumPy, MLP, Gradient Boosting, K-Means, PCA, Chart.js.
