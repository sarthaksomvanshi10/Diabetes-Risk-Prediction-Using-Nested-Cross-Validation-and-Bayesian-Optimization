
# Diabetes Prediction Machine Learning Pipeline

This repository contains the Python implementation of a comprehensive machine learning pipeline for predicting diabetes using binary classification. The pipeline leverages advanced machine learning techniques such as nested cross-validation (nCV), Bayesian optimization for hyperparameter tuning, and feature selection to identify the best-performing classifier. The project is structured with object-oriented programming (OOP) principles, ensuring modularity and scalability.

---

## Project Overview

Diabetes is a global health issue, and early diagnosis is essential for effective management. This project utilizes a dataset of medical records with features such as glucose levels, BMI, and insulin measurements to predict the likelihood of diabetes. The pipeline evaluates multiple classifiers, including Logistic Regression, Support Vector Machines, and Gaussian Naive Bayes, across various metrics. Key steps include data preprocessing, feature selection, dimensionality reduction, and model evaluation using robust statistical methods.

---

## Main Workflow

1. **Data Preprocessing**: 
   - Handling missing values, duplicate rows, and outliers.
   - Feature scaling using MinMax normalization.
   - Optional feature selection via `SelectKBest` with mutual information scoring.

2. **Pipeline Implementation**:
   - Nested cross-validation with 5 outer and 3 inner folds.
   - Bayesian hyperparameter optimization using Optuna.
   - Metrics evaluated include Matthews Correlation Coefficient (MCC), F1 Score, Precision, Recall, and more.

3. **Final Model Selection**:
   - Training the optimal model on the entire dataset.
   - Saving the model as a `.pkl` file for future use.

---

## Results Overview

- Without feature selection, Logistic Regression achieved the best performance with a median MCC of 0.442.
- With feature selection, Support Vector Machines outperformed other classifiers, achieving a median MCC of 0.455.
- The pipeline highlights the importance of robust preprocessing and optimization for high-stakes applications like medical diagnosis.

---

## Installation and Usage

### Cloning the Repository

```sh
git clone https://github.com/GiatrasKon/Diabetes-Prediction-ML-Pipeline
```

### Package Dependencies

Ensure you have the following packages installed:

- pandas
- numpy
- joblib
- sklearn
- matplotlib
- seaborn
- optuna

Install dependencies using:

```sh
pip install pandas matplotlib seaborn numpy scikit-learn joblib optuna
```

### Repository Structure

- `nCV_class.py`: Python script implementing the `NestedCrossValidation` class.
- `exploratory_data_analysis.ipynb`: Notebook for dataset exploration and visualization.
- `nCV_implementation.ipynb`: Notebook demonstrating the usage of the nCV pipeline.
- `models/`: Directory containing the final trained model files.
- `data/`: Placeholder for the input dataset (`Diabetes.csv`).
- `documents/`: Assignment description, report and professor's feedback.
- `images/`: Images produced from the analysis and included in the assignment report.

### Usage

1. **Perform Exploratory Data Analysis**: Open and execute the `exploratory_data_analysis.ipynb` notebook to inspect the dataset. This notebook provides:
    - An overview of the dataset, including summary statistics and visualization of feature distributions.
    - Feature correlation analysis and a preliminary assessment of class separability.
    - Identification of potential issues like missing values, outliers, and class imbalance.
2. **Configure and Implement the Nested Cross-Validation Pipeline**:
    - The core pipeline logic is implemented in the `nCV_class.py` file. This script defines the `NestedCrossValidation` class, which:
        - Handles data preprocessing (e.g., normalization, outlier detection, feature selection).
        - Implements nested cross-validation with Bayesian optimization.
        - Evaluates classifiers such as Logistic Regression, SVM, and Naive Bayes across multiple metrics.
        - Selects the best-performing model and saves it for deployment.
    - To execute the pipeline, open and run the `nCV_implementation.ipynb` notebook. This notebook demonstrates:
        - How to load the dataset and initialize the `NestedCrossValidation` class.
        - Configuration of pipeline parameters such as feature selection, PCA components, and the number of cross-validation iterations.
        - Analysis of performance metrics for each classifier.
3. **Train and Save the Final Model**: After identifying the best-performing classifier during nested cross-validation, the `train_final_model` method in the `nCV_class.py` script trains the final model on the entire dataset. The model is saved as a `.pkl` file in the `models/` directory.
4. **Deploy the Trained Model**: The saved model can be loaded and used for inference on new data. Ensure that the input data follows the same preprocessing steps as defined in the pipeline.

Each file in the repository plays a distinct role:

- `exploratory_data_analysis.ipynb`: Prepares the dataset and provides insights to guide model selection.
- `nCV_class.py`: Contains the reusable pipeline class for performing nested cross-validation and training models.
- `nCV_implementation.ipynb`: Demonstrates how to use the pipeline to train and evaluate models.

Follow these steps sequentially to replicate the results or adapt the pipeline for new datasets.

---