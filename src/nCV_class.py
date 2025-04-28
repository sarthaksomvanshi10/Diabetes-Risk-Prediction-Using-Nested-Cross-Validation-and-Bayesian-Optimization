# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, fbeta_score, recall_score, precision_score, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import optuna
import joblib

class NestedCrossValidation:
    def __init__(self, csv_file, outer_folds=5, inner_folds=3, n_iterations=10, random_state=42, feature_selection=False, n_components=3):
        """
        Initializing the NestedCrossValidation class.

        Parameters:
        - csv_file: Path to the CSV file containing the dataset.
        - outer_folds: Number of folds for the outer loop of cross-validation (default: 5).
        - inner_folds: Number of folds for the inner loop of cross-validation (default: 3).
        - n_iterations: Number of iterations for nested cross-validation (default: 10).
        - random_state: Random seed for reproducibility (default: 42).
        - feature_selection: Boolean indicating whether to perform feature selection (default: False).
        - n_components: Number of PCA components to retain, if any (default: 3).
        """
        self.csv_file = csv_file
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.feature_selection = feature_selection
        self.n_components = n_components

        # Defining classifiers with class_weight parameter where applicable
        self.classifiers = {
            'LogisticRegression': LogisticRegression(class_weight='balanced'),
            'GaussianNaiveBayes': GaussianNB(),
            'k-NearestNeighbors': KNeighborsClassifier(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'SupportVectorMachine': SVC(probability=True, class_weight='balanced')  # SVC with probability=True for precision-recall curve
        }

        # Defining hyperparameter spaces
        self.param_spaces = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'GaussianNaiveBayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            'k-NearestNeighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'LinearDiscriminantAnalysis': {
                'solver': ['lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            },
            'SupportVectorMachine': {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        }

        # Initializing data structures to store results
        self.results = {clf_name: [] for clf_name in self.classifiers.keys()}
        
        # Generating a list of seeds for reproducibility
        np.random.seed(self.random_state)
        self.iteration_seeds = np.random.randint(0, 10000, size=self.n_iterations)
    
    def initial_preprocessing(self):
        """
        Performing initial preprocessing on the dataset.

        Returns:
            tuple: A tuple containing the preprocessed X and y.
        """
        df = pd.read_csv(self.csv_file) # reading the CSV file into a DataFrame

        df.set_index('ID', inplace=True, drop=True) # setting the 'ID' column as the index
        df.drop_duplicates(inplace=True) # removing duplicate rows
        df.fillna(df.median(), inplace=True) # replacing missing values with the median

        # Replacing 0 values with the median of the respective columns
        columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_with_zeros:
            df[column] = df[column].replace(0, df[column].median())

        def detect_outliers(df):
            """
            Detecting outliers in a DataFrame by calculating the interquartile range (IQR) for each column
            and identifying values that fall outside of the IQR bounds.

            Parameters:
                df (pandas.DataFrame): The DataFrame to detect outliers in.

            Returns:
                dict: A dictionary where the keys are the column names and the values are the indices of the outliers.
            """
            outliers = {} # initializing an empty dictionary to store the indices of the outliers
            # Calculating the interquartile range (IQR) for each column
            for column in df.columns:
                if column != 'Outcome': # excluding the target variable
                    Q1 = df[column].quantile(0.25) # identifying the first quartile
                    Q3 = df[column].quantile(0.75) # identifying the third quartile
                    IQR = Q3 - Q1 # calculating the interquartile range
                    lower_bound = Q1 - 1.5 * IQR # calculating the lower bound
                    upper_bound = Q3 + 1.5 * IQR # calculating the upper bound
                    outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index # storing the indices of the outliers
            return outliers # returning the dictionary of outliers

        outliers = detect_outliers(df) # calling the detect_outliers function and storing the result in the 'outliers' variable
        outlier_indices = np.unique([index for indices in outliers.values() for index in indices]) # extracting the unique indices from the 'outliers' dictionary
        df_no_outliers = df.drop(index=outlier_indices) # removing the outliers from the original DataFrame
        
        original_balance = df['Outcome'].value_counts(normalize=True)[1] # calculating the original balance of the target variable
        new_balance = df_no_outliers['Outcome'].value_counts(normalize=True)[1] # calculating the new balance of the target variable
        
        # If the new balance is closer to 50% than the original balance, use the new DataFrame without outliers
        if abs(new_balance - 0.5) < abs(original_balance - 0.5):
            df = df_no_outliers

        X = df.drop(columns=['Outcome']).values # extracting the features from the DataFrame
        y = df['Outcome'].values # extracting the target variable from the DataFrame

        return X, y # returning the preprocessed X and y

    def preprocess_data(self, X_train, X_test, y_train, feature_names):
        """
        Normalizing the features and applying feature selection and PCA.

        Parameters:
        - X_train: Training features.
        - X_test: Test features.
        - y_train: Training labels.
        - feature_names: List of feature names.

        Returns:
        - X_train_scaled: Preprocessed training features.
        - X_test_scaled: Preprocessed test features.
        """
        scaler = MinMaxScaler() # creating an instance of the MinMaxScaler class
        X_train_scaled = scaler.fit_transform(X_train) # fitting the scaler to the training data
        X_test_scaled = scaler.transform(X_test) # transforming the test data using the fitted scaler

        # Applying feature selection
        if self.feature_selection:
            selector = SelectKBest(mutual_info_classif, k=5) # creating an instance of the SelectKBest class with mutual_info_classif as the score function and k=5 as the number of features to select
            X_train_scaled = selector.fit_transform(X_train_scaled, y_train) # fitting the selector to the training data
            X_test_scaled = selector.transform(X_test_scaled) # transforming the test data using the fitted selector

            # Getting the selected feature names
            selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
            print("Selected features:", selected_features)

        # Applying PCA
        if self.n_components:
            pca = PCA(n_components=self.n_components) # creating an instance of the PCA class
            X_train_scaled = pca.fit_transform(X_train_scaled) # fitting the PCA to the training data
            X_test_scaled = pca.transform(X_test_scaled) # transforming the test data using the fitted PCA

        return X_train_scaled, X_test_scaled # returning the preprocessed X_train_scaled and X_test_scaled features

    def nested_cross_validation(self):
        """
        Performing nested cross-validation on the dataset using the specified classifiers and hyperparameter spaces.
        This function iterates over a specified number of iterations and performs cross-validation using StratifiedKFold with the specified number of outer and inner folds. 
        For each outer fold, the data is split into training and test sets. 
        The training data is then preprocessed using the preprocess_data method. 
        For each classifier specified in the classifiers dictionary, the objective function is called to optimize the hyperparameters using Optuna. 
        The best hyperparameters are then used to train the classifier on the training data and make predictions on the test data. 
        The calculated metrics are stored in the results dictionary.
        """
        # For each iteration of nested cross-validation
        for iteration in range(1, self.n_iterations + 1):
            iteration_seed = self.iteration_seeds[iteration - 1] # getting the random seed for the current iteration
            print(f"Iteration {iteration}/{self.n_iterations} with seed {iteration_seed}") # printing the current iteration and the random seed
            X, y = self.initial_preprocessing() # calling the initial_preprocessing function and storing the result in the 'X' and 'y' variables

            # Getting feature names from the dataframe columns
            df = pd.read_csv(self.csv_file)
            feature_names = df.drop(columns=['ID', 'Outcome']).columns.tolist() # getting the list of feature names

            outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=iteration_seed) # creating an instance of the StratifiedKFold class for the outer loop of cross-validation
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=iteration_seed) # creating an instance of the StratifiedKFold class for the inner loop of cross-validation

            # For each outer fold of cross-validation
            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
                print(f"  Outer Fold {outer_fold}/{self.outer_folds}") # printing the current outer fold
                X_train, X_test = X[train_idx], X[test_idx] # splitting the data into training and test sets
                y_train, y_test = y[train_idx], y[test_idx] # splitting the labels into training and test sets

                X_train, X_test = self.preprocess_data(X_train, X_test, y_train, feature_names) # preprocessing the data

                # For each classifier in the classifiers dictionary
                for clf_name, clf in self.classifiers.items():
                    print(f"    Evaluating {clf_name}") # printing the name of the current classifier
                    def objective(trial):
                        """
                        Optimizes hyperparameters using Optuna for the specified classifier and returns the mean score.
                        """
                        params = {param: trial.suggest_categorical(param, values) if isinstance(values, list) else trial.suggest_float(param, *values) for param, values in self.param_spaces[clf_name].items()} # getting the hyperparameter space for the current classifier
                        clf.set_params(**params) # setting the hyperparameters for the current classifier
                        scores = [] # initializing an empty list to store the scores
                        # For each inner fold of cross-validation
                        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train), 1):
                            # print(f"      Inner Fold {inner_fold}/{self.inner_folds} for {clf_name}") # printing the current inner fold
                            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx] # splitting the data into training and validation sets
                            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx] # splitting the labels into training and validation sets
                            clf.fit(X_inner_train, y_inner_train) # fitting the classifier to the training data
                            y_pred = clf.predict(X_inner_val) # making predictions on the validation data
                            scores.append(matthews_corrcoef(y_inner_val, y_pred)) # appending the score to the scores list
                        return np.mean(scores) # returning the mean score

                    study = optuna.create_study(direction='maximize') # creating an instance of the Optuna study class for hyperparameter optimization
                    study.optimize(objective, n_trials=50) # optimizing the hyperparameters using Optuna for the specified classifier and number of trials
                    best_params = study.best_params # getting the best parameters

                    print(f"    Best params for {clf_name}: {best_params}") # printing the best parameters

                    clf.set_params(**best_params) # setting the best parameters for the current classifier
                    clf.fit(X_train, y_train) # fitting the classifier to the training data
                    y_pred = clf.predict(X_test) # making predictions on the test data

                    metrics = self.calculate_metrics(y_test, y_pred, clf, X_test, y_test) # calculating the metrics for the current classifier
                    self.results[clf_name].append(metrics) # appending the metrics to the results dictionary
    
    def calculate_metrics(self, y_true, y_pred, clf=None, X_test=None, y_test=None):
        """
        Calculating various performance metrics.

        Parameters:
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.
        - clf: Trained classifier, required for precision-recall curve.
        - X_test: Test features, required for precision-recall curve.
        - y_test: Test labels, required for precision-recall curve.

        Returns:
        - metrics: A dictionary containing the calculated metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # calculating the true negatives, false positives, false negatives, and true positives
        
        # Calculating various performance metrics based on the confusion matrix
        metrics = {
            'mcc': matthews_corrcoef(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'f2': fbeta_score(y_true, y_pred, beta=2),
            'recall': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp),
            'precision': precision_score(y_true, y_pred),
            'average_precision': average_precision_score(y_true, y_pred),
            'negative_predictive_value': tn / (tn + fn)
        }
        
        # Calculating the precision-recall curve if the classifier, test features, and test labels are provided
        if clf is not None and X_test is not None and y_test is not None:
            y_prob = clf.predict_proba(X_test)[:, 1] # getting the predicted probabilities for the test data
            precision, recall, _ = precision_recall_curve(y_test, y_prob) # calculating the precision and recall for the test data
            metrics['precision_recall_curve'] = (precision, recall) # appending the precision and recall to the metrics dictionary

        return metrics # returning the calculated metrics dictionary

    def train_final_model(self, best_clf_name):
        """
        Train the final model using the best classifier and save it.

        Parameters:
        - best_clf_name: The name of the best classifier.
        """
        # Performing initial preprocessing
        X, y = self.initial_preprocessing()

        # Getting feature names from the dataframe columns
        df = pd.read_csv(self.csv_file)
        feature_names = df.drop(columns=['ID', 'Outcome']).columns.tolist() # excluding the 'ID' and 'Outcome' columns

        # Initializing the best classifier and its parameter space
        clf = self.classifiers[best_clf_name]
        param_space = self.param_spaces[best_clf_name]

        # Creating an Optuna study for hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        
        def objective(trial):
            """
            Optimizes hyperparameters using Optuna for the specified classifier and returns the mean score.
            """
            # Suggesting hyperparameters for the trial
            params = {param: trial.suggest_categorical(param, values) if isinstance(values, list) else trial.suggest_float(param, *values) for param, values in param_space.items()} # using the param_space dictionary to suggest hyperparameters
            clf.set_params(**params) # setting the hyperparameters for the current classifier
            
            scores = [] # initializing an empty list to store the scores
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) # creating a StratifiedKFold object for cross-validation
            
            # For each fold of cross-validation
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx] # splitting the data into training and validation sets
                y_train, y_val = y[train_idx], y[val_idx] # splitting the labels into training and validation sets
                
                # Preprocessing the training and validation data
                X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val, y_train, feature_names)
                
                # Fitting the model and predicting on the validation set
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_val_scaled)
                scores.append(matthews_corrcoef(y_val, y_pred)) # appending the score to the scores list
            
            return np.mean(scores) # returning the mean score for the current trial

        # Optimizing the study
        study.optimize(objective, n_trials=50) # running 50 trials of optimization
        best_params = study.best_params # getting the best parameters

        # Setting the best parameters and fit the model on the entire dataset
        clf.set_params(**best_params) # setting the best parameters for the current classifier
        X_scaled, _ = self.preprocess_data(X, X, y, feature_names)  # applying the same preprocessing as before to the entire dataset
        clf.fit(X_scaled, y) # fitting the model on the entire dataset

        # Generating the model filename
        feature_selection_str = "FS" if self.feature_selection else "noFS" # feature selection string (no feature selection or feature selection)
        pca_str = f"PCA{self.n_components}" if self.n_components else "noPCA" # PCA string (number of components or no PCA)
        model_path = f"../models/{best_clf_name}_{feature_selection_str}_{pca_str}_final_model.pkl" # model filename

        # Saving the model
        joblib.dump(clf, model_path) #
        print(f"Final model saved to {model_path}") # printing the model filename