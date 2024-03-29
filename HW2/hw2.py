
"""
A coding assignment that combines splitting data into train, test, and validation sets using scikit-learn (sklearn):

Problem Statement:

You are given a dataset containing information about different types of iris flowers (sepal length, sepal width,
petal length, and petal width). Your task is to build a machine learning model to classify the iris flowers into
three different classes (setosa, versicolor, virginica) based on their physical characteristics.

You are to do the following:
Step 1: Load the iris dataset
    Use the following code to load the iris dataset into a pandas dataframe:

Step 2: Split the data into training, validation, and test sets
    Use the following code to split the data into training (60%), validation (20%), and test (20%) sets:

Step 3: Train a classifier (this function is provided for you)
    Use the following code to train a support vector machine (SVM) classifier on the training set:

Step 4: Evaluate the model on the validation set
    Use the following code to evaluate the performance of the trained classifier on the validation set:
    from sklearn.metrics import accuracy_score

Step 5: Evaluate the model on the test set
    Use the following code to evaluate the performance of the trained classifier on the test set:

This assignment is designed to help you understand the basics of splitting data into train, validation, and test sets,
as well as training a simple machine learning model using scikit-learn. The assignment can be extended by trying
different classifiers, tuning the parameters, or exploring the dataset further.

Step 6: Then, do the same for K-Fold cross validation.
"""
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_data():
    """
    Load the iris dataset and return it as a pandas dataframe

    Returns:
        df_out (pandas dataframe): The iris dataset
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    irisData = iris.data
    irisTarget = iris.target
    irisTarget2 = ["i"]*len(irisTarget)
    for x in range(len(irisTarget-1)):
        if irisTarget[x] == 0:
            irisTarget2[x] = "setosa"
            
        if irisTarget[x] == 1:
            irisTarget2[x] = "versicolor"
            
        if irisTarget[x] == 2:
            irisTarget2[x] = "virginica"
    
    columName = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']    
    
    #df = pd.DataFrame(data=irisData, columns=columName)
    
    df_out = pd.DataFrame(data=irisData, columns=columName)
    df_out['target'] = irisTarget
    
    return df_out
    



def split_data(df_in):
    """
    Split the data into training, validation, and test sets

    Parameters:
    df_in (pandas dataframe): The iris dataset

    Returns:
    X_train_out (numpy array): Training set features
    X_val_out (numpy array): Validation set features
    X_test_out (numpy array): Test set features
    y_train_out (numpy array): Training set targets
    y_val_out (numpy array): Validation set targets
    y_test_out (numpy array): Test set targets
    """
    x = df_in.iloc[:,:-1] #irisData
    y = df_in.iloc[:,-1] #irisTarget
    
    x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(x, y, train_size=0.6)
    x_val_out, x_test_out, y_val_out, y_test_out = train_test_split(x_test_out, y_test_out, test_size=0.5)
    
    return x_train_out, x_val_out, x_test_out, y_train_out, y_val_out, y_test_out
    
    

def train_classifier(X_train_in, y_train_in):
    """
    Train a support vector machine (SVM) classifier on the training set

    Parameters:
    X_train_in (numpy array): Training set features
    y_train_in (numpy array): Training set targets

    Returns:
    clf_out (SVM classifier): Trained SVM classifier
    """
    clf_out = SVC(kernel='linear', C=1, random_state=0)
    clf_out.fit(X_train_in, y_train_in)

    return clf_out


def evaluate_model(clf, X, y):
    """
    Evaluate the performance of the trained classifier on a given set

    Parameters:
    clf (SVM classifier): Trained SVM classifier
    X (numpy array): Features of the set to evaluate the classifier on
    y (numpy array): Targets of the set to evaluate the classifier on

    Returns:
    accuracy_out (float): Accuracy of the classifier on the given set
    """
    predict_y = clf.predict(X)
    accuracy_out  = accuracy_score(y, predict_y)
    return accuracy_out
    


def k_fold_cross_validation(df, k=5):
    """
    Perform k-fold cross validation on the iris dataset

    Parameters:
    df (pandas dataframe): The iris dataset
    k (int, optional): The number of folds. Default is 5.

    Returns:
    accuracy_out (list): A list of accuracy_out scores for each fold
    """
    # TODO do 5-fold on dataset and return the accuracy across the five folds
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    k_folds = KFold(n_splits=k, random_state=None)
    clf  = LogisticRegression(solver='liblinear')

    accuracy_out = []
    
    for train, test in k_folds.split(x):
        x_train , x_test = x.iloc[train,:], x.iloc[test,:]
        y_train , y_test = y[train], y[test]
        
        clf.fit(x_train, y_train)
        predict_y = clf.predict(x_test)
        
        acc = accuracy_score(y_test, predict_y)
        accuracy_out.append(acc)
        
    return accuracy_out

if __name__ == '__main__':
    # Load the iris dataset
    df_data = load_data()

    # Train, Val, Test
    # Split the data into training, validation, and test sets
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_data)

    # Train a classifier
    
    clf = train_classifier(X_train, y_train)

    # Evaluate the model on the validation set
    accuracy_val = evaluate_model(clf, X_val, y_val)
    print("Accuracy on validation set:", accuracy_val)

    # Evaluate the model on the test set
    accuracy_test = evaluate_model(clf, X_test, y_test)
    print("Accuracy on test set:", accuracy_test)

    # K-FOLD
    # Perform k-fold cross validation
    accuracy = k_fold_cross_validation(df_data)

    # Print the accuracy scores for each fold
    for i, acc in enumerate(accuracy):
        print("Accuracy on fold", i + 1, ":", acc)
