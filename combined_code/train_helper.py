from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn import linear_model

import numpy as np
import math

def RF_grid():
    # Number of trees in random forest
    n_estimators = [100,200,300]
    # Number of features to consider at every split
    max_features = [2,'auto']
    # Maximum number of levels in tree
    max_depth = [80,90,100]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    RF_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    return RF_grid

def RF_random_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    RF_random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    return RF_random_grid

def Logreg_grid():
    # penalty type
    penalty = ['l1','l2']
    # regularization strength
    C = np.logspace(-4,4,9)
    # Create the random grid
    logreg_grid = dict(C=C, penalty = penalty)
    return logreg_grid
