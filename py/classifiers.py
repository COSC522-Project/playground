"""
This module contains functions to speed up the classification process. It 
also contains an implementation of the maximum posterior probability (MPP) 
classifier

TO DO
-----
* We can add an option to `classify()` to specify which kind of 
  dimensionality reduction algorithm to run. 
"""

import sys
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from . import util


def classify(X_train, y_train, X_test, method='mpp', **kwargs):
    """Train and predict using specified classifier.
    
    Parameters
    ----------
    X_train : NumPy array
        The training feature matrix, shape (n_train_samples, n_classes).
    y_train : NumPy array
        The training feature labels, shape (n_train_samples,)
    X_test : NumPy array
        The testing feature matrix, shape (n_test_samples, n_classes).
    method : str
        The classification method to use. Options are ['mpp', 'knn', 
        'neural_network', 'decision_tree', 'svm', 'kmeans']. All methods use
        the sklearn library except 'mpp'.
    kwargs : dict
        Key word arguments for the classifier constructor.
        
    Returns
    -------
    y_pred : NumPy array, shape (n,)
        Class labels for each testing point.
    """
    classifiers = {
        'mpp': MPP,
        'knn': KNeighborsClassifier,
        'neural_network': MLPClassifier,
        'decision_tree': DecisionTreeClassifier,
        'svm': SVC,
        'kmeans': KMeans
    }
    clf = classifiers[method](**kwargs)
    t0 = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, time.time() - t0


def mpp(Tr, yTr, Te, cases, P):
    """Maximum posterior probability classifier -- function implementation."""
    # training process - derive the model
    covs, means = {}, {}     # dictionaries
    covsum = None

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)    # number of classes
    
    for c in classes:
        # filter out samples for the c^th class
        arr = Tr[yTr == c]  
        # calculate statistics
        covs[c] = np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis=0)  # mean along the columns
        # accumulate the covariance matrices for Case 1 and Case 2
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]
    
    # used by case 2
    covavg = covsum / classn
    # used by case 1
    varavg = np.sum(np.diagonal(covavg)) / classn
            
    # testing process - apply the learned model on test set 
    disc = np.zeros(classn)
    nr, _ = Te.shape
    y = np.zeros(nr)            # to hold labels assigned from the learned model

    for i in range(nr):
        for c in classes:
            if cases == 1:
                edist2 = util.euc2(means[c], Te[i])
                disc[c] = -edist2 / (2 * varavg) + np.log(P[c] + 0.000001)
            elif cases == 2: 
                mdist2 = util.mah2(means[c], Te[i], covavg)
                disc[c] = -mdist2 / 2 + np.log(P[c] + 0.000001)
            elif cases == 3:
                mdist2 = util.mah2(means[c], Te[i], covs[c])
                disc[c] = -mdist2 / 2 - np.log(np.linalg.det(covs[c])) / 2 + np.log(P[c] + 0.000001)
            else:
                print("Can only handle case numbers 1, 2, 3.")
                sys.exit(1)
        y[i] = disc.argmax()
            
    return y


class MPP:
    """Maximum posterior probability classifier.

    The classifier maximizes the posterior probability given by 
    P(i|x) = p(x|i) * P(i) / p(x), where i is the class number, x is a vector,
    P(i) is the prior probability of class i, p(x) is the probability 
    distribution for x, and p(x|i) is the probability distribution of x given 
    that it belongs to class i. A unimodal gaussian model is assumed for 
    p(x|i). 
    
    Attributes
    ----------
    case : int
        Specifies assumptions about the within-class covariance matrices.
        1) Sigma_i == Sigma_j == v*I, where v is the average of the 
           within-class variance along each dimension over all the classes and
           I is the identity matrix.
        2) Sigma_i == Sigma_j = S, where S is the element-wise sum of the 
           within-class class covariance matrices divided by the number of
           classes.
        3) Sigma_i != Sigma_j. In this case the covariance matrices are used
           directly.
    means : list
        List of mean vectors for each class.
    covs : list
        List of covariance matrices for each class.
    cov_avg : NumPy array, shape (nclasses, nclasses)
        The element-wise sum of the within-class class covariance matrices 
        divided by the number of classes.
    cov_avg : NumPy array, shape (nclasses, nclasses)
        The average of the diagonal elements of `cov_avg`.
    classes : list
        List of integer class labels
    """
    
    def __init__(self, case=1):
        self.case = case
        self.means, self.covs = [], []
        self.cov_avg = self.var_avg = None
        self.classes = None
        
    def fit(self, X, y, P=None):
        """Fit the classier to the training data.
        
        Parameters
        ----------
        X : NumPy array, shape (n, d)
            Array of n training points in d-dimensional space.
        y : NumPy array, shape (n,)
            Integer class labels for each training point.
        P : array-like
            List of prior probabilities for each class. If not provided, the
            prior probabilities are taken to be equal.
            
        Returns
        -------
        None. Modifies the class attributes.
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.P = np.ones(n_classes)/n_classes if P is None else P
        for c in self.classes:
            _X = X[y == c]  
            self.covs.append(np.cov(_X.T))
            self.means.append(np.mean(_X, axis=0))
        if len(self.covs) > 0:
            self.cov_avg = np.sum(np.array(self.covs), axis=0) / n_classes
            self.var_avg = np.sum(np.diagonal(self.cov_avg)) / n_classes
            
    def predict(self, X_test):
        """Make predictions on new data.
        
        Parameters
        ----------
        X_test : NumPy array, shape (n, d)
            Array of n testing points in d-dimensional space.
            
        Returns
        -------
        y_pred : NumPy array, shape (n,)
            Class labels for each testing point.
        """
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            g = np.zeros(len(self.classes))
            for c in self.classes:
                if self.case == 1:
                    edist2 = util.euc2(self.means[c], X_test[i])
                    g[c] = -edist2 / (2 * self.var_avg) + np.log(self.P[c] + 0.000001)
                elif self.case == 2: 
                    mdist2 = util.mah2(self.means[c], X_test[i], self.cov_avg)
                    g[c] = -mdist2 / 2 + np.log(self.P[c] + 0.000001)
                elif self.case == 3:
                    mdist2 = util.mah2(self.means[c], X_test[i], self.covs[c])
                    g[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs[c])) / 2 + np.log(self.P[c] + 0.000001)
                else:
                    print("Case number must be 1, 2, or 3.")
                    sys.exit(1)
            y_pred[i] = g.argmax()
        return y_pred
