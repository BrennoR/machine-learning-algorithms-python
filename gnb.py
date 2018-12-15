# Brenno Ribeiro - brenno_ribeiro@my.uri.edu
# Created on 12/13/18

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class GaussianNaiveBayes:
    """
    Simple logistic regression algorithm for binary classification.

    Parameters
    ----------
    smoothing : float, optional (default = 1e-9)
        Adds a fraction of the largest variance to each variance parameter.

    Attributes
    -------
    coefficients : array
        Array containing the weights of the model.

    Examples
    -------
    from gnb import GaussianNaiveBayes

    X = [[1, 2], [3, 4], [2, 5], [-1, 9]]
    y = [0, 1, 1, 0]

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    y_pred = gnb.predict([[2, 1], [3, 7]])
    print(y_pred)

    Notes
    -------
    Algorithm completely based off of Dr. Alvarez code shown in CSC 461
    at URI for the Gaussian Naive Bayes section of the course. Must
    make some edits to the code stilL!

    """

    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.epsilon = self.smoothing * np.var(X, axis=0).max()
        self.labels = np.unique(y)
        self.n_classes = len(self.labels)
        self.means = np.zeros((self.n_classes, X.shape[1]))
        self.variances = np.zeros((self.n_classes, X.shape[1]))
        self.priors = np.zeros(self.n_classes)
        for i, j in np.ndenumerate(self.labels):
            rows = X[j==y, :]
            self.means[i, :] = np.mean(rows, axis=0)
            self.variances[i, :] = np.var(rows, axis=0)
            self.priors[i] = len(rows) / X.shape[0]
        self.variances += self.epsilon

    def predict(self, X):
        probabilities = np.zeros((self.n_classes, X.shape[0]))
        for i, j in np.ndenumerate(self.labels):
            probabilities[i, :] = np.log(self.priors[i])
            probabilities[i, :] -= (0.5 * np.sum(np.log(2 * np.pi * self.variances[i])))
            probabilities[i, :] -= (0.5 * np.sum((((X - self.means[i]) ** 2) / self.variances[i]), axis=1))
        return self.labels[np.argmax(probabilities, axis=0)]

    @staticmethod
    def gauss(mean, var, x):
        prob = np.sum(np.log((1 / np.sqrt(2 * np.pi * var)) * np.exp((-(x - mean) ** 2) / (2 * var))))
        return prob
