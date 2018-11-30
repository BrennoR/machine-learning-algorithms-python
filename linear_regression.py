# Brenno Ribeiro - brenno_ribeiro@my.uri.edu
# Created on 11/25/18

import numpy as np


class LinearRegression:
    """
    Simple linear regression algorithm with ordinary least squares (OLS) method.

    Parameters
    ----------
    N/A

    Attributes
    -------
    intercept : array
        The intercept or bias term of the model.

    coefficients : array
        Array containing the coefficients of the model.

    Examples
    -------
    from linear_regression import LinearRegression

    X = [[1, 2], [3, 4], [2, 5], [-1, 9]]
    y = [4, 6, 8, 2]

    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict([[2, 1], [3, 7]])
    print(y_pred)

    Notes
    -------
    Will add many more features soon.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.weights = np.dot(np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)), y)

        self.coefficients = self.weights[1:]
        self.intercept = self.weights[0]

    def predict(self, X):
        X = np.array(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_hat = np.dot(X, self.weights)

        return y_hat
