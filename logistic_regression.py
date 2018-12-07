# Brenno Ribeiro - brenno_ribeiro@my.uri.edu
# Created on 12/05/18

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LogisticRegression:
    """
    Simple logistic regression algorithm for binary classification.

    Parameters
    ----------
    iterations : int, optional (default = 1000)
        Number of iterations in gradient descent.

    lr : float, optional (default = 0.01)
        Learning rate of the gradient descent optimizer.

    regularization : boolean, optional (default = False)
        Regularization is added to gradient descent when this parameter
        is set to true. Lambda values must be tuned in order for
        regularization to take place.

    lam : float, optional (default = 0)
        Lambda value used in regularization. This parameter is only used
        when regularization is set to true. The default value is set to
        zero.

    normalize: boolean, optional (default = True)
        Normalizes the data when set to true. This parameter is set to
        true by default as normalizing the data before using logistic
        regression is highly recommended.

    Attributes
    -------
    coefficients : array
        Array containing the weights of the model.

    Examples
    -------
    from logistic_regression import LogisticRegression

    X = [[1, 2], [3, 4], [2, 5], [-1, 9]]
    y = [0, 1, 1, 0]

    lr = LogisticRegression()
    lr.fit(X, y)
    y_pred = lr.predict([[2, 1], [3, 7]])
    print(y_pred)

    Notes
    -------
    This algorithm only works on binary classification of classes 0 and 1 at
    the moment. A more general multiclass structure will be added later on.
    Different versions of gradient descent will also be added.

    """

    def __init__(self, iterations=1000, lr=0.01, regularization=False, lam=0, normalize=True):
        self.iterations = iterations
        self.learning_rate = lr
        if regularization:
            self.lam = lam
        else:
            self.lam = 0
        self.normalize = normalize
        if normalize:
            self.min_max_scaler = MinMaxScaler()

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        X = self.normalize_data(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.weights = np.zeros(X.shape[1])

        # Batch gradient descent
        for i in range(self.iterations):
            f = np.dot(X, self.weights)
            grad = np.dot(X.T, (y - self.sigmoid(f)))
            self.weights += self.learning_rate * grad - self.learning_rate * self.lam * self.weights

        self.coefficients = self.weights

    def predict(self, X):
        X = np.array(X)
        X = self.normalize_data(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_hat = []
        for l in range(len(X)):
            if np.dot(X[l], self.weights) > 0:
                y_hat.append(1)
            else:
                y_hat.append(0)

        return y_hat

    @staticmethod
    def sigmoid(s):
        value = 1 / (1 + np.exp(-s))
        return value

    def normalize_data(self, x):
        if self.normalize:
            return self.min_max_scaler.fit_transform(x)
        else:
            return x
