# Brenno Ribeiro - brenno_ribeiro@my.uri.edu
# Created on 11/30/18

import numpy as np


class Perceptron:
    """
    Basic perceptron learning algorithm.

    Parameters
    ----------
    iterations : int
        Maximum number of iterations the perceptron passes through the data.

    error_threshold : int, optional (default = 0)
        Target error_threshold, the algorithm stops automatically when this
        threshold is reached.

    verbose : boolean, optional (default = False)
        When this value is set to True, the number of iterations and points
        misclassified will be printed after fit().

    Attributes
    -------
    num_iterations : int
        Number of iterations the algorithm ran.

    num_misclassified : int
        Number of points misclassified when fitting the data.

    Examples
    -------
    from perceptron import Perceptron

    X = [[1, 2], [3, 4], [2, 5], [-1, 9]]
    y = [0, 1, 0, 1]

    perceptron = Perceptron()
    perceptron.fit(X, y)
    y_pred = perceptron.predict([[2, 1], [3, 7]])
    print(y_pred)

    Notes
    -------
    Will add the pocket version of the algorithm at a later time.

    """

    def __init__(self, iterations, error_threshold=0, verbose=False):
        self.iterations = iterations
        self.error_threshold = error_threshold
        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        weights = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for i in range(self.iterations):
            error = 0

            for j in range(len(X)):
                if np.dot(X[j], weights)*y[j] <= 0:
                    weights += np.dot(X[j], y[j])
                    error += 1

            if error <= self.error_threshold:
                break

        self.weights = weights
        self.num_iterations = i + 1
        self.num_misclassified = error

        if self.verbose:
            print("num_iterations: {}, num_misclassified: {}".format(self.num_iterations, self.num_misclassified))

    def predict(self, X):
        X = np.array(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = np.sign(np.dot(X, self.weights))

        return y_pred
