# Brenno Ribeiro - brenno_ribeiro@my.uri.edu
# Created on 11/23/18

import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    Simple K-Nearest Neighbors algorithm.

    Parameters
    ----------
    k : int
        Number of neighbors.

    dist_metric : str, optional (default = 'euclidean')
        Distance metric used to calculate distances between neighbors.
        Possible values:

            - 'euclidean' : Straight line distance between points.
            - 'manhattan' : Distances between points measured along axes at right angles.
            - 'chebyshev' : Maximum distance along any coordinate dimension.

    Examples
    -------
    from knn import KNearestNeighbors

    X = [[1, 2], [3, 4], [2, 5], [-1, 9]]
    y = [0, 0, 1, 1]

    knn = KNearestNeighbors(2, dist_metric='manhattan')
    knn.fit(X, y)
    labels = knn.predict([[2, 1], [3, 7]])
    print(labels)

    Notes
    -------
    Will add an option to add weights for prediction and perhaps more distances
    at a later time.

    """

    def __init__(self, k, dist_metric='euclidean'):
        self.neighbors = k

        if dist_metric is 'euclidean':
            self.p = 2
        elif dist_metric is 'manhattan':
            self.p = 1
        elif dist_metric is 'chebyshev':
            self.p = np.inf
        else:
            self.p = 2
            print('''{} is not a valid distance metric!\n** Euclidean distance was used **'''.format(dist_metric))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = []

        for point in X:
            distances = np.sum(abs(point - self.X_train) ** self.p, axis=1) ** (1/self.p)
            dist_idx = np.argsort(distances)[0:self.neighbors]

            labels = []
            for i in dist_idx:
                labels.append(self.y_train[i])

            val = Counter(labels).most_common(1)[0][0]
            predictions.append(val)

        return predictions
