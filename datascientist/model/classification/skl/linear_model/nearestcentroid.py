from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestCentroid
import numpy as np


def _nearestcentroid(*, train, test, x_predict=None, metrics, metric='euclidean', shrink_threshold=None):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid
    """

    model = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
    model.fit(train[0], train[1])
    model_name = 'Nearest Centroid'
    y_hat = model.predict(test[0])

    if metrics == 'accuracy':
        accuracy = accuracy_score(test[1], y_hat)

    if metrics == 'f1':
        accuracy = f1_score(test[1], y_hat)

    if metrics == 'jaccard':
        accuracy = jaccard_score(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
