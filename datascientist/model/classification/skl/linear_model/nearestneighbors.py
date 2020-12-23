from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestNeighbors
import numpy as np


def _nearestneighbors(*, train, test, x_predict=None, metrics, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
    """

    model = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
    model.fit(train[0], train[1])
    model_name = 'Nearest Neighbors'
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
