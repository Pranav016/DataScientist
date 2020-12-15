from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import BallTree
import numpy as np


def _balltree(*, train, test, x_predict=None, metrics, X, leaf_size=40, metric='minkowski', **kwargs):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree
    """

    model = BallTree(X, leaf_size=leaf_size, metric=metric, **kwargs)
    model.fit(train[0], train[1])
    model_name = 'Ball Tree'
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
