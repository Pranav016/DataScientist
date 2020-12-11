from datascientist.model.classification.knn.BallTree import _BallTree
from pytest import raises
import numpy as np


def test_Balltree():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2]))

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2]))

    train=(x_train, y_train)
    test=(x_test, y_test)
    metrics = 'accuracy'
    answer = _BallTree(train, test, metrics=metrics)
    assert answer[0] == 'Ball Tree'
    assert answer[1] == 1.0
    assert answer[2] is None

    metrics = 'f1'
    answer = _BallTree(train, test, metrics=metrics)
    assert answer[0] == 'Ball Tree'
    assert answer[1] == array([1., 1., 1., 1.])
    assert answer[2] is None

    metrics = 'jaccard'
    answer = _BallTree(train, test, metrics=metrics)
    assert answer[0] == 'Ball Tree'
    assert answer[1] == array([1., 1., 1., 1.])
    assert answer[2] is None
