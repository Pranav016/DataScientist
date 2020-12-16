from datascientist.model.classification.skl.linear_model.nearestcentroid import _nearestcentroid

import numpy as np


def test_nearestcentroid():
    x_train = np.reshape([100, 120, 160, 180], (-1,1))
    y_train = np.array([0, 0, 1, 1])

    x_test = np.reshape([90, 200, 60, 70], (-1,1))
    y_test = np.array([0, 1, 0, 0])

    metrics = 'accuracy'
    answer = _nearestcentroid(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Nearest Centroid'
    assert answer[1] == 1.0
    assert answer[2] is None

    metrics = 'f1'
    answer = _nearestcentroid(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Nearest Centroid'
    assert answer[1] == 1.0
    assert answer[2] is None

    metrics = 'jaccard'
    answer = _nearestcentroid(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Nearest Centroid'
    assert answer[1] == 1.0
    assert answer[2] is None

    answer = _nearestcentroid(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert arr == any([0, 1, 0, 0])
