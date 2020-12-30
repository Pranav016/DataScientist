from datascientist.model.regression.skl.linear_model.ardregression import _ardregression

import numpy as np


def test_ardregression():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3,4], [4,5]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _ardregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**7, 4) == 1.6193
    assert answer[2] is None

    metrics = 'mse'
    answer = _ardregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**14, 3) == 3.223
    assert answer[2] is None

    metrics = 'rmse'
    answer = _ardregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**7, 3) == 1.795
    assert answer[2] is None

    answer = _ardregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([ 5.99999994,  8.00000027,  8.99999981, 11.00000013])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)