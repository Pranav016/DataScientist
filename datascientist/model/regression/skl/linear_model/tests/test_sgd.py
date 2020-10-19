from datascientist.model.regression.skl.linear_model.sgd import _sgd

import numpy as np

def test_sgd():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _sgd(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'SGDRegressor'
    assert round(answer[1] , 3) == 0.493
    assert answer[2] == None

    metrics = 'mse'
    answer = _sgd(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'SGDRegressor'
    assert round(answer[1] , 4) == 0.3055
    assert answer[2] == None

    metrics = 'rmse'
    answer = _sgd(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'SGDRegressor'
    assert round(answer[1] , 3) == 0.558
    assert answer[2] == None

    answer = _sgd(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr=np.array([5.14716575,  7.45496003,  9.14140708, 11.44920136])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
