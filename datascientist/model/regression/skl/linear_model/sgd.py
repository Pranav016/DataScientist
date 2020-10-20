from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import SGDRegressor
import numpy as np


def _sgd(*, train, test, x_predict=None, metrics, loss='squared_loss', penalty='l2',
        alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, 
        verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, 
        early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
    """
    model = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon,
        random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t,
        early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
        warm_start=warm_start, average=average)

    model.fit(train[0], train[1])
    model_name = 'SGDRegressor'
    y_hat = model.predict(test[0])

    if metrics == 'mse':
        accuracy = _mse(test[1], y_hat)
    if metrics == 'rmse':
        accuracy = _rmse(test[1], y_hat)
    if metrics == 'mae':
        accuracy = _mae(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
