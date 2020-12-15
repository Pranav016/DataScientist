from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
import numpy as np


def _perceptron(*, train, test, x_predict=None, metrics, penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001,
shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
class_weight=None, warm_start=False):
    """For for info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    """

    model = Perceptron(penalty=penalty, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
 verbose=verbose, eta0=eta0, n_jobs=n_jobs, random_state=random_state, early_stopping=early_stopping, validation_fraction=validation_fraction,
 n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start)
    model.fit(train[0], train[1])
    model_name = 'Perceptron'
    y_hat = model.predict(test[0])

    if metrics == 'f1_score':
        accuracy = f1_score(test[1], y_hat)
    if metrics == 'jaccard_score':
        accuracy = jaccard_score(test[1], y_hat)
    if metrics == 'accuracy_score':
        accuracy = accuracy_score(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
