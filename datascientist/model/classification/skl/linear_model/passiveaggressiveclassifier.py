from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np


def _passiveaggressiveclassifier(*, train, test, x_predict=None, metrics, C=1.0, fit_intercept=True, max_iter=1000,
tol=0.001, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True,
verbose=0, loss='hinge', n_jobs=None, random_state=None, warm_start=False, class_weight=None, average=False):
    """For for info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier
    """

    model = PassiveAggressiveClassifier(C=C, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, early_stopping=early_stopping,
    validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, shuffle=shuffle, verbose=verbose,
    loss=loss, n_jobs=n_jobs, random_state=random_state, warm_start=warm_start, class_weight=class_weight, average=average)
    model.fit(train[0], train[1])
    model_name = 'PassiveAggressiveClassifier'
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
