from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier
import numpy as np


def _sgdclassifier(*, train, test, x_predict=None, metrics, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
class_weight=None, warm_start=False, average=False):
    """For for info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    """

    model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t,
early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start, average=average)
    model.fit(train[0], train[1])
    model_name = 'SGDClassifier'
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
