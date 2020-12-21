from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import ComplementNB
import numpy as np


def _complementnb(*, train, test, x_predict=None, metrics, alpha=1.0, fit_prior=True, class_prior=None, norm=False):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB
    """

    model = ComplementNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, norm=norm)
    model.fit(train[0], train[1])
    model_name = 'ComplementNB'
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