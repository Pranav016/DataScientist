from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
import numpy as np


def _gaussiannb(*, train, test, x_predict=None, metrics):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
    """

    model = GaussianNB()
    model.fit(train[0], train[1])
    model_name = 'Gaussian Naive Bayes'
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