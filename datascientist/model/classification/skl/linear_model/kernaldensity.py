from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KernelDensity
import numpy as np


def _kerneldensity(*, train, test, x_predict=None, metrics, bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean', atol=0, rtol=0, breadth_first=True, leaf_size=40, metric_params=None):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
    """

    model = KernelDensity(bandwidth=bandwidth, algorithm=algorithm, kernel=kernel, metric=metric, atol=atol, rtol=rtol,
breadth_first=breadth_first, leaf_size=leaf_size, metric_params=metric_params)
    model.fit(train[0], train[1])
    model_name = 'Kernel Density'
    y_hat = model.predict(test[0])

    if metrics == 'accuracy':
        accuracy = accuracy_score(test[1], y_hat)

    if metrics == 'f1':
        accuracy = f1_score(test[1], y_hat)

    if metrics == 'jaccard':
        accuracy = jaccard_score(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
