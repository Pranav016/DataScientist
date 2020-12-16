from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NeighborhoodComponentsAnalysis
import numpy as np


def _neighborhoodcomponentsanalysis(*, train, test, x_predict=None, metrics, n_components=None, init='auto', warm_start=False, max_iter=50, tol=1e-05, callback=None, verbose=0, random_state=None):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html#sklearn.neighbors.NeighborhoodComponentsAnalysis
    """

    model = NeighborhoodComponentsAnalysis(n_components=n_components, init=init, warm_start=warm_start, max_iter=max_iter, tol=tol, callback=callback, verbose=verbose, random_state=random_state)
    model.fit(train[0], train[1])
    model_name = 'Neighborhood Components Analysis'
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
