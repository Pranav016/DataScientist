from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegressionCV
import numpy as np


def _logisticregressioncv(*, train, test, x_predict=None, metrics, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2',
scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True,
intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None):
    """For for info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
    """

    model = LogisticRegressionCV(Cs=Cs, fit_intercept=fit_intercept, cv=cv, dual=dual, penalty=penalty, scoring=scoring,
solver=solver, tol=tol, max_iter=max_iter, class_weight=class_weight, n_jobs=n_jobs, verbose=verbose, refit=refit,
intercept_scaling=intercept_scaling, multi_class=multi_class, random_state=random_state, l1_ratios=l1_ratios)
    model.fit(train[0], train[1])
    model_name = 'LogisticRegressionCV'
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
