from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm

# TODO Implement models with good default arguments

REGRESSION_MODELS = dict(
    lr=linear_model.LinearRegression(),
    svm=svm.SVR(C=0.001, epsilon=0.001, gamma='scale'),
    randomforest=ensemble.RandomForestRegressor(n_estimators=100, n_jobs=100, verbose=1),
    extratrees=ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=100, verbose=1)
)


CLASSIFICATION_MODELS = dict(
    lr=linear_model.LogisticRegression(),
    svm=svm.SVC(),
    randomforest=ensemble.RandomForestClassifier(n_estimators=200, n_jobs=100, verbose=1),
    extratrees=ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=100, verbose=1)
)
