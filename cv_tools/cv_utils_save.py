from joblib import Parallel, delayed
from sklearn.metrics._scorer import get_scorer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target


def cross_val_fit(estimator, X, y, cv, metric, n_jobs=-1):
    """Generate cross-validated estimators for all folds

    Args:
        estimator: estimator object implementing 'fit' and 'predict'
        X (array like): (n_samples, n_features)
        y (array like): (n_samples,)
        cv (int, cross-validation generator ): number of folds. You can also pass a predefined CV splitter
        metric (str): metric to return for each fold
        n_jobs (int): Number of jobs to run in parallel
    
    Returns:
        (list): A list of dictionaries containing 'score', 'scorer', and 'estimator'
                score: Calculated metric
                scorer: The scorer function used for calculating the metric
                estimator: Fitted estimator
    """
    scorer = get_scorer(metric)
 
    cv = _check_cv(y, cv)
    splits = list(cv.split(X, y))
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(_fit_estimator)(clone(estimator), X, y, train_index, test_index, scorer)
        for train_index, test_index in splits)
    return results


def _fit_estimator(estimator, X, y, train_index, test_index, scorer):
    """Helper function for fiting the estimator to the training data
       and returning the evaluated metric along with the scorer used for the metric
    """
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    predict_proba = getattr(estimator, "predict_proba", None)
    y_prob = None
    if callable(predict_proba):
        y_prob = predict_proba(X_test)
    return {'score': scorer(estimator, X_test, y_test),
            'estimator': estimator,
            'scorer': scorer,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'test_index': test_index}

def _check_cv(y, cv):
    """"Helper function to determine the CV type"""
    classification_target_types = ('binary', 'multiclass')
    if hasattr(cv, 'split'):
        return cv
    if type_of_target(y) in classification_target_types:
        return StratifiedKFold(n_splits=cv)
    else:
        return KFold(n_splits=cv)