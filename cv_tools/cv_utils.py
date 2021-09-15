from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import Parallel, delayed
from functools import reduce
import pandas as pd
import numpy as np


def voting_cv_wrapper(estimator, X, y, n_splits, n_repeats, n_jobs=-1):
    """Repeated cross validation with voting
    Args:
        estimator: estimator object implementing fit
        X (array-like): array-like of shape (n_samples, n_features)
        y (array-like): array-like of shape (n_samples,)
        n_splits (int): number of cross validation folds
        n_repeats (int): number of repeats
        n_jobs (int): number of parallel jobs to run
    Returns:
        pandas.DataFrame: containing the predictions of each repeat and the consensus
    """
    rr = cross_val_score_voting(estimator, X, y, n_splits, n_repeats, n_jobs=-1)
    repeats = []
    for r in rr:
        repeats.append(_merge_folds(r))
    df = _merge_repeats(repeats)
    df['consensus'] = df.apply(_binary_vote, axis=1)
    return df


def cross_val_score_voting(estimator, X, y, n_splits, n_repeats, n_jobs=-1):
    """Performs repeated kfold cross validation on an estimator
    Args:
        estimator: estimator object implementing 'fit'
        X (array like): Feature data to fit (n_samples, n_features)
        y (array like): target values (n_samples,)
        n_splits (int): number of folds in cross validation
        n_repeats (int): number of repeats of cross validation
        n_jobs (int): number of parallel jobs to run

    Returns:
        List of cross validation results: each element is a
                    tuple of test slice indices and results
    """

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    n_jobs = -1
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
            delayed(_fit_predict)(clone(estimator), X, y, train_index, test_index)
            for train_index, test_index in rkf.split(X, y))

    def _to_matrix(input_list, n):
        return [input_list[i:i+n] for i in range(0, len(input_list), n)]
    results = _to_matrix(results, n_splits)
    return results


def _fit_predict(estimator, X, y, train_index, test_index):
    """Helper function for fiting the estimator to the training data
       and predicting the test data
    """
    estimator.fit(X[train_index], y[train_index])
    y_pred = estimator.predict(X[test_index])
    return test_index, y_pred


def _merge_folds(folds):
    """Merges the results of a kfold cross validation
    Args:
        folds(list): A list of fold indices and classification results.
            The assumption is that the fold indices are all mutually exclusive.
            List length is number of folds and each element is a tuple of
            sample indices and results
    Returns:
        pandas.DataFrame: A data frame containing the classification outputs
    """
    y_preds = {}
    for f in folds:
        y_preds.update(dict(zip(f[0], f[1])))

    y_preds = pd.DataFrame().from_dict(y_preds,  orient='index').sort_index()
    return y_preds


def _merge_repeats(repeats):
    """Helper function for merging repeat results """
    ret = reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True),
                 repeats)
    ret.columns = list(range(len(repeats)))
    return ret


def _binary_vote(a):
    """Helper function for calculating consensus a list of binary decisions
    Args:
        a (list): liat of binary decisions (0 or 1)s
    Returns:
        int: the consensus (0 or 1)

    """
    indicator = sum(a)/len(a)
    if indicator > 0.5:
        return 1
    elif indicator == 0.5:
        return np.random.choice([0, 1])
    elif indicator < 0.5:
        return 0
