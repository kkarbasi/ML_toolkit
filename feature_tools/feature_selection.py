import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.decomposition import FactorAnalysis


class RemoveCorrelated(BaseEstimator, TransformerMixin):
    """Transformer class for decorrelating input features
    Args:
        method (str): mutual_info (for classification), pearson (for regression)
        threshold (float): Threshold over which the less correlated
                            columns (with the target) will be remove
    """

    def __init__(self, method, thresh):
        self.method = method
        self.thresh = thresh

    def fit(self, X, y=None):
        ml = DecorrHelper()
        corrs = ml.corr_with_y(X, y, method=self.method)
        self.cols_to_drop = ml.drop_by_corr_order(X, corrs, self.thresh)
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if isinstance(X, pd.DataFrame):
            X_ = X_.drop(columns=X.columns[self.cols_to_drop])
            return X_
        if isinstance(X, np.ndarray):
            X_ = np.delete(X_, self.cols_to_drop, axis=1)
            return X_
        else:
            raise ValueError('Features data type not defined')

    def get_support(self, indices=False):
        if indices:
            return self.cols_to_drop
        else:
            raise NotImplementedError(
                'Boolean mask not implemented. Try indices=True.')


class DecorrHelper:
    """Helper class for RemoveCorrelated"""

    def __init__(self):
        pass

    def corr_with_y(self, X, y, method):
        '''Method for calculating correlation of all features in X with y.

        Args:
            X (numpy.array): A numpy array (n_samples, n_features)
            y (numpy.array): A numpy array containing the target variable (n_samples,)
            method (str): pearson, mutual_info
        returns:
            numpy.array: A numpy array containing the correlations (n_features,)
        '''
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if method == 'pearson':
            def corr_func(X, y):
                return np.array([pearsonr(x, y)[0] for x in X.T])
        elif method == 'mutual_info':
            corr_func = partial(mutual_info_classif, random_state=369)
        else:
            raise ValueError('Method not defined!')
        selector = SelectKBest(corr_func, k='all')
        selector.fit(X, y)
        return selector.scores_

    def drop_low_var(self, X, threshold):
        '''Get column indices with variance greater than threshold

        Args:
            df_data (numpy.array): (n_samples, n_features)
            threshold (float): Variance threshold
        Returns:
            list: Indices of columns to keep
        '''
        selector = VarianceThreshold(threshold)
        selector.fit(X)
        return selector.get_support(indices=True)

    def drop_by_corr_order(self, X, corrs, corr_thresh, verbose=False):
        '''Get column names that needs to be dropped due to pearson correlation

        Args:
            X (numpy.array): (n_sample, n_features)
            corrs (numpy.array): Array containing the correlation of all features
                                 with the target variable (n_features)
            corr_thresh (float): Threshold over which the less correlated
                                 columns (with the target) will be removed
        Returns:
            list: Columns to remove
                '''
        # list of features to skip
        col_skip = []
        # list of features to drop because they show high correlation with
        # another feature
        col_drop = []
        sorted_corr_indices = np.argsort(corrs)[::-1]

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        for feat1 in sorted_corr_indices:
            if feat1 not in col_skip:
                for feat2 in sorted_corr_indices:
                    if feat2 not in col_skip:
                        if feat1 != feat2:
                            p = pearsonr(X[:, feat1], X[:, feat2])[0]
                            if np.abs(p) > corr_thresh:
                                col_drop.append(feat2)
                                col_skip.append(feat2)
                col_skip.append(feat1)
        return col_drop


class GroupedFactorAnalysis(BaseEstimator, TransformerMixin):
    """"sklearn style transformer class for performing factor anaysis on uniformly grouped features
    
        Args:
            set_size (int): group size
    """

    def __init__(self, set_size=7):
        self.set_size = set_size

    def fit(self, X, y=None):
        if np.mod(X.shape[1], self.set_size) != 0:
            raise ValueError('Number of input columns should be divisible by set_size')
        self.transformers = []
        indices = np.arange(0, X.shape[1]).reshape(-1, self.set_size)
        for idx in indices:
            X_ = X[:, idx]
            fa = FactorAnalysis(
                n_components=X_.shape[1],
                rotation='varimax')
            fa.fit(X_)
            self.transformers.append(fa)
        return self

    def transform(self, X):
        if X.shape[1] != len(self.transformers) * self.set_size:
            raise ValueError('Number of input columns does not match.')
        indices = np.arange(0, X.shape[1]).reshape(-1, self.set_size)
        factors = []
        for idx, fa in zip(indices, self.transformers):
            X_ = X[:, idx]
            fa_projected = fa.fit_transform(X_)
            m = fa.components_
            m1 = m**2
            m2 = np.sum(m1, axis=1)
            pvars = []
            for i in m2:
                pvars.append((100 * i) / np.sum(m2))
            pvars = np.array(pvars)
            sorted_idx = np.argsort(pvars)[::-1]
            factors.append(fa_projected[:, sorted_idx[0]])
        factors = np.vstack(factors)
        return factors.T

