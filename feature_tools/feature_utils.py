from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.decomposition import FactorAnalysis


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
