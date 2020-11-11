import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import multivariate_normal
from numpy.lib.stride_tricks import as_strided
from otawa.base import BaseCost, log_likelihood_gaussian


class CostAR(BaseCost):
    def __init__(self, order=3, alpha=1e-2, average=False):
        self.order = order
        self.alpha = alpha
        self.average = average
        self.models = {}
        self.scores = {}

    def fit(self, signal):
        """TODO: Docstring for fit.

        :signal: TODO
        :returns: TODO

        """
        # time is first dimention
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        else:
            # flatten to one dim per time step (necessary for linear model)
            signal = signal.reshape(len(signal), -1)

        # lagged values
        nshape = (signal.shape[0] - self.order, self.order, *signal.shape[1:])
        nstrides = (signal.strides[0], signal.strides[0], *signal.strides[1:])
        lagged = as_strided(signal, shape=nshape, strides=nstrides)
        lagged = lagged.reshape(len(lagged), -1)
        self.lagged = np.pad(lagged, ((self.order, 0), (0, 0)), mode='edge')
        signal[:self.order] = signal[self.order]
        self.signal = signal

        return self

    def get_model(self, start, end):
        """Value of the prediction after seeing segment."""
        if not (start, end) in self.models:
            model = Lasso(alpha=self.alpha)
            model.fit(self.lagged[start + self.order:end], self.signal[start + self.order:end])
            self.models[(start, end)] = model
        else:
            model = self.models[(start, end)]

        return model

    def score(self, start, middle, end):
        """TODO: Docstring for score.

        :start: TODO
        :middle: TODO
        :end: TODO
        :returns: TODO

        """
        if not (start, middle, end) in self.scores:
            model = self.get_model(start, middle)
            pred = model.predict(self.lagged[middle:end])
            diff = pred - self.signal[middle:end]
            score = log_likelihood_gaussian(diff) - self.likelihood(middle, end)
            if self.average:
                score /= (end - middle)
            self.scores[(start, middle, end)] = score
        else:
            score = self.scores[(start, middle, end)]
        return score

    def likelihood(self, start, end):
        model = self.get_model(start, end)
        pred = model.predict(self.lagged[start + self.order:end])
        error = pred - self.signal[start + self.order:end]

        L = log_likelihood_gaussian(error)

        return L

    def nb_params(self, start, end):
        return (
            self.get_model(start, end).sparse_coef_.getnnz()
            + self.get_model(start, end).intercept_.size
        )
