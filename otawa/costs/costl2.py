import numpy as np
from scipy.stats import multivariate_normal
from otawa.base import BaseCost, log_likelihood_gaussian

# don't forget to use itertools quand ça sera pratique : dans la boucle de
# calcul des scores


class CostL2(BaseCost):
    def __init__(self):
        self.predictions = {}
        self.scores = {}

    def fit(self, signal):
        """TODO: Docstring for fit.

        :signal: TODO
        :returns: TODO

        """
        # time is first dimention
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        assert self.signal.ndim == 2, "signal should have two dimensions: 1st=time, 2nd=variables"

        # nb elements per time step (to compute nb parameters)
        self.nb_elements = sum(signal.shape[1:])

        return self

    def prediction(self, start, end):
        """Value of the prediction after seeing segment."""
        return self.predictions.setdefault(
            (start, end), self.signal[start:end].mean(axis=0))

    def score(self, start, middle, end):
        """TODO: Docstring for score.

        :start: TODO
        :middle: TODO
        :end: TODO
        :returns: TODO

        """
        if not (start, middle, end) in self.scores:
            prediction = self.prediction(start, middle)
            diff = prediction - self.signal[middle:end]
            score = log_likelihood_gaussian(diff)
            self.scores[(start, middle, end)] = score
        else:
            score = self.scores[(start, middle, end)]
        return score

    def likelihood(self, start, end):
        error = self.signal[start:end] - self.prediction(start, end)

        # estimating sigma
        sigma_hat = np.cov(error, rowvar=False)
        L = np.sum(multivariate_normal.logpdf(
            self.signal[start:end], self.prediction(start, end), sigma_hat,
            allow_singular=True))

        return L

    def nb_params(self, start, end):
        # mean and std for each elements per time step
        return 2 * self.nb_elements
