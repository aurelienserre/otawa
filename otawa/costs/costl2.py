import numpy as np
from ..base import BaseCost, log_likelihood_gaussian
# replace by the following line once otawa transformed into package
# from otawa.base import BaseCost, log_likelihood_gaussian
# (because module `otawa` will be in `sys.path` once installed as a package
# whereas for now, it's not in `sys.path` if we import from a parent directory
# which is not the one containing `otawa`, so we need to do a relative import,
# which seems not as clean)
# replace `otawa` by the name chosen for the package (otawacpd for example)


class CostL2SOS(BaseCost):
    def __init__(self, average=False, regularize=True, diag_cov=True):
        self.average = average          # average the score over the segmants?
        self.regularize = regularize    # score regularized (by likelihood of the correct model)?
        self.diag_cov = diag_cov        # whether to use a full covariance mat, or a diag. one
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

        # compute covariance matrix of the time-series (assumed diagonal)
        if self.diag_cov:
            self.cov = np.diag(np.var(signal, axis=0, ddof=1))
        else:
            self.cov = np.cov(signal, rowvar=False)

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
            score = np.sum(diff ** 2)
            if self.regularize:
                prediction = self.prediction(middle, end)
                diff = prediction - self.signal[middle:end]
                score -= np.sum(diff ** 2)
            if self.average:
                score /= (end - middle)
            self.scores[(start, middle, end)] = score
        else:
            score = self.scores[(start, middle, end)]
        return score

    def likelihood(self, start, end):
        error = self.signal[start:end] - self.prediction(start, end)

        L = log_likelihood_gaussian(error, covariance=self.cov)

        return L

    def nb_params(self, start, end):
        # mean and std for each elements per time step
        return 2 * self.nb_elements
