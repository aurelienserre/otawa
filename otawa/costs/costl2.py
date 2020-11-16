import numpy as np
from ..base import BaseCost, log_likelihood_gaussian
# replace by the following line once otawa transformed into package
# from otawa.base import BaseCost, log_likelihood_gaussian
# (because module `otawa` will be in `sys.path` once installed as a package
# whereas for now, it's not in `sys.path` if we import from a parent directory
# which is not the one containing `otawa`, so we need to do a relative import,
# which seems not as clean)
# replace `otawa` by the name chosen for the package (otawacpd for example)


class CostL2(BaseCost):
    def __init__(self, average=False, regularize=True, const_cov=False):
        self.average = average          # average the score over the segmants?
        self.regularize = regularize    # score regularized (by likelihood of the correct model)?
        self.const_cov = const_cov      # whether the covariance is assumed constant along the whole time-series
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
        self.nb_elements = signal.shape[1]

        if self.const_cov:
            # compute covariance matrix of the time-series (assumed diagonal)
            self.cov = np.diag(np.var(signal, axis=0, ddof=1))
        else:
            # set to None, so that it will be computed on a per segment basis
            self.cov = None

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
            score = - log_likelihood_gaussian(diff, covariance=self.cov)
            if self.regularize:
                score -= - self.likelihood(middle, end)
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
        if self.const_cov:
            # only mean (i.e. one param per variable)
            return self.nb_elements
        else:
            # mean and diag cov (i.e. 2 params per variable)
            return 2 * self.nb_elements
