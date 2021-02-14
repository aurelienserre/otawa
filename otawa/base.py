import abc
from itertools import dropwhile, tee
import numpy as np


class BaseCost(abc.ABC):
    """Base class for all cost classes."""

    @abc.abstractmethod
    def fit(self, signal):
        """Set the signal (time series) on which costs will be computed."""
        pass

    @abc.abstractmethod
    def score(self, start, middle, end):
        """Compute the score of a given pair of consecutive segments."""
        pass


def log_likelihood_gaussian(diff, covariance=None, diag_cov=True):
    """Compute log likelihood with gaussian model, from diff.

    diff = y_hat - y_true, assumed to be centered.

    Parameters:
        covariance: Covariance matrix to use. If not provided, defaults to
            computing the sample covariance matrix on `diff`.
        diag_cov: if `True`, assume the covariance matrix to be diagonal; full
            otherwise. Used only if `covariance` not provided.
            Note: A full sample covariance matrix can be problematic with high
                dimensions and short segments, because when
                nb_samples < nb_variables, it will be singular. This is
                especially problematic with autoregressive models, where
                nb_variables = dimension * order.
                This could be solved by using shrinkage.
    """
    if covariance is None:
        # if no covariance provided, estimates it on `diff`
        if diag_cov:
            covariance = np.diag(np.var(diff, axis=0, ddof=1))
        else:
            covariance = np.cov(diff, rowvar=False)

    ndims = diff.shape[1]
    L = np.sum(
        - ndims * np.log(2 * np.pi) / 2
        - np.log(np.linalg.det(covariance)) / 2
        - np.einsum("ni,ij,nj->n", diff, np.linalg.inv(covariance), diff) / 2
    )

    return L


class Otawa(object):
    """Class implementing the Otawa algorithm."""

    def __init__(self, cost, jump=1, spacing=5):
        """
        Parameters:
            cost: cost object inheriting from `BaseCost`, and implementing
                the `fit` and `score` methods.
            jump: used to adjust the resolution. One every `jump` time step
                is considered as a candidate change point.
            spacing: minimum spacing between consecutive change points for a
                segmentation to to feasible
        """
        self.cost = cost
        self.signal = None
        self.jump = jump
        self.spacing = spacing

    def fit(self, signal):
        """Pass the time series to the Otawa object.

        It initializes some parameters, and also computes the scores that will
        be needed for segmentation later. So this methods accounts for most of
        the run time of OTAWA.

        Parameter:
            signal: time series on which change point detection will be
                performed. Usually assumed to be a numpy array or compatible,
                and that first dimension is time. See a specific cost's
                implementation of its fit method for more details.
        """
        self.cost.fit(signal)
        self.signal = self.cost.signal
        self.length = len(self.signal)

        # fist and last points of the signal are considered change points
        indexes = set(range(0, self.length, self.jump))
        indexes |= {self.length - 1}
        # so no other CP can be at less that `spacing` from each end of the signal
        indexes -= set(range(1, self.spacing))
        indexes -= set(range(self.length - self.spacing, self.length - 1))

        """Warning, if spacing is smaller than jump (weird anyway, doesn't
        make much sens), the last two points might be closer than `jump` time
        steps appart (but more than spacing).
        Spacing larger than jump makes more sens, and spacing multiple of
        jump should be preferred (otherwise, unexpected behaviours may
        happen)."""

        indexes = sorted(indexes)

        # compute scores of all paris of consecutive segments
        self.edges = list()
        self.nodes = set()
        for i in indexes:
            for j in dropwhile(lambda x: x < i + self.spacing, indexes):
                for k in dropwhile(lambda x: x < j + self.spacing, indexes):
                    score = self.cost.score(i, j, k)
                    self.edges.append(((i, j), (j, k), score))
                    self.nodes |= {(i, j), (j, k)}

        # add unique source and targets nodes, and corresponding edges
        source_edges = list()
        target_edges = list()
        for i, j in self.nodes:
            if i == 0:
                source_edges.append(('source', (i, j), 0))
            elif j == self.length - 1:
                target_edges.append(((i, j), 'target', 0))
        self.nodes |= {'source', 'target'}
        # necessary for the edges to be in this order for shortest path algo
        self.edges = source_edges + self.edges + target_edges

    def segmentation(self, penalty=0.):
        """
        Compute the optimal segmentation of the time series with the penalty
        value passed as argument by computing the longest path on the graph
        represented by the `edges` and `nodes` attributes.

        Parameters :
            penalty   : penalty given for each new CP introduced
        Returns :
            segmentation : list of the positions of the CPs
            score        : score corresponding to that optimal segmentation
            score_no_pen : score corresponding to that optimal segmentation
                without taking the penalty into account
        """
        # Bellman-Ford modified for DAGs
        dist = {node: -float('inf') for node in self.nodes}
        dist['source'] = 0
        pred = dict()

        for n1, n2, weight in self.edges:
            d = weight - penalty
            if dist[n2] < dist[n1] + d:
                dist[n2] = dist[n1] + d
                pred[n2] = n1

        score = dist['target']
        # penlaty should not have been applied on both edges connected to
        # source and target
        score += 2 * penalty
        path = ['target']
        current = 'target'
        while current != 'source':
            current = pred[current]
            path.append(current)
        path.reverse()

        segmentation = [seg[1] for seg in path[1:-2]]
        m = len(segmentation)
        score_no_pen = score + m * penalty

        return segmentation, score, score_no_pen

    def likelihood(self, segmentation):
        """Compute the likelihood of a given segmentation of the signal."""
        a, b = tee(segmentation)
        next(b)
        pairs = zip(a, b)

        L = 0
        for i, j in pairs:
            L += self.cost.likelihood(i, j)

        return L

    def nb_params(self, segmentation):
        """Compute the number of parameters in the piecewise model
        corresponding to a given segmentation."""
        a, b = tee(segmentation)
        next(b)
        pairs = zip(a, b)

        p = 0
        for i, j in pairs:
            p += self.cost.nb_params(i, j)

        return p
