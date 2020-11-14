import abc
from itertools import dropwhile, tee
import numpy as np
from scipy.stats import multivariate_normal


class BaseCost(abc.ABC):
    """Base class for all cost classes."""

    @abc.abstractmethod
    def fit(self, signal):
        pass

    @abc.abstractmethod
    def score(self, start, middle, end):
        pass


def log_likelihood_gaussian(diff, cov=None):
    """Compute log likelihood with gaussian model, from diff.

    diff = y_hat - y_true
    Warning: assumes diagonal covariance matrix.
    """
    # assume diagonal cov mat, because otherwise estimated full cov mat
    # might be non positive definite when nb_samples < nb_variables
    # cf. 1st comment on https://stats.stackexchange.com/a/30466/237874
    # this brings singularity problems, and det(Sigma) < 0 which doesn't work in
    # the log_likelihood term -1/2 log(det(Sigma))
    if cov is not None:
        # if no covariance provided, estimates it on `diff`
        covariance = np.diag(np.var(diff, axis=0, ddof=1))
    # slower implementation
    # L_old = np.sum(multivariate_normal.logpdf(diff, cov=covariance))
    ndims = diff.shape[1]
    L = np.sum(
        - ndims * np.log(2 * np.pi) / 2
        - np.log(np.linalg.det(covariance)) / 2
        - np.einsum("ni,ij,nj", diff, np.linalg.inv(covariance), diff) / 2
    )

    return L


class Otawa(object):

    """Docstring for Otawa. """

    def __init__(self, cost, jump=1, spacing=5):
        """TODO: to be defined.

        :Docstring for Otawa.: TODO

        """
        self.cost = cost
        self.signal = None
        self.date = None
        self.jump = jump
        self.spacing = spacing

    def fit(self, signal):
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
        make much sens), the last and second to last points might be closer
        than `jump` appart (but more than spacing).
        Spacing larger than jump makes more sens, and spacing multiple of
        jump should be preferred (otherwise, weird behaviours may happen)."""

        indexes = sorted(indexes)

        self.edges = list()
        self.nodes = set()
        for i in indexes:
            for j in dropwhile(lambda x: x < i + self.spacing, indexes):
                for k in dropwhile(lambda x: x < j + self.spacing, indexes):
                    score = self.cost.score(i, j, k)
                    self.edges.append(((i, j), (j, k), score))
                    self.nodes |= {(i, j), (j, k)}

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
        Computes the optimal segmentation of the time-series with the penalty
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
        a, b = tee(segmentation)
        next(b)
        pairs = zip(a, b)

        L = 0
        for i, j in pairs:
            L += self.cost.likelihood(i, j)

        return L

    def nb_params(self, segmentation):
        a, b = tee(segmentation)
        next(b)
        pairs = zip(a, b)

        p = 0
        for i, j in pairs:
            p += self.cost.nb_params(i, j)

        return p
