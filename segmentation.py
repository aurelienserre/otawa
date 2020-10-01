from collections import OrderedDict
import numpy as np


def min_max_pen(algo):
    m_min = 2
    m_max = algo.length // algo.spacing

    i = 0
    b_min = -(10 ** i)
    cps, _, _ = algo.segmentation(b_min)
    m = len(cps) + 1    # nb segments in the segmentation
    while m < m_max:
        i += 1
        b_min = -(10 ** i)
        cps, _, _ = algo.segmentation(b_min)
        m = len(cps) + 1    # nb segments in the segmentation

    i = 0
    b_max = 10 ** i
    cps, _, _ = algo.segmentation(b_max)
    m = len(cps) + 1    # nb segments in the segmentation
    while m > m_min:
        i += 1
        b_max = 10 ** i
        cps, _, _ = algo.segmentation(b_max)
        m = len(cps) + 1    # nb segments in the segmentation

    return b_min, b_max


def explore_penalties(algo):
    """
    explore all the optimal segmentations possible by exploring the penalty
    space using the CROPS algorithm
    Returns :
        segmentations : (OrderedDict) with number of CPs as keys, and a dict containing
        the corresponding segmentation, penalty and score as values
    """

    # dict containg the number of CPs and the scores corresponding to every value of penalty that have been explored
    results = {}
    segmentations = {}  # dict that will be returned

    def test_pen(b):
        """b is the penalty value"""

        cp_list, score, score_no_pen = algo.segmentation(penalty=b)
        m = len(cp_list) + 1
        results[b] = (m, score_no_pen)
        segmentations.setdefault(m, {'cp_list': cp_list, 'penalty': b, 'score_no_pen': score_no_pen, 'score': score})

        return m, score_no_pen

    b_min, b_max = min_max_pen(algo)
    test_pen(b_min)
    test_pen(b_max)

    # algorithm taken from https://arxiv.org/abs/1412.3617 , "Algorithm 2 : CROPS algorithm"
    b_intervals = {(b_min, b_max)}

    while b_intervals:
        b0, b1 = b_intervals.pop()

        m1, Q1 = results[b1]
        m0, Q0 = results[b0]

        if m0 > m1 + 1:
            b_int = (Q1 - Q0) / (m1 - m0)
            m_int, Q_int = test_pen(b_int)
            if m_int != m1 and m_int != m0:
                b_intervals.update([(b0, b_int), (b_int, b1)])
            else:
                segmentations[m1]['beta_min'] = b_int
        else:
            segmentations[m1]['beta_min'] = (Q1 - Q0) / (m1 - m0)

    # order the different possible segmentations
    segmentations = OrderedDict(sorted(segmentations.items()))

    return segmentations


def compute_criteria(algo, segmentations):
    for m, infos in segmentations.items():
        likelihood = algo.likelihood([0] + infos["cp_list"] + [algo.length])
        infos["likelihood"] = likelihood
        nb_params = algo.nb_params([0] + infos["cp_list"] + [algo.length])
        infos["nb_params"] = nb_params
        log_T = np.log(algo.length)
        infos["BIC"] = -2 * likelihood + nb_params * log_T


def best_seg(segs, criterion):
    best = float("inf")
    for m, infos in segs.items():
        if infos[criterion] < best:
            best = infos[criterion]
            best_m = m

    return {best_m: segs[best_m]}
