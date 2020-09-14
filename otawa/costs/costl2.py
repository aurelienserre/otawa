from otawa.base import BaseCost

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

        return self

    def score(self, start, middle, end):
        """TODO: Docstring for score.

        :start: TODO
        :middle: TODO
        :end: TODO
        :returns: TODO

        """
        if not (start, middle, end) in self.scores:
            if not (start, middle) in self.predictions:
                prediction = self.signal[start:middle].mean(axis=0)
                self.predictions[(start, middle)] = prediction
            else:
                prediction = self.predictions[(start, middle)]

            sq_diff = (prediction - self.signal[middle:end]) ** 2
            score = sq_diff.mean()
            self.scores[(start, middle, end)] = score
        else:
            score = self.scores[(start, middle, end)]
        return score
