#!/usr/bin/env python3

from collections import defaultdict

import numpy as np
from scipy.special import erf, erfinv
from scipy.optimize import minimize_scalar


class Elo:
    """
    Elo ratings and predictions based on the Bradley-Terry model.

    """
    def __init__(self, times, labels1, labels2, values, lines=0, k=None):
        """
        Provide a list of binary comparisons:
        times, labels1, labels2, values

        """
        self.values = np.array(values, ndmin=1, dtype=float)
        self.lines = np.array(lines, ndmin=1, dtype=float)

        outcomes = np.tile(
            self.values[:, np.newaxis], self.lines.size
        ) > self.lines

        outcomes = (outcomes if self.lines.size > 1 else outcomes.ravel())

        self.comparisons = np.sort(
            np.rec.fromarrays(
                [times, labels1, labels2, outcomes],
                dtype=[
                    ('time', 'M8[us]'),
                    ('label1',   'U8'),
                    ('label2',   'U8'),
                    ('outcome',   '?', self.lines.size),
                ]
            ), axis=0
        )

        self.rtg_hcap = self.rating_handicap(self.comparisons.outcome)

        self.k = (self.auto_fit_k if k is None else k)

        self.ratings, self.error = self.rate(self.k)

    @property
    def auto_fit_k(self):
        """
        Tune the k update-factor to minimize predictive error.

        """
        res = minimize_scalar(lambda x: self.rate(x)[1])
        return res.x

    def rating_handicap(self, outcomes):
        """
        Naive prior that label1 beats label2.

        """
        prob = np.count_nonzero(outcomes, axis=0) / np.size(outcomes, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        return np.sqrt(2)/2*erfinv(2*prob - 1)

    def norm_cdf(self, x, loc=0, scale=1):
        """
        Normal cumulative probability distribution.

        """
        return 0.5*(1 + erf((x - loc)/(np.sqrt(2)*scale)))

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        ratings = self.ratings[label]
        return ratings[ratings['time'] < time][-1]

    def rate(self, k):
        """
        Apply the Elo model to the list of binary comparisons.

        """
        rtg_over = defaultdict(lambda: self.rtg_hcap)
        rtg_under = defaultdict(lambda: -self.rtg_hcap)

        ratings = defaultdict(list)

        error = 0

        for (time, label1, label2, outcome) in self.comparisons:

            # lookup team ratings
            rating1 = rtg_over[label1]
            rating2 = rtg_under[label2]

            # prior and posterior outcome probabilities
            prior = self.norm_cdf(rating1 - rating2)
            post = np.where(outcome, 1, 0)

            # rating change
            rating_change = k * (post - prior)
            error += (post - prior)**2

            # update current rating
            rtg_over[label1] += rating_change
            rtg_under[label2] -= rating_change

            # record current rating
            ratings[label1].append((time, rtg_under[label1], rtg_over[label1]))
            ratings[label2].append((time, rtg_under[label2], rtg_over[label2]))

        for label in ratings.keys():
            ratings[label] = np.array(
                ratings[label],
                dtype=[
                    ('time',  'M8[us]'),
                    ('under', 'f8', self.lines.size),
                    ('over',  'f8', self.lines.size),
                ]
            )

        return ratings, error

    def prior(self, time, label1, label2):
        """
        Prior cumulative probability distribution for a comparison between
        label1 and label2 at specified 'time'.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        return self.norm_cdf(rating1['over'] - rating2['under'])
