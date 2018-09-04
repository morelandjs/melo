#!/usr/bin/env python3

from collections import defaultdict

import numpy as np
from scipy.special import erf, erfinv
from scipy.optimize import minimize_scalar
from scipy.ndimage import filters


class Melo:
    """
    Melo(times, labels1, labels2, values, lines=0, k=None)

    Margin-dependent Elo ratings and predictions.

    This class calculates Elo ratings from binary comparisons determined by a
    list of values (outcomes of each comparison) and a specified line (or
    sequence of lines) which determines the threshold of comparison.

    **Required parameters**

    These must be list-like objects of length *n_comparisons*, where the
    number of comparisons must be the same for all lists.

    - *times* -- comparison time stamp

    - *labels1* -- first entity's label name

    - *labels2* -- second entity's label name

    - *values* -- comparison value

    **Optional parameters**

    - *lines* -- comparison threshold line(s)

    - *k* -- rating update factor


    """
    def __init__(self, times, labels1, labels2, values, lines=0,
                 regress=lambda x: 1, k=None):

        self.values = np.array(values, dtype=float)
        self.lines = np.array(lines, dtype=float)
        self.regress = regress

        outcomes = np.tile(
            self.values[:, np.newaxis], self.lines.size
        ) > self.lines

        outcomes = (outcomes if self.lines.size > 1 else outcomes.ravel())
        self.dim = self.lines.size

        self.comparisons = np.sort(
            np.rec.fromarrays(
                [times, labels1, labels2, outcomes],
                dtype=[
                    ('time', 'M8[us]'),
                    ('label1',   'U8'),
                    ('label2',   'U8'),
                    ('outcome',   '?', self.dim),
                ]
            ), axis=0
        )

        self.rtg_hcap = self.rating_handicap(self.comparisons.outcome)

        self.k = (self.optimize_k() if k is None else k)

        self.ratings, self.error = self.rate(self.k)

    def optimize_k(self, kmin=1e-4, kmax=1.0):
        """
        Optimize the k update-factor to minimize predictive error.

        """
        res = minimize_scalar(lambda x: self.rate(x)[1], bounds=(kmin, kmax))

        return res.x

    def rating_handicap(self, outcomes):
        """
        Naive prior that label1 beats label2.

        """
        prob = np.count_nonzero(outcomes, axis=0) / np.size(outcomes, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        return np.sqrt(2)/2 * erfinv(2*prob - 1)

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

        try:
            return ratings[ratings['time'] < time][-1]
        except IndexError:
            return {'over': self.rtg_hcap, 'under': -self.rtg_hcap}

    def rate(self, k):
        """
        Apply the Elo model to the list of binary comparisons.

        """
        rtg_over = defaultdict(lambda: self.rtg_hcap)
        rtg_under = defaultdict(lambda: -self.rtg_hcap)
        last_played = defaultdict(lambda: self.comparisons['time'].min())

        ratings = defaultdict(list)
        error = 0

        # loop over all binary comparisons
        for (time, label1, label2, outcome) in self.comparisons:

            # lookup label ratings
            rating1 = rtg_over[label1]
            rating2 = rtg_under[label2]

            # apply rating decay to rating1
            x1 = self.regress(time - last_played[label1])
            rating1 = x1*rating1 + (1 - x1)*self.rtg_hcap

            # apply rating decay to rating2
            x2 = self.regress(time - last_played[label2])
            rating2 = x2*rating2 - (1 - x2)*self.rtg_hcap

            # prior prediction and observed outcome
            prior = self.norm_cdf(rating1 - rating2)
            observed = np.where(outcome, 1, 0)

            # rating change
            rating_change = k * (observed - prior)
            error += np.square(observed - prior).sum()

            # update current ratings
            rtg_over[label1] = rating1 + rating_change
            rtg_under[label2] = rating2 - rating_change

            # record current ratings
            for label in label1, label2:
                ratings[label].append((
                    time,
                    rtg_under[label].copy(),
                    rtg_over[label].copy(),
                ))
                last_played[label] = time

        # recast as a structured array for convenience
        for label in ratings.keys():
            ratings[label] = np.array(
                ratings[label],
                dtype=[
                    ('time',  'M8[us]'),
                    ('under', 'f8', self.dim),
                    ('over',  'f8', self.dim),
                ]
            )

        return ratings, error

    def predict_prob(self, time, label1, label2, smooth=0):
        """
        Predict the distribution of values sampled by a comparison between
        label1 and label2.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        rating_diff = rating1['over'] - rating2['under']

        if smooth > 0:
            rating_diff = filters.gaussian_filter1d(
                rating_diff, smooth, mode='nearest')

        return self.lines, self.norm_cdf(rating_diff)

    def predict_mean(self, time, label1, label2, smooth=0):
        """
        Predict the mean value for a comparison between label1 and label2.
        One can use integration by parts to calculate the mean of the CDF,

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        x, F = self.predict_prob(time, label1, label2)

        return np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])

    def predict_perc(self, time, label1, label2, smooth=0, q=[10, 50, 90]):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        q = np.array(q) / 100.

        x, F = self.predict_prob(time, label1, label2, smooth)
        F = np.sort(F)[::-1]

        indices = [np.argmin((F - p)**2) for p in q]

        return x[indices]

    @property
    def predictors(self):
        """
        Generate mean-value predictors for each binary comparison.

        The model predicts the probability that a comparison covers a given
        line, i.e. F = P(value > line).

        One can use integration by parts to calculate the mean of the CDF,

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        predictors = []

        # loop over all binary comparisons
        for (time, label1, label2, outcome) in self.comparisons[-100:]:

            # integration by parts
            mean = self.predict_mean(time, label1, label2)
            predictors.append((time, label1, label2, mean))

        # convert to structured array
        predictors = np.array(
            predictors,
            dtype=[
                ('time',  'M8[us]'),
                ('label1',    'U8'),
                ('label2',    'U8'),
                ('mean',      'f8'),
            ]
        )

        return predictors
