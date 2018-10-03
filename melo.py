#!/usr/bin/env python3

from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.special import erf, erfinv
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
                 k=0, bias=0, decay=lambda: 1, commutes=False):

        self.values = np.array(values, dtype=float)
        self.lines = np.array(lines, dtype=float, ndmin=1)
        self.lines = np.unique(np.append(-self.lines, self.lines))
        self.lines.sort()

        self.k = k
        self.bias = bias
        self.decay = decay

        outcomes = np.tile(
            self.values[:, np.newaxis], self.lines.size
        ) > self.lines

        outcomes = (outcomes if self.lines.size > 1 else outcomes.ravel())
        self.dim = self.lines.size

        self.comparisons = np.sort(
            np.rec.fromarrays(
                [times, labels1, labels2, values, outcomes],
                dtype=[
                    ('time', 'M8[us]'),
                    ('label1',   'U8'),
                    ('label2',   'U8'),
                    ('value',    'f8'),
                    ('outcome',   '?', self.dim),
                ]
            ), axis=0
        )

        self.oldest = self.comparisons['time'].min()

        self.null_rtg = self.null_rating(self.comparisons.outcome)

        self.ratings = self.rate(self.k)

    def null_rating(self, outcomes):
        """
        Assuming all labels are equal, calculate the probability that a
        comparison between label1 and label2 covers each line, i.e.

        prob = P(value1 - value2 > line).

        This probability is used to calculate a default rating difference,

        default_rtg_diff = sqrt(2)*erfiv(2*prob - 1).

        This function then returns half the default rating difference.

        """
        prob = np.sum(outcomes, axis=0) / np.size(outcomes, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        rtg_diff = np.sqrt(2)*erfinv(2*prob - 1) - self.bias

        return .5*rtg_diff

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
        times = ratings['time'] < time

        if times.sum() > 0:
            return ratings[times][-1]['rating']
        else:
            return self.null_rtg

    def regress(self, rating, elapsed_time):
        """
        Regress rating to it's null value as a function of elapsed time.

        """
        factor = self.decay(elapsed_time)

        return self.null_rtg + factor * (rating - self.null_rtg)

    def rate(self, k):
        """
        Apply the Elo model to the list of binary comparisons.

        """
        R = defaultdict(lambda: self.null_rtg)
        last_update = defaultdict(lambda: self.oldest)

        ratings = defaultdict(list)

        # loop over all binary comparisons
        for (time, label1, label2, value, outcome) in self.comparisons:

            # look up ratings
            rating1 = self.regress(R[label1].copy(), time - last_update[label1])
            rating2 = self.regress(R[label2].copy(), time - last_update[label2])

            # prior prediction and observed outcome
            rating_diff = rating1 - np.flip(rating2) + self.bias
            prior = self.norm_cdf(rating_diff)
            observed = np.where(outcome, 1, 0)

            # rating change
            rating_change = k * (observed - prior)

            # update current ratings
            R[label1] = rating1 + rating_change
            R[label2] = rating2 - np.flip(rating_change)

            # record current ratings
            for label in label1, label2:
                ratings[label].append((time, R[label].copy()))
                last_update[label] = time

        # recast as a structured array for convenience
        for label in ratings.keys():
            ratings[label] = np.array(
                ratings[label],
                dtype=[
                    ('time',  'M8[us]'),
                    ('rating', 'f8', self.dim),
                ]
            )

        return ratings

    def predict_prob(self, time, label1, label2, smooth=0):
        """
        Predict the distribution of values sampled by a comparison between
        label1 and label2.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        rating_diff = rating1 - np.flip(rating2) + self.bias

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

    def predict_perc(self, time, label1, label2, smooth=0, q=50):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        x, F = self.predict_prob(time, label1, label2, smooth)
        F = np.sort(F)[::-1]

        q = np.array(q, ndmin=1) / 100
        indices = np.argmin((F[:, np.newaxis] - q)**2, axis=0)

        return x[indices]

    def predictors(self, smooth=0, thin=1):
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
        for (time, label1, label2, value, outcome) in self.comparisons[::thin]:

            # integration by parts
            mean = self.predict_mean(time, label1, label2, smooth=smooth)
            quantiles = self.predict_perc(time, label1, label2, smooth=smooth,
                                          q=[10, 30, 50, 70, 90])

            predictors.append((time, label1, label2, mean, value, quantiles))

        # convert to structured array
        predictors = np.array(
            predictors,
            dtype=[
                ('time',  'M8[us]'),
                ('label1',    'U8'),
                ('label2',    'U8'),
                ('mean',      'f8'),
                ('value',     'f8'),
                ('quantiles', 'f8', 5),
            ]
        )

        return predictors
