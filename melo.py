#!/usr/bin/env python3

from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.special import erf, erfinv
from scipy.ndimage import filters


class Melo:
    """
    Melo(times, labels1, labels2, values, lines=0,
         statistics='Fermi', k=0, bias=0, decay=lambda x: 1
    )

    Margin-dependent Elo ratings and predictions.

    This class calculates Elo ratings from pairwise comparisons determined by a
    list of values (outcomes of each comparison) and a specified line (or
    sequence of lines) used to determine the threshold of comparison.

    **Required parameters**

    These must be list-like objects of length *n_comparisons*, where the
    number of comparisons must be the same for all lists.

    - *times* -- comparison time stamp

    - *labels1* -- first entity's label name

    - *labels2* -- second entity's label name

    - *values* -- comparison value

    **Optional parameters**

    - *lines* -- comparison threshold line(s)

    - *statistics* -- behavior of comparisons under label exchange

    - *k* -- rating update factor

    - *bias* -- bias (shift) ratings toward either label

    - *decay* -- function used to regress ratings toward the mean

    """
    def __init__(self, times, labels1, labels2, values, lines=0,
                 statistics='Fermi', k=0, bias=0, decay=lambda x: 1):

        self.times = np.array(times, dtype=str)
        self.labels1 = np.array(labels1, dtype=str)
        self.labels2 = np.array(labels2, dtype=str)
        self.values = np.array(values, dtype=float)
        self.lines = np.array(lines, dtype=float, ndmin=1)

        if statistics == 'Fermi':
            if all(self.lines != -np.flip(self.lines)):
                raise ValueError(
                    'lines must be symmetric about zero when statistics=Fermi'
                )
            self.conjugate = lambda x: -np.flip(x)
        elif statistics == 'Bose':
            self.conjugate = lambda x: x
        else:
            raise ValueError(
                'valid statistics options are Fermi or Bose'
            )

        self.k = k
        self.bias = bias
        self.decay = decay

        self.dim = self.lines.size
        outcomes = self.values[:, np.newaxis] > self.lines
        outcomes = (outcomes if self.dim > 1 else outcomes.ravel())

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

    def prob_to_cover(self, rating_diff):
        """
        Normal cumulative probability distribution.

        """
        return 0.5*(1 + erf(rating_diff/np.sqrt(2)))

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        if label not in self.ratings:
            return self.null_rtg

        ratings = self.ratings[label]
        times = ratings['time'] < time

        if times.sum() > 0:
            return ratings[times][-1]['rating']
        else:
            return self.null_rtg

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
            rating1, rating2 = [
                self.null_rtg + self.decay(elapsed) * (rating - self.null_rtg)
                for elapsed, rating in [
                        (time - last_update[label], R[label].copy())
                        for label in [label1, label2]
                ]
            ]

            # prior prediction and observed outcome
            rating_diff = rating1 + self.conjugate(rating2) + self.bias
            prior = self.prob_to_cover(rating_diff)
            observed = np.where(outcome, 1, 0)

            # rating change
            rating_change = k * (observed - prior)

            # update current ratings
            R[label1] = rating1 + rating_change
            R[label2] = rating2 + self.conjugate(rating_change)

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

        rating_diff = rating1 + self.conjugate(rating2) + self.bias

        if smooth > 0:
            rating_diff = filters.gaussian_filter1d(
                rating_diff, smooth, mode='nearest')

        return self.lines, self.prob_to_cover(rating_diff)

    def predict_mean(self, time, label1, label2, smooth=0):
        """
        Predict the mean value for a comparison between label1 and label2.
        One can use integration by parts to calculate the mean of the CDF,

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        x, F = self.predict_prob(time, label1, label2, smooth)

        return np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])

    def predict_perc(self, time, label1, label2, q=50, smooth=0):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        x, F = self.predict_prob(time, label1, label2, smooth)

        qvec = np.array(q, ndmin=1) / 100
        indices = np.argmin((np.sort(1 - F)[:, np.newaxis] - qvec)**2, axis=0)

        return x[indices][0] if np.isscalar(q) else x[indices]

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
            predictors.append((time, label1, label2, mean, value))

        # convert to structured array
        predictors = np.array(
            predictors,
            dtype=[
                ('time',  'M8[us]'),
                ('label1',    'U8'),
                ('label2',    'U8'),
                ('mean',      'f8'),
                ('value',     'f8'),
            ]
        )

        return predictors

    def rank(self, time, moment='mean'):
        """
        Rank labels according to expected mean/median expected for a
        comparison with an average label.

        """
        if moment == 'mean':
            ranked_list = [
                (label, self.predict_mean(time, label, 'avg'))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        elif moment == 'median':
            ranked_list = [
                (label, self.predict_perc(time, label, 'avg', q=50))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        else:
            raise ValueError('no such distribution moment')

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)
