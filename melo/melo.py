#!/usr/bin/env python3

from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.ndimage import filters
from scipy.special import erf, erfinv
from scipy.stats import norm


class Melo:
    """
    Melo(times, labels1, labels2, values, lines=0,
         mode='Fermi', k=0, bias=0, decay=lambda x: 1
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

    - *mode* -- behavior of comparisons under label exchange

    - *k* -- rating update factor

    - *bias* -- bias (shift) ratings toward either label

    - *decay* -- function used to regress ratings toward the mean

    """
    def __init__(self, times, labels1, labels2, values, lines=0,
                 mode='Fermi', k=0, bias=0, decay=lambda x: 1):

        self.times = np.array(times, dtype=str, ndmin=1)
        self.labels1 = np.array(labels1, dtype=str, ndmin=1)
        self.labels2 = np.array(labels2, dtype=str, ndmin=1)
        self.labels = np.union1d(labels1, labels2)
        self.values = np.array(values, dtype=float, ndmin=1)
        self.lines = np.array(lines, dtype=float, ndmin=1)

        if mode == 'Fermi':
            if all(self.lines != -np.flip(self.lines)):
                raise ValueError(
                    'lines must be symmetric about zero when mode=Fermi'
                )
            self.conjugate = lambda x: -(np.fliplr(x) if x.ndim > 1 else np.flip(x))
        elif mode == 'Bose':
            self.conjugate = lambda x: x
        else:
            raise ValueError(
                'valid mode options are Fermi or Bose'
            )

        if k < 0:
            raise ValueError('rating update factor k must be non-negative')

        self.mode = mode
        self.k = k
        self.bias = bias
        self.decay = decay

        self.dim = self.lines.size
        outcomes = self.values[:, np.newaxis] > self.lines
        self.outcomes = (outcomes if self.dim > 1 else outcomes.ravel())

        self.comparisons = np.sort(
            np.rec.fromarrays([
                self.times,
                self.labels1,
                self.labels2,
                self.values,
                self.outcomes
            ], dtype=[
                ('time', 'M8[us]'),
                ('label1',   'U8'),
                ('label2',   'U8'),
                ('value',    'f8'),
                ('outcome',   '?', self.dim),
            ] ), axis=0)

        self.oldest = self.comparisons['time'].min()
        self.null_rtg = self.null_rating(self.comparisons.outcome)
        self.ratings = self.rate(self.k)

    def null_rating(self, outcomes, q=50):
        """
        Assuming all labels are equal, calculate the probability that a
        comparison between label1 and label2 covers each line, i.e.

        prob = P(value1 - value2 > line).

        This probability is used to calculate a default rating difference,

        default_rtg_diff = sqrt(2)*erfiv(2*prob - 1).

        This function then returns half the default rating difference.

        """
        prob = np.mean(outcomes, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        rtg_diff = np.sqrt(2)*erfinv(2*prob - 1) - self.bias

        return .5*rtg_diff

    def regress(self, rating, elapsed):
        """
        Regress rating to the mean as a function of time.

        """
        return self.null_rtg + self.decay(elapsed) * (rating - self.null_rtg)

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        time = np.datetime64(time)

        if label == 'NULL':
            return self.null_rtg
        elif label not in self.ratings:
            raise ValueError("no such label in the list of comparisons")

        ratings = self.ratings[label]

        condition = ratings['time'] < time

        if any(condition):
            prior_rating = ratings[condition][-1]
            elapsed = time - prior_rating['time']
            return self.regress(prior_rating['rating'], elapsed)

        return self.null_rtg

    def rate(self, k):
        """
        Apply the margin-dependent Elo model to the list of binary comparisons.

        """
        R = defaultdict(lambda: self.null_rtg)
        last_update = defaultdict(lambda: self.oldest)

        ratings = defaultdict(list)

        # loop over all binary comparisons
        for (time, label1, label2, value, outcome) in self.comparisons:

            # look up ratings
            rating1, rating2 = [
                self.regress(rating, elapsed) for rating, elapsed in [
                    (R[label].copy(), time - last_update[label])
                    for label in [label1, label2]
                ]
            ]

            # prior prediction and observed outcome
            rating_diff = rating1 + self.conjugate(rating2) + self.bias
            prior = norm.cdf(rating_diff)
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

    def predict(self, time, label1, label2, smooth=0, neutral=False):
        """
        Predict the probability that a comparison between label1 and label2
        covers every possible value of the line.

        """
        if smooth < 0:
            raise ValueError("smoothing kernel must be non-negative")

        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)
        bias = (0 if neutral else self.bias)

        rating_diff = rating1 + self.conjugate(rating2) + bias

        if smooth > 0:
            rating_diff = filters.gaussian_filter1d(
                rating_diff, smooth, mode='nearest')

        return self.lines, norm.cdf(rating_diff)

    def probability(self, time, label1, label2, lines=0, smooth=0, neutral=False):
        """
        Predict the probability that a comparison between label1 and label2
        covers each value of the line.

        """
        return np.interp(lines, *self.predict(time, label1, label2, smooth, neutral))

    def mean(self, time, label1, label2, smooth=0, neutral=False):
        """
        Predict the mean value for a comparison between label1 and label2.

        Calculates the mean of the cumulative distribution function F(x)
        using integration by parts:

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        if smooth < 0:
            raise ValueError("smoothing kernel must be non-negative")

        x, F = self.predict(time, label1, label2, smooth, neutral)

        return np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])

    def percentile(self, time, label1, label2, q=50, smooth=0, neutral=False):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        q = np.true_divide(q, 100.0)

        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        if smooth < 0:
            raise ValueError("smoothing kernel must be non-negative")

        x, F = self.predict(time, label1, label2, smooth, neutral)

        perc = np.interp(q, np.sort(1 - F), x)

        return np.asscalar(perc) if np.isscalar(q) else perc

    def statistics(self, smooth=0, thin=1):
        """
        Calculate predicted mean and median prior to every binary comparison.
        Returns a structured array of comparisons and comparison statistics.

        """
        if smooth < 0:
            raise ValueError("smoothing kernel must be non-negative")

        if not (1 <= thin < self.comparisons.size) or not isinstance(thin, int):
            raise ValueError("thin must be an int bounded by the array dim")

        comparisons = []

        # loop over all binary comparisons
        for (time, label1, label2, value, outcome) in self.comparisons[::thin]:

            # integration by parts
            mean = self.mean(time, label1, label2, smooth=smooth)
            median = self.percentile(time, label1, label2, smooth=smooth)
            comparisons.append((time, label1, label2, mean, median, value))

        # convert to structured array
        comparisons = np.array(
            comparisons,
            dtype=[
                ('time',  'M8[us]'),
                ('label1',    'U8'),
                ('label2',    'U8'),
                ('mean',      'f8'),
                ('median',    'f8'),
                ('value',     'f8'),
            ]
        )

        return comparisons

    def rank(self, time, statistic='mean'):
        """
        Rank labels according to the specified statistic.
        Returns a rank sorted list of (label, rank) pairs.

        """
        if statistic == 'mean':
            ranked_list = [
                (label, self.mean(time, label, 'NULL', neutral=True))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        elif statistic == 'median':
            ranked_list = [
                (label, self.percentile(time, label, 'NULL', q=50, neutral=True))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        else:
            raise ValueError('no such distribution statistic')

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)

    def sample(self, time, label1, label2, smooth=0, neutral=False, size=100):
        """
        Draft random samples from the predicted probability distribution.

        """
        if smooth < 0:
            raise ValueError("smoothing kernel must be non-negative")

        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        x, F =  self.predict(time, label1, label2, smooth, neutral)
        rand = np.random.rand(size)

        return np.interp(rand, np.sort(1 - F), x)
