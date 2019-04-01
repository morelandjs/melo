#!/usr/bin/env python3

from __future__ import division

import numpy as np
from scipy import stats


class Melo:
    """
    Margin-dependent Elo (MELO)

    """
    def __init__(self, times, labels1, labels2, values,
                 lines=0, k=0, smooth=0, dist='normal',
                 prior=None, biases=None, regress=None, commutes=False):

        self.times = np.array(times, dtype=str, ndmin=1)
        self.labels1 = np.array(labels1, dtype=str, ndmin=1)
        self.labels2 = np.array(labels2, dtype=str, ndmin=1)
        self.values = np.array(values, dtype=float, ndmin=1)
        self.lines = np.array(lines, dtype=float, ndmin=1)

        self.k = k

        if smooth < 0:
            raise ValueError('smooth must be non-negative')

        self.smooth = max(smooth, 1e-9)

        if dist not in ['cauchy', 'logistic', 'normal']:
            raise ValueError('no such distribution')

        self.dist = {
            'cauchy': stats.cauchy,
            'logistic': stats.logistic,
            'normal': stats.norm,
        }[dist]

        if biases is None:
            self.biases = np.zeros_like(values)
        elif np.isscalar(biases):
            self.biases = biases * np.ones_like(values)
        else:
            self.biases = np.array(biases, dtype=float, ndmin=1)

        if regress is None:
            self.regress = lambda t: 0
        elif callable(regress):
            self.regress = regress
        else:
            raise ValueError('regress must be a callable function.')

        if commutes is True:
            self.conjugate = lambda x: x
        else:
            if all(self.lines != -np.flip(self.lines)):
                raise ValueError(
                    "lines must be symmetric when commutes is False"
                )
            self.conjugate = lambda x: -(
                np.fliplr(x) if x.ndim > 1 else np.flip(x)
            )

        self.comparisons = np.sort(
            np.rec.fromarrays([
                self.times,
                self.labels1,
                self.labels2,
                self.values,
                self.biases,
            ], dtype=[
                ('time', 'M8[us]'),
                ('label1',  'U64'),
                ('label2',  'U64'),
                ('value',    'f8'),
                ('bias',     'f8'),
            ] ), order='time', axis=0)

        self.labels = np.union1d(self.labels1, self.labels2)
        self.first_update = self.comparisons.time.min()
        self.last_update = self.comparisons.time.max()
        self.prior_rating = self.infer_prior_rating(self.values, self.lines)
        self.ratings = self.rate()

    def infer_prior_rating(self, values, lines):
        """
        Infer typical 'prior' rating from the population.

        """
        prob = np.mean(values[:, np.newaxis] - lines > 0, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        return -0.5*self.dist.isf(prob)

    def evolve(self, rating, elapsed_time):
        """
        Evolve rating to a future time and regress to the mean.

        """
        x = self.regress(elapsed_time)

        return (1 - x)*rating + x*self.prior_rating

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        time = np.datetime64(time)

        if label == 'Average':
            return self.prior_rating
        elif label not in self.ratings:
            raise ValueError("no such label in comparisons")

        ratings = self.ratings[label]

        condition = ratings.time < time

        if any(condition):
            rating = ratings[condition][-1]
            elapsed_time = time - rating.time
            return self.evolve(rating.rating, elapsed_time)

        return self.prior_rating

    def rate(self):
        """
        Calculate Elo ratings from the comparison data

        """
        # complete record of each label's ratings
        ratings = {label: [] for label in self.labels}

        # temporary variable to store each label's last rating
        prev_update = {
            label: (self.first_update, self.prior_rating)
            for label in self.labels
        }

        # loop over all binary comparisons
        for (time, label1, label2, value, bias) in self.comparisons:

            # query ratings
            rating1, rating2 = [
                self.evolve(prev_rating, time - prev_time)
                for prev_time, prev_rating in [
                        prev_update[label1],
                        prev_update[label2],
                ]
            ]

            # expected and observed comparison outcomes
            rating_diff = rating1 + self.conjugate(rating2) + bias
            expected = self.dist.cdf(rating_diff)
            observed = self.dist.sf(self.lines, loc=value, scale=self.smooth)

            # update current ratings
            rating_change = self.k * (observed - expected)
            rating1 += rating_change
            rating2 += self.conjugate(rating_change)

            # record current ratings
            for label, rating in [(label1, rating1), (label2, rating2)]:
                ratings[label].append((time, rating))
                prev_update[label] = (time, rating)

        # recast as a structured record array for convenience
        for label in ratings.keys():
            ratings[label] = np.rec.array(
                ratings[label],
                dtype=[('time', 'M8[us]'), ('rating', 'f8', self.lines.size)]
            )

        return ratings

    def predict(self, time, label1, label2, bias=0):
        """
        Predict the probability that a comparison between label1 and label2
        covers every possible value of the line.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        rating_diff = rating1 + self.conjugate(rating2) + bias

        return self.lines, self.dist.cdf(rating_diff)

    def probability(self, time, label1, label2, lines=0, bias=0):
        """
        Predict the probability that a comparison between label1 and label2
        covers each value of the line.

        """
        return np.interp(lines, *self.predict(time, label1, label2, bias=bias))

    def percentile(self, time, label1, label2, p=50, bias=0):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        p = np.true_divide(p, 100.0)

        if np.count_nonzero(p < 0.0) or np.count_nonzero(p > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        x, F = self.predict(time, label1, label2, bias=bias)

        perc = np.interp(p, np.sort(1 - F), x)

        return np.asscalar(perc) if np.isscalar(p) else perc

    def quantile(self, time, label1, label2, q=.5, bias=0):
        """
        Predict the quantiles for a comparison between label1 and label2.

        """
        q = np.asarray(q)

        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            raise ValueError("quantiles must be in the range [0, 1]")

        x, F = self.predict(time, label1, label2, bias=bias)

        perc = np.interp(q, np.sort(1 - F), x)

        return np.asscalar(perc) if np.isscalar(q) else perc

    def mean(self, time, label1, label2, bias=0):
        """
        Predict the mean value for a comparison between label1 and label2.

        Calculates the mean of the cumulative distribution function F(x)
        using integration by parts:

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        x, F = self.predict(time, label1, label2, bias=bias)

        return np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])

    def median(self, time, label1, label2, bias=0):
        """
        Predict the median value for a comparison between label1 and label2.

        """
        return self.quantile(time, label1, label2, q=.5, bias=bias)

    def residuals(self, predict='mean', standardize=False):
        """
        Returns an array of observed residuals,

        residual = y_pred - y_obs,

        or standardized residuals,

        std residual = (y_pred - y_obs) / sigma_pred.

        """
        residuals = []

        for (time, label1, label2, observed, bias) in self.comparisons:

            if predict == 'mean':
                predicted = self.mean(time, label1, label2, bias=bias)
            elif predict == 'median':
                predicted = self.quantile(time, label1, label2, q=.5, bias=bias)
            else:
                raise ValueError("predict options are 'mean' and 'median'")

            residual = predicted - observed

            if standardize is True:
                qlo, qhi = self.quantile(time, label1, label2,
                                         q=[.159, .841], bias=bias)
                residual /= .5*abs(qhi - qlo)

            residuals.append(residual)

        return np.array(residuals)

    def entropy(self):
        """
        Returns the simulation's total cross entropy:

        S = -\Sum obs*log(pred) + (1 - obs)*log(1 - pred).

        """
        entropy = 0

        for (time, label1, label2, observed, bias) in self.comparisons:

            lines, pred = self.predict(time, label1, label2, bias=bias)
            obs = np.heaviside(observed - lines, .5)

            entropy += -np.sum(
                obs*np.log(pred) + (1 - obs)*np.log(1 - pred)
            )

        return entropy

    def quantiles(self):
        """
        Returns an array of observed quantiles.

        """
        quantiles = []

        for (time, label1, label2, observed, bias) in self.comparisons:

            quantile = self.probability(time, label1, label2,
                                        lines=observed, bias=bias)

            quantiles.append(quantile)

        return np.array(quantiles)

    def rank(self, time, statistic='mean'):
        """
        Rank labels according to the specified statistic.
        Returns a rank sorted list of (label, rank) pairs.

        """
        if statistic == 'mean':
            ranked_list = [
                (label, self.mean(time, label, 'Average', bias=0))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        elif statistic == 'median':
            ranked_list = [
                (label, self.quantile(time, label, 'Average', q=.5, bias=0))
                for label in np.union1d(self.labels1, self.labels2)
            ]
        else:
            raise ValueError('no such distribution statistic')

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)

    def sample(self, time, label1, label2, bias=0, size=100):
        """
        Draw random samples from the predicted probability distribution.

        """
        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        x, F =  self.predict(time, label1, label2, bias=bias)
        rand = np.random.rand(size)

        return np.interp(rand, np.sort(1 - F), x)
